"""CLI entry point for tokenisers.benchmarks.colab."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from tokenisers.benchmarks.colab import (
    BpbEvaluator,
    ByteBudgetTrainer,
    ColabOrchestrator,
    ProxyLanguageModel,
)
from tokenisers.data.pipeline import DataPipeline, OfflineFallbackGenerator, SigmorphonLoader
from tokenisers.models import (
    BpeTokenizer,
    MorphemeBpeTokenizer,
    SuperBpeTokenizer,
    UnigramTokenizer,
)


REMOTE_WORKER_TEMPLATE = '''
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

PAYLOAD = __PAYLOAD_JSON__

class ProxyLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=2, n_heads=4, d_ffn=256, max_seq_len=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ffn,
                batch_first=True, norm_first=True
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, x):
        seq_len = min(x.shape[1], self.max_seq_len)
        x = x[:, :seq_len]
        h = self.embedding(x)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=True)
        return self.lm_head(self.norm(h))

@torch.no_grad()
def eval_bpb(model, token_ids, utf8_bytes, seq_len, device):
    model.eval()
    if len(token_ids) < 2:
        token_ids = token_ids * 2
    total_nll = 0.0
    total_toks = 0
    for i in range(0, len(token_ids) - 1, seq_len):
        chunk = token_ids[i : i + seq_len + 1]
        if len(chunk) < 2:
            continue
        t = torch.tensor([chunk], dtype=torch.long, device=device)
        logits = model(t[:, :-1])
        targets = t[:, 1:]
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum")
        total_nll += float(nll.item())
        total_toks += targets.numel()
    loss_nats = total_nll / max(1, total_toks)
    bpt = utf8_bytes / max(1, len(token_ids))
    bpb = loss_nats / (math.log(2.0) * bpt)
    return {"loss_nats": loss_nats, "bytes_per_token": bpt, "bpb": bpb}

def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hw_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    byte_budget = PAYLOAD["byte_budget"]
    seq_len = PAYLOAD["seq_len"]
    batch_size = PAYLOAD["batch_size"]
    results = {"device": device, "hardware": hw_name, "byte_budget": byte_budget, "models": {}}

    for name, item in PAYLOAD["tokenizers"].items():
        torch.manual_seed(42)
        v_size = item["vocab_size"]
        train_ids = item["train_ids"]
        train_bytes = item["train_bytes"]
        bpt = max(1e-3, train_bytes / max(1, len(train_ids)))

        model = ProxyLM(vocab_size=v_size, d_model=128, n_layers=2, n_heads=4, d_ffn=256).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

        if len(train_ids) < seq_len + 1:
            train_ids = (train_ids * ((seq_len + 2) // max(1, len(train_ids)) + 2))

        bytes_consumed = 0
        tokens_consumed = 0
        cursor = 0
        steps = 0
        curve = []
        model.train()

        while bytes_consumed < byte_budget:
            batch = []
            for _ in range(batch_size):
                if cursor + seq_len + 1 > len(train_ids):
                    cursor = 0
                batch.append(train_ids[cursor : cursor + seq_len + 1])
                cursor += seq_len
            t = torch.tensor(batch, dtype=torch.long, device=device)
            opt.zero_grad()
            logits = model(t[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), t[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            step_toks = batch_size * seq_len
            step_bytes = max(1, int(round(step_toks * bpt)))
            tokens_consumed += step_toks
            bytes_consumed += step_bytes
            steps += 1
            if steps == 1 or bytes_consumed >= byte_budget or steps % 5 == 0:
                curve.append({
                    "step": steps,
                    "bytes_consumed": bytes_consumed,
                    "tokens_consumed": tokens_consumed,
                    "train_loss_nats": round(float(loss.item()), 4),
                    "train_bpb": round(float(loss.item()) / (math.log(2.0) * bpt), 4),
                })

        val_domains = {}
        for dom_name, dom_data in item["val_domains"].items():
            val_domains[dom_name] = eval_bpb(
                model, dom_data["ids"], dom_data["bytes"], seq_len, device
            )

        macro_bpb = sum(d["bpb"] for d in val_domains.values()) / max(1, len(val_domains))
        results["models"][name] = {
            "vocab_size": v_size,
            "steps": steps,
            "bytes_consumed": bytes_consumed,
            "tokens_consumed": tokens_consumed,
            "train_bytes_per_token": bpt,
            "val_domains": val_domains,
            "macro_val_bpb": macro_bpb,
            "curve": curve,
        }

    print(json.dumps(results))

if __name__ == "__main__":
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrinsic Proxy-LM Bits-Per-Byte Benchmark Suite")
    parser.add_argument("--vocab-size", type=int, default=400, help="Target vocabulary size")
    parser.add_argument("--byte-budget", type=int, default=150_000, help="Fixed UTF-8 training byte budget per tokenizer")
    parser.add_argument("--gpu", type=str, default="T4", help="Colab GPU accelerator type (e.g. T4, L4, A100)")
    parser.add_argument("--local", action="store_true", help="Force local execution without Colab")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results/colab", help="Output directory")
    args = parser.parse_args()

    pipeline = DataPipeline(force_offline=True)
    corpus = pipeline.sample_preset("smoke", max_bytes=80_000)
    morph_samples = SigmorphonLoader().load_builtin_samples()
    lexicon = {item["word"]: item["segments"] for item in morph_samples}

    v_size = args.vocab_size
    v_switch = max(260, int(v_size * 0.75))

    models: Dict[str, Any] = {
        "bpe_frequency": BpeTokenizer.train(corpus=corpus, vocab_size=v_size, scorer="frequency"),
        "wordpiece_pmi": BpeTokenizer.train(corpus=corpus, vocab_size=v_size, scorer="wordpiece"),
        "mdl_bits": BpeTokenizer.train(corpus=corpus, vocab_size=v_size, scorer="mdl"),
        "length_penalized": BpeTokenizer.train(corpus=corpus, vocab_size=v_size, scorer="length_penalized", beta=0.5),
        "super_bpe": SuperBpeTokenizer.train(corpus=corpus, v_switch=v_switch, v_final=v_size),
        "morpheme_bpe": MorphemeBpeTokenizer.train(corpus=corpus, vocab_size=v_size, morpheme_lexicon=lexicon, lambda_weight=2.0),
        "unigram_lm": UnigramTokenizer.train(corpus=corpus, vocab_size=v_size),
    }

    gen = OfflineFallbackGenerator()
    train_text = "\n".join(corpus[:40])
    train_bytes = len(train_text.encode("utf-8"))

    val_texts = {
        "english": "\n".join(gen.SAMPLE_PROSE * 4),
        "multilingual": "\n".join(gen.SAMPLE_MULTILINGUAL * 3),
        "code": "\n".join(gen.SAMPLE_CODE * 4),
    }

    payload: Dict[str, Any] = {
        "byte_budget": args.byte_budget,
        "seq_len": 64,
        "batch_size": 8,
        "tokenizers": {},
    }

    for name, tok in models.items():
        t_ids = tok.encode(train_text)
        val_doms = {}
        for d_name, d_txt in val_texts.items():
            val_doms[d_name] = {
                "ids": tok.encode(d_txt),
                "bytes": len(d_txt.encode("utf-8")),
            }
        payload["tokenizers"][name] = {
            "vocab_size": tok.vocab_size,
            "train_ids": t_ids,
            "train_bytes": train_bytes,
            "val_domains": val_doms,
        }

    # Add BPE-Dropout stochastic training variant
    bpe_tok = models["bpe_frequency"]
    payload["tokenizers"]["bpe_dropout_p0.1"] = {
        "vocab_size": bpe_tok.vocab_size,
        "train_ids": bpe_tok.encode(train_text, dropout=0.1, seed=42),
        "train_bytes": train_bytes,
        "val_domains": payload["tokenizers"]["bpe_frequency"]["val_domains"],
    }

    script_code = REMOTE_WORKER_TEMPLATE.replace("__PAYLOAD_JSON__", json.dumps(payload))
    orchestrator = ColabOrchestrator(mock=args.local)
    results = orchestrator.run_ephemeral_script(script_code, gpu=args.gpu, timeout=240)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "extrinsic_bpb_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(out_dir / "extrinsic_bpb_leaderboard.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tokenizer",
            "vocab_size",
            "bytes_consumed",
            "tokens_consumed",
            "train_bpt",
            "val_english_bpb",
            "val_multilingual_bpb",
            "val_code_bpb",
            "macro_val_bpb",
        ])
        for m_name, m_res in results["models"].items():
            vd = m_res["val_domains"]
            writer.writerow([
                m_name,
                m_res["vocab_size"],
                m_res["bytes_consumed"],
                m_res["tokens_consumed"],
                f"{m_res['train_bytes_per_token']:.4f}",
                f"{vd['english']['bpb']:.4f}",
                f"{vd['multilingual']['bpb']:.4f}",
                f"{vd['code']['bpb']:.4f}",
                f"{m_res['macro_val_bpb']:.4f}",
            ])

    md_lines = [
        "# Extrinsic Proxy-LM Validation Bits-Per-Byte (BPB) Leaderboard",
        "",
        f"- **Compute Backend**: `{results['hardware']}` (`{results['device']}`)",
        f"- **Fixed UTF-8 Training Budget**: `{results['byte_budget']:,}` bytes per tokenizer",
        "",
        "| Tokenizer Variant | Train Bytes/Token | Tokens Trained (FLOPs proxy) | English BPB (↓) | Multilingual BPB (↓) | Code BPB (↓) | Macro Val BPB (↓) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    sorted_models = sorted(
        results["models"].items(),
        key=lambda kv: kv[1]["macro_val_bpb"],
    )
    for m_name, m_res in sorted_models:
        vd = m_res["val_domains"]
        md_lines.append(
            f"| `{m_name}` | {m_res['train_bytes_per_token']:.3f} | {m_res['tokens_consumed']:,} | "
            f"{vd['english']['bpb']:.3f} | {vd['multilingual']['bpb']:.3f} | {vd['code']['bpb']:.3f} | "
            f"**{m_res['macro_val_bpb']:.3f}** |"
        )
    md_path = out_dir / "extrinsic_bpb_leaderboard.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
