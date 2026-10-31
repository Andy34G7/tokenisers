"""CLI: extrinsic proxy-LM bits-per-byte benchmark (Colab GPU, or local with --local)."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Any, Dict, List

import numpy as np

from tokenisers.benchmarks.colab import ColabError, ColabOrchestrator, run_local
from tokenisers.benchmarks.common import (
    DEFAULT_MORPH_LANGS,
    add_common_args,
    assert_disjoint,
    check_memory,
    find_duplicates,
    start_provenance,
    load_dataset,
    load_morphology,
    run_metadata,
    train_suite,
    write_json,
)
from tokenisers.benchmarks.report import render_colab

DOMAINS = ("english", "multilingual", "code")
DROPOUT_VARIANT = "bpe_dropout_p0.1"
# --preset big: defaults only, so any explicit flag still wins. The small tier is the parser's defaults.
# 64 x 512 = 32,768 tokens per step; warmup 2% of steps, cosine decay (ByteBudgetTrainer), bf16 on CUDA.
BIG_PRESET: Dict[str, Any] = {
    "n_layers": 8, "d_model": 512, "n_heads": 8, "d_ffn": 2048, "seq_len": 512, "batch_size": 64,
    "lr": 6e-4, "weight_decay": 0.1, "warmup_frac": 0.02, "bf16": True, "byte_budget": 100_000_000,
    "gpu": "L4", "concurrency": 3, "seeds": [0, 1, 2], "output_dir": "benchmarks/results/colab_big",
    # bpe_dropout_p0.1 is derived from bpe_frequency (kept on by default); wordpiece_pmi, wp_bpe_base and
    # wp_bpe_ft are left out.
    "variants": ["bpe_frequency", "wp_bpe_pig", "wp_bpe_scp", "wp_bpe_npmi", "wp_bpe_ppig", "mdl_bits",
                 "length_penalized", "super_bpe", "morpheme_bpe", "unigram_lm"],
}


def _encode_docs(tok: Any, docs: List[str], dtype: Any, dropout: float | None = None) -> np.ndarray:
    """Ids of every document, concatenated. With ``dropout``, document i uses BPE-dropout seed i."""
    kw = (lambda i: {"dropout": dropout, "seed": i}) if dropout else (lambda i: {})
    parts = [np.asarray(tok.encode(d, **kw(i)), dtype=dtype) for i, d in enumerate(docs)]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)


def _token_bytes(tok: Any) -> np.ndarray:
    return np.array([len(tok.id_to_bytes(i) or b"") for i in range(tok.vocab_size)], dtype=np.int32)


def _lm_train_docs(train: List[str], byte_budget: int, seed: int) -> List[str]:
    """Shuffled train-split documents covering ``byte_budget`` bytes plus 10%, each used once (no
    repetition). They come from the split the tokenizers were trained on, i.e. they are not held out
    from tokenizer training; only the val split is held out."""
    docs = list(train)
    random.Random(seed).shuffle(docs)
    out: List[str] = []
    n = 0
    for d in docs:
        if n >= byte_budget * 1.1:
            break
        out.append(d)
        n += len(d.encode("utf-8"))
    if n < byte_budget * 1.1:
        raise SystemExit(
            f"--byte-budget {byte_budget:,} needs {int(byte_budget * 1.1):,} train bytes but the train split has "
            f"{n:,}; prepare a larger dataset (--train-bytes) or lower the budget"
        )
    return out


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrinsic proxy-LM bits-per-byte benchmark")
    add_common_args(parser, "benchmarks/results/colab")
    parser.add_argument("--byte-budget", type=int, default=10_000_000,
                        help="UTF-8 bytes each proxy LM trains on: train-split documents (the tokenizers' "
                             "training text), each used at most once")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="LM seeds (mean ± sd reported)")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ffn", type=int, default=512)
    parser.add_argument("--gpu", default="T4", help="Colab accelerator (T4, L4, A100, H100)")
    parser.add_argument("--timeout", type=float, default=6 * 3600, help="Colab job wall-clock limit (s)")
    parser.add_argument("--poll-interval", type=float, default=60.0,
                        help="Seconds between status polls of the detached Colab job")
    parser.add_argument("--colab-bin", default=None, help="colab CLI executable (default ~/.local/bin/colab)")
    parser.add_argument("--local", action="store_true", help="Run the proxy LMs locally instead of on Colab")
    parser.add_argument("--allow-local-fallback", action="store_true",
                        help="If Colab fails, run locally and record the fallback in the results")
    parser.add_argument("--preset", choices=["small", "big"], default="small",
                        help="small: the default tier above; big: 8x512 LM, 100 MB budget, L4, 11 variants, "
                             "concurrency 3, output benchmarks/results/colab_big (explicit flags override)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="(variant, seed) runs trained at once on the one GPU, as separate processes")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-frac", type=float, default=None,
                        help="Warmup as a fraction of steps (default: 10%% of steps, at most 200)")
    parser.add_argument("--bf16", action="store_true", help="bf16 autocast when the GPU supports it")
    parser.add_argument("--no-dropout-variant", action="store_true",
                        help=f"Skip {DROPOUT_VARIANT} (bpe_frequency trained on BPE-dropout encodings)")
    if parser.parse_known_args(argv)[0].preset == "big":
        parser.set_defaults(**BIG_PRESET)
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    start = start_provenance(__package__, argv)  # before the tree can change during a long run

    if args.preset == "big" and args.train_bytes < args.byte_budget * 1.1:  # before any work
        sys.exit(f"--preset big needs {int(args.byte_budget * 1.1):,} bytes of LM text but the tokenizer train "
                 f"split is {args.train_bytes:,}. A separate LM text pool (same sources, disjoint from val/test) "
                 "is not implemented yet: see docs/TODO-big-lm.md. Raising --train-bytes would also change "
                 "the tokenizers, so it is not a fix.")
    if not args.skip_memory_check:  # before downloading/preparing anything
        check_memory(args.variants, args.train_bytes)
    ds = load_dataset(args)
    train = ds.texts("train")
    val = {d: ds.texts("val", domain=d) for d in DOMAINS}
    assert_disjoint(train, [t for texts in val.values() for t in texts], "validation")
    lm_docs = _lm_train_docs(train, args.byte_budget, args.data_seed)

    lexicon, _, morph_prov = load_morphology(args, train, DEFAULT_MORPH_LANGS) if "morpheme_bpe" in args.variants \
        else ({}, [], None)
    print(f"Training {len(args.variants)} tokenizers at V={args.vocab_size} on {len(train)} train docs")
    models, records = train_suite(args.variants, train, args.vocab_size, lexicon)
    duplicates = find_duplicates(models)

    job_dir = Path(tempfile.mkdtemp(prefix="tokenisers-bpb-"))
    try:
        tokenizers: Dict[str, Any] = {}
        unique = {n: t for n, t in models.items() if n not in duplicates}
        if not args.no_dropout_variant and "bpe_frequency" in unique:
            unique[DROPOUT_VARIANT] = unique["bpe_frequency"]
        for name, tok in unique.items():
            dtype = np.uint16 if tok.vocab_size <= 65536 else np.uint32
            dropout = 0.1 if name == DROPOUT_VARIANT else None
            arrays = {"train": _encode_docs(tok, lm_docs, dtype, dropout), "token_bytes": _token_bytes(tok)}
            for d, texts in val.items():
                arrays[f"val_{d}"] = _encode_docs(tok, texts, dtype)
            np.savez(job_dir / f"{name}.npz", **arrays)
            tokenizers[name] = {"vocab_size": tok.vocab_size}
        lm = {"d_model": args.d_model, "n_layers": args.n_layers, "n_heads": args.n_heads,
              "d_ffn": args.d_ffn, "max_seq_len": args.seq_len}
        train_cfg = {"byte_budget": args.byte_budget, "seq_len": args.seq_len, "batch_size": args.batch_size,
                     "lr": args.lr, "eval_stride": max(1, args.seq_len // 2)}
        if args.preset != "small" or args.weight_decay != 0.01 or args.warmup_frac is not None or args.bf16:
            train_cfg.update(weight_decay=args.weight_decay, warmup_frac=args.warmup_frac, bf16=args.bf16)
        write_json(job_dir / "config.json", {"lm": lm, "train": train_cfg, "seeds": args.seeds,
                                             "domains": list(DOMAINS), "tokenizers": tokenizers,
                                             "concurrency": args.concurrency})

        fallback = None
        if args.local:
            result = run_local(job_dir)
        else:
            try:
                result = ColabOrchestrator(colab_bin=args.colab_bin).run_job(
                    job_dir, gpu=args.gpu, timeout=args.timeout, poll_interval=args.poll_interval)
            except ColabError as e:
                if not args.allow_local_fallback:
                    sys.exit(f"Colab run failed (pass --local or --allow-local-fallback to run locally): {e}")
                print(f"Colab run failed, running locally: {e}", file=sys.stderr)
                fallback = {"from": "colab", "requested_gpu": args.gpu, "error": str(e)}
                result = run_local(job_dir)
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

    meta = run_metadata(
        args, ds, "proxy_lm_bpb", "val", records, duplicates, start=start,
        backend=result["backend"], fallback=fallback, lm=lm, lm_architecture=result["architecture"],
        train=train_cfg, byte_budget=args.byte_budget,
        seeds=args.seeds, val_domains=list(DOMAINS), lm_train_docs=len(lm_docs),
        lm_seconds=result["seconds"], morphology=morph_prov,
        variants_extra={DROPOUT_VARIANT: "bpe_frequency tokenizer; LM train ids encoded with BPE-dropout p=0.1, "
                                         "validation ids deterministic"} if DROPOUT_VARIANT in tokenizers else {},
    )
    doc = {"meta": meta, "models": result["models"]}
    out = Path(args.output_dir)
    write_json(out / "extrinsic_bpb_results.json", doc)
    md = render_colab(doc)
    (out / "extrinsic_bpb_leaderboard.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
