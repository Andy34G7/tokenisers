"""Proxy-LM tier: init/warmup, sliding-window BPB, seeds, one implementation, truthful backend
(B11, B12, B13, B15, B18). Everything runs on CPU with tiny models."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tokenisers.benchmarks import colab  # noqa: E402
from tokenisers.benchmarks.colab import lm  # noqa: E402


def test_initial_loss_is_ln_vocab():
    """B13: with N(0, 0.02) init (tied embeddings included) the untrained loss is ~ln V, not ~100 nats."""
    for vocab in (1000, 16384):
        torch.manual_seed(0)
        model = lm.ProxyLanguageModel(vocab_size=vocab, d_model=128, n_layers=2, n_heads=4, d_ffn=256)
        x = torch.randint(0, vocab, (4, 65))
        logits = model(x[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab), x[:, 1:].reshape(-1)).item()
        assert abs(loss - math.log(vocab)) < 0.1, (vocab, loss)


def test_trainer_warms_up_and_records_curve():
    """B13: the learning rate ramps up linearly, then decays; the first logged loss is ~ln V."""
    torch.manual_seed(0)
    model = lm.ProxyLanguageModel(vocab_size=50, d_model=16, n_layers=1, n_heads=2, d_ffn=32)
    trainer = lm.ByteBudgetTrainer(model, byte_budget=4000, lr=1e-2, warmup_steps=5)
    ids = np.arange(2000) % 50
    curve = trainer.train_on_token_stream(ids, total_utf8_bytes=8000, seq_len=16, batch_size=4, log_every=1)
    lrs = [c["lr"] for c in curve]
    assert lrs[:5] == pytest.approx([2e-3, 4e-3, 6e-3, 8e-3, 1e-2])
    assert lrs[-1] < 1e-2
    assert abs(curve[0]["train_loss_nats"] - math.log(50)) < 0.1
    assert trainer.bytes_consumed >= 4000 and trainer.epochs < 1  # ~1000 of the 2000 tokens: no repetition


class LateSighted(torch.nn.Module):
    """Predicts the next id of an arange stream perfectly, but only at window positions >= k."""

    def __init__(self, vocab, k):
        super().__init__()
        self.vocab, self.k = vocab, k
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        logits = torch.zeros(*x.shape, self.vocab)
        nxt = (x + 1) % self.vocab
        pos = torch.arange(x.shape[1]).expand_as(x)
        logits.scatter_(2, nxt.unsqueeze(-1), torch.where(pos >= self.k, 50.0, 0.0).unsqueeze(-1))
        return logits


def test_bpb_scores_every_token_once_with_carried_context():
    """B18: sliding windows score each token after the first exactly once with >= seq_len/2 context,
    so a model that needs 8 tokens of context is blind only in the very first window."""
    vocab, n, seq = 20, 200, 16
    ids = np.arange(n) % vocab
    token_bytes = np.full(vocab, 2)
    res = lm.BpbEvaluator().evaluate_validation_bpb(LateSighted(vocab, k=8), ids, 2 * n, seq_len=seq,
                                                    token_bytes=token_bytes)
    assert res["scored_tokens"] == n - 1
    blind = 8  # only positions 0..7 of the first window
    expected_nll = blind * math.log(vocab)
    assert res["loss_nats"] * res["scored_tokens"] == pytest.approx(expected_nll, rel=1e-3)
    assert res["bpb"] == pytest.approx(expected_nll / (math.log(2) * 2 * (n - 1)), rel=1e-3)


def test_bpb_matches_full_context_when_window_covers_stream():
    torch.manual_seed(0)
    model = lm.ProxyLanguageModel(vocab_size=30, d_model=16, n_layers=1, n_heads=2, d_ffn=32, max_seq_len=128)
    ids = np.random.default_rng(0).integers(0, 30, 60)
    res = lm.BpbEvaluator().evaluate_validation_bpb(model, ids, 60, seq_len=128)
    with torch.no_grad():
        x = torch.from_numpy(ids).long()[None]
        ref = torch.nn.functional.cross_entropy(model(x[:, :-1])[0], x[0, 1:], reduction="sum").item()
    assert res["loss_nats"] * 59 == pytest.approx(ref, rel=1e-5)


def _job(tmp_path: Path, seeds=(0, 1), val_code_len=80) -> Path:
    job = tmp_path / "job"
    job.mkdir(parents=True)
    rng = np.random.default_rng(0)
    cfg = {"lm": {"d_model": 16, "n_layers": 1, "n_heads": 2, "d_ffn": 32, "max_seq_len": 16},
           "train": {"byte_budget": 3000, "seq_len": 16, "batch_size": 4, "lr": 3e-3, "eval_stride": 8},
           "seeds": list(seeds), "domains": ["english", "code"], "tokenizers": {"tok": {"vocab_size": 40}}}
    (job / "config.json").write_text(json.dumps(cfg))
    np.savez(job / "tok.npz", train=rng.integers(0, 40, 2000).astype(np.uint16),
             val_english=rng.integers(0, 40, 100).astype(np.uint16), val_code=rng.integers(0, 40, val_code_len).astype(np.uint16),
             token_bytes=np.full(40, 2, dtype=np.int32))
    return job


def test_run_job_reports_mean_and_sd_over_seeds(tmp_path):
    """B12: >= 2 seeds give mean ± sd per domain and overall; the curve and initial losses are kept."""
    res = colab.run_local(_job(tmp_path, seeds=(0, 1, 2)))
    m = res["models"]["tok"]
    assert res["backend"]["kind"] == "local" and res["backend"]["device"] == "cpu"
    by_seed = list(m["macro_val_bpb_by_seed"].values())
    assert len(by_seed) == 3 and m["macro_val_bpb"] == pytest.approx(np.mean(by_seed))
    assert m["macro_val_bpb_sd"] == pytest.approx(np.std(by_seed, ddof=1))
    assert set(m["val_domains"]) == {"english", "code"} and m["val_domains"]["code"]["scored_tokens"] == 79
    assert all(abs(x - math.log(40)) < 0.1 for x in m["initial_loss_nats"])
    assert m["epochs"] < 1


def test_lm_train_docs_never_cycle():
    """B12: the LM uses each train-split document at most once; a budget larger than the split is an error."""
    from tokenisers.benchmarks.colab.__main__ import _lm_train_docs

    docs = ["x" * 100] * 50
    assert sum(len(d) for d in _lm_train_docs(docs, 1000, 0)) >= 1100
    with pytest.raises(SystemExit, match="byte-budget"):
        _lm_train_docs(docs, 10_000, 0)


@pytest.mark.parametrize("n_tokens", [0, 1])
def test_empty_validation_domain_is_an_error_not_bpb_zero(tmp_path, n_tokens):
    """A domain with nothing to score must not enter the macro average as BPB 0. run_job refuses it
    before training; the evaluator refuses it too."""
    job = _job(tmp_path, seeds=(0,), val_code_len=n_tokens)
    with pytest.raises(ValueError, match=r"\['code'\] have fewer than 2 tokens"):
        colab.run_local(job)
    assert not (job / "result.json").exists()
    model = lm.ProxyLanguageModel(vocab_size=40, d_model=16, n_layers=1, n_heads=2, d_ffn=32)
    with pytest.raises(ValueError, match="too short to score"):
        lm.BpbEvaluator().evaluate_validation_bpb(model, np.zeros(n_tokens, dtype=np.int64), 2 * n_tokens)


def test_single_seed_has_no_standard_deviation(tmp_path):
    """With one seed the sd is undefined: stored as null and rendered without '± 0.000'."""
    from tokenisers.benchmarks.report import render_colab

    res = colab.run_local(_job(tmp_path, seeds=(0,)))
    m = res["models"]["tok"]
    assert m["macro_val_bpb_sd"] is None and m["val_domains"]["english"]["bpb_sd"] is None
    assert res["architecture"] == lm.ARCHITECTURE and "no positional encoding" in lm.ARCHITECTURE
    meta = {"seeds": [0], "val_domains": ["english", "code"], "duplicates": {}, "backend": res["backend"],
            "lm_architecture": res["architecture"]}
    md = render_colab({"meta": meta, "models": res["models"]})
    assert "±" not in md.split("| :--- |")[1] and "(single seed: no sd)" in md
    assert f"**{m['macro_val_bpb']:.3f}**" in md
    assert "no positional encoding" in md


def test_one_implementation_runs_locally_and_remotely(tmp_path, monkeypatch):
    """B15: the colab package re-exports lm.py, and the orchestrator ships lm.py itself to the VM.
    A fake `colab` CLI runs the uploaded driver in a separate interpreter: same code, same numbers."""
    assert colab.ProxyLanguageModel is lm.ProxyLanguageModel and colab.LM_SOURCE == Path(lm.__file__)
    fake = _fake_colab(tmp_path)
    monkeypatch.setenv("FAKE_COLAB_ROOT", str(tmp_path / "vm"))
    remote = colab.ColabOrchestrator(colab_bin=str(fake)).run_job(_job(tmp_path / "a", seeds=(0,)), gpu="T4")
    local = colab.run_local(_job(tmp_path / "b", seeds=(0,)))
    assert remote["backend"]["kind"] == "colab" and remote["backend"]["requested_gpu"] == "T4"
    assert remote["models"]["tok"]["macro_val_bpb"] == pytest.approx(local["models"]["tok"]["macro_val_bpb"])
    calls = [json.loads(line)[0] for line in (tmp_path / "vm" / "calls.log").read_text().splitlines()]
    assert calls[0] == "new" and calls[-1] == "stop" and "exec" in calls
    assert (tmp_path / "vm" / "content" / "lm.py").read_bytes() == colab.LM_SOURCE.read_bytes()


def _fake_colab(tmp_path: Path) -> Path:
    """A stand-in for the colab CLI: /content on the 'VM' is $FAKE_COLAB_ROOT/content."""
    fake = tmp_path / "colab"
    fake.write_text(f"""#!{sys.executable}
import json, os, shutil, subprocess, sys
root = os.environ["FAKE_COLAB_ROOT"]
os.makedirs(root, exist_ok=True)
args = sys.argv[1:]
with open(os.path.join(root, "calls.log"), "a") as f:
    f.write(json.dumps(args) + "\\n")
if args[0] == os.environ.get("FAKE_COLAB_FAIL"):
    sys.exit("simulated colab failure")
vm = lambda p: os.path.join(root, p.lstrip("/"))
if args[0] == "upload":
    os.makedirs(os.path.dirname(vm(args[4])), exist_ok=True)
    shutil.copy(args[3], vm(args[4]))
elif args[0] == "exec":
    code = open(args[args.index("-f") + 1]).read().replace("/content", vm("/content"))
    subprocess.run([sys.executable, "-I", "-c", code], check=True, env={{**os.environ, "CUDA_VISIBLE_DEVICES": ""}})
elif args[0] == "download":
    shutil.copy(vm(args[3]), args[4])
""")
    fake.chmod(0o755)
    return fake


@pytest.fixture
def cli_args(tmp_path):
    return ["--offline", "--cache-dir", str(tmp_path / "cache"), "--train-bytes", "30000", "--vocab-size", "280",
            "--variants", "bpe_frequency", "wp_bpe_base", "--byte-budget", "3000", "--seeds", "0", "1",
            "--seq-len", "16", "--batch-size", "4", "--d-model", "16", "--n-layers", "1", "--n-heads", "2",
            "--d-ffn", "32", "--output-dir", str(tmp_path / "out")]


def test_cli_fails_when_colab_fails(tmp_path, monkeypatch, cli_args, require_tokenisers):
    """B11: no silent local fallback."""
    from tokenisers.benchmarks.colab.__main__ import main

    monkeypatch.setenv("FAKE_COLAB_ROOT", str(tmp_path / "vm"))
    monkeypatch.setenv("FAKE_COLAB_FAIL", "new")
    with pytest.raises(SystemExit, match="Colab run failed"):
        main(cli_args + ["--colab-bin", str(_fake_colab(tmp_path))])
    assert not (tmp_path / "out" / "extrinsic_bpb_results.json").exists()


def test_cli_records_local_fallback_and_val_split(tmp_path, monkeypatch, cli_args, require_tokenisers):
    """B11/B12/B2: an allowed fallback is recorded; the LM is scored on the val split, trained on train text."""
    from tokenisers.benchmarks.colab.__main__ import main
    from tokenisers.data.pipeline import DataPipeline

    monkeypatch.setenv("FAKE_COLAB_ROOT", str(tmp_path / "vm"))
    monkeypatch.setenv("FAKE_COLAB_FAIL", "exec")
    main(cli_args + ["--colab-bin", str(_fake_colab(tmp_path)), "--allow-local-fallback"])
    doc = json.loads((tmp_path / "out" / "extrinsic_bpb_results.json").read_text())
    meta = doc["meta"]
    assert meta["backend"]["kind"] == "local" and meta["fallback"]["from"] == "colab"
    assert "simulated colab failure" in meta["fallback"]["error"]
    assert meta["eval_split"] == "val" and meta["seeds"] == [0, 1]
    assert set(doc["models"]) == {"bpe_frequency", "wp_bpe_base", "bpe_dropout_p0.1"}
    md = (tmp_path / "out" / "extrinsic_bpb_leaderboard.md").read_text()
    assert "FALLBACK after Colab failed" in md and "±" in md
    calls = [json.loads(line)[0] for line in (tmp_path / "vm" / "calls.log").read_text().splitlines()]
    assert calls[-1] == "stop"  # the session is released even though exec failed
    ds = DataPipeline(cache_dir=tmp_path / "cache", force_offline=True).prepare(train_bytes=30000)
    val_bytes = sum(len(t.encode()) for t in ds.texts("val"))
    m = doc["models"]["bpe_frequency"]
    assert sum(m["val_domains"][d]["bytes_per_token"] * (m["val_domains"][d]["scored_tokens"] + 1)
               for d in meta["val_domains"]) == pytest.approx(val_bytes)
