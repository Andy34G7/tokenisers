"""Big proxy-LM tier (--preset big): config resolution, size, init, per-run files, concurrency, resume.
Everything runs on CPU; only the size/init checks build the full-width model."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tokenisers.benchmarks.colab import lm  # noqa: E402
from tokenisers.benchmarks.colab.__main__ import BIG_PRESET, parse_args  # noqa: E402

from test_bench_proxy_lm import _job  # noqa: E402

SMALL = {"n_layers": 2, "d_model": 128, "n_heads": 4, "d_ffn": 512, "seq_len": 128, "batch_size": 16,
         "lr": 3e-3, "byte_budget": 10_000_000, "seeds": [0, 1, 2], "gpu": "T4", "concurrency": 1,
         "weight_decay": 0.01, "warmup_frac": None, "bf16": False, "output_dir": "benchmarks/results/colab"}


def test_small_preset_is_unchanged():
    args = parse_args([])
    assert {k: getattr(args, k) for k in SMALL} == SMALL
    assert args.preset == "small" and "wordpiece_pmi" in args.variants


def test_big_preset_resolves_and_flags_override():
    args = parse_args(["--preset", "big"])
    assert {k: getattr(args, k) for k in BIG_PRESET} == BIG_PRESET
    assert args.batch_size * args.seq_len == 32_768 and len(args.variants) + 1 == 11  # + bpe_dropout_p0.1
    assert not {"wordpiece_pmi", "wp_bpe_base", "wp_bpe_ft"} & set(args.variants)
    over = parse_args(["--preset", "big", "--gpu", "A100", "--concurrency", "2", "--lr", "1e-3"])
    assert (over.gpu, over.concurrency, over.lr, over.d_model) == ("A100", 2, 1e-3, 512)


def _big_model(vocab=16384):
    torch.manual_seed(0)
    p = BIG_PRESET
    return lm.ProxyLanguageModel(vocab_size=vocab, d_model=p["d_model"], n_layers=p["n_layers"],
                                 n_heads=p["n_heads"], d_ffn=p["d_ffn"], max_seq_len=p["seq_len"])


def test_big_parameter_count_and_initial_loss():
    model = _big_model()
    emb = model.embedding.weight.numel()
    non_emb = sum(p.numel() for p in model.parameters()) - emb  # lm_head is tied to the embedding
    assert emb == 16384 * 512 and abs(emb - 8.4e6) < 0.05e6
    assert 24.5e6 < non_emb < 25.5e6
    ids = torch.randint(0, 16384, (2, 65))
    with torch.no_grad():
        loss = torch.nn.functional.cross_entropy(model(ids[:, :-1]).reshape(-1, 16384), ids[:, 1:].reshape(-1))
    assert abs(float(loss) - math.log(16384)) < 0.1


def test_concurrent_runs_aggregate_like_sequential(tmp_path):
    seq = lm.run_job(str(_job(tmp_path / "a", seeds=(0, 1))), device="cpu")
    job = _job(tmp_path / "b", seeds=(0, 1))
    cfg = json.loads((job / "config.json").read_text())
    cfg["concurrency"] = 2
    (job / "config.json").write_text(json.dumps(cfg))
    par = lm.run_job(str(job), device="cpu")
    assert sorted(p.name for p in (job / "runs").iterdir()) == ["tok__s0.json", "tok__s1.json"]
    a, b = seq["models"]["tok"], par["models"]["tok"]
    assert b["macro_val_bpb_by_seed"] == pytest.approx(a["macro_val_bpb_by_seed"], rel=1e-5)
    assert b["steps"] == a["steps"] and b["initial_loss_nats"] == pytest.approx(a["initial_loss_nats"], rel=1e-5)


def test_job_resumes_from_finished_run_files(tmp_path, monkeypatch):
    """A dropped job loses only the runs in flight: finished run files are reused, not retrained."""
    job = _job(tmp_path, seeds=(0, 1))
    first = lm.run_job(str(job), device="cpu")
    (job / "runs" / "tok__s1.json").unlink()  # s1 was in flight when the session dropped
    calls = []
    real = lm._run_one
    monkeypatch.setattr(lm, "_run_one", lambda d, n, s, dev=None: calls.append((n, s)) or real(d, n, s, dev))
    again = lm.run_job(str(job), device="cpu")
    assert calls == [("tok", 1)]
    assert again["models"]["tok"]["macro_val_bpb_by_seed"] == first["models"]["tok"]["macro_val_bpb_by_seed"]


def test_trainer_warmup_fraction_and_weight_decay():
    model = lm.ProxyLanguageModel(vocab_size=40, d_model=16, n_layers=1, n_heads=2, d_ffn=32)
    tr = lm.ByteBudgetTrainer(model, byte_budget=4000, lr=1e-3, device="cpu", weight_decay=0.1, warmup_frac=0.5)
    curve = tr.train_on_token_stream(list(range(40)) * 50, seq_len=8, batch_size=2,
                                     token_bytes=__import__("numpy").full(40, 2), log_every=1)
    assert tr.optimizer.param_groups[0]["weight_decay"] == 0.1
    lrs = [c["lr"] for c in curve]
    peak = lrs.index(max(lrs))
    assert abs(peak + 1 - round(len(lrs) * 0.5)) <= 1 and lrs[-1] < max(lrs)


def test_big_preset_fails_loudly_without_enough_lm_text(monkeypatch):
    from tokenisers.benchmarks.colab import __main__ as cli

    monkeypatch.setattr(cli, "load_dataset", lambda *a: pytest.fail("loaded data before the LM-text check"))
    with pytest.raises(SystemExit, match="LM text pool"):
        cli.main(["--preset", "big", "--offline"])


def test_finished_runs_are_downloaded_and_reused_after_a_drop(tmp_path, monkeypatch):
    """Each finished run file is downloaded as the job runs; a new session after a drop uploads the
    held run files, and the VM job trains only the missing runs."""
    from tokenisers.benchmarks import colab
    from test_bench_proxy_lm import _fake_colab

    monkeypatch.setenv("FAKE_COLAB_ROOT", str(tmp_path / "vm1"))
    job = _job(tmp_path, seeds=(0, 1))
    first = colab.ColabOrchestrator(colab_bin=str(_fake_colab(tmp_path))).run_job(job, poll_interval=0.05)
    assert sorted(p.name for p in (job / "runs").iterdir()) == ["tok__s0.json", "tok__s1.json"]
    (job / "runs" / "tok__s1.json").unlink()  # s1 was in flight when the session dropped
    monkeypatch.setenv("FAKE_COLAB_ROOT", str(tmp_path / "vm2"))  # a fresh VM
    again = colab.ColabOrchestrator(colab_bin=str(_fake_colab(tmp_path))).run_job(job, poll_interval=0.05)
    log = (tmp_path / "vm2" / "content" / "job.log").read_text()
    assert "tok seed=1" in log and "tok seed=0" not in log, log
    assert again["models"]["tok"]["macro_val_bpb_by_seed"] == first["models"]["tok"]["macro_val_bpb_by_seed"]
