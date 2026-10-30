"""Throughput re-measurement: saved tokenizers are verified by id hash, then timed in several full
passes, locally or on a (fake) Colab CPU runtime through ColabOrchestrator."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from tokenisers.benchmarks import colab, throughput
from tokenisers.benchmarks.report import render_throughput
from test_bench_proxy_lm import _fake_colab  # noqa: E402  (tests/e2e is on sys.path)

VARIANTS = ("bpe_frequency", "super_bpe", "unigram_lm")


@pytest.fixture(scope="module")
def local_run(require_tokenisers, tmp_path_factory):
    """Prepares a job offline (2 vocabulary sizes x 3 variants) and runs it locally through the CLI."""
    tmp = tmp_path_factory.mktemp("throughput")
    throughput.cli(["--job-dir", str(tmp / "job"), "--offline", "--cache-dir", str(tmp / "cache"),
                    "--train-bytes", "30000", "--vocab-sizes", "280", "300", "--variants", *VARIANTS,
                    "--repeats", "1", "--passes", "2", "--output-dir", str(tmp / "out")])
    return tmp / "job", json.loads((tmp / "out" / "throughput.json").read_text())


def test_local_run_verifies_and_times_every_tokenizer_twice(local_run):
    job, doc = local_run
    cfg = json.loads((job / "config.json").read_text())
    assert [(t["vocab_size"], t["name"]) for t in cfg["tokenizers"]] == [(v, n) for v in (280, 300) for n in VARIANTS]
    assert {t["class"] for t in cfg["tokenizers"]} == {"BpeTokenizer", "SuperBpeTokenizer", "UnigramTokenizer"}
    assert set(doc["models"]) == {"280", "300"} and set(doc["models"]["300"]) == set(VARIANTS)
    for m in (m for v in doc["models"].values() for m in v.values()):
        assert len(m["encode_mb_per_sec"]) == len(m["decode_mb_per_sec"]) == 2
        assert min(m["encode_mb_per_sec"] + m["decode_mb_per_sec"]) > 0
    meta = doc["meta"]
    assert meta["backend"]["kind"] == "local" and meta["backend"]["build"] is None
    assert len(meta["pass_load"]) == 2 and meta["pass_load"][0]["before"]["loadavg"]
    assert meta["prepared"]["synthetic_data"] and meta["repeats"] == 1
    md = render_throughput(doc)
    m = doc["models"]["300"]["super_bpe"]
    assert f"| `super_bpe` | {m['vocab_size']} | {m['encode_mb_per_sec'][0]:.2f} / {m['encode_mb_per_sec'][1]:.2f} |" in md
    assert "SYNTHETIC" in md and "run-to-run spread" in md


def _log(tmp_path: Path) -> list:
    return [json.loads(line) for line in (tmp_path / "vm" / "calls.log").read_text().splitlines()]


def _remote(local_run, tmp_path, monkeypatch, tamper=None):
    job = tmp_path / "job"
    shutil.copytree(local_run[0], job, ignore=shutil.ignore_patterns("tokenizers", "result.json"))
    if tamper:
        cfg = json.loads((job / "config.json").read_text())
        cfg["tokenizers"][tamper]["ids_sha256"] = "0" * 64
        (job / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("FAKE_COLAB_ROOT", str(tmp_path / "vm"))
    files = [job / f for f in ("config.json", "docs.json", "tokenizers.tar.gz")]
    return colab.ColabOrchestrator(colab_bin=str(_fake_colab(tmp_path))).run_job(
        job, gpu=None, poll_interval=0.1, script=Path(throughput.__file__), files=files)


def test_colab_cpu_run_ships_the_job_file_and_times_the_same_tokenizers(local_run, tmp_path, monkeypatch):
    res = _remote(local_run, tmp_path, monkeypatch)
    assert res["backend"]["kind"] == "colab" and res["backend"]["requested_gpu"] is None
    assert res["models"].keys() == local_run[1]["models"].keys() and len(res["passes"]) == 2
    calls = _log(tmp_path)
    new = next(c for c in calls if c[0] == "new")
    assert "--gpu" not in new and new[2].startswith("tokenisers-throughput-")  # a CPU runtime
    assert calls[-1][0] == "stop"
    vm = tmp_path / "vm" / "content"
    assert (vm / "throughput.py").read_bytes() == Path(throughput.__file__).read_bytes()
    assert json.loads((vm / "status.json").read_text()) == {"state": "ok"}


def test_colab_run_fails_when_a_tokenizer_encodes_differently(local_run, tmp_path, monkeypatch):
    with pytest.raises(colab.ColabError, match=r"(?s)remote job failed \(error\).*super_bpe at V=280: ids .* differ"):
        _remote(local_run, tmp_path, monkeypatch, tamper=1)
    assert _log(tmp_path)[-1][0] == "stop"
