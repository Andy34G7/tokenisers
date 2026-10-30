"""Shared benchmark plumbing: duplicate tokenizers, README tables from JSON, memory preflight (B10)."""

from __future__ import annotations

import importlib.util
import json

import pytest

from tokenisers.benchmarks import common
from tokenisers.benchmarks.report import render_colab, render_intrinsic, render_tokka


def _gen_module(repo_root):
    spec = importlib.util.spec_from_file_location("gen_readme_tables", repo_root / "scripts" / "gen_readme_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_readme_tables_match_committed_results(repo_root):
    """B10: README leaderboards are exactly what scripts/gen_readme_tables.py renders from the result JSON."""
    gen = _gen_module(repo_root)
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    assert gen.render_readme(readme, repo_root) == readme, "run: uv run python scripts/gen_readme_tables.py"


def _meta(**kw):
    meta = {"requested_vocab_size": 300, "synthetic_data": False, "train_bytes": 4_000_000, "eval_split": "test",
            "eval_bytes": 300_000, "git_commit": "0123456789abcdef", "git_dirty": False,
            "created_utc": "2026-10-30T00:00:00Z", "duplicates": {"wp_bpe_ft": "wordpiece_pmi"}, "host": {"cpu": "x"}}
    meta.update(kw)
    return meta


def test_generator_renders_json_and_detects_stale_readme(repo_root, tmp_path):
    """B10: a section is replaced by the rendered JSON; --check fails on a README that differs."""
    gen = _gen_module(repo_root)
    row = {"vocab_size": 300, "en": {"bytes_per_token": 4.0}, "macro_avg": {
        "bytes_per_token": 3.25, "subword_fertility": 1.5, "continued_word_rate": 20.0},
        "micro_avg": {"bytes_per_token": 3.5}, "unique_tokens": 250}
    (tmp_path / "benchmarks/results/tokka").mkdir(parents=True)
    (tmp_path / "benchmarks/results/tokka/tokka_matrix.json").write_text(json.dumps(
        {"meta": _meta(), "models": {"wordpiece_pmi": row, "wp_bpe_ft": row}}))
    readme = "x\n" + "".join(f"<!-- BEGIN GENERATED: {s} -->\nold\n<!-- END GENERATED: {s} -->\n"
                             for s in ("tokka", "intrinsic", "colab"))
    out = gen.render_readme(readme, tmp_path)
    assert "| `wordpiece_pmi` = `wp_bpe_ft` | 300 | 3.250 | 3.500 | 1.500 | 20.00 | 250 |" in out
    assert out.count("`wp_bpe_ft`") == 1  # one row for the duplicate pair
    assert "No results committed" in out.split("BEGIN GENERATED: intrinsic")[1]
    path = tmp_path / "README.md"
    path.write_text(readme)
    assert gen.main(["--check", "--readme", str(path)]) == 1


def test_find_duplicates_detects_identical_tokenizers(require_tokenisers):
    """Identical trainings are one result; different ones are kept apart."""
    tk = require_tokenisers
    corpus = ["the cat sat on the mat with the hat " * 20]
    models = {
        "a": tk.BpeTokenizer.train(corpus, vocab_size=266),
        "b": tk.BpeTokenizer.train(corpus, vocab_size=266),
        "c": tk.BpeTokenizer.train(corpus, vocab_size=262),
        "u": tk.UnigramTokenizer.train(corpus, vocab_size=266),
    }
    assert common.find_duplicates(models) == {"b": "a"}


def test_leaderboards_state_data_and_backend():
    """B11: the rendered header says whether data were synthetic and which backend ran the LM."""
    intr = {"meta": _meta(synthetic_data=True, duplicates={}), "models": {"bpe": {
        "vocab_size": 300, "bytes_per_token": 1.0, "fertility": 2.0, "renyi_e_1_0": 0.5, "renyi_e_2_5": 0.4,
        "morph_f1": 0.3, "identifier_f1": 0.2, "encode_mb_per_sec": 1.0, "decode_mb_per_sec": 2.0}}}
    assert "SYNTHETIC" in render_intrinsic(intr)
    col = {"meta": _meta(duplicates={}, eval_split="val", val_domains=["english"], seeds=[0, 1, 2], byte_budget=10,
                         backend={"kind": "colab", "device": "cuda", "gpu": "Tesla T4"}, lm={"d_model": 8}),
           "models": {"bpe": {"vocab_size": 300, "train_bytes_per_token": 3.0, "tokens_consumed": 10,
                              "val_domains": {"english": {"bpb": 1.25, "bpb_sd": 0.01}},
                              "macro_val_bpb": 1.25, "macro_val_bpb_sd": 0.01}}}
    md = render_colab(col)
    assert "Colab GPU `Tesla T4`" in md and "1.250 ± 0.010" in md
    col["meta"]["backend"] = {"kind": "local", "device": "cpu", "cpu": "Some CPU"}
    assert "local CPU `Some CPU`" in render_colab(col)
    assert render_tokka({"meta": _meta(duplicates={}), "models": {}}).count("|") > 0


def test_memory_preflight_refuses_runs_that_would_not_fit(monkeypatch):
    """The CLIs stop before training when the estimated peak exceeds available memory."""
    mb = 2**20
    monkeypatch.setattr(common, "available_memory", lambda: 3 * 1024 * mb)
    common.check_memory(["bpe_frequency", "unigram_lm"], 4 * mb)  # ~1.3 GiB: fine
    with pytest.raises(SystemExit, match="unigram_lm"):
        common.check_memory(["bpe_frequency", "unigram_lm"], 50 * mb)  # ~9 GiB
    common.check_memory(["bpe_frequency"], 35 * mb)  # BPE alone: ~2.6 GiB
    monkeypatch.setattr(common, "available_memory", lambda: None)  # unknown platform: no check
    common.check_memory(["unigram_lm"], 10**12)


def test_memory_estimate_covers_the_largest_vocabulary():
    """Measured at V=65536 on 4 MB of the real corpus (peak RSS growth per training byte): super_bpe
    100, wp_bpe_npmi/mdl_bits 47, unigram_lm 139. The estimate must cover these with a margin, since
    65536 is a supported size."""
    measured = {"super_bpe": 100, "wp_bpe_npmi": 47, "mdl_bits": 47, "unigram_lm": 139, "bpe_frequency": 38}
    for variant, per_byte in measured.items():
        assert common.estimate_train_memory([variant], 10**8) - common.BASE_MEMORY >= 1.2 * per_byte * 10**8, variant


def test_system_memory_keeps_a_reserve_for_the_desktop():
    """Outside a cgroup limit the benchmark may not plan on all of MemAvailable."""
    gib = 2**30
    meminfo = f"MemTotal: {15 * gib // 1024} kB\nMemFree: 1 kB\nMemAvailable: {int(9.5 * gib) // 1024} kB\n"
    usable = common.usable_system_memory(meminfo)
    assert usable == int(9.5 * gib) - int(0.15 * 15 * gib)  # 15% of 15 GiB > the 2 GiB floor
    assert common.usable_system_memory(f"MemTotal: {4 * gib // 1024} kB\nMemAvailable: {3 * gib // 1024} kB\n") == gib
    assert common.usable_system_memory("MemTotal: 1 kB\n") is None
    # the reviewer's scenario: a ~9.1 GiB unigram_lm estimate on a 15 GiB desktop with 9.5 GiB available
    assert common.estimate_train_memory(["unigram_lm"], 50 * 2**20) > usable


def test_memory_preflight_runs_before_the_dataset_is_prepared(monkeypatch, tmp_path):
    """A run that cannot fit stops before downloading/preparing its dataset."""
    from tokenisers.benchmarks.colab import __main__ as colab_cli
    from tokenisers.benchmarks.intrinsic import __main__ as intrinsic_cli
    from tokenisers.benchmarks.tokka import __main__ as tokka_cli

    monkeypatch.setattr(common, "available_memory", lambda: 2**30)
    for cli in (tokka_cli, intrinsic_cli, colab_cli):
        monkeypatch.setattr(cli, "load_dataset", lambda args: pytest.fail("dataset prepared before the check"))
        with pytest.raises(SystemExit, match="needs ~"):
            cli.main(["--offline", "--cache-dir", str(tmp_path), "--train-bytes", str(50 * 2**20)])


def test_assert_disjoint():
    common.assert_disjoint(["a", "b"], ["c"])
    with pytest.raises(RuntimeError, match="1 test documents"):
        common.assert_disjoint(["a", "b"], ["b", "c"], "test")
