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
    common.check_memory(["bpe_frequency"], 50 * mb)  # BPE alone: ~2.8 GiB
    monkeypatch.setattr(common, "available_memory", lambda: None)  # unknown platform: no check
    common.check_memory(["unigram_lm"], 10**12)


def test_assert_disjoint():
    common.assert_disjoint(["a", "b"], ["c"])
    with pytest.raises(RuntimeError, match="1 test documents"):
        common.assert_disjoint(["a", "b"], ["b", "c"], "test")
