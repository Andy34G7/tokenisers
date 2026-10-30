"""Tokka-bench tier: vendored metric definitions, held-out languages, aggregation (B1-B4, B7, B8, B14)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

EN = ("Tokenizers split text into subword units. Subword units trade vocabulary size against "
      "sequence length, and every language pays a different price for the same vocabulary. ") * 3
ZH = "分词器把文本切分成子词单元。子词单元在词表大小和序列长度之间取得平衡。" * 3


@pytest.fixture(scope="module")
def tok(require_tokenisers):
    return require_tokenisers.BpeTokenizer.train(corpus=[EN, ZH], vocab_size=300)


def test_language_metrics_equal_vendored_tokka_bench(tok, repo_root):
    """B8: every per-language number equals vendored tokka-bench's _compute_language_metrics_from_ids."""
    from tokenisers.benchmarks.tokka import TokkaBenchRunner, language_info

    sys.path.append(str(repo_root / "vendor" / "tokka-bench" / "src"))
    from tokka_bench.fast_benchmark import _compute_language_metrics_from_ids

    class Proto:  # the tokka-bench tokenizer protocol
        def encode(self, text, add_special_tokens=True):
            return tok.encode(text)

    runner = TokkaBenchRunner()
    for lang, text in (("en", EN), ("zh", ZH), ("python", "def f(x):\n    return x + 1\n")):
        ours = runner.evaluate_tokenizer(tok, text_sample=text, lang_code=lang)
        ref = _compute_language_metrics_from_ids(Proto(), text, language_info(lang), tok.encode(text))
        for key in ("bytes_per_token", "total_bytes", "total_tokens", "unique_tokens", "subword_fertility",
                    "continued_word_rate", "word_split_pct", "continuation_token_pct"):
            assert ours[key] == pytest.approx(ref[key]), (lang, key)
        assert ours["total_units"] == ref["total_units"]


def test_continued_word_rate_is_continuation_token_share(tok):
    """B8: tokka's continued_word_rate is continuation tokens / tokens (%), not the share of split words."""
    from tokenisers.benchmarks.tokka import TokkaBenchRunner

    words = EN.split()
    counts = [len(tok.encode(w)) for w in words]
    expected = 100.0 * sum(c - 1 for c in counts) / sum(counts)
    res = TokkaBenchRunner().evaluate_tokenizer(tok, text_sample=EN, lang_code="en")
    assert res["continued_word_rate"] == pytest.approx(expected)
    assert res["word_split_pct"] == pytest.approx(100.0 * sum(c > 1 for c in counts) / len(counts))
    # fertility = tokens of the running text / words
    assert res["subword_fertility"] == pytest.approx(len(tok.encode(EN)) / len(words))


def test_matrix_unique_tokens_is_union_and_micro_is_token_weighted(tok):
    """B7: unique tokens are the union over languages; B3: micro bytes/token = total bytes / total tokens."""
    from tokenisers.benchmarks.tokka import TokkaBenchRunner

    texts = {"en": EN, "de": EN.replace("units", "Einheiten"), "zh": ZH}
    row = TokkaBenchRunner().evaluate_matrix({"bpe": tok}, texts)["bpe"]
    ids = {lang: tok.encode(t) for lang, t in texts.items()}
    union = set().union(*map(set, ids.values()))
    assert row["unique_tokens"] == len(union)
    assert row["unique_tokens"] < sum(row[lang]["unique_tokens"] for lang in texts)
    total_bytes = sum(len(t.encode("utf-8")) for t in texts.values())
    assert row["micro_avg"]["bytes_per_token"] == pytest.approx(total_bytes / sum(map(len, ids.values())))
    assert row["macro_avg"]["bytes_per_token"] == pytest.approx(
        sum(row[lang]["bytes_per_token"] for lang in texts) / len(texts))
    assert row["vocab_size"] == tok.vocab_size


def test_language_texts_join_held_out_documents_per_language():
    """B3: each language is its real held-out documents joined by newlines, keyed like tokka-bench."""
    from tokenisers.benchmarks.tokka.__main__ import language_texts

    docs = [{"text": "Erstes Dokument.", "lang": "deu_Latn"}, {"text": "def f(): pass", "lang": "python"},
            {"text": "Zweites Dokument.", "lang": "deu_Latn"}, {"text": "日本語の文書です。", "lang": "jpn_Jpan"}]
    texts = language_texts(docs)
    assert texts == {"de": "Erstes Dokument.\nZweites Dokument.", "python": "def f(): pass", "ja": "日本語の文書です。"}
    assert language_texts(docs, max_bytes=8)["ja"] == "日本"  # UTF-8 safe truncation


def test_cli_trains_on_train_split_and_evaluates_on_test_split(require_tokenisers, tmp_path, monkeypatch):
    """B1/B2/B4: the CLI trains only on the dataset's train split, evaluates the test split and records config."""
    from tokenisers.benchmarks.tokka import __main__ as cli
    from tokenisers.data.pipeline import DataPipeline

    seen = {}
    real_train_suite = cli.train_suite

    def spy(variants, corpus, vocab_size, lexicon, **kw):
        seen["corpus"] = list(corpus)
        return real_train_suite(variants, corpus, vocab_size, lexicon, **kw)

    monkeypatch.setattr(cli, "train_suite", spy)
    out = tmp_path / "out"
    cli.main(["--offline", "--cache-dir", str(tmp_path / "cache"), "--train-bytes", "30000", "--vocab-size", "280",
              "--variants", "bpe_frequency", "unigram_lm", "--output-dir", str(out),
              "--min-docs-per-lang", "1", "--min-bytes-per-lang", "0"])

    ds = DataPipeline(cache_dir=tmp_path / "cache", force_offline=True).prepare(train_bytes=30000)
    assert seen["corpus"] == ds.texts("train")
    doc = json.loads((out / "tokka_matrix.json").read_text())
    meta = doc["meta"]
    assert meta["eval_split"] == "test" and meta["synthetic_data"] is True
    assert meta["requested_vocab_size"] == 280
    assert meta["variants"]["unigram_lm"]["vocab_size"] == doc["models"]["unigram_lm"]["vocab_size"]
    assert meta["dataset"]["key"] == ds.manifest["key"] and meta["git_commit"]
    test_by_lang = cli.language_texts(ds.docs("test"))
    assert meta["language_bytes"] == {k: len(v.encode()) for k, v in test_by_lang.items()}
    assert set(doc["models"]["bpe_frequency"]) >= set(test_by_lang) | {"macro_avg", "micro_avg"}
    assert not set(seen["corpus"]) & set(ds.texts("test"))
    assert (out / "tokka_leaderboard.md").read_text().count("| `") == 2


def test_default_vocab_size_is_16k(capsys):
    """B14: every CLI defaults to a 16,384-token vocabulary."""
    from tokenisers.benchmarks.colab import __main__ as colab_cli
    from tokenisers.benchmarks.common import DEFAULT_VOCAB_SIZE
    from tokenisers.benchmarks.intrinsic import __main__ as intrinsic_cli
    from tokenisers.benchmarks.tokka import __main__ as tokka_cli

    assert DEFAULT_VOCAB_SIZE == 16384
    for cli in (tokka_cli, intrinsic_cli, colab_cli):
        with pytest.raises(SystemExit):
            cli.main(["--help"])
        assert "default 16384" in capsys.readouterr().out


def test_packaged_tokka_metrics_are_the_vendored_files(repo_root):
    """The copy shipped in the package is byte-identical to vendor/tokka-bench."""
    from tokenisers.benchmarks.tokka import _tokka_bench

    pkg = Path(_tokka_bench.__file__).parent
    for name in ("metrics.py", "unicode_utils.py"):
        vendored = repo_root / "vendor" / "tokka-bench" / "src" / "tokka_bench" / name
        assert (pkg / name).read_bytes() == vendored.read_bytes(), name


def test_tokka_works_outside_a_source_checkout(tmp_path, require_tokenisers):
    """An installed copy of the package (no vendor/ directory above it) imports and runs the tokka suite."""
    import tokenisers

    site = tmp_path / "site"
    shutil.copytree(Path(tokenisers.__file__).parent, site / "tokenisers",
                    ignore=shutil.ignore_patterns("__pycache__"))
    code = (
        "import tokenisers, tokenisers.benchmarks.tokka as t\n"
        f"assert tokenisers.__file__.startswith({str(site)!r}), tokenisers.__file__\n"
        "tok = tokenisers.BpeTokenizer.train(['one two one two'] * 4, vocab_size=260)\n"
        "print(t.TokkaBenchRunner().evaluate_tokenizer(tok, 'one two', lang_code='en')['total_units'])\n"
    )
    env = {**os.environ, "PYTHONPATH": str(site)}
    out = subprocess.run([sys.executable, "-c", code], cwd=site, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "2"


def test_macro_average_leaves_out_languages_with_too_little_held_out_text(require_tokenisers, tmp_path, capsys):
    """A language with one short held-out document is reported but does not get a full macro weight;
    the leaderboard shows every language's held-out docs and bytes."""
    from tokenisers.benchmarks.tokka import __main__ as cli

    args = ["--offline", "--cache-dir", str(tmp_path / "cache"), "--train-bytes", "30000", "--vocab-size", "280",
            "--variants", "bpe_frequency", "--output-dir", str(tmp_path / "out")]
    # the synthetic test split has 2 English documents and 1 of every other language
    with pytest.raises(SystemExit, match="No language has >= 3 held-out documents"):
        cli.main(args)
    capsys.readouterr()
    cli.main(args + ["--min-docs-per-lang", "2", "--min-bytes-per-lang", "0"])
    assert "warning: de has only 1 held-out docs" in capsys.readouterr().err
    doc = json.loads((tmp_path / "out" / "tokka_matrix.json").read_text())
    row, meta = doc["models"]["bpe_frequency"], doc["meta"]
    assert meta["macro_languages"] == ["en"] and meta["language_docs"]["en"] == 2
    assert row["macro_avg"]["bytes_per_token"] == pytest.approx(row["en"]["bytes_per_token"])
    langs = [k for k in meta["language_bytes"]]
    assert row["micro_avg"]["bytes_per_token"] == pytest.approx(
        sum(row[k]["total_bytes"] for k in langs) / sum(row[k]["total_tokens"] for k in langs))
    md = (tmp_path / "out" / "tokka_leaderboard.md").read_text()
    assert "macro = unweighted mean over the 1 languages with >= 2 held-out documents" in md
    assert f"| de | 1 | {meta['language_bytes']['de']:,} | no |" in md
    assert f"| en | 2 | {meta['language_bytes']['en']:,} | yes |" in md
