"""Intrinsic tier: byte offsets, held-out morphology, throughput, Rényi, identifiers (B5, B6, B9, B16, B17)."""

from __future__ import annotations

import json
import math
import time

import pytest

from tokenisers.benchmarks.intrinsic import (
    IntrinsicEvaluator,
    canonical_identifier_cuts,
    extract_identifier_cuts,
    mine_identifiers,
)


class Pieces:
    """Fake tokenizer: encodes a word into fixed pieces (raw bytes per id)."""

    def __init__(self, pieces):
        self.pieces = [p.encode() for p in pieces]
        self.vocab_size = len(self.pieces)

    def encode(self, text):
        assert text.encode() == b"".join(self.pieces)
        return list(range(len(self.pieces)))

    def id_to_bytes(self, i):
        return self.pieces[i]


def test_morph_f1_uses_raw_token_bytes(require_tokenisers):
    """B5: offsets come from id_to_bytes, so a non-ASCII word scores the same as its manual byte cuts."""
    word = "megszentségteleníthetetlen"
    tok = require_tokenisers.UnigramTokenizer.train(corpus=[f"{word} szentség telen ít hetetlen " * 20],
                                                    vocab_size=300)
    ids = tok.encode(word)
    cuts, off = set(), 0
    for i in ids[:-1]:
        off += len(tok.decode_bytes([i]))
        cuts.add(off)
    gold = {"word": word, "segments": ["meg", "szent", "ség", "telen", "ít", "hetetlen"]}
    gold["gold_cuts"] = {3, 8, 12, 17, 20}
    ev = IntrinsicEvaluator()
    assert ev.compute_morphological_f1(tok, [gold])["f1"] == pytest.approx(ev.evaluate_cut_f1(cuts, gold["gold_cuts"])["f1"])


def test_morph_f1_requires_byte_offsets():
    """B5: a tokenizer without id_to_bytes is an error, not a silent str(id)-length fallback."""

    class NoBytes:
        vocab_size = 300

        def encode(self, text):
            return [104, 299]

    with pytest.raises(TypeError, match="id_to_bytes"):
        IntrinsicEvaluator().compute_morphological_f1(NoBytes(), [{"word": "hi", "gold_cuts": {1}}])


def test_morph_f1_skips_canonical_entries():
    tok = Pieces(["un", "happi", "ness"])
    gold = [{"word": "unhappiness", "gold_cuts": {2, 7}},
            {"word": "unhappiness", "gold_cuts": None}]  # canonical (non-surface) SIGMORPHON entry
    res = IntrinsicEvaluator().compute_morphological_f1(tok, gold)
    assert res["f1"] == 1.0 and res["n"] == 1


def test_lexicon_is_sigmorphon_train_minus_test_gold(require_tokenisers, tmp_path):
    """B6: gold = SIGMORPHON test split; the morpheme_bpe lexicon = train-split words seen in the corpus,
    never a gold word."""
    import argparse

    from tokenisers.benchmarks.common import load_morphology
    from tokenisers.data.pipeline import SIGMORPHON_COMMIT

    d = tmp_path / f"sigmorphon2022-{SIGMORPHON_COMMIT[:12]}"
    d.mkdir()
    (d / "eng.word.train.tsv").write_text(
        "walked\twalk @@ed\t010\nplaying\tplay @@ing\t010\nunseen\tun @@seen\t010\nhoped\thope @@ed\t010\n"
        "hopes\thope @@s\t010\n")
    (d / "eng.word.test.gold.tsv").write_text("playing\tplay @@ing\t010\njumped\tjump @@ed\t010\n")
    args = argparse.Namespace(cache_dir=str(tmp_path), offline=False)  # files are cached: no download
    corpus = ["she walked while playing", "they hoped", "he hopes"]
    lexicon, gold, prov = load_morphology(args, corpus, ["eng"])
    assert {g["word"] for g in gold} == {"playing", "jumped"}
    # "unseen": not in the corpus; "playing": gold; "hoped": canonical (hope+ed != hoped), no byte offsets
    assert lexicon == {"walked": ["walk", "ed"], "hopes": ["hope", "s"]}
    assert prov["lexicon_words_dropped_as_gold"] == 1
    assert prov["gold"][0]["split"] == "test" and prov["lexicon"][0]["split"] == "train"


def test_cli_never_adds_gold_words_to_training_text(require_tokenisers, tmp_path, monkeypatch):
    """B6/B2: the tokenizer corpus is exactly the train split; evaluation text is the test split."""
    from tokenisers.benchmarks.intrinsic import __main__ as cli
    from tokenisers.data.pipeline import DataPipeline

    seen = {}
    real = cli.train_suite

    def spy(variants, corpus, vocab_size, lexicon, **kw):
        seen["corpus"], seen["lexicon"] = list(corpus), dict(lexicon)
        return real(variants, corpus, vocab_size, lexicon, **kw)

    evaluated = {}
    real_suite = cli.IntrinsicEvaluator.evaluate_suite

    def spy_suite(self, models, text, gold, *a, **kw):
        evaluated["text"], evaluated["gold"] = list(text), gold
        return real_suite(self, models, text, gold, *a, **kw)

    monkeypatch.setattr(cli, "train_suite", spy)
    monkeypatch.setattr(cli.IntrinsicEvaluator, "evaluate_suite", spy_suite)
    out = tmp_path / "out"
    cli.main(["--offline", "--cache-dir", str(tmp_path / "cache"), "--train-bytes", "30000", "--vocab-size", "280",
              "--variants", "bpe_frequency", "morpheme_bpe", "--output-dir", str(out), "--throughput-repeats", "5"])
    ds = DataPipeline(cache_dir=tmp_path / "cache", force_offline=True).prepare(train_bytes=30000)
    assert seen["corpus"] == ds.texts("train")
    assert evaluated["text"] == ds.texts("test")
    gold_words = {g["word"] for g in evaluated["gold"]}
    assert not gold_words & set(seen["lexicon"])
    doc = json.loads((out / "intrinsic_leaderboard.json").read_text())
    assert doc["meta"]["eval_split"] == "test"
    # offline: empty lexicon, so morpheme_bpe == bpe_frequency and is reported once
    assert doc["meta"]["duplicates"] == {"morpheme_bpe": "bpe_frequency"}
    assert list(doc["models"]) == ["bpe_frequency"]
    assert doc["models"]["bpe_frequency"]["throughput_repeats"] == 5


def test_throughput_is_per_document_median_without_floor():
    """B9: one encode call per document, median of >= 5 timed passes, no 0.1 MB/s floor."""
    calls = []

    class Slow:
        def encode(self, text):
            calls.append(text)
            time.sleep(0.01)
            return [0]

        def decode(self, ids):
            return ""

    docs = ["a" * 100, "b" * 100]
    res = IntrinsicEvaluator().benchmark_throughput(Slow(), docs)
    assert res["throughput_repeats"] >= 5
    assert len(calls) == 2 * (res["throughput_repeats"] + 1)  # one warm-up pass
    assert res["encode_mb_per_sec"] < 0.1  # ~0.01 MB/s; the old code reported max(0.1, ...)
    assert res["throughput_bytes"] == 200 and res["throughput_docs"] == 2


def test_renyi_uses_actual_vocab_and_is_not_clamped(monkeypatch):
    """B16: |V| is the tokenizer's real vocabulary size and values are returned as computed."""
    tok = Pieces(["a", "b", "c", "d"])  # vocab_size 4, all used once
    ev = IntrinsicEvaluator()
    assert ev.compute_renyi_efficiency(tok, "abcd") == pytest.approx({"e_1_0": 1.0, "e_2_0": 1.0, "e_2_5": 1.0})
    counts = [3, 1, 0, 0]
    p = [0.75, 0.25]
    assert ev.compute_renyi_from_counts(counts, 2.0) == pytest.approx(-math.log2(sum(x * x for x in p)) / 2)
    # a (hypothetically) broken entropy must show up, not be clamped into the expected order
    monkeypatch.setattr(IntrinsicEvaluator, "compute_renyi_from_counts", lambda self, c, alpha=1.0: alpha / 10)
    assert ev.compute_renyi_efficiency(tok, "abcd") == {"e_1_0": 0.1, "e_2_0": 0.2, "e_2_5": 0.25}


def test_identifier_gold_rule_does_not_make_underscores_segments():
    """B17: '_' is a delimiter, not a segment; cutting on either side of it is the same boundary."""
    assert extract_identifier_cuts("get_user_profile") == {4, 9}
    assert extract_identifier_cuts("getUserProfile") == {3, 7}
    assert extract_identifier_cuts("HTTPServer") == {4}
    assert extract_identifier_cuts("calculate_loss_v2") == {10, 15}
    assert extract_identifier_cuts("__init__") == set()
    assert canonical_identifier_cuts("get_user", {3}) == canonical_identifier_cuts("get_user", {4}) == {4}
    assert canonical_identifier_cuts("_private", {1}) == set()

    ev = IntrinsicEvaluator()
    for pieces in (["get", "_user", "_profile"], ["get_", "user_", "profile"], ["get", "_", "user", "_", "profile"]):
        assert ev.compute_identifier_f1(Pieces(pieces), ["get_user_profile"])["f1"] == 1.0, pieces
    assert ev.compute_identifier_f1(Pieces(["get_user_profile"]), ["get_user_profile"])["recall"] == 0.0


def test_identifiers_are_mined_from_code_by_frequency():
    """B17: the identifier set comes from held-out code, most frequent multi-part identifiers first."""
    code = ["def get_user(user_id):\n    return load_user(user_id)\n", "x = get_user(1)\nuserId = 2\n"]
    assert mine_identifiers(code) == ["get_user", "user_id", "load_user", "userId"]
    assert mine_identifiers(code, max_items=2) == ["get_user", "user_id"]


class ByteTok:
    """Fake byte-level tokenizer: one token per UTF-8 byte, so every byte boundary is a cut."""

    vocab_size = 256

    def encode(self, text):
        return list(text.encode())

    def decode(self, ids):
        return bytes(ids).decode()

    def id_to_bytes(self, i):
        return bytes([i])


def _meta():
    return {"requested_vocab_size": 256, "train_bytes": 1, "eval_split": "test", "eval_bytes": 1, "duplicates": {}}


def test_morph_f1_headline_weights_languages_equally():
    """The headline is the unweighted mean over languages; a language with many gold words (hun in
    SIGMORPHON) must not dominate it. Per-language and word-weighted values are reported too."""
    from tokenisers.benchmarks.report import render_intrinsic

    gold = [{"word": "ab", "gold_cuts": {1}, "lang": "eng"}]  # byte cuts {1}: F1 1.0
    gold += [{"word": "abc", "gold_cuts": set(), "lang": "hun"}] * 3  # cuts {1, 2} vs none: F1 0.0
    res = IntrinsicEvaluator().evaluate_suite({"bytes": ByteTok()}, ["ab abc"], gold, repeats=1)["bytes"]
    assert res["morph_f1_by_lang"] == {"eng": 1.0, "hun": 0.0}
    assert res["morph_words_by_lang"] == {"eng": 1, "hun": 3} and res["morph_words"] == 4
    assert res["morph_f1"] == pytest.approx(0.5) and res["morph_f1_word_weighted"] == pytest.approx(0.25)
    md = render_intrinsic({"meta": _meta(), "models": {"bytes": res}})
    assert "then over the 2 languages unweighted" in md
    assert "| eng (1 words) | hun (3 words) | Word-weighted |" in md
    assert "| `bytes` | 1.000 | 0.000 | 0.250 |" in md


def test_morph_and_identifier_f1_are_undefined_without_items():
    """No gold words / no identifiers: F1 is None and renders as n/a, not a score of 0.000."""
    from tokenisers.benchmarks.report import render_intrinsic

    res = IntrinsicEvaluator().evaluate_suite({"bytes": ByteTok()}, ["the cat sat"], gold_annotations=[],
                                              identifiers=[], repeats=1)["bytes"]
    assert res["morph_words"] == 0 and res["identifiers"] == 0
    assert res["morph_f1"] is None and res["morph_f1_word_weighted"] is None and res["identifier_f1"] is None
    row = next(line for line in render_intrinsic({"meta": _meta(), "models": {"bytes": res}}).splitlines()
               if line.startswith("| `bytes`"))
    assert "| n/a | n/a |" in row and "0.000" not in row
