"""HF export parity: an exported tokenizer.json, loaded by Hugging Face ``tokenizers`` and by
``transformers``, must produce exactly the ids of native encoding for every exportable variant."""

import json
import subprocess
import sys

import pytest
import tokenizers
import transformers

from tokenisers import _core

DOCS = [
    "The quick brown fox jumps over the lazy dog. Don't stop!",
    "hello, world! hello, world! hello,world -- (parens) [brackets] {braces}",
    "def foo(x):\n    return x + 1  # comment\n\n\tif x: pass",
    "price: $123.45, abc123 xyz789 1234567 3.14159e-10",
    "Héllo wörld, naïve café résumé façade",
    "日本語のテキスト、中文文本。한국어 텍스트",
    "中文　文本、中文。日本語　の　テキスト",
    "emoji 🚀🔥💯 party 🎉🎉 👩‍👩‍👧",
    "   leading spaces   and   runs\t\ttabs\n\nnewlines\r\n",
    "non breaking space and em thin　ideographic",
    "it's they're we've I'm y'all",
]
CORPUS = [f"{d} {r}" for i, d in enumerate(DOCS) for r in range(5 + i)]

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "hello, world!",
    "hello,world",
    "def foo(x):\n    return x + 1  # comment",
    'fn main() { println!("{}", a[0] + b.c); }',
    "abc123 xyz789",
    "Héllo wörld, naïve café",
    "日本語のテキスト 中文文本",
    "中文　文本、中文。",
    "emoji 🚀🔥💯 party 🎉 👩‍👩‍👧",
    "a  b   c    d",
    "   leading",
    "trailing   ",
    "it's they're",
    "tabs\t\tand\n\nnewlines\r\n",
    "non breaking　ideo line\u0085nel ogham math",
    "zero​width﻿bom",
    " ",
    "x",
]

BPE_SCORERS = [
    "frequency",
    "wordpiece",
    "wp-bpe-base",
    "wp-bpe-ft",
    "wp-bpe-pig",
    "wp-bpe-scp",
    "wp-bpe-npmi",
    "wp-bpe-ppig",
    "mdl",
    "length_penalized",
]


def _train(variant):
    if variant == "super_bpe":
        return _core.train_super_bpe(CORPUS, 330, 460)
    if variant == "unigram":
        return _core.train_unigram(CORPUS, 500)
    if variant == "morpheme":
        return _core.train_morpheme_guided(CORPUS, 420, {"jumps": ["jump", "s"]})
    return _core.train_bpe(CORPUS, 420, scorer_type=variant)


VARIANTS = BPE_SCORERS + ["super_bpe", "unigram", "morpheme"]


@pytest.fixture(scope="module", params=VARIANTS)
def exported(request, tmp_path_factory):
    model = _train(request.param)
    out_dir = tmp_path_factory.mktemp(request.param.replace("-", "_"))
    model.export_hf_json(str(out_dir / "tokenizer.json"))
    return request.param, model, out_dir


def _assert_same_ids(name, native_encode, hf_encode):
    mismatches = [t for t in TEXTS if native_encode(t) != hf_encode(t)]
    assert not mismatches, f"[{name}] HF ids differ from native for {mismatches!r}"


def test_hf_tokenizers_ids_equal_native(exported):
    name, model, out_dir = exported
    hf = tokenizers.Tokenizer.from_file(str(out_dir / "tokenizer.json"))
    _assert_same_ids(name, model.encode, lambda t: hf.encode(t, add_special_tokens=False).ids)


def test_transformers_ids_equal_native(exported):
    name, model, out_dir = exported
    fast = transformers.PreTrainedTokenizerFast(tokenizer_file=str(out_dir / "tokenizer.json"))
    _assert_same_ids(name, model.encode, lambda t: fast.encode(t, add_special_tokens=False))
    auto = transformers.AutoTokenizer.from_pretrained(str(out_dir))
    _assert_same_ids(name, model.encode, lambda t: auto.encode(t, add_special_tokens=False))


def test_hf_decode_round_trips(exported):
    name, model, out_dir = exported
    hf = tokenizers.Tokenizer.from_file(str(out_dir / "tokenizer.json"))
    for t in TEXTS:
        assert hf.decode(model.encode(t), skip_special_tokens=False) == t, f"[{name}] {t!r}"


SPECIALS = ["<|endoftext|>", "<pad>"]
SPECIAL_TEXTS = ["hello<|endoftext|>world", "<pad><pad> x", "a <|endoftext|> 日本<pad>語", "none"]


def _split_specials_then_encode(model, text):
    """Reference: split out special tokens (leftmost-longest, as HF does), encode the rest."""
    ids, start, i = [], 0, 0
    while i < len(text):
        hit = max((s for s in SPECIALS if text.startswith(s, i)), key=len, default=None)
        if hit is None:
            i += 1
            continue
        ids += model.encode(text[start:i]) if i > start else []
        ids.append(model.token_to_id(hit))
        i = start = i + len(hit)
    return ids + (model.encode(text[start:]) if start < len(text) else [])


def _special_model(tmp_path):
    model = _core.train_bpe(CORPUS, 420, special_tokens=SPECIALS)
    model.export_hf_json(str(tmp_path / "tokenizer.json"))
    return model, tokenizers.Tokenizer.from_file(str(tmp_path / "tokenizer.json"))


def test_hf_splits_special_tokens_then_applies_bpe(tmp_path):
    model, hf = _special_model(tmp_path)
    for t in SPECIAL_TEXTS:
        assert hf.encode(t, add_special_tokens=False).ids == _split_specials_then_encode(model, t), t


@pytest.mark.xfail(
    strict=True,
    reason="needs native special-token splitting in the core encoder (C10, rust-merge branch); "
    "remove this marker once that is merged",
)
def test_native_special_token_encoding_equals_hf(tmp_path):
    model, hf = _special_model(tmp_path)
    for t in SPECIAL_TEXTS:
        assert model.encode(t) == hf.encode(t, add_special_tokens=False).ids, t


_DETERMINISM_SCRIPT = """
import sys
from tokenisers import _core
corpus = [l for l in sys.stdin.read().split("\\x1e") if l]
_core.train_unigram(corpus, 500).export_hf_json(sys.argv[1])
"""


def test_unigram_export_is_identical_across_processes(tmp_path):
    """Hash seeds differ per process, so train in two fresh interpreters (different corpus
    order) and require byte-identical tokenizer.json files."""
    outputs = []
    for i, corpus in enumerate([CORPUS, CORPUS[::-1]]):
        path = tmp_path / f"run{i}.json"
        subprocess.run(
            [sys.executable, "-c", _DETERMINISM_SCRIPT, str(path)],
            input="\x1e".join(corpus),
            text=True,
            check=True,
        )
        outputs.append(path.read_bytes())
    assert outputs[0] == outputs[1]


def test_reloaded_super_bpe_keeps_whole_text_pre_tokenization(tmp_path):
    model = _train("super_bpe")
    model.export_hf_json(str(tmp_path / "a" / "tokenizer.json"))
    reloaded = _core.PyBpeModel.from_hf_json(str(tmp_path / "a" / "tokenizer.json"))
    reloaded.export_hf_json(str(tmp_path / "b" / "tokenizer.json"))
    a = json.loads((tmp_path / "a" / "tokenizer.json").read_text())
    b = json.loads((tmp_path / "b" / "tokenizer.json").read_text())
    assert a["pre_tokenizer"] == b["pre_tokenizer"] == {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": False,
        "use_regex": False,
    }
    hf = tokenizers.Tokenizer.from_file(str(tmp_path / "b" / "tokenizer.json"))
    _assert_same_ids("super_bpe reloaded", reloaded.encode, lambda t: hf.encode(t).ids)
