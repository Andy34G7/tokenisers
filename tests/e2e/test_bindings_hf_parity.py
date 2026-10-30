"""HF parity through the high-level classes (U1/C10/P2): ``save`` writes a tokenizer.json that
Hugging Face ``tokenizers`` and ``transformers`` load to the same ids, the same token strings and
the same special-token ids as native encoding, and ``from_pretrained`` reloads it unchanged."""

import pytest
import tokenizers
import transformers

import tokenisers

SPECIALS = ["<|endoftext|>", "<pad>"]
CORPUS = [
    f"{d} {r}"
    for i, d in enumerate([
        "The quick brown fox jumps over the lazy dog. Don't stop!",
        "def foo(x):\n    return x + 1  # comment\n\n\tif x: pass",
        "Héllo wörld, naïve café résumé façade",
        "日本語のテキスト、中文文本。한국어 텍스트",
        "中文　文本、中文。日本語　の　テキスト",
        "emoji 🚀🔥💯 party 🎉🎉 👩‍👩‍👧",
        "   leading spaces   and   runs\t\ttabs\n\nnewlines\r\n",
        "non breaking space and thin　ideographic",
    ])
    for r in range(4 + i)
]
TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "def foo(x):\n    return x + 1",
    "Héllo wörld, naïve café",
    "日本語のテキスト 中文文本",
    "中文　文本、中文。",
    "emoji 🚀🔥 👩‍👩‍👧",
    "a  b   c\t\td\r\n",
    "non breaking space thin　ideo\u0085nel ogham",
    "hello<|endoftext|>world",
    "<pad><pad> x <|endoftext|>日本<pad>語",
    " ",
]

MAKERS = {
    "bpe+specials": lambda: tokenisers.BpeTokenizer.train(CORPUS, 420, special_tokens=SPECIALS),
    "wordpiece+specials": lambda: tokenisers.WordPieceTokenizer.train(CORPUS, 420, special_tokens=SPECIALS),
    "wp-bpe-npmi": lambda: tokenisers.WpBpeNpmiTokenizer.train(CORPUS, 420),
    "mdl": lambda: tokenisers.MdlTokenizer.train(CORPUS, 420),
    "super_bpe": lambda: tokenisers.SuperBpeTokenizer.train(CORPUS, vocab_size=460),
    "morpheme": lambda: tokenisers.MorphemeBpeTokenizer.train(CORPUS, 420, morpheme_lexicon={}),
    "unigram": lambda: tokenisers.UnigramTokenizer.train(CORPUS, 500),
}


@pytest.fixture(scope="module", params=list(MAKERS))
def saved(request, tmp_path_factory):
    tok = MAKERS[request.param]()
    out = tmp_path_factory.mktemp(request.param.replace("+", "_").replace("-", "_"))
    tok.save(str(out))
    return request.param, tok, out


def test_hf_ids_equal_native(saved):
    name, tok, out = saved
    hf = tokenizers.Tokenizer.from_file(str(out / "tokenizer.json"))
    auto = transformers.AutoTokenizer.from_pretrained(str(out))
    for text in TEXTS:
        ids = tok.encode(text)
        assert hf.encode(text, add_special_tokens=False).ids == ids, (name, text)
        assert auto.encode(text, add_special_tokens=False) == ids, (name, text)
        assert hf.decode(ids, skip_special_tokens=False) == text, (name, text)


def test_get_vocab_equals_hf_vocab(saved):
    """P2: token strings are the tokenizer.json keys (byte-level printable, specials by content)."""
    name, tok, out = saved
    hf = tokenizers.Tokenizer.from_file(str(out / "tokenizer.json"))
    assert hf.get_vocab(with_added_tokens=True) == tok.get_vocab(), name
    assert hf.get_vocab_size(with_added_tokens=True) == tok.vocab_size


@pytest.mark.parametrize("saved", ["bpe+specials", "wordpiece+specials"], indirect=True)
def test_special_tokens_use_reserved_slots(saved):
    """P10/C10: specials are inside vocab_size, encoded as single ids, as HF added_tokens."""
    name, tok, out = saved
    assert tok.vocab_size == 420
    assert [tok.token_to_id(s) for s in SPECIALS] == [418, 419]
    hf = tokenizers.Tokenizer.from_file(str(out / "tokenizer.json"))
    assert [hf.token_to_id(s) for s in SPECIALS] == [418, 419]
    assert tok.encode("<pad>") == [419]


def test_from_pretrained_round_trips(saved, tmp_path):
    name, tok, out = saved
    reloaded = type(tok).from_pretrained(str(out))
    assert reloaded.get_vocab() == tok.get_vocab()
    for text in TEXTS:
        assert reloaded.encode(text) == tok.encode(text), (name, text)
    reloaded.save(str(tmp_path))
    assert (tmp_path / "tokenizer.json").read_bytes() == (out / "tokenizer.json").read_bytes()


def test_wrapper_ids_equal_hf_fast_tokenizer(saved):
    name, tok, out = saved
    wrapper = tokenisers.DynamicResearchTokenizer(tok)
    fast = transformers.PreTrainedTokenizerFast(tokenizer_file=str(out / "tokenizer.json"))
    for text in TEXTS:
        if "<" in text and "specials" not in name:
            continue  # the wrapper adds its own <pad>/</s>; the bare model has none
        assert wrapper(text)["input_ids"] == fast.encode(text, add_special_tokens=False), (name, text)
