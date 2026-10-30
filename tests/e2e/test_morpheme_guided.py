"""Morpheme-guided BPE: occurrence-level boundary prior (U5) and a strict train() signature (P8)."""

import math
import warnings

import pytest

from tokenisers import BpeTokenizer, MorphemeBpeTokenizer

# "unhappy" is the only annotated word; the others share its pieces so plain BPE learns
# tokens across un|happy, through merges of already-merged tokens (un + happy).
CORPUS = ["unhappy " * 100, "happy " * 40, "unhappiness " * 20, "undo " * 30]
LEXICON = {"unhappy": ["un", "happy"]}


def _tokens(tok):
    """Vocabulary in id order, without the binding's <unused_N> padding of plain BPE."""
    toks = [tok.id_to_bytes(i) for i in range(tok.vocab_size)]
    return [t for t in toks if not t.startswith(b"<unused_")]


def _pieces(tok, text):
    return [tok.id_to_bytes(i) for i in tok.encode(text)]


def _crossing(tok, word: bytes, boundary: int):
    """Learned tokens that occur inside `word` spanning byte offset `boundary`."""
    out = []
    for t in _tokens(tok)[256:]:
        for p in range(len(word) - len(t) + 1):
            if p < boundary < p + len(t) and word[p : p + len(t)] == t:
                out.append(t)
    return out


def _train_quiet(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return MorphemeBpeTokenizer.train(**kwargs)


def test_strong_prior_learns_no_token_across_a_lexicon_boundary():
    plain = BpeTokenizer.train(CORPUS, vocab_size=296)
    assert _crossing(plain, b"unhappy", 2), "plain BPE must cross un|happy for the test to mean anything"

    tok = _train_quiet(corpus=CORPUS, vocab_size=296, morpheme_lexicon=LEXICON, lambda_weight=50.0)
    assert _crossing(tok, b"unhappy", 2) == []
    assert _pieces(tok, "unhappy") == [b"un", b"happy"]
    assert tok.decode(tok.encode("unhappy unhappiness")) == "unhappy unhappiness"


def test_review_probe_no_merged_token_spans_un_h():
    # Reviewer repro: lambda=50 used to learn unhap/unhappy/unhappiness from merged-token pairs.
    corpus = ["unhappy unhappy unhappiness unhappy redo undo"] * 50
    lex = {"unhappy": ["un", "happy"], "unhappiness": ["un", "happi", "ness"]}
    tok = _train_quiet(corpus=corpus, vocab_size=275, morpheme_lexicon=lex, lambda_weight=50.0)
    assert [t for t in _tokens(tok)[256:] if b"nh" in t] == []
    assert _pieces(tok, "unhappy") == [b"un", b"happy"]
    assert _pieces(tok, "unhappiness") == [b"un", b"happi", b"ness"]


@pytest.mark.parametrize("lexicon, lam", [({}, 50.0), (LEXICON, 0.0), ({"zzz": ["z", "zz"]}, 50.0)])
def test_without_effective_prior_training_equals_plain_bpe(lexicon, lam):
    plain = BpeTokenizer.train(CORPUS, vocab_size=266)
    tok = MorphemeBpeTokenizer.train(CORPUS, vocab_size=266, morpheme_lexicon=lexicon, lambda_weight=lam)
    assert tok.vocab_size == 266
    assert _tokens(tok) == _tokens(plain)


@pytest.mark.parametrize(
    "scorer, params", [("wordpiece", {"alpha": 0.25}), ("length_penalized", {"beta": 2.0}), ("mdl", {"penalty": 4.0})]
)
def test_empty_lexicon_matches_train_bpe_with_the_same_scorer_params(scorer, params, sample_english_text):
    corpus = [sample_english_text * 3]
    tok = _train_quiet(corpus=corpus, vocab_size=300, scorer=scorer, **params)
    reference = BpeTokenizer.train(corpus, vocab_size=300, scorer=scorer, min_frequency=1, **params)
    assert _tokens(tok) == _tokens(reference)


@pytest.mark.parametrize(
    "scorer, params, lexicon",
    [
        ("wordpiece", {"alpha": 0.25}, {}),
        ("length_penalized", {"beta": 2.0}, {}),
        # MDL's entry cost is a constant offset: it only reorders merges once the prior scales scores.
        ("mdl", {"penalty": 0.0}, LEXICON),
    ],
)
def test_inner_scorer_parameters_change_training(scorer, params, lexicon, sample_english_text):
    corpus = CORPUS if lexicon else [sample_english_text * 3]
    vocab = 270 if lexicon else 300
    common = {"corpus": corpus, "vocab_size": vocab, "morpheme_lexicon": lexicon, "scorer": scorer}
    default = _train_quiet(**common)
    tuned = _train_quiet(**common, **params)
    assert _tokens(tuned) != _tokens(default), f"{params} had no effect"


def test_unknown_keyword_raises_type_error():
    with pytest.raises(TypeError):
        MorphemeBpeTokenizer.train(CORPUS, vocab_size=300, morpheme_lexicon=LEXICON, entry_cost=3.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lambda_weight": -1.0},
        {"lambda_weight": math.nan},
        {"lambda_weight": math.inf},
        {"alpha": 0.5},  # alpha only applies to wordpiece
        {"scorer": "wordpiece", "alpha": -1.0},
        {"scorer": "unknown"},
        {"vocab_size": 257, "special_tokens": ["<a>", "<b>"]},
    ],
)
def test_invalid_arguments_raise_value_error(kwargs):
    args = {"corpus": CORPUS, "vocab_size": 300, "morpheme_lexicon": LEXICON, **kwargs}
    with pytest.raises(ValueError):
        MorphemeBpeTokenizer.train(**args)


def test_special_tokens_are_kept_inside_vocab_size_and_encoded_whole():
    tok = MorphemeBpeTokenizer.train(
        CORPUS, vocab_size=270, morpheme_lexicon=LEXICON, special_tokens=["<|endoftext|>", "<pad>"]
    )
    assert tok.vocab_size == 270
    eot = _tokens(tok).index(b"<|endoftext|>")
    ids = tok.encode("unhappy<|endoftext|>undo")
    assert eot in ids
    assert tok.decode(ids) == "unhappy<|endoftext|>undo"


def test_min_frequency_is_honoured_and_early_stop_warns():
    with pytest.warns(UserWarning, match="stopped at 256 tokens"):
        tok = MorphemeBpeTokenizer.train(CORPUS, vocab_size=300, morpheme_lexicon=LEXICON, min_frequency=10_000)
    assert tok.vocab_size == 256


def test_non_surface_segmentations_are_ignored_with_a_warning():
    lex = {"happiness": ["happy", "ness"]}  # canonical, not a byte split of the word
    with pytest.warns(UserWarning, match="do not concatenate"):
        tok = MorphemeBpeTokenizer.train(CORPUS, vocab_size=266, morpheme_lexicon=lex, lambda_weight=50.0)
    assert _tokens(tok) == _tokens(BpeTokenizer.train(CORPUS, vocab_size=266))
