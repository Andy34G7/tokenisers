"""
Unit and integration tests for wp-bpe variants (wp-bpe-base, wp-bpe-ft, wp-bpe-pig).
"""

from __future__ import annotations

import pytest
from tokenisers import (
    WpBpeBaseTokenizer,
    WpBpeFtTokenizer,
    WpBpePigTokenizer,
    BpeTokenizer,
)


@pytest.fixture
def sample_corpus():
    return [
        "the internationalization of natural language processing systems",
        "understanding tokenization algorithms with mutual information",
        "subword units and byte level byte pair encoding mechanisms",
        "pointwise information gain measures correlation strength across tokens",
    ] * 6


def test_wp_bpe_variants_instantiate_and_train(require_tokenisers, sample_corpus):
    t_base = WpBpeBaseTokenizer.train(sample_corpus, vocab_size=280)
    t_ft = WpBpeFtTokenizer.train(sample_corpus, vocab_size=280, min_frequency=2)
    t_pig = WpBpePigTokenizer.train(sample_corpus, vocab_size=280)

    assert t_base.vocab_size == 280
    assert t_ft.vocab_size == 280
    assert t_pig.vocab_size == 280


def test_wp_bpe_lossless_roundtrip(require_tokenisers, sample_corpus):
    test_texts = [
        "hello world! 12345",
        "internationalization & misunderstandings",
        "def compute_gain(x, y): return x * y",
        "Съешь же ещё этих мягких французских булок",
        "道可道，非常道",
    ]

    for TokenizerCls in (WpBpeBaseTokenizer, WpBpeFtTokenizer, WpBpePigTokenizer):
        tok = TokenizerCls.train(sample_corpus, vocab_size=285)
        for text in test_texts:
            encoded = tok.encode(text)
            decoded = tok.decode(encoded)
            assert decoded == text, f"{TokenizerCls.__name__} failed lossless roundtrip on: {text}"


def test_wp_bpe_distinct_vocabularies(require_tokenisers):
    """Verify that uncalibrated PMI, thresholded PMI, and PIG yield distinct merge orders / vocabularies."""
    corpus = [
        "the internationalization of natural language processing systems",
        "understanding tokenization algorithms with mutual information",
        "rare_unique_pair_xyzz occurs_once_only",
    ] * 3

    t_base = WpBpeBaseTokenizer.train(corpus, vocab_size=275)
    t_ft = WpBpeFtTokenizer.train(corpus, vocab_size=275, min_frequency=2)
    t_pig = WpBpePigTokenizer.train(corpus, vocab_size=275)

    vocab_base = set(t_base.get_vocab().keys())
    vocab_ft = set(t_ft.get_vocab().keys())
    vocab_pig = set(t_pig.get_vocab().keys())

    # With singletons in the corpus, uncalibrated PMI merges rare unique pairs first,
    # while frequency-thresholded PMI filters them out, and PIG weights by frequency.
    assert vocab_base != vocab_pig
    assert vocab_ft != vocab_base
    assert vocab_ft != vocab_pig
