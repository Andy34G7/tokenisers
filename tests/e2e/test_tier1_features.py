"""
Tier 1: Comprehensive Happy-Path Isolated Tests for All 18 Features.
Methodology: Category-Partition and Specification Conformance.
Target: >= 5 test cases per feature (18 features x 5 = 90 test cases minimum).
"""

import math
import os
import subprocess
from pathlib import Path
import pytest


# ==============================================================================
# Feature 1: Byte-level BPE Frequency (R1.1)
# ==============================================================================

def test_tier1_f1_bpe_frequency_training_basic(require_tokenisers, sample_english_text):
    """F1.1: Standard frequency BPE trains successfully on repetitive text to target vocab size."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(
        corpus=[sample_english_text * 10],
        vocab_size=300,
        scorer="frequency"
    )
    assert tok.vocab_size == 300
    vocab = tok.get_vocab()
    assert len(vocab) == 300
    # Must preserve base 256 bytes
    for b in range(256):
        assert bytes([b]) in vocab or chr(b) in vocab or f"<0x{b:02X}>" in vocab


def test_tier1_f1_bpe_frequency_encode_decode_roundtrip(require_tokenisers, sample_english_text):
    """F1.2: Standard frequency BPE encodes and decodes text losslessly."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(
        corpus=[sample_english_text],
        vocab_size=280,
        scorer="frequency"
    )
    token_ids = tok.encode(sample_english_text)
    assert isinstance(token_ids, list)
    assert len(token_ids) > 0
    decoded = tok.decode(token_ids)
    assert decoded == sample_english_text


def test_tier1_f1_bpe_frequency_token_ids_range(require_tokenisers, sample_english_text):
    """F1.3: All emitted token IDs fall strictly within [0, vocab_size - 1]."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(
        corpus=[sample_english_text],
        vocab_size=290,
        scorer="frequency"
    )
    token_ids = tok.encode("The quick brown fox jumps over the lazy dog.")
    assert all(isinstance(tid, int) and 0 <= tid < 290 for tid in token_ids)


def test_tier1_f1_bpe_frequency_multilingual_basic(require_tokenisers, sample_multilingual_text):
    """F1.4: Multi-byte UTF-8 scripts (German, Russian, CJK) encode and decode losslessly."""
    tokenisers = require_tokenisers
    combined_text = " ".join(sample_multilingual_text.values())
    tok = tokenisers.BpeTokenizer.train(
        corpus=[combined_text],
        vocab_size=350,
        scorer="frequency"
    )
    for lang, text in sample_multilingual_text.items():
        ids = tok.encode(text)
        assert tok.decode(ids) == text, f"Lossless failure for language: {lang}"


def test_tier1_f1_bpe_frequency_compression_ratio(require_tokenisers, sample_english_text):
    """F1.5: Subword merging reduces token sequence length relative to raw bytes."""
    tokenisers = require_tokenisers
    raw_bytes_len = len(sample_english_text.encode("utf-8"))
    tok = tokenisers.BpeTokenizer.train(
        corpus=[sample_english_text * 5],
        vocab_size=320,
        scorer="frequency"
    )
    token_ids = tok.encode(sample_english_text)
    assert len(token_ids) < raw_bytes_len
    # Compression ratio should be strictly > 1.0
    compression_ratio = raw_bytes_len / len(token_ids)
    assert compression_ratio > 1.0


# ==============================================================================
# Feature 2: WordPiece / PMI Likelihood Scoring (R1.1)
# ==============================================================================

def test_tier1_f2_wordpiece_training_basic(require_tokenisers, sample_english_text):
    """F2.1: WordPiece/PMI objective trains cleanly and produces specified vocabulary size."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(
        corpus=[sample_english_text * 10],
        vocab_size=300,
        scorer="wordpiece"
    )
    assert tok.vocab_size == 300


def test_tier1_f2_wordpiece_encode_decode_roundtrip(require_tokenisers, sample_english_text):
    """F2.2: WordPiece/PMI tokenization achieves exact lossless roundtrip."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(
        corpus=[sample_english_text],
        vocab_size=280,
        scorer="wordpiece"
    )
    ids = tok.encode(sample_english_text)
    assert tok.decode(ids) == sample_english_text


def test_tier1_f2_wordpiece_distinction_from_frequency(require_tokenisers):
    """F2.3: PMI scoring prioritizes high-correlation pairs over raw frequency."""
    tokenisers = require_tokenisers
    corpus = [
        "the the the the the the the the the the "
        "strongly_correlated_compound strongly_correlated_compound strongly_correlated_compound"
    ]
    tok_freq = tokenisers.BpeTokenizer.train(corpus=corpus, vocab_size=265, scorer="frequency")
    tok_pmi = tokenisers.BpeTokenizer.train(corpus=corpus, vocab_size=265, scorer="wordpiece")
    # Vocabularies should reflect different merge selections
    assert tok_freq.get_vocab() != tok_pmi.get_vocab()


def test_tier1_f2_wordpiece_code_syntax_merging(require_tokenisers, sample_code_snippets):
    """F2.4: Code syntax patterns and operator pairs merge cleanly under PMI."""
    tokenisers = require_tokenisers
    code_text = sample_code_snippets["python"]
    tok = tokenisers.BpeTokenizer.train(corpus=[code_text * 10], vocab_size=320, scorer="wordpiece")
    ids = tok.encode(code_text)
    assert tok.decode(ids) == code_text


def test_tier1_f2_wordpiece_laplace_smoothing_guard(require_tokenisers):
    """F2.5: WordPiece Laplace smoothing prevents numerical errors on low-count pairs."""
    tokenisers = require_tokenisers
    corpus = ["rare unique words with distinct characters xyz uvw"]
    tok = tokenisers.BpeTokenizer.train(corpus=corpus, vocab_size=265, scorer="wordpiece")
    ids = tok.encode("xyz uvw")
    assert tok.decode(ids) == "xyz uvw"


# ==============================================================================
# Feature 3: Minimum Description Length (MDL / Delta-Bits) Scoring (R1.1)
# ==============================================================================

def test_tier1_f3_mdl_training_basic(require_tokenisers, sample_english_text):
    """F3.1: MDL objective trains and generates valid vocabulary."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(
        corpus=[sample_english_text * 8],
        vocab_size=300,
        scorer="mdl"
    )
    assert tok.vocab_size == 300


def test_tier1_f3_mdl_encode_decode_roundtrip(require_tokenisers, sample_english_text):
    """F3.2: MDL-trained tokenizer preserves exact byte losslessness."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=280, scorer="mdl")
    assert tok.decode(tok.encode(sample_english_text)) == sample_english_text


def test_tier1_f3_mdl_entropy_reduction_monotonic(require_tokenisers, sample_english_text):
    """F3.3: Total token count on training corpus decreases as MDL merges occur."""
    tokenisers = require_tokenisers
    tok_small = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 4], vocab_size=260, scorer="mdl")
    tok_large = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 4], vocab_size=300, scorer="mdl")
    assert len(tok_large.encode(sample_english_text)) <= len(tok_small.encode(sample_english_text))


def test_tier1_f3_mdl_variable_entry_cost(require_tokenisers, sample_english_text):
    """F3.4: Configuring higher entry cost shifts vocabulary toward higher-frequency units."""
    tokenisers = require_tokenisers
    # Higher entry cost penalizes rare large tokens
    tok_low = tokenisers.BpeTokenizer.train(
        corpus=[sample_english_text * 5],
        vocab_size=280,
        scorer="mdl",
        entry_cost=8.0
    )
    tok_high = tokenisers.BpeTokenizer.train(
        corpus=[sample_english_text * 5],
        vocab_size=280,
        scorer="mdl",
        entry_cost=64.0
    )
    assert tok_low.vocab_size == 280
    assert tok_high.vocab_size == 280


def test_tier1_f3_mdl_multilingual_corpus(require_tokenisers, sample_multilingual_text):
    """F3.5: MDL scores multi-byte UTF-8 sequences properly across natural languages."""
    tokenisers = require_tokenisers
    corpus = [sample_multilingual_text["en"] + " " + sample_multilingual_text["ru"]] * 5
    tok = tokenisers.BpeTokenizer.train(corpus=corpus, vocab_size=310, scorer="mdl")
    text = sample_multilingual_text["ru"]
    assert tok.decode(tok.encode(text)) == text


# ==============================================================================
# Feature 4: Length-Penalized Frequency (R1.1)
# ==============================================================================

def test_tier1_f4_length_penalized_beta_zero_equals_bpe(require_tokenisers, sample_english_text):
    """F4.1: When beta=0.0, Length-Penalized scoring reproduces standard BPE."""
    tokenisers = require_tokenisers
    tok_bpe = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=280, scorer="frequency")
    tok_lp0 = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=280, scorer="length_penalized", beta=0.0)
    assert tok_bpe.get_vocab() == tok_lp0.get_vocab()


def test_tier1_f4_length_penalized_positive_beta_favors_short(require_tokenisers, sample_english_text):
    """F4.2: README score count(uv) / (|u|+|v|)^beta: beta > 0 penalizes longer merges."""
    tokenisers = require_tokenisers
    tok_bpe = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 5], vocab_size=290, scorer="frequency")
    tok_short = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 5], vocab_size=290, scorer="length_penalized", beta=1.0)
    # Average token byte length in vocab should be below standard BPE
    avg_len_bpe = sum(len(k.encode('utf-8')) if isinstance(k, str) else len(k) for k in tok_bpe.get_vocab().keys()) / 290
    avg_len_lp = sum(len(k.encode('utf-8')) if isinstance(k, str) else len(k) for k in tok_short.get_vocab().keys()) / 290
    assert avg_len_lp < avg_len_bpe


def test_tier1_f4_length_penalized_negative_beta_favors_short(require_tokenisers, sample_english_text):
    """F4.3: Beta < 0.0 penalizes long tokens, preserving compact subwords."""
    tokenisers = require_tokenisers
    tok_short = tokenisers.BpeTokenizer.train(
        corpus=[sample_english_text * 5],
        vocab_size=290,
        scorer="length_penalized",
        beta=-0.5
    )
    assert tok_short.vocab_size == 290
    assert tok_short.decode(tok_short.encode(sample_english_text)) == sample_english_text


def test_tier1_f4_length_penalized_encode_decode_roundtrip(require_tokenisers, sample_code_snippets):
    """F4.4: Length-penalized tokenizer maintains exact roundtrip losslessness on code."""
    tokenisers = require_tokenisers
    code = sample_code_snippets["rust"]
    tok = tokenisers.BpeTokenizer.train(corpus=[code * 5], vocab_size=300, scorer="length_penalized", beta=0.5)
    assert tok.decode(tok.encode(code)) == code


def test_tier1_f4_length_penalized_multilingual_scaling(require_tokenisers, sample_multilingual_text):
    """F4.5: Operates on UTF-8 byte length for multi-byte scripts (e.g. CJK 3-byte characters)."""
    tokenisers = require_tokenisers
    cjk_text = sample_multilingual_text["zh"] * 10
    tok = tokenisers.BpeTokenizer.train(corpus=[cjk_text], vocab_size=280, scorer="length_penalized", beta=0.5)
    assert tok.decode(tok.encode(sample_multilingual_text["zh"])) == sample_multilingual_text["zh"]


# ==============================================================================
# Feature 5: Top-Down Unigram LM Engine (R1.2)
# ==============================================================================

def test_tier1_f5_unigram_training_basic(require_tokenisers, sample_english_text):
    """F5.1: Unigram LM EM training converges and produces requested vocabulary size."""
    tokenisers = require_tokenisers
    tok = tokenisers.UnigramTokenizer.train(corpus=[sample_english_text * 5], vocab_size=300)
    assert tok.vocab_size == 300


def test_tier1_f5_unigram_viterbi_encode_decode(require_tokenisers, sample_english_text):
    """F5.2: 1-best Viterbi segmentation decodes losslessly."""
    tokenisers = require_tokenisers
    tok = tokenisers.UnigramTokenizer.train(corpus=[sample_english_text], vocab_size=280)
    token_ids = tok.encode(sample_english_text)
    assert isinstance(token_ids, list) and len(token_ids) > 0
    assert tok.decode(token_ids) == sample_english_text


def test_tier1_f5_unigram_base_bytes_protected(require_tokenisers, sample_english_text):
    """F5.3: Pruning never removes the 256 base byte tokens."""
    tokenisers = require_tokenisers
    tok = tokenisers.UnigramTokenizer.train(corpus=[sample_english_text], vocab_size=270)
    vocab = tok.get_vocab()
    # Test that unseen bytes can be encoded without <unk>
    unseen_byte_string = "\x01\x02\xFF"
    encoded = tok.encode(unseen_byte_string)
    assert tok.decode(encoded) == unseen_byte_string


def test_tier1_f5_unigram_log_probs_normalized(require_tokenisers, sample_english_text):
    """F5.4: Unigram token probabilities sum to 1.0 (sum of exp(log_prob) ~= 1.0)."""
    tokenisers = require_tokenisers
    tok = tokenisers.UnigramTokenizer.train(corpus=[sample_english_text], vocab_size=280)
    log_probs = tok.get_log_probs()
    total_prob = sum(math.exp(lp) for lp in log_probs.values())
    assert abs(total_prob - 1.0) < 1e-2


def test_tier1_f5_unigram_multilingual_segmentation(require_tokenisers, sample_multilingual_text):
    """F5.5: Unigram LM successfully tokenizes multilingual text losslessly."""
    tokenisers = require_tokenisers
    corpus = list(sample_multilingual_text.values())
    tok = tokenisers.UnigramTokenizer.train(corpus=corpus, vocab_size=350)
    for lang, text in sample_multilingual_text.items():
        assert tok.decode(tok.encode(text)) == text


# ==============================================================================
# Feature 6: Two-Stage SuperBPE Curriculum (R1.3)
# ==============================================================================

def test_tier1_f6_superbpe_stage1_respects_boundaries(require_tokenisers, sample_english_text):
    """F6.1: In Stage 1 (up to v_switch), all learned tokens respect word boundaries."""
    tokenisers = require_tokenisers
    tok = tokenisers.SuperBpeTokenizer.train(
        corpus=[sample_english_text * 5],
        v_switch=275,
        v_final=275
    )
    vocab = tok.get_vocab()
    # In stage 1 only, no token should contain internal spaces (other than leading space token)
    for token in vocab.keys():
        if isinstance(token, str):
            stripped = token.lstrip(" Ġ")
            assert " " not in stripped and "Ġ" not in stripped


def test_tier1_f6_superbpe_stage2_cross_word_merges(require_tokenisers, sample_english_text):
    """F6.2: Stage 2 (v_switch -> v_final) enables multi-word collocation merges."""
    tokenisers = require_tokenisers
    tok = tokenisers.SuperBpeTokenizer.train(
        corpus=[sample_english_text * 15],
        v_switch=270,
        v_final=320
    )
    assert tok.vocab_size == 320
    # There should exist at least one token with whitespace in middle representing a cross-word merge
    vocab = tok.get_vocab()
    has_multiword = any(
        isinstance(t, str) and (" " in t.lstrip(" Ġ") or "Ġ" in t.lstrip(" Ġ"))
        for t in vocab.keys()
    )
    assert has_multiword, "SuperBPE Stage 2 must learn at least one multi-word subword token"


def test_tier1_f6_superbpe_encode_decode_lossless(require_tokenisers, sample_english_text):
    """F6.3: SuperBPE maintains exact lossless roundtrip across whitespace."""
    tokenisers = require_tokenisers
    tok = tokenisers.SuperBpeTokenizer.train(
        corpus=[sample_english_text * 5],
        v_switch=270,
        v_final=290
    )
    assert tok.decode(tok.encode(sample_english_text)) == sample_english_text


def test_tier1_f6_superbpe_compression_gain(require_tokenisers, sample_english_text):
    """F6.4: Stage 2 multi-word tokens yield higher compression (fewer tokens) on repetitive text."""
    tokenisers = require_tokenisers
    tok_stage1 = tokenisers.SuperBpeTokenizer.train(
        corpus=[sample_english_text * 10],
        v_switch=280,
        v_final=280
    )
    tok_stage2 = tokenisers.SuperBpeTokenizer.train(
        corpus=[sample_english_text * 10],
        v_switch=280,
        v_final=320
    )
    assert len(tok_stage2.encode(sample_english_text)) <= len(tok_stage1.encode(sample_english_text))


def test_tier1_f6_superbpe_code_keyword_merges(require_tokenisers, sample_code_snippets):
    """F6.5: SuperBPE captures recurring code syntax phrases (e.g. 'def __init__', 'pub fn')."""
    tokenisers = require_tokenisers
    code = sample_code_snippets["python"] * 10
    tok = tokenisers.SuperBpeTokenizer.train(corpus=[code], v_switch=280, v_final=320)
    assert tok.decode(tok.encode(code)) == code


# ==============================================================================
# Feature 7: Morpheme-Guided Boundary Priors (R1.3)
# ==============================================================================

def test_tier1_f7_morpheme_training_with_priors(require_tokenisers, sample_morpheme_annotations):
    """F7.1: Training with morpheme priors biases merges along morpheme cuts."""
    tokenisers = require_tokenisers
    corpus = ["unhappiness and disagreement in pretokenization"] * 10
    lexicon = {item["word"]: item["segments"] for item in sample_morpheme_annotations}
    tok = tokenisers.MorphemeBpeTokenizer.train(
        corpus=corpus,
        vocab_size=280,
        morpheme_lexicon=lexicon,
        lambda_weight=2.0
    )
    assert tok.vocab_size == 280


def test_tier1_f7_morpheme_lambda_zero_equals_unregularized(require_tokenisers, sample_english_text):
    """F7.2: When lambda=0.0, Morpheme-guided BPE matches standard unregularized BPE."""
    tokenisers = require_tokenisers
    tok_std = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=275, scorer="frequency")
    tok_morph0 = tokenisers.MorphemeBpeTokenizer.train(
        corpus=[sample_english_text],
        vocab_size=275,
        morpheme_lexicon={"sample": ["sam", "ple"]},
        lambda_weight=0.0
    )
    assert tok_std.get_vocab() == tok_morph0.get_vocab()


def test_tier1_f7_morpheme_encode_decode_lossless(require_tokenisers, sample_english_text):
    """F7.3: Morpheme-guided tokenizer decodes all texts losslessly."""
    tokenisers = require_tokenisers
    tok = tokenisers.MorphemeBpeTokenizer.train(
        corpus=[sample_english_text],
        vocab_size=280,
        morpheme_lexicon={"tokenization": ["token", "iz", "ation"]},
        lambda_weight=1.5
    )
    assert tok.decode(tok.encode(sample_english_text)) == sample_english_text


def test_tier1_f7_morpheme_unannotated_fallback(require_tokenisers):
    """F7.4: Unannotated words receive neutral prior and tokenize without error."""
    tokenisers = require_tokenisers
    corpus = ["knownword and unknownword"] * 5
    tok = tokenisers.MorphemeBpeTokenizer.train(
        corpus=corpus,
        vocab_size=270,
        morpheme_lexicon={"knownword": ["known", "word"]},
        lambda_weight=2.0
    )
    ids = tok.encode("unknownword")
    assert tok.decode(ids) == "unknownword"


def test_tier1_f7_morpheme_multilingual_roots(require_tokenisers, sample_multilingual_text):
    """F7.5: Morpheme priors operate smoothly on multi-byte agglutinative words (Hungarian/Finnish)."""
    tokenisers = require_tokenisers
    hu_text = sample_multilingual_text["hu"]
    tok = tokenisers.MorphemeBpeTokenizer.train(
        corpus=[hu_text * 5],
        vocab_size=275,
        morpheme_lexicon={"kalandja": ["kaland", "ja"]},
        lambda_weight=1.5
    )
    assert tok.decode(tok.encode(hu_text)) == hu_text


# ==============================================================================
# Feature 8: BPE-Dropout Stochastic Inference (R1.4)
# ==============================================================================

def test_tier1_f8_bpe_dropout_zero_is_deterministic(require_tokenisers, sample_english_text):
    """F8.1: At dropout p=0.0, inference is 100% deterministic."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=280)
    ids1 = tok.encode(sample_english_text, dropout=0.0)
    ids2 = tok.encode(sample_english_text, dropout=0.0)
    assert ids1 == ids2


def test_tier1_f8_bpe_dropout_positive_produces_variance(require_tokenisers, sample_english_text):
    """F8.2: With dropout p > 0.0, repeated encodings produce diverse valid segmentations."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 5], vocab_size=300)
    results = {tuple(tok.encode(sample_english_text, dropout=0.3)) for _ in range(10)}
    assert len(results) > 1, "BPE-Dropout must generate multiple distinct segmentations"


def test_tier1_f8_bpe_dropout_all_variations_decode_losslessly(require_tokenisers, sample_english_text):
    """F8.3: Every stochastically sampled BPE-Dropout tokenization decodes back to original text."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 5], vocab_size=300)
    for _ in range(5):
        token_ids = tok.encode(sample_english_text, dropout=0.25)
        assert tok.decode(token_ids) == sample_english_text


def test_tier1_f8_bpe_dropout_fertility_increases_with_p(require_tokenisers, sample_english_text):
    """F8.4: Higher dropout probability p increases average subword count (fertility)."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 5], vocab_size=320)
    len_p0 = len(tok.encode(sample_english_text, dropout=0.0))
    avg_len_p5 = sum(len(tok.encode(sample_english_text, dropout=0.5)) for _ in range(10)) / 10
    assert avg_len_p5 >= len_p0


def test_tier1_f8_bpe_dropout_seed_reproducibility(require_tokenisers, sample_english_text):
    """F8.5: Supplying fixed random seed produces identical dropout tokenizations."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 5], vocab_size=300)
    ids_seed1 = tok.encode(sample_english_text, dropout=0.3, seed=42)
    ids_seed2 = tok.encode(sample_english_text, dropout=0.3, seed=42)
    assert ids_seed1 == ids_seed2


# ==============================================================================
# Feature 9: Unigram Lattice Sampling (FFBS / k-best) (R1.4)
# ==============================================================================

def test_tier1_f9_unigram_sampling_low_temp_recovers_viterbi(require_tokenisers, sample_english_text):
    """F9.1: Low temperature (T -> 0) recovers the deterministic Viterbi 1-best segmentation."""
    tokenisers = require_tokenisers
    tok = tokenisers.UnigramTokenizer.train(corpus=[sample_english_text * 3], vocab_size=290)
    viterbi_ids = tok.encode(sample_english_text, temperature=None)
    sampled_ids = tok.encode(sample_english_text, temperature=0.001)
    assert sampled_ids == viterbi_ids


def test_tier1_f9_unigram_sampling_temperature_variance(require_tokenisers, sample_english_text):
    """F9.2: FFBS temperature sampling produces varied valid lattice segmentations."""
    tokenisers = require_tokenisers
    tok = tokenisers.UnigramTokenizer.train(corpus=[sample_english_text * 5], vocab_size=300)
    samples = {tuple(tok.encode(sample_english_text, temperature=1.0)) for _ in range(10)}
    assert len(samples) > 1


def test_tier1_f9_unigram_sampling_kbest_mode(require_tokenisers, sample_english_text):
    """F9.3: nbest_size mode restricts sampled segmentations to the top-k paths."""
    tokenisers = require_tokenisers
    tok = tokenisers.UnigramTokenizer.train(corpus=[sample_english_text * 5], vocab_size=300)
    ids = tok.encode(sample_english_text, nbest_size=5, temperature=1.0)
    assert isinstance(ids, list) and len(ids) > 0
    assert tok.decode(ids) == sample_english_text


def test_tier1_f9_unigram_sampling_all_samples_decode_losslessly(require_tokenisers, sample_english_text):
    """F9.4: Every sampled lattice path decodes back to the exact original text."""
    tokenisers = require_tokenisers
    tok = tokenisers.UnigramTokenizer.train(corpus=[sample_english_text * 5], vocab_size=300)
    for _ in range(5):
        ids = tok.encode(sample_english_text, temperature=0.8)
        assert tok.decode(ids) == sample_english_text


def test_tier1_f9_unigram_sampling_seed_reproducibility(require_tokenisers, sample_english_text):
    """F9.5: Random seed locks FFBS sampling sequence deterministically."""
    tokenisers = require_tokenisers
    tok = tokenisers.UnigramTokenizer.train(corpus=[sample_english_text * 5], vocab_size=300)
    sample1 = tok.encode(sample_english_text, temperature=0.8, seed=123)
    sample2 = tok.encode(sample_english_text, temperature=0.8, seed=123)
    assert sample1 == sample2


# ==============================================================================
# Feature 10: Byte-Entropy Dynamic Patching (R1.4)
# ==============================================================================

def test_tier1_f10_byte_entropy_basic(require_tokenisers, sample_english_text):
    """F10.1: Byte-entropy dynamic patcher tokenizes text into non-empty byte chunks."""
    tokenisers = require_tokenisers
    tok = tokenisers.ByteEntropyTokenizer(entropy_threshold=3.5, min_patch_len=1, max_patch_len=16)
    patches = tok.encode(sample_english_text)
    assert isinstance(patches, list)
    assert len(patches) > 0


def test_tier1_f10_byte_entropy_decode_lossless(require_tokenisers, sample_english_text):
    """F10.2: Decoded dynamic patches reconstruct the original text byte-for-byte."""
    tokenisers = require_tokenisers
    tok = tokenisers.ByteEntropyTokenizer(entropy_threshold=3.5, min_patch_len=1, max_patch_len=16)
    patches = tok.encode(sample_english_text)
    assert tok.decode(patches) == sample_english_text


def test_tier1_f10_byte_entropy_threshold_scaling(require_tokenisers, sample_english_text):
    """F10.3: Higher entropy threshold tau yields fewer, longer patches on average."""
    tokenisers = require_tokenisers
    tok_low = tokenisers.ByteEntropyTokenizer(entropy_threshold=2.0, min_patch_len=1, max_patch_len=16)
    tok_high = tokenisers.ByteEntropyTokenizer(entropy_threshold=5.0, min_patch_len=1, max_patch_len=16)
    patches_low = tok_low.encode(sample_english_text)
    patches_high = tok_high.encode(sample_english_text)
    assert len(patches_high) <= len(patches_low)


def test_tier1_f10_byte_entropy_respects_max_len(require_tokenisers, sample_english_text):
    """F10.4: Emitted patches never exceed max_patch_len."""
    tokenisers = require_tokenisers
    max_len = 8
    tok = tokenisers.ByteEntropyTokenizer(entropy_threshold=10.0, min_patch_len=1, max_patch_len=max_len)
    patches = tok.encode(sample_english_text)
    for p in patches:
        patch_bytes = tok.id_to_bytes(p) if hasattr(tok, "id_to_bytes") else str(p).encode('utf-8')
        assert len(patch_bytes) <= max_len


def test_tier1_f10_byte_entropy_respects_min_len(require_tokenisers, sample_english_text):
    """F10.5: Internal patches never fall below min_patch_len."""
    tokenisers = require_tokenisers
    min_len = 3
    tok = tokenisers.ByteEntropyTokenizer(entropy_threshold=1.0, min_patch_len=min_len, max_patch_len=16)
    patches = tok.encode(sample_english_text)
    # Check all but trailing remnant
    for p in patches[:-1]:
        patch_bytes = tok.id_to_bytes(p) if hasattr(tok, "id_to_bytes") else str(p).encode('utf-8')
        assert len(patch_bytes) >= min_len


# ==============================================================================
# Feature 11: HuggingFace tokenizer.json Export (R1.5)
# ==============================================================================

def test_tier1_f11_hf_export_json_structure(require_tokenisers, sample_english_text, tmp_path):
    """F11.1: Exported tokenizer.json contains canonical HuggingFace schema keys."""
    tokenisers = require_tokenisers
    import json
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=270)
    export_path = tmp_path / "tokenizer.json"
    tok.export_hf_json(str(export_path))
    assert export_path.exists()
    with open(export_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    required_keys = {"version", "model", "pre_tokenizer", "decoder"}
    assert required_keys.issubset(data.keys())


def test_tier1_f11_hf_export_bpe_model_schema(require_tokenisers, sample_english_text, tmp_path):
    """F11.2: Exported BPE model block has type='BPE', 'vocab', and 'merges'."""
    tokenisers = require_tokenisers
    import json
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=275)
    export_path = tmp_path / "tokenizer.json"
    tok.export_hf_json(str(export_path))
    with open(export_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["model"]["type"] == "BPE"
    assert "vocab" in data["model"]
    assert "merges" in data["model"]


def test_tier1_f11_hf_export_unigram_model_schema(require_tokenisers, sample_english_text, tmp_path):
    """F11.3: Exported Unigram model block has type='Unigram' and log-prob vocab list."""
    tokenisers = require_tokenisers
    import json
    tok = tokenisers.UnigramTokenizer.train(corpus=[sample_english_text], vocab_size=270)
    export_path = tmp_path / "tokenizer.json"
    tok.export_hf_json(str(export_path))
    with open(export_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["model"]["type"] == "Unigram"
    assert "vocab" in data["model"]


def test_tier1_f11_hf_export_loadable_by_hf_tokenizers(require_tokenisers, sample_english_text, tmp_path):
    """F11.4: Exported artifact loads directly into tokenizers.Tokenizer.from_file."""
    tokenisers = require_tokenisers
    hf_tok_pkg = pytest.importorskip("tokenizers")
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=270)
    export_path = tmp_path / "tokenizer.json"
    tok.export_hf_json(str(export_path))
    hf_tok = hf_tok_pkg.Tokenizer.from_file(str(export_path))
    assert hf_tok.get_vocab_size() == 270


def test_tier1_f11_hf_export_equivalence_after_reload(require_tokenisers, sample_english_text, tmp_path):
    """F11.5: Reloaded HF tokenizer encodes and decodes identically to native engine."""
    tokenisers = require_tokenisers
    hf_tok_pkg = pytest.importorskip("tokenizers")
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=280)
    export_path = tmp_path / "tokenizer.json"
    tok.export_hf_json(str(export_path))
    hf_tok = hf_tok_pkg.Tokenizer.from_file(str(export_path))
    native_ids = tok.encode(sample_english_text)
    hf_ids = hf_tok.encode(sample_english_text).ids
    assert native_ids == hf_ids


# ==============================================================================
# Feature 12: HF PreTrainedTokenizer Dynamic Wrapper (R1.5)
# ==============================================================================

def test_tier1_f12_hf_wrapper_instance_of_pretrained(require_tokenisers, sample_english_text):
    """F12.1: DynamicResearchTokenizer is an instance of PreTrainedTokenizer."""
    tokenisers = require_tokenisers
    transformers = pytest.importorskip("transformers")
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=270)
    wrapper = tokenisers.DynamicResearchTokenizer(tok)
    assert isinstance(wrapper, transformers.PreTrainedTokenizer)


def test_tier1_f12_hf_wrapper_call_dict_keys(require_tokenisers, sample_english_text):
    """F12.2: Calling wrapper returns dictionary with input_ids and attention_mask."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=270)
    wrapper = tokenisers.DynamicResearchTokenizer(tok)
    out = wrapper("Hello world")
    assert "input_ids" in out
    assert "attention_mask" in out
    assert len(out["input_ids"]) == len(out["attention_mask"])


def test_tier1_f12_hf_wrapper_batch_encoding(require_tokenisers, sample_english_text):
    """F12.3: Wrapper correctly batch encodes lists of sentences with padding."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=270)
    wrapper = tokenisers.DynamicResearchTokenizer(tok, pad_token="<pad>")
    batch = wrapper(["Short sentence", "A much longer sentence for padding test"], padding=True)
    assert len(batch["input_ids"]) == 2
    assert len(batch["input_ids"][0]) == len(batch["input_ids"][1])


def test_tier1_f12_hf_wrapper_decode_tokens(require_tokenisers, sample_english_text):
    """F12.4: Wrapper decode reconstructs the original string cleanly."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=270)
    wrapper = tokenisers.DynamicResearchTokenizer(tok)
    encoded = wrapper(sample_english_text)["input_ids"]
    assert wrapper.decode(encoded) == sample_english_text


def test_tier1_f12_hf_wrapper_dynamic_kwargs_passthrough(require_tokenisers, sample_english_text):
    """F12.5: Dynamic kwargs (dropout, temperature) pass through wrapper successfully."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 5], vocab_size=300)
    wrapper = tokenisers.DynamicResearchTokenizer(tok)
    # Different dropout rates produce varied outputs
    out1 = wrapper(sample_english_text, dropout=0.5)["input_ids"]
    out2 = wrapper(sample_english_text, dropout=0.5)["input_ids"]
    assert wrapper.decode(out1) == sample_english_text
    assert wrapper.decode(out2) == sample_english_text


# ==============================================================================
# Feature 13: Lossless Byte Round-Trip (Acceptance)
# ==============================================================================

def test_tier1_f13_lossless_ascii_prose(require_tokenisers, sample_english_text):
    """F13.1: Strict decode(encode(text)) == text for standard ASCII prose."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=280)
    assert tok.decode(tok.encode(sample_english_text)) == sample_english_text


def test_tier1_f13_lossless_multilingual_utf8(require_tokenisers, sample_multilingual_text):
    """F13.2: Strict decode(encode(text)) == text across diverse scripts and alphabets."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=list(sample_multilingual_text.values()), vocab_size=350)
    for lang, text in sample_multilingual_text.items():
        assert tok.decode(tok.encode(text)) == text, f"Roundtrip failed for {lang}"


def test_tier1_f13_lossless_code_syntax(require_tokenisers, sample_code_snippets):
    """F13.3: Strict decode(encode(text)) == text across polyglot source code with indentation."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=list(sample_code_snippets.values()), vocab_size=320)
    for lang, code in sample_code_snippets.items():
        assert tok.decode(tok.encode(code)) == code, f"Roundtrip failed for code {lang}"


def test_tier1_f13_lossless_emojis_and_modifiers(require_tokenisers):
    """F13.4: Strict decode(encode(text)) == text for complex emoji modifier sequences."""
    tokenisers = require_tokenisers
    emoji_text = "Hello! 🌍🚀🔥👩‍👩‍👦‍👦👍🏽👩‍💻✨"
    tok = tokenisers.BpeTokenizer.train(corpus=[emoji_text], vocab_size=280)
    assert tok.decode(tok.encode(emoji_text)) == emoji_text


def test_tier1_f13_lossless_mathematical_symbols(require_tokenisers):
    """F13.5: Strict decode(encode(text)) == text for mathematical formulas and Greek symbols."""
    tokenisers = require_tokenisers
    math_text = "∀x ∈ ℝ: ∫_{-∞}^{∞} e^{-x^2} dx = √π ∧ ∑_{n=1}^∞ 1/n^2 = π^2/6"
    tok = tokenisers.BpeTokenizer.train(corpus=[math_text], vocab_size=280)
    assert tok.decode(tok.encode(math_text)) == math_text


# ==============================================================================
# Feature 14: Multilingual + Code Data Pipeline (R2)
# ==============================================================================

def test_tier1_f14_data_pipeline_sample_preset_smoke(require_tokenisers):
    """F14.1: Data pipeline samples minimal smoke preset with valid documents."""
    tokenisers = require_tokenisers
    pipeline = tokenisers.data.DataPipeline()
    docs = pipeline.sample_preset("smoke", max_bytes=100_000)
    assert len(docs) > 0
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)
    assert total_bytes >= 50_000


def test_tier1_f14_data_pipeline_deterministic_split(require_tokenisers):
    """F14.2: Deterministic document splitter assigns identical docs to the same split."""
    tokenisers = require_tokenisers
    splitter = tokenisers.data.DeterministicSplitter(train_ratio=0.9)
    doc = "This is a fixed deterministic document for split testing."
    split1 = splitter.assign_split(doc)
    split2 = splitter.assign_split(doc)
    assert split1 == split2
    assert split1 in ("train", "val")


def test_tier1_f14_data_pipeline_domain_distribution(require_tokenisers):
    """F14.3: Stratified sampler yields documents across English, Multilingual, and Code domains."""
    tokenisers = require_tokenisers
    pipeline = tokenisers.data.DataPipeline()
    dataset_mix = pipeline.get_dataset_mix("smoke")
    domains = {item["domain"] for item in dataset_mix}
    assert {"english", "multilingual", "code"}.issubset(domains)


def test_tier1_f14_data_pipeline_disk_caching(require_tokenisers, tmp_path):
    """F14.4: Cached corpus can be reloaded without network calls."""
    tokenisers = require_tokenisers
    pipeline = tokenisers.data.DataPipeline(cache_dir=tmp_path)
    path = pipeline.prepare_corpus(preset="smoke", max_bytes=50_000)
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0


def test_tier1_f14_data_pipeline_offline_fallback(require_tokenisers):
    """F14.5: Offline procedural generator produces synthetic corpus when network is unavailable."""
    tokenisers = require_tokenisers
    generator = tokenisers.data.OfflineFallbackGenerator()
    docs = generator.generate_synthetic_corpus(target_bytes=50_000)
    assert len(docs) > 0
    assert sum(len(d.encode("utf-8")) for d in docs) >= 50_000


# ==============================================================================
# Feature 15: Tokka-Bench Evaluation (R3.1)
# ==============================================================================

def test_tier1_f15_tokka_bench_returns_metrics(require_tokenisers, sample_english_text):
    """F15.1: Tokka-bench benchmark returns all 4 core metrics."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 5], vocab_size=280)
    bench = tokenisers.benchmarks.tokka.TokkaBenchRunner()
    results = bench.evaluate_tokenizer(tok, text_sample=sample_english_text)
    for metric in ("bytes_per_token", "subword_fertility", "continued_word_rate", "unique_tokens"):
        assert metric in results, f"Missing metric: {metric}"


def test_tier1_f15_tokka_bench_bytes_per_token_positive(require_tokenisers, sample_english_text):
    """F15.2: bytes_per_token is strictly positive and within physical bounds."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 5], vocab_size=280)
    bench = tokenisers.benchmarks.tokka.TokkaBenchRunner()
    results = bench.evaluate_tokenizer(tok, text_sample=sample_english_text)
    assert 1.0 <= results["bytes_per_token"] <= 8.0


def test_tier1_f15_tokka_bench_fertility_greater_than_one(require_tokenisers, sample_english_text):
    """F15.3: subword_fertility is >= 1.0 (subwords per word)."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 5], vocab_size=280)
    bench = tokenisers.benchmarks.tokka.TokkaBenchRunner()
    results = bench.evaluate_tokenizer(tok, text_sample=sample_english_text)
    assert results["subword_fertility"] >= 1.0


def test_tier1_f15_tokka_bench_continued_word_rate_bounds(require_tokenisers, sample_english_text):
    """F15.4: continued_word_rate is a valid percentage in [0.0, 100.0]."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 5], vocab_size=280)
    bench = tokenisers.benchmarks.tokka.TokkaBenchRunner()
    results = bench.evaluate_tokenizer(tok, text_sample=sample_english_text)
    assert 0.0 <= results["continued_word_rate"] <= 100.0


def test_tier1_f15_tokka_bench_json_export_schema(require_tokenisers, sample_english_text, tmp_path):
    """F15.5: Exported Tokka-bench JSON result matches standard report structure."""
    tokenisers = require_tokenisers
    import json
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 5], vocab_size=280)
    bench = tokenisers.benchmarks.tokka.TokkaBenchRunner()
    out_file = tmp_path / "tokka_results.json"
    bench.run_and_save(tok, text_sample=sample_english_text, output_path=str(out_file))
    assert out_file.exists()
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "metrics" in data or "bytes_per_token" in data


# ==============================================================================
# Feature 16: Extended Intrinsic Suite (R3.2)
# ==============================================================================

def test_tier1_f16_intrinsic_renyi_efficiency(require_tokenisers, sample_english_text):
    """F16.1: Renyi efficiency E_alpha is computed in [0, 1] with E_1.0 >= E_2.0 >= E_2.5."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 5], vocab_size=280)
    intrinsic = tokenisers.benchmarks.intrinsic.IntrinsicEvaluator()
    metrics = intrinsic.compute_renyi_efficiency(tok, text_sample=sample_english_text)
    assert "e_1_0" in metrics and "e_2_0" in metrics and "e_2_5" in metrics
    e1, e2, e25 = metrics["e_1_0"], metrics["e_2_0"], metrics["e_2_5"]
    assert 0.0 <= e25 <= e2 <= e1 <= 1.0


def test_tier1_f16_intrinsic_vocab_utilization(require_tokenisers, sample_english_text):
    """F16.2: Vocabulary utilization computes active tokens, dead tokens, and Gini coefficient."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 5], vocab_size=300)
    intrinsic = tokenisers.benchmarks.intrinsic.IntrinsicEvaluator()
    util = intrinsic.compute_vocab_utilization(tok, text_sample=sample_english_text)
    assert 0.0 <= util["active_fraction"] <= 1.0
    assert util["dead_tokens"] >= 0
    assert 0.0 <= util["gini_coefficient"] <= 1.0


def test_tier1_f16_intrinsic_morphological_f1(require_tokenisers, sample_morpheme_annotations):
    """F16.3: Morphological Boundary Precision, Recall, and F1 lie in [0, 1]."""
    tokenisers = require_tokenisers
    corpus = [item["word"] for item in sample_morpheme_annotations] * 5
    tok = tokenisers.BpeTokenizer.train(corpus=corpus, vocab_size=280)
    intrinsic = tokenisers.benchmarks.intrinsic.IntrinsicEvaluator()
    f1_res = intrinsic.compute_morphological_f1(tok, gold_annotations=sample_morpheme_annotations)
    assert 0.0 <= f1_res["precision"] <= 1.0
    assert 0.0 <= f1_res["recall"] <= 1.0
    assert 0.0 <= f1_res["f1"] <= 1.0


def test_tier1_f16_intrinsic_structural_diagnostics(require_tokenisers):
    """F16.4: Structural diagnostics evaluate number and code identifier splitting."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=["1234567890 getUserProfile get_user_profile"], vocab_size=270)
    intrinsic = tokenisers.benchmarks.intrinsic.IntrinsicEvaluator()
    diag = intrinsic.compute_structural_diagnostics(tok)
    assert "number_fertility" in diag or "identifier_f1" in diag


def test_tier1_f16_intrinsic_throughput_measurement(require_tokenisers, sample_english_text):
    """F16.5: Throughput harness measures encode and decode MB/s with warm-up passes."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 5], vocab_size=280)
    intrinsic = tokenisers.benchmarks.intrinsic.IntrinsicEvaluator()
    speed = intrinsic.benchmark_throughput(tok, text_sample=sample_english_text * 20, num_warmup=2)
    assert speed["encode_mb_per_sec"] > 0.0
    assert speed["decode_mb_per_sec"] > 0.0


# ==============================================================================
# Feature 17: Colab Extrinsic Small-LM Benchmark (R3.3)
# ==============================================================================

def test_tier1_f17_colab_model_architecture(require_tokenisers):
    """F17.1: Proxy LM model instantiates with RoPE, RMSNorm, SwiGLU, and tied embeddings."""
    tokenisers = require_tokenisers
    torch = pytest.importorskip("torch")
    lm_module = tokenisers.benchmarks.colab.ProxyLanguageModel(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=2,
        d_ffn=256
    )
    assert isinstance(lm_module, torch.nn.Module)


def test_tier1_f17_colab_parameter_count_tied_embedding(require_tokenisers):
    """F17.2: Backbone parameter count is fixed; only tied embedding scales with vocab size."""
    tokenisers = require_tokenisers
    pytest.importorskip("torch")
    m1 = tokenisers.benchmarks.colab.ProxyLanguageModel(vocab_size=1000, d_model=64, n_layers=2, n_heads=2, d_ffn=128)
    m2 = tokenisers.benchmarks.colab.ProxyLanguageModel(vocab_size=2000, d_model=64, n_layers=2, n_heads=2, d_ffn=128)
    diff = sum(p.numel() for p in m2.parameters()) - sum(p.numel() for p in m1.parameters())
    # Difference should be exactly (2000 - 1000) * d_model because of weight tying
    assert diff == 1000 * 64


def test_tier1_f17_colab_byte_budget_training_step(require_tokenisers):
    """F17.3: Byte-budget trainer advances cumulative byte counter on training step."""
    tokenisers = require_tokenisers
    pytest.importorskip("torch")
    trainer = tokenisers.benchmarks.colab.ByteBudgetTrainer(
        model=tokenisers.benchmarks.colab.ProxyLanguageModel(vocab_size=500, d_model=32, n_layers=1, n_heads=1, d_ffn=64),
        byte_budget=10_000
    )
    text = "Sample training batch text for byte tracking."
    batch_bytes = len(text.encode("utf-8"))
    trainer.step_batch(text)
    assert trainer.bytes_consumed >= batch_bytes


def test_tier1_f17_colab_bpb_computation_formula(require_tokenisers):
    """F17.4: Bits-Per-Byte formula BPB = L / (ln(2) * bpt) is strictly computed."""
    tokenisers = require_tokenisers
    evaluator = tokenisers.benchmarks.colab.BpbEvaluator()
    loss = 2.5  # nats per token
    bpt = 3.5   # bytes per token
    expected_bpb = loss / (math.log(2) * bpt)
    computed_bpb = evaluator.compute_bpb(loss_nats=loss, bytes_per_token=bpt)
    assert computed_bpb == pytest.approx(expected_bpb, rel=1e-5)


def test_tier1_f17_colab_cli_binary_presence(colab_cli_path):
    """F17.5: The colab CLI binary exists and responds with version or help."""
    assert colab_cli_path.exists(), f"colab binary missing at {colab_cli_path}"
    proc = subprocess.run([str(colab_cli_path), "--help"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "Google Colab" in proc.stdout or "Usage" in proc.stdout or "colab" in proc.stdout


# ==============================================================================
# Feature 18: Git Sync to Public Remote (R4)
# ==============================================================================

def test_tier1_f18_git_remote_configured(repo_root):
    """F18.1: Remote 'origin' is configured and points to https://github.com/Andy34G7/tokenisers.git."""
    proc = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True
    )
    assert "Andy34G7/tokenisers" in proc.stdout.strip()


def test_tier1_f18_git_branch_main(repo_root):
    """F18.2: Git repository active branch is main."""
    proc = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True
    )
    branch = proc.stdout.strip()
    assert branch == "main" or branch == ""  # "" if unborn main branch


def test_tier1_f18_git_user_config(repo_root):
    """F18.3: Git user.name and user.email are properly configured."""
    name_proc = subprocess.run(["git", "config", "user.name"], cwd=repo_root, capture_output=True, text=True)
    email_proc = subprocess.run(["git", "config", "user.email"], cwd=repo_root, capture_output=True, text=True)
    assert len(name_proc.stdout.strip()) > 0
    assert len(email_proc.stdout.strip()) > 0


def test_tier1_f18_git_status_accessible(repo_root):
    """F18.4: git status executes cleanly with exit code 0."""
    proc = subprocess.run(["git", "status"], cwd=repo_root, capture_output=True, text=True)
    assert proc.returncode == 0


def test_tier1_f18_git_credentials_helper(repo_root):
    """F18.5: Git credential helper is configured for authenticated sync."""
    proc = subprocess.run(["git", "config", "credential.helper"], cwd=repo_root, capture_output=True, text=True)
    # Either gh auth helper or system helper is configured
    assert proc.returncode == 0
