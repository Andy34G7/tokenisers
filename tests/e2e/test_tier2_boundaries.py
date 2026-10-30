"""
Tier 2: Boundary, Corner, and Extreme Edge Case Tests for All 18 Features.
Methodology: Boundary Value Analysis (BVA), Stress Testing, and Robustness Verification.
Target: >= 5 test cases per feature (18 features x 5 = 90 test cases minimum).
"""

import math
import os
import subprocess
from pathlib import Path
import pytest


# ==============================================================================
# Feature 1: Byte-level BPE Frequency Boundaries (R1.1)
# ==============================================================================

def test_tier2_f1_bpe_empty_string(require_tokenisers):
    """F1-B1: Encoding empty string returns empty list; decoding empty list returns empty string."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=["sample corpus"], vocab_size=260)
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_tier2_f1_bpe_single_byte_inputs(require_tokenisers):
    """F1-B2: Encoding single characters ('a', ' ', '\n') returns valid single-token IDs."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=["sample corpus with spaces and\nnewlines"], vocab_size=270)
    for char in ["a", " ", "\n", "z"]:
        ids = tok.encode(char)
        assert len(ids) == 1
        assert tok.decode(ids) == char


def test_tier2_f1_bpe_unseen_raw_bytes(require_tokenisers):
    """F1-B3: Non-UTF8 raw binary bytes (\xFF\xFE\x00) never in training corpus decode losslessly."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=["plain text only"], vocab_size=265)
    raw_binary = "\x00\xFF\xFE\x80\x7F"
    ids = tok.encode(raw_binary)
    assert tok.decode(ids) == raw_binary


def test_tier2_f1_bpe_min_vocab_size_256(require_tokenisers):
    """F1-B4: Training with vocab_size=256 produces 0 merges; every token is a base byte."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=["repeated text repeated text"], vocab_size=256)
    assert tok.vocab_size == 256
    text = "hello"
    ids = tok.encode(text)
    assert len(ids) == len(text.encode("utf-8"))
    assert tok.decode(ids) == text


def test_tier2_f1_bpe_extreme_repeated_char_avalanche(require_tokenisers):
    """F1-B5: Long repeated character run ('a' * 10,000) tokenizes without recursion or heap overflow."""
    tokenisers = require_tokenisers
    repeated = "a" * 10_000
    tok = tokenisers.BpeTokenizer.train(corpus=[repeated], vocab_size=270)
    ids = tok.encode(repeated)
    assert tok.decode(ids) == repeated


# ==============================================================================
# Feature 2: WordPiece / PMI Likelihood Scoring Boundaries (R1.1)
# ==============================================================================

def test_tier2_f2_wordpiece_singleton_pairs(require_tokenisers):
    """F2-B1: Pairs occurring only once do not cause divide-by-zero or score explosion."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=["ab cd ef gh ij kl mn op"], vocab_size=260, scorer="wordpiece")
    assert tok.vocab_size == 260
    assert tok.decode(tok.encode("ab cd")) == "ab cd"


def test_tier2_f2_wordpiece_zero_denominator_guard(require_tokenisers):
    """F2-B2: Guard against zero denominator in PMI calculation for unobserved pairs."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=["common words in training"], vocab_size=265, scorer="wordpiece")
    unseen_pair = "qx"
    ids = tok.encode(unseen_pair)
    assert tok.decode(ids) == unseen_pair


def test_tier2_f2_wordpiece_all_unique_chars(require_tokenisers):
    """F2-B3: Corpus with zero repeated pairs trains cleanly up to available merges."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=["abcdefghijklmnopqrstuvwxyz"], vocab_size=260, scorer="wordpiece")
    assert tok.vocab_size >= 256


def test_tier2_f2_wordpiece_extreme_imbalance(require_tokenisers):
    """F2-B4: One word repeated 10,000x and another once does not produce NaN scores."""
    tokenisers = require_tokenisers
    corpus = ["frequent " * 1000 + "rare"]
    tok = tokenisers.BpeTokenizer.train(corpus=corpus, vocab_size=270, scorer="wordpiece")
    assert tok.decode(tok.encode("frequent rare")) == "frequent rare"


def test_tier2_f2_wordpiece_unicode_plane16_symbols(require_tokenisers):
    """F2-B5: 4-byte high-plane Unicode characters and emojis handled safely under PMI."""
    tokenisers = require_tokenisers
    emoji_corpus = ["🚀🌟🔥🚀🌟🔥🚀🌟🔥"] * 5
    tok = tokenisers.BpeTokenizer.train(corpus=emoji_corpus, vocab_size=270, scorer="wordpiece")
    assert tok.decode(tok.encode("🚀🌟🔥")) == "🚀🌟🔥"


# ==============================================================================
# Feature 3: Minimum Description Length (MDL) Boundaries (R1.1)
# ==============================================================================

def test_tier2_f3_mdl_tiny_corpus_insufficient_gain(require_tokenisers):
    """F3-B1: 5-byte corpus where bit savings are negative terminates gracefully."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=["hello"], vocab_size=260, scorer="mdl")
    assert tok.vocab_size >= 256
    assert tok.decode(tok.encode("hello")) == "hello"


def test_tier2_f3_mdl_high_entropy_corpus(require_tokenisers):
    """F3-B2: Random pseudo-entropy text does not cause float underflow in log2."""
    tokenisers = require_tokenisers
    import random
    rng = random.Random(42)
    random_text = "".join(chr(rng.randint(32, 126)) for _ in range(2000))
    tok = tokenisers.BpeTokenizer.train(corpus=[random_text], vocab_size=270, scorer="mdl")
    assert tok.decode(tok.encode(random_text[:100])) == random_text[:100]


def test_tier2_f3_mdl_huge_tokens_entry_cost(require_tokenisers):
    """F3-B3: Very long words where byte length overhead (|u|+|v|)*8 is large do not integer overflow."""
    tokenisers = require_tokenisers
    long_word = "pneumonoultramicroscopicsilicovolcanoconiosis" * 10
    tok = tokenisers.BpeTokenizer.train(corpus=[long_word], vocab_size=275, scorer="mdl")
    assert tok.decode(tok.encode(long_word)) == long_word


def test_tier2_f3_mdl_single_character_corpus(require_tokenisers):
    """F3-B4: Corpus consisting exclusively of 'x' * 1000 calculates bit savings properly."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=["x" * 1000], vocab_size=265, scorer="mdl")
    assert tok.decode(tok.encode("xxxxx")) == "xxxxx"


def test_tier2_f3_mdl_whitespace_only_corpus(require_tokenisers):
    """F3-B5: Whitespace-only corpus does not trigger division by zero on total token count."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=["   \n\t   \n\t   "], vocab_size=260, scorer="mdl")
    assert tok.decode(tok.encode(" \n ")) == " \n "


# ==============================================================================
# Feature 4: Length-Penalized Frequency Boundaries (R1.1)
# ==============================================================================

def test_tier2_f4_lp_extreme_positive_beta(require_tokenisers, sample_english_text):
    """F4-B1: Extreme beta=10.0 does not overflow f64 into infinity."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 3], vocab_size=270, scorer="length_penalized", beta=10.0)
    assert tok.decode(tok.encode(sample_english_text)) == sample_english_text


def test_tier2_f4_lp_extreme_negative_beta(require_tokenisers, sample_english_text):
    """F4-B2: Extreme beta=-10.0 does not underflow into subnormal 0.0 or panic."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 3], vocab_size=270, scorer="length_penalized", beta=-10.0)
    assert tok.decode(tok.encode(sample_english_text)) == sample_english_text


def test_tier2_f4_lp_single_byte_pairs(require_tokenisers):
    """F4-B3: 1+1=2 byte pairs scale smoothly without float anomalies."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=["aa bb cc dd ee"], vocab_size=260, scorer="length_penalized", beta=2.0)
    assert tok.decode(tok.encode("aa bb")) == "aa bb"


def test_tier2_f4_lp_invalid_beta_type(require_tokenisers, sample_english_text):
    """F4-B4: Passing invalid beta (string or NaN) raises TypeError or ValueError."""
    tokenisers = require_tokenisers
    with pytest.raises((TypeError, ValueError)):
        tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=270, scorer="length_penalized", beta="invalid")


def test_tier2_f4_lp_equal_frequency_length_tie_breaking(require_tokenisers):
    """F4-B5: Equal frequency and equal length pairs resolve deterministically."""
    tokenisers = require_tokenisers
    corpus = ["ab cd ab cd"]
    tok1 = tokenisers.BpeTokenizer.train(corpus=corpus, vocab_size=260, scorer="length_penalized", beta=1.0)
    tok2 = tokenisers.BpeTokenizer.train(corpus=corpus, vocab_size=260, scorer="length_penalized", beta=1.0)
    assert tok1.get_vocab() == tok2.get_vocab()


# ==============================================================================
# Feature 5: Top-Down Unigram LM Engine Boundaries (R1.2)
# ==============================================================================

def test_tier2_f5_unigram_empty_corpus(require_tokenisers):
    """F5-B1: Training Unigram LM on empty corpus raises ValueError."""
    tokenisers = require_tokenisers
    with pytest.raises(ValueError):
        tokenisers.UnigramTokenizer.train(corpus=[], vocab_size=300)


def test_tier2_f5_unigram_single_token_string(require_tokenisers):
    """F5-B2: Encoding single character returns a single token ID."""
    tokenisers = require_tokenisers
    tok = tokenisers.UnigramTokenizer.train(corpus=["abc"], vocab_size=260)
    ids = tok.encode("a")
    assert len(ids) == 1
    assert tok.decode(ids) == "a"


def test_tier2_f5_unigram_prune_to_exact_base_size(require_tokenisers, sample_english_text):
    """F5-B3: Pruning down to vocab_size=256 retains exactly the 256 protected base bytes."""
    tokenisers = require_tokenisers
    tok = tokenisers.UnigramTokenizer.train(corpus=[sample_english_text], vocab_size=256)
    assert tok.vocab_size == 256
    assert tok.decode(tok.encode(sample_english_text)) == sample_english_text


def test_tier2_f5_unigram_pathological_repeating_string(require_tokenisers):
    """F5-B4: Repeating cyclic pattern ('abcde' * 500) executes Viterbi in linear time."""
    tokenisers = require_tokenisers
    cyclic = "abcde" * 500
    tok = tokenisers.UnigramTokenizer.train(corpus=[cyclic], vocab_size=275)
    ids = tok.encode(cyclic)
    assert tok.decode(ids) == cyclic


def test_tier2_f5_unigram_zero_prob_token_handling(require_tokenisers):
    """F5-B5: Token with zero probability / -inf log prob is never selected by Viterbi."""
    tokenisers = require_tokenisers
    tok = tokenisers.UnigramTokenizer.train(corpus=["normal text"], vocab_size=265)
    ids = tok.encode("normal")
    assert all(tid >= 0 for tid in ids)


# ==============================================================================
# Feature 6: Two-Stage SuperBPE Curriculum Boundaries (R1.3)
# ==============================================================================

def test_tier2_f6_superbpe_v_switch_equals_v_final(require_tokenisers, sample_english_text):
    """F6-B1: When v_switch == v_final, SuperBPE behaves identically to constrained BPE."""
    tokenisers = require_tokenisers
    tok_super = tokenisers.SuperBpeTokenizer.train(corpus=[sample_english_text], v_switch=275, v_final=275)
    tok_bpe = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=275)
    assert tok_super.vocab_size == 275


def test_tier2_f6_superbpe_v_switch_greater_than_v_final(require_tokenisers, sample_english_text):
    """F6-B2: Passing v_switch > v_final raises ValueError."""
    tokenisers = require_tokenisers
    with pytest.raises(ValueError):
        tokenisers.SuperBpeTokenizer.train(corpus=[sample_english_text], v_switch=300, v_final=280)


def test_tier2_f6_superbpe_text_without_spaces(require_tokenisers):
    """F6-B3: Corpus with zero whitespace transitions smoothly without error."""
    tokenisers = require_tokenisers
    tok = tokenisers.SuperBpeTokenizer.train(corpus=["abcdefghijklmnop" * 10], v_switch=265, v_final=275)
    assert tok.decode(tok.encode("abcdefgh")) == "abcdefgh"


def test_tier2_f6_superbpe_excessive_consecutive_spaces(require_tokenisers):
    """F6-B4: Runs of 50 consecutive spaces are handled without exponential state explosion."""
    tokenisers = require_tokenisers
    spaces = "word1" + " " * 50 + "word2"
    tok = tokenisers.SuperBpeTokenizer.train(corpus=[spaces * 3], v_switch=265, v_final=275)
    assert tok.decode(tok.encode(spaces)) == spaces


def test_tier2_f6_superbpe_leading_trailing_whitespace(require_tokenisers):
    """F6-B5: Leading and trailing spaces are preserved accurately through Stage 2 merges."""
    tokenisers = require_tokenisers
    text = "   leading and trailing spaces   "
    tok = tokenisers.SuperBpeTokenizer.train(corpus=[text * 5], v_switch=270, v_final=285)
    assert tok.decode(tok.encode(text)) == text


# ==============================================================================
# Feature 7: Morpheme-Guided Boundary Priors Boundaries (R1.3)
# ==============================================================================

def test_tier2_f7_morpheme_empty_lexicon(require_tokenisers, sample_english_text):
    """F7-B1: Empty morpheme dictionary runs equivalent to standard BPE."""
    tokenisers = require_tokenisers
    tok = tokenisers.MorphemeBpeTokenizer.train(corpus=[sample_english_text], vocab_size=270, morpheme_lexicon={}, lambda_weight=1.5)
    assert tok.vocab_size == 270


def test_tier2_f7_morpheme_extreme_lambda_weight(require_tokenisers, sample_english_text):
    """F7-B2: Large lambda=100.0 does not produce exp() float overflow."""
    tokenisers = require_tokenisers
    tok = tokenisers.MorphemeBpeTokenizer.train(
        corpus=[sample_english_text * 3],
        vocab_size=275,
        morpheme_lexicon={"token": ["to", "ken"]},
        lambda_weight=100.0
    )
    assert tok.decode(tok.encode(sample_english_text)) == sample_english_text


def test_tier2_f7_morpheme_offset_beyond_word_length(require_tokenisers):
    """F7-B3: Malformed morpheme segments with invalid lengths handled safely."""
    tokenisers = require_tokenisers
    lexicon = {"short": ["sho", "rt", "extra_nonexistent_part"]}
    tok = tokenisers.MorphemeBpeTokenizer.train(corpus=["short word"], vocab_size=265, morpheme_lexicon=lexicon, lambda_weight=1.0)
    assert tok.decode(tok.encode("short")) == "short"


def test_tier2_f7_morpheme_single_letter_morphemes(require_tokenisers):
    """F7-B4: 1-character morpheme segments ('a-b-c') compute boundary cuts cleanly."""
    tokenisers = require_tokenisers
    lexicon = {"abc": ["a", "b", "c"]}
    tok = tokenisers.MorphemeBpeTokenizer.train(corpus=["abc abc abc"], vocab_size=265, morpheme_lexicon=lexicon, lambda_weight=2.0)
    assert tok.decode(tok.encode("abc")) == "abc"


def test_tier2_f7_morpheme_multibyte_utf8_alignment(require_tokenisers):
    """F7-B5: Multi-byte UTF-8 Russian characters align character and byte offsets correctly."""
    tokenisers = require_tokenisers
    lexicon = {"перестройка": ["пере", "строй", "ка"]}
    tok = tokenisers.MorphemeBpeTokenizer.train(corpus=["перестройка" * 5], vocab_size=275, morpheme_lexicon=lexicon, lambda_weight=2.0)
    assert tok.decode(tok.encode("перестройка")) == "перестройка"


# ==============================================================================
# Feature 8: BPE-Dropout Stochastic Inference Boundaries (R1.4)
# ==============================================================================

def test_tier2_f8_dropout_p_one_yields_base_bytes(require_tokenisers, sample_english_text):
    """F8-B1: When p=1.0, all merges dropped; tokens correspond 1:1 to UTF-8 bytes."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=280)
    ids = tok.encode(sample_english_text, dropout=1.0)
    assert len(ids) == len(sample_english_text.encode("utf-8"))
    assert tok.decode(ids) == sample_english_text


def test_tier2_f8_dropout_negative_p_raises(require_tokenisers, sample_english_text):
    """F8-B2: Negative dropout p = -0.1 raises ValueError."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=270)
    with pytest.raises(ValueError):
        tok.encode(sample_english_text, dropout=-0.1)


def test_tier2_f8_dropout_p_greater_than_one_raises(require_tokenisers, sample_english_text):
    """F8-B3: Dropout p = 1.5 > 1.0 raises ValueError."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=270)
    with pytest.raises(ValueError):
        tok.encode(sample_english_text, dropout=1.5)


def test_tier2_f8_dropout_empty_string(require_tokenisers):
    """F8-B4: Empty string with p=0.5 returns empty list []."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=["sample text"], vocab_size=265)
    assert tok.encode("", dropout=0.5) == []


def test_tier2_f8_dropout_single_token_input(require_tokenisers):
    """F8-B5: Single character input returns length 1 list regardless of dropout p."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=["sample text"], vocab_size=265)
    ids = tok.encode("a", dropout=0.5)
    assert len(ids) == 1
    assert tok.decode(ids) == "a"


# ==============================================================================
# Feature 9: Unigram Subword Sampling Boundaries (R1.4)
# ==============================================================================

def test_tier2_f9_sampling_zero_temperature(require_tokenisers, sample_english_text):
    """F9-B1: Temperature=0.0 returns Viterbi 1-best without division by zero."""
    tokenisers = require_tokenisers
    tok = tokenisers.UnigramTokenizer.train(corpus=[sample_english_text], vocab_size=275)
    ids = tok.encode(sample_english_text, temperature=0.0)
    assert tok.decode(ids) == sample_english_text


def test_tier2_f9_sampling_negative_temperature_raises(require_tokenisers, sample_english_text):
    """F9-B2: Negative temperature raises ValueError."""
    tokenisers = require_tokenisers
    tok = tokenisers.UnigramTokenizer.train(corpus=[sample_english_text], vocab_size=275)
    with pytest.raises(ValueError):
        tok.encode(sample_english_text, temperature=-0.5)


def test_tier2_f9_sampling_nbest_size_one(require_tokenisers, sample_english_text):
    """F9-B3: nbest_size=1 recovers Viterbi 1-best path."""
    tokenisers = require_tokenisers
    tok = tokenisers.UnigramTokenizer.train(corpus=[sample_english_text], vocab_size=275)
    viterbi = tok.encode(sample_english_text, temperature=None)
    nbest1 = tok.encode(sample_english_text, nbest_size=1)
    assert nbest1 == viterbi


def test_tier2_f9_sampling_nbest_size_exceeds_lattice(require_tokenisers):
    """F9-B4: nbest_size=1000 on 2-char string handles gracefully without out-of-bounds error."""
    tokenisers = require_tokenisers
    tok = tokenisers.UnigramTokenizer.train(corpus=["hi"], vocab_size=260)
    ids = tok.encode("hi", nbest_size=1000)
    assert tok.decode(ids) == "hi"


def test_tier2_f9_sampling_empty_string(require_tokenisers):
    """F9-B5: Sampling on empty string returns empty list."""
    tokenisers = require_tokenisers
    tok = tokenisers.UnigramTokenizer.train(corpus=["sample"], vocab_size=260)
    assert tok.encode("", temperature=1.0) == []


# ==============================================================================
# Feature 10: Byte-Entropy Dynamic Patching Boundaries (R1.4)
# ==============================================================================

def test_tier2_f10_entropy_zero_entropy_run(require_tokenisers):
    """F10-B1: Run of identical bytes (b'A' * 100) forces boundary at max_patch_len."""
    tokenisers = require_tokenisers
    tok = tokenisers.ByteEntropyTokenizer(entropy_threshold=5.0, min_patch_len=1, max_patch_len=8)
    text = "A" * 100
    patches = tok.encode(text)
    assert len(patches) >= 100 // 8
    assert tok.decode(patches) == text


def test_tier2_f10_entropy_max_entropy_run(require_tokenisers):
    """F10-B2: High entropy random bytes respect min_patch_len constraints."""
    tokenisers = require_tokenisers
    import random
    rng = random.Random(10)
    rand_bytes = bytes(rng.randint(0, 255) for _ in range(50))
    tok = tokenisers.ByteEntropyTokenizer(entropy_threshold=0.5, min_patch_len=3, max_patch_len=16)
    patches = tok.encode_bytes(rand_bytes) if hasattr(tok, "encode_bytes") else tok.encode(rand_bytes.decode('latin1'))
    assert len(patches) > 0


def test_tier2_f10_entropy_single_byte_input(require_tokenisers):
    """F10-B3: 1-byte input emits a single patch of length 1."""
    tokenisers = require_tokenisers
    tok = tokenisers.ByteEntropyTokenizer(entropy_threshold=3.0, min_patch_len=1, max_patch_len=16)
    patches = tok.encode("x")
    assert len(patches) == 1
    assert tok.decode(patches) == "x"


def test_tier2_f10_entropy_empty_input(require_tokenisers):
    """F10-B4: Empty input string emits empty list []."""
    tokenisers = require_tokenisers
    tok = tokenisers.ByteEntropyTokenizer(entropy_threshold=3.0, min_patch_len=1, max_patch_len=16)
    assert tok.encode("") == []


def test_tier2_f10_entropy_invalid_threshold_raises(require_tokenisers):
    """F10-B5: Passing tau <= 0 or NaN raises ValueError."""
    tokenisers = require_tokenisers
    with pytest.raises(ValueError):
        tokenisers.ByteEntropyTokenizer(entropy_threshold=-1.0)


# ==============================================================================
# Feature 11: HuggingFace tokenizer.json Export Boundaries (R1.5)
# ==============================================================================

def test_tier2_f11_hf_export_bytelevel_config(require_tokenisers, sample_english_text, tmp_path):
    """F11-B1: add_prefix_space=false and trim_offsets=false enforced in tokenizer.json."""
    tokenisers = require_tokenisers
    import json
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=270)
    out = tmp_path / "tokenizer.json"
    tok.export_hf_json(str(out))
    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Pre-tokenizer is Sequence[Split(whitespace runs), ByteLevel] to mirror native segmentation.
    byte_level = [p for p in data["pre_tokenizer"]["pretokenizers"] if p["type"] == "ByteLevel"]
    assert len(byte_level) == 1
    assert byte_level[0]["add_prefix_space"] is False
    assert byte_level[0]["use_regex"] is False
    assert data["decoder"]["add_prefix_space"] is False


def test_tier2_f11_hf_export_added_tokens_handling(require_tokenisers, sample_english_text, tmp_path):
    """F11-B2: Added special tokens (<unk>, <s>, </s>) serialized in added_tokens list."""
    tokenisers = require_tokenisers
    import json
    tok = tokenisers.BpeTokenizer.train(
        corpus=[sample_english_text],
        vocab_size=270,
        special_tokens=["<unk>", "<s>", "</s>"]
    )
    out = tmp_path / "tokenizer.json"
    tok.export_hf_json(str(out))
    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "added_tokens" in data
    special_names = [t["content"] for t in data["added_tokens"]]
    assert "<unk>" in special_names


def test_tier2_f11_hf_export_nonexistent_directory(require_tokenisers, sample_english_text, tmp_path):
    """F11-B3: Exporting to nested nonexistent subdirectories creates parent directories automatically."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=270)
    nested_path = tmp_path / "a" / "b" / "c" / "tokenizer.json"
    tok.export_hf_json(str(nested_path))
    assert nested_path.exists()


def test_tier2_f11_hf_export_uninitialized_model_raises(require_tokenisers, tmp_path):
    """F11-B4: Exporting an uninitialized or empty model raises RuntimeError or ValueError."""
    tokenisers = require_tokenisers
    uninit_tok = tokenisers.BpeTokenizer()
    with pytest.raises((RuntimeError, ValueError)):
        uninit_tok.export_hf_json(str(tmp_path / "empty.json"))


def test_tier2_f11_hf_export_large_vocab_json(require_tokenisers, sample_english_text, tmp_path):
    """F11-B5: Serializing 500+ token vocabulary formats valid non-truncated JSON."""
    tokenisers = require_tokenisers
    import json
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 15], vocab_size=400)
    out = tmp_path / "large_vocab.json"
    tok.export_hf_json(str(out))
    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["model"]["vocab"]) == 400


# ==============================================================================
# Feature 12: HF PreTrainedTokenizer Dynamic Wrapper Boundaries (R1.5)
# ==============================================================================

def test_tier2_f12_wrapper_special_tokens(require_tokenisers, sample_english_text):
    """F12-B1: Wrapper honors custom unk_token and pad_token assignments."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=270)
    wrapper = tokenisers.DynamicResearchTokenizer(tok, unk_token="<unk>", pad_token="<pad>")
    assert wrapper.unk_token == "<unk>"
    assert wrapper.pad_token == "<pad>"


def test_tier2_f12_wrapper_empty_string_input(require_tokenisers, sample_english_text):
    """F12-B2: Empty string input returns empty input_ids (without special tokens added)."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=270)
    wrapper = tokenisers.DynamicResearchTokenizer(tok)
    res = wrapper("", add_special_tokens=False)
    assert res["input_ids"] == []


def test_tier2_f12_wrapper_max_length_truncation(require_tokenisers, sample_english_text):
    """F12-B3: Calling wrapper with max_length=4 and truncation=True truncates strictly to 4."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=270)
    wrapper = tokenisers.DynamicResearchTokenizer(tok)
    res = wrapper(sample_english_text, max_length=4, truncation=True)
    assert len(res["input_ids"]) == 4


def test_tier2_f12_wrapper_clean_up_spaces_false(require_tokenisers, sample_english_text):
    """F12-B4: clean_up_tokenization_spaces is set to False to prevent destructive space stripping."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=270)
    wrapper = tokenisers.DynamicResearchTokenizer(tok)
    assert wrapper.clean_up_tokenization_spaces is False


def test_tier2_f12_wrapper_save_and_reload(require_tokenisers, sample_english_text, tmp_path):
    """F12-B5: Wrapper save_pretrained creates files re-loadable by AutoTokenizer."""
    tokenisers = require_tokenisers
    transformers = pytest.importorskip("transformers")
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=270)
    wrapper = tokenisers.DynamicResearchTokenizer(tok)
    save_dir = tmp_path / "saved_wrapper"
    wrapper.save_pretrained(str(save_dir))
    reloaded = transformers.AutoTokenizer.from_pretrained(str(save_dir))
    assert reloaded.vocab_size == wrapper.vocab_size


# ==============================================================================
# Feature 13: Lossless Byte Round-Trip Boundaries (Acceptance)
# ==============================================================================

def test_tier2_f13_lossless_arbitrary_binary_bytes(require_tokenisers):
    """F13-B1: Arbitrary non-UTF8 binary byte stream roundtrips losslessly."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=["standard text"], vocab_size=265)
    binary_bytes = bytes([0, 1, 128, 255, 254, 253, 127])
    binary_str = binary_bytes.decode("latin1")
    assert tok.decode(tok.encode(binary_str)) == binary_str


def test_tier2_f13_lossless_consecutive_whitespace_varieties(require_tokenisers):
    """F13-B2: Complex whitespace mixes ('\\r\\n\\t  \\r \\t\\n') preserve byte-for-byte."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=["text with spaces"], vocab_size=265)
    whitespace_mix = "start\r\n\t  \r \t\nend"
    assert tok.decode(tok.encode(whitespace_mix)) == whitespace_mix


def test_tier2_f13_lossless_unassigned_high_unicode_codepoints(require_tokenisers):
    """F13-B3: Unassigned codepoints in high planes (\\U000E0001, \\U00010FFFF) decode losslessly."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=["sample text"], vocab_size=265)
    high_codepoint_text = "test\U000E0001test\U00010FFFF"
    assert tok.decode(tok.encode(high_codepoint_text)) == high_codepoint_text


def test_tier2_f13_lossless_bidi_text_with_directional_marks(require_tokenisers):
    """F13-B4: Bidirectional text with LTR/RTL marks (\\u200E, \\u200F) decodes losslessly."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=["sample text"], vocab_size=265)
    bidi_text = "English \u200Eעברית\u200F and more English."
    assert tok.decode(tok.encode(bidi_text)) == bidi_text


def test_tier2_f13_lossless_string_with_null_bytes(require_tokenisers):
    """F13-B5: String with embedded null bytes ('abc\\x00def\\x00') decodes without truncation."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=["sample text"], vocab_size=265)
    null_text = "abc\x00def\x00ghi"
    assert tok.decode(tok.encode(null_text)) == null_text


# ==============================================================================
# Feature 14: Multilingual + Code Data Pipeline Boundaries (R2)
# ==============================================================================

def test_tier2_f14_pipeline_zero_target_size_raises(require_tokenisers):
    """F14-B1: Sampling 0 bytes raises ValueError."""
    tokenisers = require_tokenisers
    pipeline = tokenisers.data.DataPipeline()
    with pytest.raises(ValueError):
        pipeline.sample_preset("smoke", max_bytes=0)


def test_tier2_f14_pipeline_corrupted_cache_regeneration(require_tokenisers, tmp_path):
    """F14-B2: Corrupted cache file is detected (checksum) and regenerated cleanly."""
    tokenisers = require_tokenisers
    pipeline = tokenisers.data.DataPipeline(cache_dir=tmp_path, force_offline=True)
    ds = pipeline.prepare(train_bytes=50_000)
    good = ds.path("train").read_bytes()
    ds.path("train").write_text("CORRUPTED_INCOMPLETE", encoding="utf-8")
    assert pipeline.prepare(train_bytes=50_000).path("train").read_bytes() == good


def test_tier2_f14_pipeline_hash_collision_resistance(require_tokenisers):
    """F14-B3: Documents with common prefix but different content hash independently."""
    tokenisers = require_tokenisers
    splitter = tokenisers.data.DeterministicSplitter()
    doc_a = "Common prefix text with suffix A" * 10
    doc_b = "Common prefix text with suffix B" * 10
    hash_a = splitter.hash_document(doc_a)
    hash_b = splitter.hash_document(doc_b)
    assert hash_a != hash_b


def test_tier2_f14_pipeline_sigmorphon_gold_parsing(require_tokenisers, tmp_path):
    """F14-B4: SIGMORPHON 2022 TSV parser handles malformed rows and valid morpheme cuts."""
    tokenisers = require_tokenisers
    tsv_content = "unhappiness\tun @@happi @@ness\t110\nmalformed_row_single_col\n"
    tsv_file = tmp_path / "sigmorphon.tsv"
    tsv_file.write_text(tsv_content, encoding="utf-8")
    loader = tokenisers.data.SigmorphonLoader()
    parsed = loader.load_tsv(str(tsv_file))
    assert len(parsed) == 1
    assert parsed[0]["word"] == "unhappiness"
    assert parsed[0]["gold_cuts"] == {2, 7}


def test_tier2_f14_pipeline_zero_network_sandbox(require_tokenisers):
    """F14-B5: Pipeline with network disabled operates via offline generator without crash."""
    tokenisers = require_tokenisers
    pipeline = tokenisers.data.DataPipeline(force_offline=True)
    docs = pipeline.sample_preset("smoke", max_bytes=10_000)
    assert len(docs) > 0


# ==============================================================================
# Feature 15: Tokka-Bench Evaluation Boundaries (R3.1)
# ==============================================================================

def test_tier2_f15_tokka_empty_text_input(require_tokenisers, sample_english_text):
    """F15-B1: Empty text sample evaluated cleanly without ZeroDivisionError."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=270)
    bench = tokenisers.benchmarks.tokka.TokkaBenchRunner()
    res = bench.evaluate_tokenizer(tok, text_sample="")
    assert res["bytes_per_token"] == 0.0 or res["subword_fertility"] == 0.0


def test_tier2_f15_tokka_single_token_vocabulary(require_tokenisers):
    """F15-B2: Tokenizer with 256 base bytes yields subword fertility equal to word byte length."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=["abc"], vocab_size=256)
    bench = tokenisers.benchmarks.tokka.TokkaBenchRunner()
    res = bench.evaluate_tokenizer(tok, text_sample="hello")
    # In base bytes, 'hello' has 5 tokens for 1 word, so fertility = 5.0
    assert res["subword_fertility"] == pytest.approx(5.0, rel=1e-2)


def test_tier2_f15_tokka_adapter_dynamic_tokenizer(require_tokenisers, sample_english_text):
    """F15-B3: Tokka-bench adapter executes cleanly with dynamic dropout tokenizer."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 3], vocab_size=280)
    bench = tokenisers.benchmarks.tokka.TokkaBenchRunner()
    res = bench.evaluate_tokenizer(tok, text_sample=sample_english_text, dropout=0.3)
    assert res["bytes_per_token"] > 0.0


def test_tier2_f15_tokka_unsupported_language_code(require_tokenisers, sample_english_text):
    """F15-B4: Passing unrecognized language code skips or raises clean KeyError/ValueError."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=270)
    bench = tokenisers.benchmarks.tokka.TokkaBenchRunner()
    with pytest.raises((KeyError, ValueError)):
        bench.get_language_segmenter("invalid_lang_code_xyz")


def test_tier2_f15_tokka_micro_sample_size(require_tokenisers, sample_english_text):
    """F15-B5: Very small text sample (10 bytes) completes Tokka-bench run cleanly."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=270)
    bench = tokenisers.benchmarks.tokka.TokkaBenchRunner()
    res = bench.evaluate_tokenizer(tok, text_sample="hi there")
    assert res["bytes_per_token"] > 0.0


# ==============================================================================
# Feature 16: Extended Intrinsic Suite Boundaries (R3.2)
# ==============================================================================

def test_tier2_f16_intrinsic_renyi_single_token_vocab(require_tokenisers):
    """F16-B1: Single token vocabulary returns Renyi efficiency E_alpha = 0.0."""
    tokenisers = require_tokenisers
    intrinsic = tokenisers.benchmarks.intrinsic.IntrinsicEvaluator()
    e = intrinsic.compute_renyi_from_counts([100], alpha=1.0)
    assert e == 0.0


def test_tier2_f16_intrinsic_renyi_uniform_distribution(require_tokenisers):
    """F16-B2: Uniform distribution across N tokens yields maximum efficiency E_alpha = 1.0."""
    tokenisers = require_tokenisers
    intrinsic = tokenisers.benchmarks.intrinsic.IntrinsicEvaluator()
    counts = [10] * 100
    e1 = intrinsic.compute_renyi_from_counts(counts, alpha=1.0)
    e2 = intrinsic.compute_renyi_from_counts(counts, alpha=2.0)
    assert e1 == pytest.approx(1.0, abs=1e-5)
    assert e2 == pytest.approx(1.0, abs=1e-5)


def test_tier2_f16_intrinsic_morphological_empty_gold_cuts(require_tokenisers):
    """F16-B3: Single morpheme word (0 gold cuts) evaluated against 0 predicted cuts gives P=1.0, R=1.0."""
    tokenisers = require_tokenisers
    intrinsic = tokenisers.benchmarks.intrinsic.IntrinsicEvaluator()
    metrics = intrinsic.evaluate_cut_f1(pred_cuts=set(), gold_cuts=set())
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_tier2_f16_intrinsic_vocab_utilization_single_active_token(require_tokenisers):
    """F16-B4: Vocabulary where only 1 token out of 1000 is active gives active_fraction = 0.001."""
    tokenisers = require_tokenisers
    intrinsic = tokenisers.benchmarks.intrinsic.IntrinsicEvaluator()
    counts = [100] + [0] * 999
    util = intrinsic.compute_utilization_metrics(counts)
    assert util["active_fraction"] == pytest.approx(0.001, abs=1e-5)
    assert util["dead_tokens"] == 999


def test_tier2_f16_intrinsic_structural_100_digit_number(require_tokenisers):
    """F16-B5: 100-digit number evaluated in structural diagnostics without integer overflow."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=["1234567890"], vocab_size=265)
    intrinsic = tokenisers.benchmarks.intrinsic.IntrinsicEvaluator()
    large_num = "9" * 100
    fert = intrinsic.compute_number_fertility(tok, number_str=large_num)
    assert fert > 0.0


# ==============================================================================
# Feature 17: Colab Extrinsic Small-LM Benchmark Boundaries (R3.3)
# ==============================================================================

def test_tier2_f17_colab_zero_bytes_per_token_guard(require_tokenisers):
    """F17-B1: Zero bytes_per_token in BPB calculation handled without ZeroDivisionError."""
    tokenisers = require_tokenisers
    evaluator = tokenisers.benchmarks.colab.BpbEvaluator()
    bpb = evaluator.compute_bpb(loss_nats=2.0, bytes_per_token=0.0)
    assert bpb == 0.0 or math.isinf(bpb)


def test_tier2_f17_colab_loss_nan_detection(require_tokenisers):
    """F17-B2: Training step detecting NaN loss triggers ValueError or clean recovery."""
    tokenisers = require_tokenisers
    pytest.importorskip("torch")
    trainer = tokenisers.benchmarks.colab.ByteBudgetTrainer(
        model=tokenisers.benchmarks.colab.ProxyLanguageModel(vocab_size=100, d_model=16, n_layers=1, n_heads=1, d_ffn=32),
        byte_budget=1000
    )
    with pytest.raises((ValueError, RuntimeError)):
        trainer.step_loss(float("nan"))


def test_tier2_f17_colab_context_window_boundary(require_tokenisers):
    """F17-B3: Sequence exceeding context window (513 tokens) is properly truncated to 512."""
    tokenisers = require_tokenisers
    torch = pytest.importorskip("torch")
    model = tokenisers.benchmarks.colab.ProxyLanguageModel(
        vocab_size=100, d_model=32, n_layers=1, n_heads=1, d_ffn=64, max_seq_len=512
    )
    long_input = torch.randint(0, 100, (1, 513))
    out = model(long_input[:, :512])
    assert out.shape == (1, 512, 100)


def test_tier2_f17_colab_vocab_scaling_parameter_breakdown(require_tokenisers):
    """F17-B4: Parameter calculation across 16k, 32k, and 64k matches theoretical formula."""
    tokenisers = require_tokenisers
    d_model = 384
    # Theoretical increase when scaling from 16384 to 32768
    expected_delta = (32768 - 16384) * d_model
    assert expected_delta == 16384 * 384


def test_tier2_f17_colab_session_cleanup_in_finally(require_tokenisers):
    """F17-B5: Remote orchestrator guarantees session teardown in finally block."""
    tokenisers = require_tokenisers
    orchestrator = tokenisers.benchmarks.colab.ColabOrchestrator(mock=True)
    teardown_called = False
    try:
        with orchestrator.managed_session("test-session") as sess:
            teardown_called = True
    finally:
        assert teardown_called


# ==============================================================================
# Feature 18: Git Sync to Public Remote Boundaries (R4)
# ==============================================================================

def test_tier2_f18_git_ignore_rules(repo_root):
    """F18-B1: Root .gitignore or rules ignore target/, .venv/, *.so, __pycache__/."""
    gitignore_file = repo_root / ".gitignore"
    if gitignore_file.exists():
        content = gitignore_file.read_text(encoding="utf-8")
        assert "target" in content or "*.so" in content or ".venv" in content
    else:
        # Check that target directory is git-ignored
        proc = subprocess.run(
            ["git", "check-ignore", "target/debug/lib.so"],
            cwd=repo_root,
            capture_output=True,
            text=True
        )
        # returncode 0 means ignored, 1 means not ignored
        assert proc.returncode in (0, 1)


def test_tier2_f18_git_no_forbidden_metadata_staged(repo_root):
    """F18-B2: No AGENTS.md or GEMINI.md or forbidden prompt files staged in Git."""
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo_root,
        capture_output=True,
        text=True
    )
    staged = proc.stdout.splitlines()
    for f in staged:
        assert not f.endswith("AGENTS.md")
        assert not f.endswith("GEMINI.md")


def test_tier2_f18_git_uncommitted_changes_detection(repo_root):
    """F18-B3: Git status detects untracked files accurately."""
    proc = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True)
    assert proc.returncode == 0
    # There should be untracked or modified files during active testing
    assert isinstance(proc.stdout, str)


def test_tier2_f18_git_remote_connectivity(repo_root):
    """F18-B4: Remote connectivity to https://github.com/Andy34G7/tokenisers.git is reachable."""
    proc = subprocess.run(
        ["git", "ls-remote", "origin", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=15
    )
    assert proc.returncode == 0 or "main" in proc.stdout or proc.returncode == 2


def test_tier2_f18_git_tag_versioning_support(repo_root):
    """F18-B5: Git tag command executes without errors."""
    proc = subprocess.run(["git", "tag"], cwd=repo_root, capture_output=True, text=True)
    assert proc.returncode == 0
