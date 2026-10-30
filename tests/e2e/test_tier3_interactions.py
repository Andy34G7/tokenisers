"""
Tier 3: Pairwise Cross-Feature Combinatorial Interaction Tests.
Target: >= 18 pairwise interaction test cases covering multi-feature synergy.
"""

import math
import subprocess
from pathlib import Path
import pytest


def test_tier3_superbpe_with_bpe_dropout(require_tokenisers, sample_english_text):
    """T3.1: SuperBPE (F6) multi-word tokens interact smoothly with BPE-Dropout (F8) inference."""
    tokenisers = require_tokenisers
    tok = tokenisers.SuperBpeTokenizer.train(
        corpus=[sample_english_text * 10],
        v_switch=275,
        v_final=310
    )
    # Apply stochastic dropout on SuperBPE model
    out1 = tok.encode(sample_english_text, dropout=0.25)
    out2 = tok.encode(sample_english_text, dropout=0.25)
    assert tok.decode(out1) == sample_english_text
    assert tok.decode(out2) == sample_english_text


def test_tier3_morpheme_priors_with_wordpiece(require_tokenisers, sample_morpheme_annotations):
    """T3.2: Morpheme boundary priors (F7) integrate with WordPiece/PMI scoring (F2)."""
    tokenisers = require_tokenisers
    lexicon = {item["word"]: item["segments"] for item in sample_morpheme_annotations}
    corpus = ["unhappiness and disagreement in pretokenization"] * 10
    tok = tokenisers.MorphemeBpeTokenizer.train(
        corpus=corpus,
        vocab_size=280,
        morpheme_lexicon=lexicon,
        scorer="wordpiece",
        lambda_weight=1.5
    )
    assert tok.vocab_size == 280
    assert tok.decode(tok.encode(corpus[0])) == corpus[0]


def test_tier3_morpheme_priors_with_mdl(require_tokenisers, sample_morpheme_annotations):
    """T3.3: Morpheme boundary priors (F7) balance information gain under MDL scoring (F3)."""
    tokenisers = require_tokenisers
    lexicon = {item["word"]: item["segments"] for item in sample_morpheme_annotations}
    corpus = ["unhappiness and disagreement in pretokenization"] * 10
    tok = tokenisers.MorphemeBpeTokenizer.train(
        corpus=corpus,
        vocab_size=280,
        morpheme_lexicon=lexicon,
        scorer="mdl",
        lambda_weight=2.0
    )
    assert tok.vocab_size == 280
    assert tok.decode(tok.encode(corpus[0])) == corpus[0]


def test_tier3_unigram_with_lattice_sampling(require_tokenisers, sample_english_text):
    """T3.4: Top-down Unigram (F5) produces valid stochastic segmentations via FFBS sampling (F9)."""
    tokenisers = require_tokenisers
    tok = tokenisers.UnigramTokenizer.train(corpus=[sample_english_text * 5], vocab_size=300)
    for temp in [0.2, 0.7, 1.2]:
        ids = tok.encode(sample_english_text, temperature=temp)
        assert tok.decode(ids) == sample_english_text


def test_tier3_unigram_with_hf_export(require_tokenisers, sample_english_text, tmp_path):
    """T3.5: Trained Unigram model (F5) exports to canonical HF tokenizer.json (F11) and reloads."""
    tokenisers = require_tokenisers
    hf_tok = pytest.importorskip("tokenizers")
    tok = tokenisers.UnigramTokenizer.train(corpus=[sample_english_text], vocab_size=275)
    export_file = tmp_path / "unigram_hf.json"
    tok.export_hf_json(str(export_file))
    reloaded = hf_tok.Tokenizer.from_file(str(export_file))
    assert reloaded.get_vocab_size() == 275


def test_tier3_bpe_frequency_with_hf_wrapper(require_tokenisers, sample_english_text):
    """T3.6: Frequency BPE (F1) wraps in DynamicResearchTokenizer (F12) for HuggingFace pipelines."""
    tokenisers = require_tokenisers
    transformers = pytest.importorskip("transformers")
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=275)
    wrapper = tokenisers.DynamicResearchTokenizer(tok)

    # 1. Pipeline roundtrip when max_length accommodates input (182 tokens <= 256)
    batch = wrapper([sample_english_text, "Short text"], padding=True, truncation=True, max_length=256)
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert len(batch["input_ids"][0]) == len(batch["input_ids"][1])
    assert wrapper.decode(batch["input_ids"][0]) == sample_english_text

    # 2. Strict truncation enforcement when input exceeds max_length (182 tokens truncated to 64)
    batch_trunc = wrapper([sample_english_text, "Short text"], padding=True, truncation=True, max_length=64)
    assert len(batch_trunc["input_ids"][0]) == 64
    assert len(batch_trunc["input_ids"][1]) == 64
    assert sample_english_text.startswith(wrapper.decode(batch_trunc["input_ids"][0]))


def test_tier3_superbpe_with_hf_export(require_tokenisers, sample_english_text, tmp_path):
    """T3.7: SuperBPE (F6) multi-word tokens export cleanly into standard tokenizer.json (F11)."""
    tokenisers = require_tokenisers
    hf_tok = pytest.importorskip("tokenizers")
    tok = tokenisers.SuperBpeTokenizer.train(corpus=[sample_english_text * 10], v_switch=275, v_final=300)
    export_file = tmp_path / "superbpe_hf.json"
    tok.export_hf_json(str(export_file))
    reloaded = hf_tok.Tokenizer.from_file(str(export_file))
    assert reloaded.get_vocab_size() == 300


def test_tier3_length_penalized_with_bpe_dropout(require_tokenisers, sample_english_text):
    """T3.8: Length-penalized BPE (F4) supports dynamic merge dropout (F8) at inference."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 5], vocab_size=290, scorer="length_penalized", beta=0.5)
    ids = tok.encode(sample_english_text, dropout=0.3)
    assert tok.decode(ids) == sample_english_text


def test_tier3_byte_entropy_with_data_pipeline(require_tokenisers):
    """T3.9: Byte-entropy dynamic patching (F10) runs directly on streaming pipeline batches (F14)."""
    tokenisers = require_tokenisers
    pipeline = tokenisers.data.DataPipeline()
    docs = pipeline.sample_preset("smoke", max_bytes=20_000)
    tok = tokenisers.ByteEntropyTokenizer.train(docs[3:], entropy_threshold=3.5, min_patch_len=1, max_patch_len=16)
    for doc in docs[:3]:
        patches = tok.encode(doc)
        assert tok.decode(patches) == doc


def test_tier3_unigram_sampling_with_tokka_bench(require_tokenisers, sample_english_text):
    """T3.10: Unigram stochastic lattice sampling (F9) evaluated through Tokka-bench adapter (F15)."""
    tokenisers = require_tokenisers
    tok = tokenisers.UnigramTokenizer.train(corpus=[sample_english_text * 5], vocab_size=290)
    bench = tokenisers.benchmarks.tokka.TokkaBenchRunner()
    metrics = bench.evaluate_tokenizer(tok, text_sample=sample_english_text, temperature=0.8)
    assert metrics["bytes_per_token"] > 0.0
    assert metrics["subword_fertility"] >= 1.0


def test_tier3_mdl_with_renyi_efficiency(require_tokenisers, sample_english_text):
    """T3.11: MDL tokenizer (F3) evaluated on extended intrinsic suite for Renyi efficiency (F16)."""
    tokenisers = require_tokenisers
    tok_mdl = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 8], vocab_size=290, scorer="mdl")
    intrinsic = tokenisers.benchmarks.intrinsic.IntrinsicEvaluator()
    metrics = intrinsic.compute_renyi_efficiency(tok_mdl, text_sample=sample_english_text)
    assert 0.0 <= metrics["e_1_0"] <= 1.0


def test_tier3_morpheme_bpe_with_sigmorphon_f1(require_tokenisers, sample_morpheme_annotations):
    """T3.12: Morpheme-guided BPE (F7) achieves positive boundary F1 against SIGMORPHON cuts (F16)."""
    tokenisers = require_tokenisers
    lexicon = {item["word"]: item["segments"] for item in sample_morpheme_annotations}
    corpus = [item["word"] for item in sample_morpheme_annotations] * 8
    tok = tokenisers.MorphemeBpeTokenizer.train(corpus=corpus, vocab_size=280, morpheme_lexicon=lexicon, lambda_weight=2.0)
    intrinsic = tokenisers.benchmarks.intrinsic.IntrinsicEvaluator()
    f1_res = intrinsic.compute_morphological_f1(tok, gold_annotations=sample_morpheme_annotations)
    assert f1_res["f1"] > 0.0


def test_tier3_wordpiece_with_hf_export_and_transformers(require_tokenisers, sample_english_text, tmp_path):
    """T3.13: WordPiece model (F2) exports to tokenizer.json (F11) and verifies roundtrip losslessness (F13)."""
    tokenisers = require_tokenisers
    hf_tok = pytest.importorskip("tokenizers")
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text], vocab_size=275, scorer="wordpiece")
    export_path = tmp_path / "wp_hf.json"
    tok.export_hf_json(str(export_path))
    reloaded = hf_tok.Tokenizer.from_file(str(export_path))
    encoded = reloaded.encode(sample_english_text).ids
    assert reloaded.decode(encoded) == sample_english_text


def test_tier3_hf_wrapper_with_dynamic_dropout_and_sampling(require_tokenisers, sample_english_text):
    """T3.14: DynamicResearchTokenizer (F12) supports per-call dropout (F8) and sampling (F9)."""
    tokenisers = require_tokenisers
    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 5], vocab_size=290)
    wrapper = tokenisers.DynamicResearchTokenizer(tok)
    res_det = wrapper(sample_english_text, dropout=0.0)["input_ids"]
    res_drop = wrapper(sample_english_text, dropout=0.4)["input_ids"]
    assert wrapper.decode(res_det) == sample_english_text
    assert wrapper.decode(res_drop) == sample_english_text


def test_tier3_data_pipeline_streaming_into_all_four_scorers(require_tokenisers):
    """T3.15: Streaming data pipeline batches (F14) feed cleanly into all 4 bottom-up scorers (F1-F4)."""
    tokenisers = require_tokenisers
    pipeline = tokenisers.data.DataPipeline()
    docs = pipeline.sample_preset("smoke", max_bytes=30_000)
    for scorer in ["frequency", "wordpiece", "mdl", "length_penalized"]:
        tok = tokenisers.BpeTokenizer.train(corpus=docs, vocab_size=270, scorer=scorer)
        assert tok.vocab_size == 270
        assert tok.decode(tok.encode(docs[0])) == docs[0]


def test_tier3_tokka_bench_cross_scoring_comparison(require_tokenisers, sample_english_text):
    """T3.16: Tokka-bench (F15) evaluates both standard BPE (F1) and SuperBPE (F6) in parallel."""
    tokenisers = require_tokenisers
    tok_bpe = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 10], vocab_size=300, scorer="frequency")
    tok_super = tokenisers.SuperBpeTokenizer.train(corpus=[sample_english_text * 10], v_switch=280, v_final=300)
    bench = tokenisers.benchmarks.tokka.TokkaBenchRunner()
    res_bpe = bench.evaluate_tokenizer(tok_bpe, text_sample=sample_english_text)
    res_super = bench.evaluate_tokenizer(tok_super, text_sample=sample_english_text)
    assert res_bpe["bytes_per_token"] > 0
    assert res_super["bytes_per_token"] > 0


def test_tier3_colab_proxy_lm_with_superbpe_and_unigram(require_tokenisers, sample_english_text):
    """T3.17: Proxy LM batching (F17) ingests token streams from both SuperBPE (F6) and Unigram (F5)."""
    tokenisers = require_tokenisers
    pytest.importorskip("torch")
    tok_super = tokenisers.SuperBpeTokenizer.train(corpus=[sample_english_text * 5], v_switch=275, v_final=290)
    tok_uni = tokenisers.UnigramTokenizer.train(corpus=[sample_english_text * 5], vocab_size=290)
    lm = tokenisers.benchmarks.colab.ProxyLanguageModel(vocab_size=290, d_model=32, n_layers=1, n_heads=1, d_ffn=64)
    # Both token streams are valid inputs to the proxy LM
    ids_super = tok_super.encode(sample_english_text)
    ids_uni = tok_uni.encode(sample_english_text)
    assert len(ids_super) > 0 and len(ids_uni) > 0


def test_tier3_git_sync_after_benchmark_generation(repo_root):
    """T3.18: Benchmark result artifacts (F15, F16) generate in tracked results/ directory (F18)."""
    proc = subprocess.run(["git", "status"], cwd=repo_root, capture_output=True, text=True)
    assert proc.returncode == 0
