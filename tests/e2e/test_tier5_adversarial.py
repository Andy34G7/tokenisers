"""
Tier 5: Adversarial Stress & Full End-to-End Benchmark Verification Suite.
"""

from __future__ import annotations

from pathlib import Path
import pytest


def test_tier5_adv1_tokka_cjk_segmentation_fertility_bounds(require_tokenisers):
    """Adv-1: Unspaced CJK text is segmented at ideograph/kana granularity rather than treated as 1 word."""
    tokenisers = require_tokenisers
    cjk_sample = "道可道，非常道；名可名，非常名。无名天地之始，有名万物之母。"
    tok = tokenisers.BpeTokenizer.train(corpus=[cjk_sample * 5], vocab_size=280)
    bench = tokenisers.benchmarks.tokka.TokkaBenchRunner()

    # Evaluate both with explicit lang_code="zh" and auto-detection
    res_zh = bench.evaluate_tokenizer(tok, text_sample=cjk_sample, lang_code="zh")
    res_auto = bench.evaluate_tokenizer(tok, text_sample=cjk_sample)

    assert 1.0 <= res_zh["subword_fertility"] <= 4.0
    assert 1.0 <= res_auto["subword_fertility"] <= 4.0
    assert res_zh["bytes_per_token"] > 0.0


def test_tier5_adv2_data_pipeline_train_and_val_splits(require_tokenisers, tmp_path):
    """Adv-2: DataPipeline.prepare writes train/val/test split files that share no document."""
    tokenisers = require_tokenisers
    pipeline = tokenisers.data.DataPipeline(cache_dir=tmp_path, force_offline=True)
    ds = pipeline.prepare(train_bytes=20_000)
    texts = {split: ds.texts(split) for split in ("train", "val", "test")}

    assert sum(len(t.encode("utf-8")) for t in texts["train"]) >= 20_000
    assert texts["val"] and texts["test"]
    assert len({ds.path(s) for s in texts}) == 3
    assert not set(texts["train"]) & set(texts["val"])
    assert not set(texts["train"]) & set(texts["test"])
    assert not set(texts["val"]) & set(texts["test"])


def test_tier5_adv3_sigmorphon_builtin_loader_gold_cuts(require_tokenisers):
    """Adv-3: SigmorphonLoader.load_offline_fixture returns valid multilingual morphological cuts."""
    tokenisers = require_tokenisers
    loader = tokenisers.data.SigmorphonLoader()
    samples = loader.load_offline_fixture()
    assert len(samples) >= 10
    for item in samples:
        word_bytes = len(item["word"].encode("utf-8"))
        assert len(item["gold_cuts"]) == len(item["segments"]) - 1
        for cut in item["gold_cuts"]:
            assert 0 < cut < word_bytes


def test_tier5_adv4_intrinsic_evaluate_suite_exports_leaderboards(require_tokenisers, sample_english_text, tmp_path):
    """Adv-4: IntrinsicEvaluator.evaluate_suite exports JSON, CSV, and Markdown leaderboards."""
    tokenisers = require_tokenisers
    corpus = [sample_english_text * 4]
    gold = tokenisers.data.SigmorphonLoader().load_offline_fixture()
    models = {
        "bpe_freq": tokenisers.BpeTokenizer.train(corpus=corpus, vocab_size=275, scorer="frequency"),
        "super_bpe": tokenisers.SuperBpeTokenizer.train(corpus=corpus, v_switch=265, v_final=275),
    }
    evaluator = tokenisers.benchmarks.intrinsic.IntrinsicEvaluator()
    res = evaluator.evaluate_suite(models, sample_english_text, gold, output_dir=tmp_path)

    assert "bpe_freq" in res and "super_bpe" in res
    assert (tmp_path / "intrinsic_leaderboard.json").exists()
    assert (tmp_path / "intrinsic_leaderboard.csv").exists()
    assert (tmp_path / "intrinsic_leaderboard.md").exists()


def test_tier5_adv5_colab_byte_budget_real_training_loop(require_tokenisers, sample_english_text):
    """Adv-5: ByteBudgetTrainer executes real causal gradient updates and BpbEvaluator computes validation BPB."""
    tokenisers = require_tokenisers
    pytest.importorskip("torch")

    tok = tokenisers.BpeTokenizer.train(corpus=[sample_english_text * 5], vocab_size=280)
    ids = tok.encode(sample_english_text * 3)
    raw_bytes = len((sample_english_text * 3).encode("utf-8"))

    model = tokenisers.benchmarks.colab.ProxyLanguageModel(
        vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_heads=2, d_ffn=64, max_seq_len=64
    )
    trainer = tokenisers.benchmarks.colab.ByteBudgetTrainer(model=model, byte_budget=1000, lr=5e-3)
    history = trainer.train_on_token_stream(ids, total_utf8_bytes=raw_bytes, seq_len=16, batch_size=2)

    assert len(history) >= 1
    assert trainer.bytes_consumed >= 1000
    assert history[-1]["train_bpb"] > 0.0

    evaluator = tokenisers.benchmarks.colab.BpbEvaluator()
    val_metrics = evaluator.evaluate_validation_bpb(model, ids[:64], utf8_bytes=raw_bytes // 3, seq_len=16)
    assert val_metrics["bpb"] > 0.0
    assert val_metrics["bytes_per_token"] > 0.0
