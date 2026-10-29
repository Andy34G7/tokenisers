"""CLI entry point for tokenisers.benchmarks.intrinsic."""

from __future__ import annotations

import argparse
from pathlib import Path
from tokenisers.benchmarks.intrinsic import IntrinsicEvaluator
from tokenisers.data.pipeline import DataPipeline, SigmorphonLoader
from tokenisers.models import (
    BpeTokenizer,
    MorphemeBpeTokenizer,
    SuperBpeTokenizer,
    UnigramTokenizer,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extended Intrinsic Tokenizer Benchmark Suite")
    parser.add_argument("--vocab-size", type=int, default=400, help="Target vocabulary size")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results/intrinsic", help="Directory for output JSON/CSV/Markdown")
    args = parser.parse_args()

    pipeline = DataPipeline(force_offline=True)
    corpus = pipeline.sample_preset("smoke", max_bytes=120_000)
    eval_text = "\n".join(corpus[:80])
    gold_annotations = SigmorphonLoader().load_builtin_samples()
    lexicon = {item["word"]: item["segments"] for item in gold_annotations}

    v_size = args.vocab_size
    v_switch = max(260, int(v_size * 0.75))
    morph_corpus = corpus + [item["word"] for item in gold_annotations] * 15

    models = {
        "bpe_frequency": BpeTokenizer.train(corpus=morph_corpus, vocab_size=v_size, scorer="frequency"),
        "wordpiece_pmi": BpeTokenizer.train(corpus=morph_corpus, vocab_size=v_size, scorer="wordpiece"),
        "wp_bpe_base": BpeTokenizer.train(corpus=morph_corpus, vocab_size=v_size, scorer="wp-bpe-base"),
        "wp_bpe_ft": BpeTokenizer.train(corpus=morph_corpus, vocab_size=v_size, scorer="wp-bpe-ft", min_frequency=2),
        "wp_bpe_pig": BpeTokenizer.train(corpus=morph_corpus, vocab_size=v_size, scorer="wp-bpe-pig"),
        "mdl_bits": BpeTokenizer.train(corpus=morph_corpus, vocab_size=v_size, scorer="mdl"),
        "length_penalized": BpeTokenizer.train(corpus=morph_corpus, vocab_size=v_size, scorer="length_penalized", beta=0.5),
        "super_bpe": SuperBpeTokenizer.train(corpus=morph_corpus, v_switch=v_switch, v_final=v_size),
        "morpheme_bpe": MorphemeBpeTokenizer.train(
            corpus=morph_corpus, vocab_size=v_size, morpheme_lexicon=lexicon, lambda_weight=2.5
        ),
        "unigram_lm": UnigramTokenizer.train(corpus=morph_corpus, vocab_size=v_size),
    }

    evaluator = IntrinsicEvaluator()
    evaluator.evaluate_suite(models, eval_text, gold_annotations, output_dir=args.output_dir)
    md_file = Path(args.output_dir) / "intrinsic_leaderboard.md"
    print(md_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
