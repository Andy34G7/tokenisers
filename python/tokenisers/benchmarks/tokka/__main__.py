"""CLI entry point for tokenisers.benchmarks.tokka."""

from __future__ import annotations

import argparse
from pathlib import Path
from tokenisers.benchmarks.tokka import TokkaBenchRunner
from tokenisers.data.pipeline import DataPipeline, OfflineFallbackGenerator, SigmorphonLoader
from tokenisers.models import (
    BpeTokenizer,
    MorphemeBpeTokenizer,
    SuperBpeTokenizer,
    UnigramTokenizer,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokka-Bench Multilingual & Code Tokenizer Benchmark Runner")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Path to HF tokenizer.json (optional; if omitted, trains & evaluates full suite)")
    parser.add_argument("--corpus-path", type=str, default=None, help="Path to plain text evaluation corpus")
    parser.add_argument("--lang", type=str, default="en", help="Language code (e.g. en, de, zh, code)")
    parser.add_argument("--vocab-size", type=int, default=512, help="Vocab size when training full matrix")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results/tokka", help="Directory for output JSON/CSV/Markdown")
    args = parser.parse_args()

    runner = TokkaBenchRunner()

    if args.tokenizer_path:
        import tokenizers  # type: ignore
        tok = tokenizers.Tokenizer.from_file(args.tokenizer_path)
        text = (
            Path(args.corpus_path).read_text(encoding="utf-8")
            if args.corpus_path
            else "\n".join(OfflineFallbackGenerator().SAMPLE_PROSE)
        )
        out_file = Path(args.output_dir) / "tokka_single.json"
        metrics = runner.run_and_save(tok, text_sample=text, output_path=str(out_file), lang_code=args.lang)
        print(f"Tokka-bench single result saved to: {out_file}")
        print(metrics)
        return

    pipeline = DataPipeline(force_offline=True)
    corpus = pipeline.sample_preset("smoke", max_bytes=120_000)
    morph_samples = SigmorphonLoader().load_builtin_samples()
    lexicon = {item["word"]: item["segments"] for item in morph_samples}
    v_size = args.vocab_size
    v_switch = max(260, int(v_size * 0.75))

    models = {
        "bpe_frequency": BpeTokenizer.train(corpus=corpus, vocab_size=v_size, scorer="frequency"),
        "wordpiece_pmi": BpeTokenizer.train(corpus=corpus, vocab_size=v_size, scorer="wordpiece"),
        "wp_bpe_base": BpeTokenizer.train(corpus=corpus, vocab_size=v_size, scorer="wp-bpe-base"),
        "wp_bpe_ft": BpeTokenizer.train(corpus=corpus, vocab_size=v_size, scorer="wp-bpe-ft", min_frequency=2),
        "wp_bpe_pig": BpeTokenizer.train(corpus=corpus, vocab_size=v_size, scorer="wp-bpe-pig"),
        "mdl_bits": BpeTokenizer.train(corpus=corpus, vocab_size=v_size, scorer="mdl"),
        "length_penalized": BpeTokenizer.train(corpus=corpus, vocab_size=v_size, scorer="length_penalized", beta=0.5),
        "super_bpe": SuperBpeTokenizer.train(corpus=corpus, v_switch=v_switch, v_final=v_size),
        "morpheme_bpe": MorphemeBpeTokenizer.train(corpus=corpus, vocab_size=v_size, morpheme_lexicon=lexicon, lambda_weight=2.0),
        "unigram_lm": UnigramTokenizer.train(corpus=corpus, vocab_size=v_size),
    }

    gen = OfflineFallbackGenerator()
    domain_texts = {
        "en": "\n".join(gen.SAMPLE_PROSE * 5),
        "de": gen.SAMPLE_MULTILINGUAL[0] * 5,
        "fr": gen.SAMPLE_MULTILINGUAL[1] * 5,
        "ru": gen.SAMPLE_MULTILINGUAL[2] * 5,
        "zh": gen.SAMPLE_MULTILINGUAL[3] * 5,
        "ja": gen.SAMPLE_MULTILINGUAL[4] * 5,
        "ar": gen.SAMPLE_MULTILINGUAL[5] * 5,
        "hi": gen.SAMPLE_MULTILINGUAL[6] * 5,
        "hu": gen.SAMPLE_MULTILINGUAL[7] * 5,
        "fi": gen.SAMPLE_MULTILINGUAL[8] * 5,
        "code": "\n".join(gen.SAMPLE_CODE * 5),
    }

    runner.evaluate_matrix(models, domain_texts, output_dir=args.output_dir)
    md_file = Path(args.output_dir) / "tokka_leaderboard.md"
    print(md_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
