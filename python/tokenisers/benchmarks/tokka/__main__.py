"""CLI: Tokka-Bench matrix on held-out text of every corpus language (natural and code)."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, List

from tokenisers.benchmarks.common import (
    DEFAULT_MORPH_LANGS,
    add_common_args,
    assert_disjoint,
    check_memory,
    find_duplicates,
    start_provenance,
    load_dataset,
    load_morphology,
    run_metadata,
    train_suite,
)
from tokenisers.benchmarks.tokka import TokkaBenchRunner, short_lang

# tokka-bench samples ~2 MB per language; longer held-out text is truncated (UTF-8 safe).
MAX_BYTES_PER_LANG = 2 * 1024 * 1024
# A language enters the macro average only with at least this much held-out text: one short
# document says little about a language, yet macro weights every language equally.
MIN_DOCS_PER_LANG = 3
MIN_BYTES_PER_LANG = 10_000


class _HfTokenizer:
    """`tokenizers.Tokenizer` with the encode/vocab_size interface the runner uses."""

    def __init__(self, path: str) -> None:
        import tokenizers

        self._tok = tokenizers.Tokenizer.from_file(path)
        self.vocab_size = self._tok.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text, add_special_tokens=False).ids


def language_texts(docs: List[Dict[str, Any]], max_bytes: int = MAX_BYTES_PER_LANG) -> Dict[str, str]:
    """{language key: held-out documents joined by newlines}, as tokka-bench builds its samples."""
    by_lang: Dict[str, List[str]] = {}
    for d in docs:
        by_lang.setdefault(short_lang(d["lang"]), []).append(d["text"])
    out = {}
    for lang, texts in by_lang.items():
        raw = "\n".join(texts).encode("utf-8")
        out[lang] = raw[:max_bytes].decode("utf-8", errors="ignore")
    return out


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tokka-Bench multilingual & code tokenizer benchmark")
    add_common_args(parser, "benchmarks/results/tokka")
    parser.add_argument("--tokenizer-path", default=None,
                        help="Evaluate this HF tokenizer.json instead of training the suite")
    parser.add_argument("--max-bytes-per-lang", type=int, default=MAX_BYTES_PER_LANG)
    parser.add_argument("--min-docs-per-lang", type=int, default=MIN_DOCS_PER_LANG,
                        help="Held-out documents a language needs to enter the macro average")
    parser.add_argument("--min-bytes-per-lang", type=int, default=MIN_BYTES_PER_LANG,
                        help="Held-out UTF-8 bytes a language needs to enter the macro average")
    args = parser.parse_args(argv)
    start = start_provenance(__package__, argv)  # before the tree can change during a long run

    if not args.tokenizer_path and not args.skip_memory_check:  # before downloading/preparing anything
        check_memory(args.variants, args.train_bytes)
    ds = load_dataset(args)
    test_docs = ds.docs("test")
    texts = language_texts(test_docs, args.max_bytes_per_lang)
    docs_per_lang = Counter(short_lang(d["lang"]) for d in test_docs)
    bytes_per_lang = {lang: len(t.encode("utf-8")) for lang, t in texts.items()}
    macro_langs = [lang for lang in texts
                   if docs_per_lang[lang] >= args.min_docs_per_lang and bytes_per_lang[lang] >= args.min_bytes_per_lang]
    for lang in texts:
        if lang not in macro_langs:
            print(f"warning: {lang} has only {docs_per_lang[lang]} held-out docs / {bytes_per_lang[lang]:,} bytes; "
                  "reported, but left out of the macro average", file=sys.stderr)
    if not macro_langs:
        raise SystemExit(f"No language has >= {args.min_docs_per_lang} held-out documents and >= "
                         f"{args.min_bytes_per_lang:,} bytes: prepare a larger dataset (--train-bytes) or lower "
                         "--min-docs-per-lang / --min-bytes-per-lang")
    runner = TokkaBenchRunner()

    if args.tokenizer_path:
        models = {Path(args.tokenizer_path).stem: _HfTokenizer(args.tokenizer_path)}
        records: Dict[str, Any] = {}
        duplicates: Dict[str, str] = {}
    else:
        train = ds.texts("train")
        assert_disjoint(train, [d["text"] for d in test_docs], "test")
        lexicon = {}
        if "morpheme_bpe" in args.variants:
            lexicon, _, _ = load_morphology(args, train, DEFAULT_MORPH_LANGS)
        print(f"Training {len(args.variants)} tokenizers at V={args.vocab_size} on {len(train)} train docs")
        models, records = train_suite(args.variants, train, args.vocab_size, lexicon)
        duplicates = find_duplicates(models)
        models = {n: t for n, t in models.items() if n not in duplicates}

    meta = run_metadata(
        args, ds, "tokka", "test", records, duplicates, start=start,
        tokka_bench="vendor/tokka-bench (tokka_bench.metrics.calculate_word_metrics)",
        language_bytes=bytes_per_lang,
        language_docs={lang: docs_per_lang[lang] for lang in texts},
        macro_languages=macro_langs,
        macro_min_docs=args.min_docs_per_lang,
        macro_min_bytes=args.min_bytes_per_lang,
        tokenizer_path=args.tokenizer_path,
    )
    runner.evaluate_matrix(models, texts, output_dir=args.output_dir, meta=meta, macro_langs=macro_langs)
    print((Path(args.output_dir) / "tokka_leaderboard.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
