"""CLI: Tokka-Bench matrix on held-out text of every corpus language (natural and code)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from tokenisers.benchmarks.common import (
    DEFAULT_MORPH_LANGS,
    add_common_args,
    assert_disjoint,
    check_memory,
    find_duplicates,
    load_dataset,
    load_morphology,
    run_metadata,
    train_suite,
)
from tokenisers.benchmarks.tokka import TokkaBenchRunner, short_lang

# tokka-bench samples ~2 MB per language; longer held-out text is truncated (UTF-8 safe).
MAX_BYTES_PER_LANG = 2 * 1024 * 1024


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
    args = parser.parse_args(argv)

    ds = load_dataset(args)
    test_docs = ds.docs("test")
    texts = language_texts(test_docs, args.max_bytes_per_lang)
    runner = TokkaBenchRunner()

    if args.tokenizer_path:
        models = {Path(args.tokenizer_path).stem: _HfTokenizer(args.tokenizer_path)}
        records: Dict[str, Any] = {}
        duplicates: Dict[str, str] = {}
    else:
        train = ds.texts("train")
        assert_disjoint(train, [d["text"] for d in test_docs], "test")
        if not args.skip_memory_check:
            check_memory(args.variants, ds.manifest["splits"]["train"]["text_bytes"])
        lexicon = {}
        if "morpheme_bpe" in args.variants:
            lexicon, _, _ = load_morphology(args, train, DEFAULT_MORPH_LANGS)
        print(f"Training {len(args.variants)} tokenizers at V={args.vocab_size} on {len(train)} train docs")
        models, records = train_suite(args.variants, train, args.vocab_size, lexicon)
        duplicates = find_duplicates(models)
        models = {n: t for n, t in models.items() if n not in duplicates}

    meta = run_metadata(
        args, ds, "tokka", "test", records, duplicates,
        tokka_bench="vendor/tokka-bench (tokka_bench.metrics.calculate_word_metrics)",
        language_bytes={lang: len(t.encode("utf-8")) for lang, t in texts.items()},
        tokenizer_path=args.tokenizer_path,
    )
    runner.evaluate_matrix(models, texts, output_dir=args.output_dir, meta=meta)
    print((Path(args.output_dir) / "tokka_leaderboard.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
