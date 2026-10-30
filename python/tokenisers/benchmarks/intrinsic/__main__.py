"""CLI: extended intrinsic suite on held-out text, SIGMORPHON test gold and held-out code identifiers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

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
from tokenisers.benchmarks.intrinsic import IntrinsicEvaluator, mine_identifiers


THROUGHPUT_BYTES = 1024 * 1024


def throughput_docs(ds) -> List[str]:
    """Test documents, topped up with val then train documents to at least 1 MiB when the test split is smaller."""
    docs: List[str] = []
    n = 0
    for split in ("test", "val", "train"):
        for d in ds.texts(split):
            if n >= THROUGHPUT_BYTES and split != "test":
                return docs
            docs.append(d)
            n += len(d.encode("utf-8"))
    return docs


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extended intrinsic tokenizer benchmark suite")
    add_common_args(parser, "benchmarks/results/intrinsic")
    parser.add_argument("--morph-langs", nargs="+", default=list(DEFAULT_MORPH_LANGS),
                        help="SIGMORPHON 2022 languages (gold: test split; morpheme_bpe lexicon: train split)")
    parser.add_argument("--max-identifiers", type=int, default=2000)
    parser.add_argument("--throughput-repeats", type=int, default=5)
    args = parser.parse_args(argv)
    start = start_provenance(__package__, argv)  # before the tree can change during a long run

    if not args.skip_memory_check:  # before downloading/preparing anything
        check_memory(args.variants, args.train_bytes)
    ds = load_dataset(args)
    train = ds.texts("train")
    test = ds.texts("test")
    assert_disjoint(train, test, "test")
    lexicon, gold, morph_prov = load_morphology(args, train, args.morph_langs)
    identifiers = mine_identifiers(ds.texts("test", domain="code"), args.max_identifiers)

    print(f"Training {len(args.variants)} tokenizers at V={args.vocab_size} on {len(train)} train docs")
    models, records = train_suite(args.variants, train, args.vocab_size, lexicon)
    duplicates = find_duplicates(models)
    models = {n: t for n, t in models.items() if n not in duplicates}

    meta = run_metadata(
        args, ds, "intrinsic", "test", records, duplicates, start=start,
        morphology=morph_prov,
        identifier_source=f"{len(identifiers)} most frequent multi-part identifiers of the test split's code",
    )
    IntrinsicEvaluator().evaluate_suite(
        models, test, gold, output_dir=args.output_dir, identifiers=identifiers, meta=meta,
        repeats=args.throughput_repeats, throughput_docs=throughput_docs(ds),
    )
    print((Path(args.output_dir) / "intrinsic_leaderboard.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
