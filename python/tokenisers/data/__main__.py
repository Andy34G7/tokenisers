"""CLI: prepare (download, de-duplicate, split, cache) a dataset and print its manifest summary."""

from __future__ import annotations

import argparse
import logging

from tokenisers.data.pipeline import (
    PRESET_BYTE_SIZES,
    SIGMORPHON_LANGUAGES,
    SPLITS,
    DataPipeline,
    SigmorphonLoader,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the tokenisers training/evaluation corpus")
    parser.add_argument(
        "--preset",
        default="small",
        choices=list(PRESET_BYTE_SIZES),
        help="Train-split size preset (smoke=100KB, small=50MB, medium=100MB, large=200MB)",
    )
    parser.add_argument("--train-bytes", type=int, default=None, help="Train-split bytes; overrides --preset")
    parser.add_argument("--seed", type=int, default=0, help="Split seed")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--cache-dir", default=None, help="Cache directory (default $TOKENISERS_CACHE_DIR or ~/.cache/tokenisers/data)")
    parser.add_argument(
        "--offline",
        "--force-offline",
        action="store_true",
        help="Generate the SYNTHETIC test corpus instead of downloading real data",
    )
    parser.add_argument("--allow-partial", action="store_true", help="Keep going when a source fails mid-stream")
    parser.add_argument(
        "--sigmorphon",
        nargs="*",
        default=None,
        metavar="LANG",
        help=f"Also download SIGMORPHON 2022 train/dev/test for these languages (no LANG = all of {' '.join(SIGMORPHON_LANGUAGES)})",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    train_bytes = args.train_bytes if args.train_bytes is not None else PRESET_BYTE_SIZES[args.preset]
    pipeline = DataPipeline(cache_dir=args.cache_dir, force_offline=args.offline)
    ds = pipeline.prepare(
        train_bytes=train_bytes,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        allow_partial=args.allow_partial,
    )
    m = ds.manifest
    label = "SYNTHETIC offline test corpus" if m["synthetic"] else "real data"
    print(f"Dataset ({label}, complete={m['complete']}): {ds.root}")
    for split in SPLITS:
        info = m["splits"][split]
        print(f"  {split:5s} {info['docs']:>8,d} docs {info['text_bytes']:>13,d} bytes  {ds.path(split)}")
    for fb in m["fallbacks"]:
        print(f"  FALLBACK {fb['domain']}: {fb['wanted']} -> {fb['used']} ({fb['reason']})")
    for err in m["errors"]:
        print(f"  ERROR {err['domain']}/{err['lang']} {err['repo']}: {err['error']}")

    if args.sigmorphon is not None:
        loader = SigmorphonLoader(cache_dir=args.cache_dir, offline=args.offline)
        for lang in args.sigmorphon or SIGMORPHON_LANGUAGES:
            for split in ("train", "dev", "test"):
                entries = loader.load(lang, split, surface_only=False)
                surface = sum(e["surface"] for e in entries)
                print(f"  SIGMORPHON {lang} {split}: {len(entries):,d} words ({surface:,d} surface) {loader.path(lang, split)}")


if __name__ == "__main__":
    main()
