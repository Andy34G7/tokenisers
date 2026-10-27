"""CLI entry point for tokenisers.data."""

from __future__ import annotations

import argparse
from pathlib import Path
from tokenisers.data.pipeline import DataPipeline, PRESET_BYTE_SIZES


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenisers Data Pipeline Corpus Preparation")
    parser.add_argument(
        "--preset",
        type=str,
        default="smoke",
        choices=list(PRESET_BYTE_SIZES.keys()),
        help="Dataset preset size (smoke=100KB, small=50MB, medium=100MB, large=200MB)",
    )
    parser.add_argument("--max-bytes", type=int, default=None, help="Override max bytes to sample")
    parser.add_argument("--cache-dir", type=str, default=None, help="Directory to cache corpus")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Target split")
    parser.add_argument("--force-offline", action="store_true", help="Force synthetic offline generation")
    parser.add_argument("--stream-remote", action="store_true", help="Stream live from HuggingFace Hub")
    args = parser.parse_args()

    target_bytes = args.max_bytes if args.max_bytes is not None else PRESET_BYTE_SIZES[args.preset]
    pipeline = DataPipeline(cache_dir=args.cache_dir, force_offline=args.force_offline)
    path = pipeline.prepare_corpus(
        preset=args.preset,
        max_bytes=target_bytes,
        split=args.split,
        stream_remote=args.stream_remote,
    )
    size_bytes = Path(path).stat().st_size
    print(f"Prepared corpus ({args.split}) at: {path} ({size_bytes:,} bytes)")


if __name__ == "__main__":
    main()
