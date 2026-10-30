#!/usr/bin/env python3
"""Regenerate the README.md leaderboard sections from the committed benchmark result JSON.

Each section sits between ``<!-- BEGIN GENERATED: <name> -->`` and ``<!-- END GENERATED: <name> -->``.

    uv run python scripts/gen_readme_tables.py           # rewrite README.md
    uv run python scripts/gen_readme_tables.py --check   # exit 1 if README.md is stale
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional

from tokenisers.benchmarks.report import render_colab, render_intrinsic, render_tokka

ROOT = Path(__file__).resolve().parents[1]

SECTIONS = {
    "tokka": ("benchmarks/results/tokka/tokka_matrix.json", render_tokka, "tokenisers.benchmarks.tokka"),
    "intrinsic": ("benchmarks/results/intrinsic/intrinsic_leaderboard.json", render_intrinsic,
                  "tokenisers.benchmarks.intrinsic"),
    "colab": ("benchmarks/results/colab/extrinsic_bpb_results.json", render_colab, "tokenisers.benchmarks.colab"),
}


def render_readme(text: str, root: Path = ROOT) -> str:
    for name, (rel, render, module) in SECTIONS.items():
        begin, end = f"<!-- BEGIN GENERATED: {name} -->", f"<!-- END GENERATED: {name} -->"
        i = text.index(begin) + len(begin)
        j = text.index(end, i)
        path = root / rel
        if path.is_file():
            body = render(json.loads(path.read_text(encoding="utf-8")))
        else:
            body = (f"_No results committed (`{rel}` is missing). Run `uv run python -m {module}`, then "
                    "`uv run python scripts/gen_readme_tables.py`._")
        text = text[:i] + "\n" + body + "\n" + text[j:]
    return text


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Only check that README.md is up to date")
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    args = parser.parse_args(argv)
    current = args.readme.read_text(encoding="utf-8")
    fresh = render_readme(current)
    if args.check:
        if fresh != current:
            print(f"{args.readme} is stale: run uv run python scripts/gen_readme_tables.py", file=sys.stderr)
            return 1
        return 0
    args.readme.write_text(fresh, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
