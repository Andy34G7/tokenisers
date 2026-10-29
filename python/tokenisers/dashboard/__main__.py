"""
CLI entry point for running the Tokenisers Lab Benchmark Dashboard.
Usage:
    python -m tokenisers.dashboard --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import argparse
import sys
import webbrowser

from tokenisers.dashboard.server import start_server


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tokenisers.dashboard",
        description="Artificial Analysis-style Benchmark Intelligence Dashboard for Tokenisers Lab",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port number to bind (default: 8765)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open dashboard in the default web browser upon launch",
    )

    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/"
    print("=" * 72)
    print("  TOKENISERS LAB — ALGORITHMIC BENCHMARK INTELLIGENCE DASHBOARD")
    print(f"  Live UI available at: {url}")
    print("  Endpoints:")
    print(f"    - Overview & Visual Dashboard: {url}")
    print(f"    - Benchmarks Unified JSON API: {url}api/benchmarks")
    print(f"    - Live Tokenizer Playground API: {url}api/tokenize")
    print("=" * 72)

    if args.open:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        start_server(host=args.host, port=args.port, blocking=True)
    except KeyboardInterrupt:
        print("\nExiting dashboard.")
        sys.exit(0)


if __name__ == "__main__":
    main()
