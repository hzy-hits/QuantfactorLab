#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.autoresearch.session_state import ensure_session_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize Factor Lab autoresearch session files.")
    parser.add_argument("--market", choices=["cn", "us"], required=True)
    parser.add_argument("--goal", default=None, help="Optional session objective to seed autoresearch.md")
    args = parser.parse_args()

    paths = ensure_session_files(args.market, goal=args.goal)
    print(f"Session dir: {paths.session_dir}")
    print(f"Context:     {paths.session_doc}")
    print(f"Benchmark:   {paths.benchmark_script}")
    print(f"Checks:      {paths.checks_script}")
    print(f"Run log:     {paths.log_file}")


if __name__ == "__main__":
    main()
