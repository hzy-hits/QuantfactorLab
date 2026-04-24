#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.autoresearch.dashboard import export_dashboard_bundle
from src.autoresearch.session_state import session_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Factor Lab autoresearch dashboard artifacts.")
    parser.add_argument("--market", choices=["cn", "us"], required=True)
    parser.add_argument("--session-log", default=None, help="Optional path to autoresearch.jsonl")
    parser.add_argument("--output-dir", default=None, help="Optional export directory")
    args = parser.parse_args()

    paths = session_paths(args.market)
    log_path = Path(args.session_log) if args.session_log else paths.log_file
    output_dir = Path(args.output_dir) if args.output_dir else None
    bundle = export_dashboard_bundle(market=args.market, log_path=log_path, output_dir=output_dir)
    print(f"Markdown: {bundle['markdown']}")
    print(f"JSON:     {bundle['json']}")
    print(f"HTML:     {bundle['html']}")


if __name__ == "__main__":
    main()
