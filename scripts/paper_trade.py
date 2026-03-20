#!/usr/bin/env python3
"""Paper trading tracker — record picks, evaluate returns, report performance.

Usage:
    python scripts/paper_trade.py record                    # Record today's picks
    python scripts/paper_trade.py evaluate                  # Evaluate yesterday's picks
    python scripts/paper_trade.py report                    # Print performance
    python scripts/paper_trade.py backfill --start 2026-03-01  # Backfill history

Cron:
    30 4 * * 2-6 cd /path/to/factor-lab && python3 scripts/paper_trade.py record >> logs/paper.log 2>&1
    15 7 * * 2-6 cd /path/to/factor-lab && python3 scripts/paper_trade.py evaluate >> logs/paper.log 2>&1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.paper.tracker import record, evaluate, report, backfill


def main():
    parser = argparse.ArgumentParser(description="Paper trading tracker")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("record", help="Record today's top/bottom picks")
    sub.add_parser("evaluate", help="Evaluate returns for previous picks")
    sub.add_parser("report", help="Print performance summary")

    bf = sub.add_parser("backfill", help="Backfill historical picks and returns")
    bf.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    bf.add_argument("--end", help="End date (default: latest)")

    args = parser.parse_args()

    if args.cmd == "record":
        record()
    elif args.cmd == "evaluate":
        evaluate()
    elif args.cmd == "report":
        report()
    elif args.cmd == "backfill":
        backfill(args.start, args.end)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
