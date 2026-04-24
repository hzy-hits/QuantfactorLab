#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.loop import FactorSession
from src.autoresearch.dashboard import export_dashboard_bundle
from src.autoresearch.session_state import ensure_session_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a resumable Factor Lab autoresearch session.")
    parser.add_argument("--market", choices=["cn", "us"], required=True)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--time-budget-minutes", type=int, default=55)
    parser.add_argument("--model", default="claude-opus-4-6")
    parser.add_argument("--goal", default=None, help="Optional session objective for autoresearch.md")
    parser.add_argument("--output", default=None, help="Optional markdown output path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    paths = ensure_session_files(args.market, goal=args.goal)
    output = Path(args.output) if args.output else (
        Path(__file__).resolve().parents[1]
        / "reports"
        / f"autoresearch_{args.market}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )

    session = FactorSession(
        market=args.market,
        budget=args.budget,
        model=args.model,
        time_budget_minutes=args.time_budget_minutes,
        session_context_path=paths.session_doc,
        experiments_file=paths.log_file,
        journal_path=paths.journal_file,
        checks_script_path=paths.checks_script,
    )
    result = session.run()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.summary, encoding="utf-8")
    dashboard_bundle = export_dashboard_bundle(market=args.market, log_path=paths.log_file)
    print(f"\nReport written to {output}")
    print(f"Session context: {paths.session_doc}")
    print(f"Session log:     {paths.log_file}")
    print(f"Dashboard HTML:  {dashboard_bundle['html']}")
    print(f"Dashboard JSON:  {dashboard_bundle['json']}")


if __name__ == "__main__":
    main()
