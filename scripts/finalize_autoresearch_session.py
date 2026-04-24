#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.autoresearch.finalize import apply_finalize_plan, build_finalize_plan
from src.autoresearch.session_state import session_paths
from src.paths import FACTOR_LAB_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize kept autoresearch experiments into reviewable branches.")
    parser.add_argument("--market", choices=["cn", "us"], required=True)
    parser.add_argument("--session-log", default=None, help="Optional path to autoresearch.jsonl")
    parser.add_argument("--apply", action="store_true", help="Actually create branches and commits")
    args = parser.parse_args()

    paths = session_paths(args.market)
    log_path = Path(args.session_log) if args.session_log else paths.log_file
    plans = build_finalize_plan(log_path, args.market)
    if not plans:
        print("No kept experiments found.")
        return

    print(f"Finalize plan for {len(plans)} kept experiments:")
    for plan in plans:
        print(f"- {plan.branch_name} <= {plan.experiment_name} [{plan.output_rel_dir}]")

    if not args.apply:
        print("Dry run only. Re-run with --apply to create branches.")
        return

    created = apply_finalize_plan(plans, FACTOR_LAB_ROOT)
    print("Created branches:")
    for branch in created:
        print(f"- {branch}")


if __name__ == "__main__":
    main()
