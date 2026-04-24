#!/usr/bin/env python3
"""Generate Factor Lab report section for pipeline daily reports.

Reads experiments.jsonl + DuckDB registry + research_journal.md to produce
a markdown section that can be appended to the existing pipeline report.

Usage:
    # Print markdown to stdout
    python scripts/generate_factor_report.py --date 2026-03-20

    # Append to existing report file
    python scripts/generate_factor_report.py --date 2026-03-20 --append-to /path/to/report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, date as date_type
from pathlib import Path

import duckdb

FACTOR_LAB_DIR = Path(__file__).resolve().parent.parent
FACTOR_LAB_DB = FACTOR_LAB_DIR / "data" / "factor_lab.duckdb"
EXPERIMENTS_FILE = FACTOR_LAB_DIR / "experiments.jsonl"
JOURNAL_FILE = FACTOR_LAB_DIR / "research_journal.md"
REPORTS_DIR = FACTOR_LAB_DIR / "reports"
RUNTIME_AUTORESEARCH_DIR = FACTOR_LAB_DIR / "runtime" / "autoresearch"


def load_experiments(target_date: str) -> list[dict]:
    """Load experiments from JSONL for a given date."""
    experiments = []
    if EXPERIMENTS_FILE.exists():
        for line in EXPERIMENTS_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                ts = entry.get("ts", "")
                if ts.startswith(target_date):
                    experiments.append(entry)
            except json.JSONDecodeError:
                continue

    experiments.extend(_load_experiments_from_runtime_logs(target_date))

    fallback = _load_experiments_from_reports(target_date)
    if fallback and len(fallback) > len(experiments):
        return fallback
    return experiments


def _load_experiments_from_runtime_logs(target_date: str) -> list[dict]:
    experiments: list[dict] = []
    if not RUNTIME_AUTORESEARCH_DIR.exists():
        return experiments

    for log_path in sorted(RUNTIME_AUTORESEARCH_DIR.glob("*/autoresearch.jsonl")):
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = entry.get("ts", "")
            if ts.startswith(target_date):
                experiments.append(entry)
    return experiments


def _parse_markdown_table(lines: list[str], header: str) -> list[list[str]]:
    for idx, line in enumerate(lines):
        if line.strip() != header:
            continue
        rows = []
        for row in lines[idx + 2:]:
            if not row.startswith("|"):
                break
            cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
            rows.append(cells)
        return rows
    return []


def _load_experiments_from_reports(target_date: str) -> list[dict]:
    target_token = target_date.replace("-", "")
    experiments = []

    for path in sorted(REPORTS_DIR.glob(f"autoresearch_*_{target_token}*.md")):
        market = "cn" if "_cn_" in path.name else "us" if "_us_" in path.name else ""
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        oos_rows = _parse_markdown_table(lines, "| Name | Formula | IS IC | OOS |")
        oos_by_name = {
            row[0]: row[3]
            for row in oos_rows
            if len(row) >= 4
        }

        exp_rows = _parse_markdown_table(
            lines,
            "| # | Name | Formula | IC | IC_IR | Sharpe | Gates |",
        )
        for row in exp_rows:
            if len(row) < 7:
                continue
            name = row[1]
            experiments.append({
                "ts": f"{target_date}T00:00:00+08:00",
                "market": market,
                "name": name,
                "formula": row[2].strip("`"),
                "gates": row[6],
                "oos": oos_by_name.get(name),
                "status": "evaluated",
            })

    return experiments


def load_promoted(target_date: str) -> list[dict]:
    """Load recently promoted factors from registry."""
    if not FACTOR_LAB_DB.exists():
        return []
    con = duckdb.connect(str(FACTOR_LAB_DB), read_only=True)
    try:
        df = con.execute("""
            SELECT name, formula, market, direction,
                   ic_7d, ic_ir_7d, hypothesis, promoted_at
            FROM factor_registry
            WHERE status = 'promoted'
              AND CAST(promoted_at AS DATE) = ?
            ORDER BY ic_ir_7d DESC NULLS LAST
        """, [target_date]).fetchdf()
        return df.to_dict("records")
    except Exception:
        return []
    finally:
        con.close()


def load_composite_stats() -> dict[str, dict]:
    """Load current composite stats per market."""
    if not FACTOR_LAB_DB.exists():
        return {}
    con = duckdb.connect(str(FACTOR_LAB_DB), read_only=True)
    stats = {}
    try:
        for market in ("cn", "us"):
            row = con.execute("""
                SELECT COUNT(*) AS n,
                       AVG(ic_ir_7d) AS avg_ir
                FROM factor_registry
                WHERE market = ? AND status = 'promoted'
            """, [market]).fetchone()
            if row and row[0] > 0:
                stats[market] = {"n": row[0], "avg_ir": round(row[1] or 0, 2)}
    except Exception:
        pass
    finally:
        con.close()
    return stats


def extract_journal_highlights() -> str:
    """Extract the latest session log entry from research_journal.md."""
    if not JOURNAL_FILE.exists():
        return ""
    text = JOURNAL_FILE.read_text()
    # Find the last "### Session" block
    lines = text.split("\n")
    session_start = -1
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("### Session"):
            session_start = i
            break
    if session_start < 0:
        return ""
    # Take up to 10 lines from the session entry
    session_lines = lines[session_start : session_start + 10]
    return "\n".join(session_lines)


def generate_report(target_date: str) -> str:
    """Generate the Factor Lab markdown section."""
    experiments = load_experiments(target_date)
    promoted = load_promoted(target_date)
    composites = load_composite_stats()
    journal = extract_journal_highlights()

    total = len(experiments)
    gates_pass = sum(1 for e in experiments if e.get("gates") == "PASS")
    oos_pass = sum(1 for e in experiments if e.get("oos") == "PASS")
    new_promoted = len(promoted)

    if total == 0 and new_promoted == 0:
        return ""  # No session ran, skip section

    lines = [
        "",
        "---",
        "",
        "## Factor Lab 因子实验报告",
        "",
        f"**Session**: {target_date} | "
        f"**实验数**: {total} | "
        f"**IS Gates 通过**: {gates_pass} | "
        f"**OOS 通过**: {oos_pass} | "
        f"**新 Promoted**: {new_promoted}",
        "",
    ]

    # New promoted factors table
    if promoted:
        lines.append("### 新发现因子")
        lines.append("")
        lines.append("| 名称 | 公式 | 市场 | IC | IC_IR | 假设 |")
        lines.append("|------|------|------|-----|-------|------|")
        for f in promoted:
            ic = f"{ f['ic_7d']:.3f}" if f.get("ic_7d") else "N/A"
            ir = f"{ f['ic_ir_7d']:.2f}" if f.get("ic_ir_7d") else "N/A"
            market = (f.get("market") or "").upper()
            name = f.get("name") or "unnamed"
            formula = f"`{f.get('formula', '')}`"
            hyp = f.get("hypothesis") or ""
            # Truncate long hypothesis
            if len(hyp) > 40:
                hyp = hyp[:37] + "..."
            lines.append(f"| {name} | {formula} | {market} | {ic} | {ir} | {hyp} |")
        lines.append("")

    # Composite stats
    if composites:
        lines.append("### Composite 状态")
        lines.append("")
        parts = []
        for mkt, stats in composites.items():
            parts.append(f"**{mkt.upper()}**: IC_IR={stats['avg_ir']:.2f} ({stats['n']} factors)")
        lines.append(" | ".join(parts))
        lines.append("")

    # Journal highlights
    if journal:
        lines.append("### 研究笔记")
        lines.append("")
        for jline in journal.split("\n"):
            if jline.strip():
                lines.append(f"> {jline}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate Factor Lab report section")
    parser.add_argument("--date", default=str(date_type.today()), help="Target date YYYY-MM-DD")
    parser.add_argument("--append-to", type=str, help="Append to this report file")
    args = parser.parse_args()

    section = generate_report(args.date)
    if not section:
        print("No Factor Lab data for this date, skipping.", file=sys.stderr)
        return

    if args.append_to:
        report_path = Path(args.append_to)
        if report_path.exists():
            content = report_path.read_text()
            content += section
            report_path.write_text(content)
            print(f"Appended Factor Lab section to {report_path}", file=sys.stderr)
        else:
            print(f"Report file not found: {report_path}", file=sys.stderr)
            sys.exit(1)
    else:
        print(section)


if __name__ == "__main__":
    main()
