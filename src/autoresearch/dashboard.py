from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from src.paths import FACTOR_LAB_ROOT


EXPORT_ROOT = FACTOR_LAB_ROOT / "reports" / "autoresearch_exports"


def load_session_runs(log_path: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not log_path.exists():
        return runs
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            run = json.loads(line)
            if not run.get("decision"):
                if run.get("oos") == "PASS":
                    run["decision"] = "keep"
                elif run.get("gates") == "PASS":
                    run["decision"] = "candidate"
                else:
                    run["decision"] = "revert"
            if not run.get("checks_status"):
                run["checks_status"] = "passed" if run.get("decision") == "keep" else "skipped"
            if not run.get("status"):
                run["status"] = (
                    "kept" if run.get("decision") == "keep"
                    else "candidate" if run.get("decision") == "candidate"
                    else "reverted"
                )
            runs.append(run)
        except json.JSONDecodeError:
            continue
    return runs


def _median_abs_deviation(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    med = median(values)
    deviations = [abs(v - med) for v in values]
    return float(median(deviations))


def build_dashboard_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(runs)
    keep_count = sum(1 for run in runs if run.get("decision") == "keep")
    revert_count = sum(1 for run in runs if run.get("decision") == "revert")
    candidate_count = sum(1 for run in runs if run.get("decision") == "candidate")
    gates_pass = sum(1 for run in runs if run.get("gates") == "PASS")
    oos_pass = sum(1 for run in runs if run.get("oos") == "PASS")
    metrics = [float(run.get("is_ic_ir", 0.0)) for run in runs if run.get("is_ic_ir") is not None]
    best_run = max(runs, key=lambda run: float(run.get("is_ic_ir", float("-inf"))), default=None)
    mad = _median_abs_deviation(metrics)
    confidence = None
    if best_run is not None and mad > 0:
        confidence = abs(float(best_run.get("is_ic_ir", 0.0)) - median(metrics)) / mad
    return {
        "total_runs": total,
        "keep_count": keep_count,
        "revert_count": revert_count,
        "candidate_count": candidate_count,
        "gates_pass": gates_pass,
        "oos_pass": oos_pass,
        "best_run": best_run,
        "confidence_score": round(confidence, 2) if confidence is not None and math.isfinite(confidence) else None,
        "session_branch": runs[-1].get("session_branch", "") if runs else "",
        "base_ref": runs[-1].get("base_ref", "") if runs else "",
        "merge_base": runs[-1].get("merge_base", "") if runs else "",
    }


def render_dashboard_markdown(summary: dict[str, Any], runs: list[dict[str, Any]], market: str) -> str:
    lines = [
        f"# Autoresearch Dashboard — {market.upper()}",
        "",
        f"- Total runs: {summary['total_runs']}",
        f"- Kept: {summary['keep_count']}",
        f"- Reverted: {summary['revert_count']}",
        f"- Pending candidates: {summary['candidate_count']}",
        f"- Gates pass: {summary['gates_pass']}",
        f"- OOS pass: {summary['oos_pass']}",
    ]
    if summary.get("session_branch"):
        lines.append(f"- Branch: {summary['session_branch']}")
    if summary.get("base_ref"):
        lines.append(f"- Base ref: {summary['base_ref']}")
    if summary.get("confidence_score") is not None:
        lines.append(f"- Confidence score: {summary['confidence_score']:.2f}x")
    lines.extend(
        [
            "",
            "| # | Name | IC | IC_IR | Gates | OOS | Decision | Checks |",
            "|---|------|----|-------|-------|-----|----------|--------|",
        ]
    )
    for idx, run in enumerate(runs, 1):
        lines.append(
            f"| {idx} | {run.get('name', '')} | {run.get('is_ic', '')} | {run.get('is_ic_ir', '')} | "
            f"{run.get('gates', '')} | {run.get('oos', '')} | {run.get('decision', '')} | {run.get('checks_status', '')} |"
        )
    return "\n".join(lines) + "\n"


def render_dashboard_html(summary: dict[str, Any], runs: list[dict[str, Any]], market: str) -> str:
    def esc(value: Any) -> str:
        return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    rows = []
    for run in runs:
        decision = esc(run.get("decision", ""))
        rows.append(
            "<tr>"
            f"<td>{esc(run.get('name', ''))}</td>"
            f"<td>{esc(run.get('is_ic', ''))}</td>"
            f"<td>{esc(run.get('is_ic_ir', ''))}</td>"
            f"<td>{esc(run.get('gates', ''))}</td>"
            f"<td>{esc(run.get('oos', ''))}</td>"
            f"<td class='decision {decision}'>{decision}</td>"
            f"<td>{esc(run.get('checks_status', ''))}</td>"
            f"<td><code>{esc(run.get('formula', ''))}</code></td>"
            "</tr>"
        )

    confidence = summary.get("confidence_score")
    confidence_text = f"{confidence:.2f}x" if confidence is not None else "n/a"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Autoresearch Dashboard — {market.upper()}</title>
  <style>
    body {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin: 24px; background: #f6f3ee; color: #1f2328; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }}
    .card {{ background: #fffdf8; border: 1px solid #d0c7b8; padding: 12px; border-radius: 10px; }}
    .label {{ font-size: 12px; color: #6b5f52; text-transform: uppercase; }}
    .value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; }}
    th, td {{ border: 1px solid #ded6ca; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #efe7da; }}
    .decision.keep {{ color: #0a6c2f; font-weight: 700; }}
    .decision.revert {{ color: #b42318; font-weight: 700; }}
    .decision.candidate {{ color: #935f00; font-weight: 700; }}
    code {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>Autoresearch Dashboard — {market.upper()}</h1>
  <div class="cards">
    <div class="card"><div class="label">Runs</div><div class="value">{summary['total_runs']}</div></div>
    <div class="card"><div class="label">Kept</div><div class="value">{summary['keep_count']}</div></div>
    <div class="card"><div class="label">Reverted</div><div class="value">{summary['revert_count']}</div></div>
    <div class="card"><div class="label">Candidates</div><div class="value">{summary['candidate_count']}</div></div>
    <div class="card"><div class="label">OOS Pass</div><div class="value">{summary['oos_pass']}</div></div>
    <div class="card"><div class="label">Confidence</div><div class="value">{confidence_text}</div></div>
  </div>
  <p>Branch: <code>{esc(summary.get('session_branch', ''))}</code> | Base: <code>{esc(summary.get('base_ref', ''))}</code></p>
  <table>
    <thead>
      <tr><th>Name</th><th>IC</th><th>IC_IR</th><th>Gates</th><th>OOS</th><th>Decision</th><th>Checks</th><th>Formula</th></tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""


def export_dashboard_bundle(
    *,
    market: str,
    log_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    runs = load_session_runs(log_path)
    summary = build_dashboard_summary(runs)
    target_dir = (output_dir or (EXPORT_ROOT / market)).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = target_dir / "summary.md"
    json_path = target_dir / "dashboard.json"
    html_path = target_dir / "dashboard.html"

    markdown_path.write_text(render_dashboard_markdown(summary, runs, market), encoding="utf-8")
    json_path.write_text(
        json.dumps({"summary": summary, "runs": runs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    html_path.write_text(render_dashboard_html(summary, runs, market), encoding="utf-8")

    return {"markdown": markdown_path, "json": json_path, "html": html_path}
