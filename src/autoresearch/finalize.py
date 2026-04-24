from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.autoresearch.dashboard import load_session_runs


@dataclass(frozen=True)
class FinalizeBranchPlan:
    branch_name: str
    experiment_name: str
    experiment_slug: str
    merge_base: str
    output_rel_dir: Path
    record: dict[str, Any]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "experiment"


def build_finalize_plan(log_path: Path, market: str) -> list[FinalizeBranchPlan]:
    runs = load_session_runs(log_path)
    latest_branch = next((run.get("session_branch") for run in reversed(runs) if run.get("session_branch")), "")
    latest_merge_base = next((run.get("merge_base") for run in reversed(runs) if run.get("merge_base")), "")
    kept = [run for run in runs if run.get("decision") == "keep"]
    plans: list[FinalizeBranchPlan] = []
    for run in kept:
        if latest_branch and not run.get("session_branch"):
            run["session_branch"] = latest_branch
        if latest_merge_base and not run.get("merge_base"):
            run["merge_base"] = latest_merge_base
        session_id = str(run.get("session_id") or "session")
        slug = _slugify(run.get("name") or run.get("formula") or "experiment")
        branch_name = f"autoresearch/{market}/{session_id}/{slug}"
        output_rel_dir = Path("reports") / "autoresearch_exports" / "finalized" / session_id / slug
        plans.append(
            FinalizeBranchPlan(
                branch_name=branch_name,
                experiment_name=run.get("name") or slug,
                experiment_slug=slug,
                merge_base=run.get("merge_base") or "HEAD",
                output_rel_dir=output_rel_dir,
                record=run,
            )
        )
    return plans


def _manifest_markdown(plan: FinalizeBranchPlan) -> str:
    run = plan.record
    return "\n".join(
        [
            f"# Finalized Factor Proposal — {plan.experiment_name}",
            "",
            f"- Branch: `{plan.branch_name}`",
            f"- Session branch: `{run.get('session_branch', '')}`",
            f"- Merge base: `{plan.merge_base}`",
            f"- Market: `{run.get('market', '')}`",
            f"- Formula: `{run.get('formula', '')}`",
            f"- IC: `{run.get('is_ic', '')}`",
            f"- IC_IR: `{run.get('is_ic_ir', '')}`",
            f"- Gates: `{run.get('gates', '')}`",
            f"- OOS: `{run.get('oos', '')}`",
            f"- Checks: `{run.get('checks_status', '')}`",
            f"- Decision: `{run.get('decision', '')}`",
            "",
            "## Rationale",
            "",
            "This experiment was marked `keep` by the autoresearch session and exported as a reviewable branch artifact.",
            "",
        ]
    ) + "\n"


def _run_git(args: list[str], repo_root: Path) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def apply_finalize_plan(plans: list[FinalizeBranchPlan], repo_root: Path) -> list[str]:
    created: list[str] = []
    for plan in plans:
        with tempfile.TemporaryDirectory(prefix="factor_lab_finalize_") as tmpdir:
            worktree = Path(tmpdir)
            _run_git(["worktree", "add", "-B", plan.branch_name, str(worktree), plan.merge_base], repo_root)
            try:
                out_dir = worktree / plan.output_rel_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                md_path = out_dir / "proposal.md"
                json_path = out_dir / "proposal.json"
                md_path.write_text(_manifest_markdown(plan), encoding="utf-8")
                json_path.write_text(json.dumps(plan.record, ensure_ascii=False, indent=2), encoding="utf-8")
                _run_git(["add", str(plan.output_rel_dir)], worktree)
                message = (
                    f"factor-lab finalize: keep {plan.experiment_name} "
                    f"(IC_IR={plan.record.get('is_ic_ir', '')}, OOS={plan.record.get('oos', '')})"
                )
                _run_git(["commit", "-m", message], worktree)
                created.append(plan.branch_name)
            finally:
                _run_git(["worktree", "remove", "--force", str(worktree)], repo_root)
                worktree_git = repo_root / ".git" / "worktrees" / worktree.name
                if worktree_git.exists():
                    shutil.rmtree(worktree_git, ignore_errors=True)
    return created
