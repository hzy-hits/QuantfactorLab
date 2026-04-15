from __future__ import annotations

import os
from pathlib import Path


FACTOR_LAB_ROOT = Path(
    os.environ.get("FACTOR_LAB_ROOT", Path(__file__).resolve().parents[1])
).expanduser().resolve()


def _pick_repo_root(
    env_var: str,
    sibling_name: str,
    legacy_path: str,
    *,
    legacy_group: str,
) -> Path:
    candidates: list[Path] = []

    direct = os.environ.get(env_var)
    if direct:
        candidates.append(Path(direct).expanduser())

    stack_root = os.environ.get("QUANT_STACK_ROOT")
    if stack_root:
        candidates.append(Path(stack_root).expanduser() / sibling_name)

    candidates.append(FACTOR_LAB_ROOT.parent / sibling_name)
    candidates.append(FACTOR_LAB_ROOT.parents[1] / legacy_group / sibling_name)
    candidates.append(Path(legacy_path).expanduser())

    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)

    for candidate in ordered:
        if candidate.exists():
            return candidate
    return ordered[0]


QUANT_CN_ROOT = _pick_repo_root(
    "QUANT_CN_ROOT",
    "quant-research-cn",
    "~/coding/rust/quant-research-cn",
    legacy_group="rust",
)

QUANT_US_ROOT = _pick_repo_root(
    "QUANT_US_ROOT",
    "quant-research-v1",
    "~/coding/python/quant-research-v1",
    legacy_group="python",
)

FACTOR_LAB_DB = FACTOR_LAB_ROOT / "data" / "factor_lab.duckdb"
QUANT_CN_DB = QUANT_CN_ROOT / "data" / "quant_cn.duckdb"
QUANT_CN_REPORT_DB = QUANT_CN_ROOT / "data" / "quant_cn_report.duckdb"
QUANT_US_DB = QUANT_US_ROOT / "data" / "quant.duckdb"
QUANT_US_REPORT_DB = QUANT_US_ROOT / "data" / "quant_report.duckdb"
