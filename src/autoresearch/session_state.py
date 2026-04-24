from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path

from src.paths import FACTOR_LAB_ROOT


AUTORESEARCH_ROOT = FACTOR_LAB_ROOT / "runtime" / "autoresearch"


@dataclass(frozen=True)
class SessionPaths:
    market: str
    session_dir: Path
    session_doc: Path
    benchmark_script: Path
    checks_script: Path
    log_file: Path
    journal_file: Path


def _market_label(market: str) -> str:
    return "A-Share (CN)" if market == "cn" else "US Equity"


def session_paths(market: str, root: Path | None = None) -> SessionPaths:
    base = (root or AUTORESEARCH_ROOT).resolve()
    session_dir = base / market
    return SessionPaths(
        market=market,
        session_dir=session_dir,
        session_doc=session_dir / "autoresearch.md",
        benchmark_script=session_dir / "autoresearch.sh",
        checks_script=session_dir / "autoresearch.checks.sh",
        log_file=session_dir / "autoresearch.jsonl",
        journal_file=FACTOR_LAB_ROOT / "research_journal.md",
    )


def _write_if_missing(path: Path, content: str) -> None:
    if path.exists():
        return
    path.write_text(content, encoding="utf-8")


def _mark_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _session_doc_template(market: str, goal: str | None) -> str:
    label = _market_label(market)
    objective = goal or f"Discover and validate new {label} alpha factors with durable OOS edge."
    return f"""# Factor Lab Autoresearch Session — {label}

## Objective
{objective}

## Optimization Target
- Promote factors that pass IS gates and OOS validation.
- Prefer factors that improve composite quality without exceeding correlation limits.
- Treat `eval_factor.py` as the canonical evaluator. Do not bypass its gates.

## Files In Scope
- [eval_factor.py]({FACTOR_LAB_ROOT / "eval_factor.py"})
- [src/agent/loop.py]({FACTOR_LAB_ROOT / "src/agent/loop.py"})
- [research_journal.md]({FACTOR_LAB_ROOT / "research_journal.md"})
- [experiments.jsonl]({FACTOR_LAB_ROOT / "experiments.jsonl"})

## Benchmark Harness
- Run [autoresearch.sh](./autoresearch.sh) with `FORMULA='rank(...)'`.
- The script emits `METRIC is_ic=...`, `METRIC is_ic_ir=...`, `METRIC gates_pass=...`.

## Checks Harness
- Run [autoresearch.checks.sh](./autoresearch.checks.sh) after promising results.
- Checks must stay green before anything is considered keep-worthy.

## Notes
- This file is the resumable session context for autoresearch mode.
- Update assumptions, promising factor families, and dead ends here.
"""


def _benchmark_script_template(market: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

ROOT="{FACTOR_LAB_ROOT}"
cd "$ROOT"

MARKET="{market}"
FORMULA="${{FORMULA:?set FORMULA='rank(...)'}}"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

if ! uv run python eval_factor.py --market "$MARKET" --formula "$FORMULA" >"$TMP" 2>&1; then
  cat "$TMP"
  exit 1
fi

python - "$TMP" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(errors="ignore")

def extract(name: str):
    m = re.search(rf"^{{name}}:\\s+(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""

is_ic = extract("is_ic")
is_ic_ir = extract("is_ic_ir")
gates = extract("gates")
max_corr = extract("max_corr")
print(f"METRIC is_ic={{is_ic or 'nan'}}")
print(f"METRIC is_ic_ir={{is_ic_ir or 'nan'}}")
print(f"METRIC max_corr={{max_corr or 'nan'}}")
print(f"METRIC gates_pass={{1 if gates == 'PASS' else 0}}")
PY

cat "$TMP"
"""


def _checks_script_template(market: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

ROOT="{FACTOR_LAB_ROOT}"
cd "$ROOT"

MARKET="{market}"
uv run python eval_factor.py --show-registry --market "$MARKET" >/dev/null
uv run python eval_factor.py --eval-composite --market "$MARKET" >/dev/null
"""


def ensure_session_files(
    market: str,
    *,
    goal: str | None = None,
    root: Path | None = None,
) -> SessionPaths:
    if market not in {"cn", "us"}:
        raise ValueError("market must be 'cn' or 'us'")

    paths = session_paths(market, root=root)
    paths.session_dir.mkdir(parents=True, exist_ok=True)

    _write_if_missing(paths.session_doc, _session_doc_template(market, goal))
    _write_if_missing(paths.benchmark_script, _benchmark_script_template(market))
    _write_if_missing(paths.checks_script, _checks_script_template(market))
    paths.log_file.touch(exist_ok=True)

    _mark_executable(paths.benchmark_script)
    _mark_executable(paths.checks_script)

    return paths


def load_session_context(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()
