# Autoresearch Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable Claude Code to autonomously mine factors overnight via `eval_factor.py` CLI + `program.md` instructions.

**Architecture:** `eval_factor.py` wraps existing modules (parser → compute → backtest → gates) into a single CLI. `program.md` gives Claude Code the autoresearch-style loop instructions. `research_journal.md` persists cross-session insights.

**Tech Stack:** Python 3.11+, DuckDB, pandas, scipy, argparse. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-20-autoresearch-mode-design.md`

---

### Task 1: eval_factor.py — Core CLI with IS Evaluation

The main deliverable. Wires existing modules into a grep-friendly CLI tool.

**Files:**
- Create: `eval_factor.py` (project root)
- Read: `src/agent/loop.py:51-72` (MARKET_CONFIGS), `src/agent/loop.py:79-117` (_load_prices, _compute_forward_returns)
- Read: `src/dsl/parser.py:373` (parse), `src/dsl/compute.py:276` (compute_factor)
- Read: `src/backtest/walk_forward.py:219` (walk_forward_backtest)
- Read: `src/backtest/gates.py:68` (check_gates)

- [ ] **Step 1: Create eval_factor.py with argument parsing and data loading**

```python
#!/usr/bin/env python3
"""Factor evaluation CLI — single entry point for autoresearch mode.

Usage:
    uv run python eval_factor.py --market cn --formula "rank(delta(volume, 5))"
    uv run python eval_factor.py --market cn --formula "..." --oos-check
    uv run python eval_factor.py --market cn --formula "..." --promote --name "x" --hypothesis "y" --direction long
    uv run python eval_factor.py --show-registry --market cn
    uv run python eval_factor.py --eval-composite --market cn
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pickle
import duckdb
import numpy as np
import pandas as pd

from src.dsl.parser import parse, DSLParseError
from src.dsl.compute import compute_factor
from src.backtest.walk_forward import walk_forward_backtest, run_oos_check
from src.backtest.gates import check_gates, GateResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACTOR_LAB_DB = str(_PROJECT_ROOT / "data" / "factor_lab.duckdb")
CACHE_DIR = _PROJECT_ROOT / "data" / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Market configuration — mirrors src/agent/loop.py:51-72
# ---------------------------------------------------------------------------
MARKET_CONFIGS = {
    "cn": {
        "db_path": "$QUANT_CN_ROOT/data/quant_cn.duckdb",
        "table": "prices",
        "sym_col": "ts_code",
        "date_col": "trade_date",
        "close_col": "close",
        "vol_col": "vol",
        "cost_per_trade": 0.003,
        "oos_start": "2025-10-01",
    },
    "us": {
        "db_path": "$QUANT_US_ROOT/data/quant.duckdb",
        "table": "prices_daily",
        "sym_col": "symbol",
        "date_col": "date",
        "close_col": "adj_close",
        "vol_col": "volume",
        "cost_per_trade": 0.001,
        "oos_start": "2025-10-01",
    },
}

# ---------------------------------------------------------------------------
# Data caching — avoids 10-30s DB loads on every experiment
# ---------------------------------------------------------------------------
def _cached_load(market: str, loader, name: str, max_age_hours: int = 24):
    """Cache DataFrames to pickle for fast repeated access."""
    cache_file = CACHE_DIR / f"{market}_{name}.pkl"
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < max_age_hours:
            return pickle.load(open(cache_file, "rb"))
    data = loader(market)
    pickle.dump(data, open(cache_file, "wb"))
    return data


# ---------------------------------------------------------------------------
# Data loading — adapted from src/agent/loop.py:79-117
# Uses SELECT * to include all available columns (amount, turnover_rate, etc.)
# so DSL formulas can reference any column that exists in the pipeline DB.
# ---------------------------------------------------------------------------
def _load_prices_raw(market: str) -> pd.DataFrame:
    """Load raw prices from pipeline DuckDB. Uses SELECT * for full feature access."""
    cfg = MARKET_CONFIGS[market]
    con = duckdb.connect(cfg["db_path"], read_only=True)
    sql = f"""
        SELECT *
        FROM {cfg['table']}
        WHERE {cfg['close_col']} > 0
        ORDER BY {cfg['sym_col']}, {cfg['date_col']}
    """
    try:
        df = con.execute(sql).fetchdf()
    finally:
        con.close()
    # Normalize key column names for compute_factor compatibility
    rename = {}
    if cfg["close_col"] != "close":
        rename[cfg["close_col"]] = "close"
    if cfg["vol_col"] != "volume":
        rename[cfg["vol_col"]] = "volume"
    if rename:
        df = df.rename(columns=rename)
    return df


def load_prices(market: str) -> pd.DataFrame:
    return _cached_load(market, _load_prices_raw, "prices")


def _compute_fwd_raw(market: str) -> pd.DataFrame:
    """Compute 5-day forward returns via DuckDB window function."""
    cfg = MARKET_CONFIGS[market]
    con = duckdb.connect(cfg["db_path"], read_only=True)
    sql = f"""
        SELECT {cfg['sym_col']}, {cfg['date_col']},
               LEAD({cfg['close_col']}, 5) OVER w / {cfg['close_col']} - 1 AS fwd_5d
        FROM {cfg['table']}
        WHERE {cfg['close_col']} > 0
        WINDOW w AS (PARTITION BY {cfg['sym_col']} ORDER BY {cfg['date_col']})
        ORDER BY {cfg['sym_col']}, {cfg['date_col']}
    """
    try:
        df = con.execute(sql).fetchdf()
    finally:
        con.close()
    return df


def compute_forward_returns(market: str) -> pd.DataFrame:
    return _cached_load(market, _compute_fwd_raw, "fwd")


# ---------------------------------------------------------------------------
# Promoted factor loading (for correlation gate)
# ---------------------------------------------------------------------------
def load_promoted_factor_values(market: str, prices: pd.DataFrame) -> tuple[list[pd.Series], list[str]]:
    """Load promoted factors from registry, re-compute their values, return as list of Series.

    Returns (factor_value_series_list, factor_name_list).
    """
    cfg = MARKET_CONFIGS[market]
    con = duckdb.connect(FACTOR_LAB_DB, read_only=True)
    try:
        promoted = con.execute(
            "SELECT factor_id, name, formula FROM factor_registry WHERE market=? AND status='promoted'",
            [market],
        ).fetchdf()
    except duckdb.CatalogException:
        return [], []
    finally:
        con.close()

    if promoted.empty:
        return [], []

    series_list = []
    name_list = []
    for _, row in promoted.iterrows():
        try:
            ast = parse(row["formula"])
            vals = compute_factor(ast, prices, sym_col=cfg["sym_col"], date_col=cfg["date_col"])
            series_list.append(vals["factor_value"])
            name_list.append(row["name"] or row["factor_id"][:7])
        except Exception:
            continue  # skip broken promoted factors

    return series_list, name_list


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def print_results(
    market: str,
    formula: str,
    bt,
    gate_result: GateResult,
    elapsed: float,
) -> None:
    """Print grep-friendly results to stdout."""
    print("---")
    print(f"market:           {market}")
    print(f"formula:          {formula}")
    print(f"is_ic:            {bt.avg_ic:.4f}")
    print(f"is_ic_ir:         {bt.avg_ic_ir:.3f}")
    print(f"is_sharpe:        {bt.avg_sharpe:.3f}")
    print(f"is_turnover:      {bt.avg_turnover:.4f}")
    print(f"is_monotonicity:  {bt.avg_monotonicity:.3f}")

    corr_info = gate_result.details.get("correlation", {})
    print(f"max_corr:         {corr_info.get('value', 0.0):.3f}")
    print(f"max_corr_with:    {corr_info.get('most_correlated', 'none')}")

    gate_str = "PASS" if gate_result.passed else "FAIL"
    detail_parts = []
    for gname, ginfo in gate_result.details.items():
        detail_parts.append(f"{gname}={'PASS' if ginfo['passed'] else 'FAIL'}")
    print(f"gates:            {gate_str}")
    print(f"gate_detail:      {' '.join(detail_parts)}")
    print(f"eval_seconds:     {elapsed:.1f}")


# ---------------------------------------------------------------------------
# Ensure DuckDB tables exist — reuses schema from daily_pipeline.py:69
# ---------------------------------------------------------------------------
def ensure_tables():
    """Create factor_registry table if it doesn't exist."""
    from src.mining.daily_pipeline import init_db
    init_db()


# ---------------------------------------------------------------------------
# Promote to DuckDB — adapts src/mining/daily_pipeline.py:265
# ---------------------------------------------------------------------------
def promote_factor(
    market: str,
    formula: str,
    name: str,
    hypothesis: str,
    direction: str,
    bt,
) -> str:
    """Write factor to DuckDB registry as promoted. Populates IS metrics. Returns factor_id."""
    ensure_tables()
    factor_id = hashlib.sha256(formula.encode()).hexdigest()[:16]
    con = duckdb.connect(FACTOR_LAB_DB)
    try:
        existing = con.execute(
            "SELECT factor_id FROM factor_registry WHERE factor_id=?", [factor_id]
        ).fetchone()

        # Use 5d IS metrics for the 7d columns (closest horizon we have)
        ic_val = bt.avg_ic if bt else None
        icir_val = bt.avg_ic_ir if bt else None
        mono_val = bt.avg_monotonicity if bt else None

        if existing:
            con.execute("""
                UPDATE factor_registry SET
                    name=?, hypothesis=?, direction=?,
                    status='promoted', promoted_at=CURRENT_TIMESTAMP,
                    watchlist_at=NULL, retired_at=NULL, retire_reason=NULL,
                    health_watch_count=0,
                    ic_7d=?, ic_ir_7d=?, mono_7d=?
                WHERE factor_id=?
            """, [name, hypothesis, direction, ic_val, icir_val, mono_val, factor_id])
        else:
            con.execute("""
                INSERT INTO factor_registry
                    (factor_id, market, name, hypothesis, formula, direction,
                     status, promoted_at, ic_7d, ic_ir_7d, mono_7d)
                VALUES (?, ?, ?, ?, ?, ?, 'promoted', CURRENT_TIMESTAMP, ?, ?, ?)
            """, [factor_id, market, name, hypothesis, formula, direction,
                  ic_val, icir_val, mono_val])
    finally:
        con.close()

    return factor_id


# ---------------------------------------------------------------------------
# Show registry
# ---------------------------------------------------------------------------
def show_registry(market: str) -> None:
    """Print promoted factors for a market."""
    con = duckdb.connect(FACTOR_LAB_DB, read_only=True)
    try:
        df = con.execute("""
            SELECT name, formula, status, composite_score,
                   ic_7d, ic_ir_7d, promoted_at
            FROM factor_registry
            WHERE market=?
            ORDER BY composite_score DESC NULLS LAST
        """, [market]).fetchdf()
    except duckdb.CatalogException:
        print("---")
        print("registry:         empty (no tables)")
        return
    finally:
        con.close()

    print("---")
    print(f"market:           {market}")
    print(f"promoted_count:   {len(df[df['status'] == 'promoted'])}")
    print(f"total_count:      {len(df)}")
    print()
    for _, row in df.iterrows():
        status = row["status"]
        name = row["name"] or "unnamed"
        formula = row["formula"]
        print(f"  [{status:>10s}] {name}: {formula}")


# ---------------------------------------------------------------------------
# Eval composite
# ---------------------------------------------------------------------------
def eval_composite(market: str) -> None:
    """Evaluate the IC_IR-weighted composite of all promoted factors."""
    cfg = MARKET_CONFIGS[market]
    sym_col, date_col = cfg["sym_col"], cfg["date_col"]

    prices = load_prices(market)
    fwd = compute_forward_returns(market)

    con = duckdb.connect(FACTOR_LAB_DB, read_only=True)
    try:
        promoted = con.execute(
            "SELECT factor_id, name, formula, ic_ir_7d FROM factor_registry WHERE market=? AND status='promoted'",
            [market],
        ).fetchdf()
    except duckdb.CatalogException:
        print("---")
        print("composite:        no promoted factors")
        return
    finally:
        con.close()

    if promoted.empty:
        print("---")
        print("composite:        no promoted factors")
        return

    # Compute each factor and combine
    factor_dfs = {}
    weights = {}
    for _, row in promoted.iterrows():
        try:
            ast = parse(row["formula"])
            vals = compute_factor(ast, prices, sym_col=sym_col, date_col=date_col)
            fid = row["factor_id"]
            factor_dfs[fid] = vals
            # IC_IR-weighted (fallback to 1.0 if NULL/NaN)
            ir = row["ic_ir_7d"]
            w = abs(ir) if ir is not None and not (isinstance(ir, float) and np.isnan(ir)) else 1.0
            weights[fid] = max(w, 0.01)
        except Exception:
            continue

    if not factor_dfs:
        print("---")
        print("composite:        all factor computations failed")
        return

    # Normalize weights
    total_w = sum(weights.values())
    for k in weights:
        weights[k] /= total_w

    # Merge all factor values into composite
    base = None
    for fid, fdf in factor_dfs.items():
        renamed = fdf.rename(columns={"factor_value": f"f_{fid}"})
        if base is None:
            base = renamed
        else:
            base = base.merge(renamed, on=[sym_col, date_col], how="outer")

    # Weighted sum
    composite_vals = pd.Series(0.0, index=base.index)
    for fid in factor_dfs:
        col = f"f_{fid}"
        if col in base.columns:
            composite_vals += base[col].fillna(0) * weights[fid]

    base["factor_value"] = composite_vals
    composite_df = base[[sym_col, date_col, "factor_value"]].dropna()

    # Backtest composite
    bt = walk_forward_backtest(
        composite_df, fwd,
        sym_col=sym_col, date_col=date_col,
        oos_start=cfg["oos_start"],
        cost_per_trade=cfg["cost_per_trade"],
    )

    print("---")
    print(f"composite_ic:     {bt.avg_ic:.4f}")
    print(f"composite_ic_ir:  {bt.avg_ic_ir:.3f}")
    print(f"composite_sharpe: {bt.avg_sharpe:.3f}")
    print(f"n_factors:        {len(factor_dfs)}")
    weights_str = ", ".join(f"{fid[:7]}={w:.2f}" for fid, w in weights.items())
    print(f"weights:          {weights_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Factor evaluation CLI for autoresearch mode")
    parser.add_argument("--market", choices=["cn", "us"], help="Target market")
    parser.add_argument("--formula", type=str, help="DSL formula to evaluate")
    parser.add_argument("--name", type=str, default="", help="Factor name")
    parser.add_argument("--hypothesis", type=str, default="", help="Economic hypothesis")
    parser.add_argument("--direction", choices=["long", "short"], default="long")
    parser.add_argument("--oos-check", action="store_true", help="Run OOS validation (PASS/FAIL)")
    parser.add_argument("--promote", action="store_true", help="Promote factor to registry")
    parser.add_argument("--show-registry", action="store_true", help="Show promoted factors")
    parser.add_argument("--eval-composite", action="store_true", help="Evaluate promoted factor composite")

    args = parser.parse_args()

    # Dispatch modes
    if args.show_registry:
        if not args.market:
            print("ERROR: --market required", file=sys.stderr)
            return 1
        show_registry(args.market)
        return 0

    if args.eval_composite:
        if not args.market:
            print("ERROR: --market required", file=sys.stderr)
            return 1
        eval_composite(args.market)
        return 0

    # Formula-based modes require --market and --formula
    if not args.market or not args.formula:
        print("ERROR: --market and --formula required", file=sys.stderr)
        return 1

    cfg = MARKET_CONFIGS[args.market]
    sym_col, date_col = cfg["sym_col"], cfg["date_col"]

    t0 = time.time()

    # 1. Parse DSL
    try:
        ast = parse(args.formula)
    except DSLParseError as e:
        print(f"PARSE_ERROR: {e}", file=sys.stderr)
        return 1

    # 2. Load data (cached — first call ~10s, subsequent <0.5s)
    try:
        prices = load_prices(args.market)
        fwd = compute_forward_returns(args.market)
    except Exception as e:
        print(f"DATA_ERROR: {e}", file=sys.stderr)
        return 2

    # 3. Compute factor values
    try:
        factor_values = compute_factor(ast, prices, sym_col=sym_col, date_col=date_col)
    except Exception as e:
        print(f"COMPUTE_ERROR: {e}", file=sys.stderr)
        return 2

    if factor_values.empty or factor_values["factor_value"].isna().all():
        print("COMPUTE_ERROR: factor produced all NaN values", file=sys.stderr)
        return 2

    # 4. Walk-forward backtest
    try:
        bt = walk_forward_backtest(
            factor_values, fwd,
            sym_col=sym_col, date_col=date_col,
            oos_start=cfg["oos_start"],
            cost_per_trade=cfg["cost_per_trade"],
        )
    except Exception as e:
        print(f"BACKTEST_ERROR: {e}", file=sys.stderr)
        return 3

    # 5. Check gates (with correlation against promoted factors)
    promoted_series, promoted_names = load_promoted_factor_values(args.market, prices)
    gate_result = check_gates(
        bt, args.market,
        existing_factors=promoted_series if promoted_series else None,
        candidate_values=factor_values["factor_value"] if promoted_series else None,
        candidate_dates=factor_values[date_col] if promoted_series else None,
    )

    # Update correlation gate with actual factor names
    if promoted_names and "correlation" in gate_result.details:
        corr_detail = gate_result.details["correlation"]
        mc = corr_detail.get("most_correlated", "")
        if mc and mc.startswith("factor_"):
            idx = int(mc.split("_")[1])
            if idx < len(promoted_names):
                corr_detail["most_correlated"] = promoted_names[idx]

    elapsed = time.time() - t0

    # 6. Print IS results
    print_results(args.market, args.formula, bt, gate_result, elapsed)

    # 7. OOS check
    if args.oos_check:
        if not gate_result.passed:
            print(f"oos_result:       SKIP (gates failed)")
        else:
            oos_pass = run_oos_check(
                factor_values, fwd,
                sym_col=sym_col, date_col=date_col,
                oos_start=cfg["oos_start"],
                market=args.market,
                cost_per_trade=cfg["cost_per_trade"],
            )
            print(f"oos_result:       {'PASS' if oos_pass else 'FAIL'}")

    # 8. Promote (only if gates passed — defense in depth)
    if args.promote:
        if not args.name:
            print("ERROR: --name required for --promote", file=sys.stderr)
            return 1
        if not gate_result.passed:
            print("PROMOTE_SKIP:     gates did not pass", file=sys.stderr)
            return 1
        factor_id = promote_factor(
            args.market, args.formula, args.name,
            args.hypothesis, args.direction, bt,
        )
        print(f"promoted_id:      {factor_id}")

    # 9. Append to experiments.jsonl
    log_entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "market": args.market,
        "formula": args.formula,
        "name": args.name or "",
        "is_ic": round(bt.avg_ic, 4),
        "is_ic_ir": round(bt.avg_ic_ir, 3),
        "gates": "PASS" if gate_result.passed else "FAIL",
        "status": "promoted" if args.promote and gate_result.passed else "evaluated",
        "eval_seconds": round(elapsed, 1),
    }
    log_path = Path(__file__).parent / "experiments.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke test — parse a simple formula against CN data**

Run:
```bash
cd $FACTOR_LAB_ROOT && uv run python eval_factor.py --market cn --formula "rank(delta(volume, 5))" 2>&1
```

Expected: grep-friendly output with `is_ic:`, `gates:`, etc. or a clear error message.

- [ ] **Step 3: Smoke test — parse a simple formula against US data**

Run:
```bash
cd $FACTOR_LAB_ROOT && uv run python eval_factor.py --market us --formula "rank(delta(volume, 5))" 2>&1
```

Expected: output with different metrics (US has stricter gates).

- [ ] **Step 4: Test error handling — invalid formula**

Run:
```bash
cd $FACTOR_LAB_ROOT && uv run python eval_factor.py --market cn --formula "invalid(((" 2>&1; echo "exit=$?"
```

Expected: stderr with `PARSE_ERROR:`, exit code 1.

- [ ] **Step 5: Test error handling — missing feature**

Run:
```bash
cd $FACTOR_LAB_ROOT && uv run python eval_factor.py --market cn --formula "rank(nonexistent_feature)" 2>&1; echo "exit=$?"
```

Expected: stderr with `COMPUTE_ERROR:`, exit code 2.

- [ ] **Step 6: Test OOS check mode**

Run:
```bash
cd $FACTOR_LAB_ROOT && uv run python eval_factor.py --market cn --formula "rank(delta(volume, 5))" --oos-check 2>&1
```

Expected: output includes `oos_result: PASS` or `FAIL` or `SKIP`.

- [ ] **Step 7: Test show-registry mode**

Run:
```bash
cd $FACTOR_LAB_ROOT && uv run python eval_factor.py --show-registry --market cn 2>&1
```

Expected: list of promoted factors or "empty".

- [ ] **Step 8: Test eval-composite mode**

Run:
```bash
cd $FACTOR_LAB_ROOT && uv run python eval_factor.py --eval-composite --market cn 2>&1
```

Expected: composite IC/IR metrics or "no promoted factors".

- [ ] **Step 9: Fix any issues found during smoke testing**

Debug and fix based on test results. Common issues:
- Column name mismatches between markets
- NaN handling in factor computation
- Empty DataFrames from data loading

- [ ] **Step 10: Test promote mode (creates a test factor, then verify in DB)**

Run:
```bash
cd $FACTOR_LAB_ROOT && uv run python eval_factor.py \
  --market cn \
  --formula "rank(delta(volume, 5))" \
  --name "test_volume_delta" \
  --hypothesis "test hypothesis" \
  --direction long \
  --promote 2>&1
```

Then verify:
```bash
cd $FACTOR_LAB_ROOT && python -c "
import duckdb
con = duckdb.connect('data/factor_lab.duckdb', read_only=True)
print(con.execute(\"SELECT name, formula, status FROM factor_registry WHERE name='test_volume_delta'\").fetchdf())
con.close()
"
```

- [ ] **Step 11: Commit**

```bash
cd $FACTOR_LAB_ROOT
git add eval_factor.py
git commit -m "feat: add eval_factor.py CLI for autoresearch mode"
```

---

### Task 2: program.md — Claude Code Autonomous Mining Instructions

**Files:**
- Create: `program.md` (project root)
- Read: `spec.md`, `FACTORS.md`

- [ ] **Step 1: Write program.md**

```markdown
# Factor Lab — Autonomous Research Mode

You are a senior quantitative researcher running an autonomous factor mining session.
Your tools are `eval_factor.py` for evaluation and `WebSearch` for literature.
You maintain `research_journal.md` as your evolving research log.

## Setup

1. Read `spec.md` for architecture context.
2. Read `FACTORS.md` for available DSL operators and features.
3. Read `research_journal.md` for prior session insights (if exists).
4. Check data availability:
   ```bash
   uv run python eval_factor.py --show-registry --market cn
   uv run python eval_factor.py --show-registry --market us
   ```
5. Record session start time. You have **8 hours**.
6. Initialize `experiments.jsonl` if it doesn't exist.

## The Research Loop

LOOP UNTIL 8 HOURS ELAPSED:

### Per-Experiment Flow

1. **Generate hypothesis**: Think of an economic reason why a pattern should predict returns.
2. **Write DSL formula**: Translate the hypothesis into a DSL expression.
3. **Evaluate**:
   ```bash
   uv run python eval_factor.py --market {cn|us} --formula "YOUR_FORMULA" > run.log 2>&1
   ```
4. **Read results**:
   ```bash
   grep "^is_ic:\|^is_ic_ir:\|^gates:\|^max_corr:" run.log
   ```
5. **If gates PASS** → run OOS check:
   ```bash
   uv run python eval_factor.py --market {cn|us} --formula "YOUR_FORMULA" --oos-check > oos.log 2>&1
   grep "^oos_result:" oos.log
   ```
6. **If OOS PASS** → promote:
   ```bash
   uv run python eval_factor.py --market {cn|us} --formula "YOUR_FORMULA" \
     --name "factor_name" --hypothesis "your hypothesis" --direction long --promote
   ```
7. **Log experiment** to `experiments.jsonl` (append one JSON line):
   ```json
   {"ts":"TIMESTAMP","n":N,"market":"cn","formula":"...","source":"discover","is_ic":0.032,"gates":"PASS","oos":"PASS","status":"promoted","description":"..."}
   ```
8. **If error** (non-zero exit): read stderr, fix if trivial, skip if fundamental.
9. Return to step 1.

### Research Modes (rotate based on progress)

**Discovery (first ~2 hours)**:
- Generate diverse, independent hypotheses
- Cover different feature families: volume, price, momentum, volatility, flow
- When stuck, search papers: use WebSearch for "quantitative factor [topic] alpha"
- Try academic factors: Amihud illiquidity, Kyle's lambda, lottery demand, etc.

**Evolution (hours 2-5)**:
- Look at near-miss factors (IC close to threshold but gates failed)
- Mutate: change windows (5→10→20), swap features, add conditions
- Combine: multiply two near-miss factors, use if_then for regime conditioning
- Example: if `rank(delta(volume, 5))` has IC=0.018, try:
  - `rank(delta(volume, 10))` — different window
  - `rank(delta(volume, 5)) * rank(-ret_5d)` — add reversal signal
  - `if_then(volume > ts_mean(volume, 20), rank(delta(volume, 5)), 0)` — condition

**Composite Optimization (hours 5-7)**:
- Check composite performance: `uv run python eval_factor.py --eval-composite --market cn`
- Try adding new factors and see if composite IC_IR improves
- Try if removing weak factors improves the composite

**Wrap-up (hours 7-8)**:
- Update `research_journal.md` with session summary
- List open questions for next session
- Note any data requests for the human

## Research Journal Updates

Every ~10 experiments, update `research_journal.md` with:
- What patterns you've observed (which features/operators work, which don't)
- Near-miss factors worth revisiting
- New hypotheses generated from results
- Exploration coverage (which areas are saturated, which are untouched)

## Constraints

**What you CAN do:**
- Generate any valid DSL formula (see FACTORS.md for operators and features)
- Search papers for inspiration (WebSearch)
- Write data requests in the journal for the human
- Alternate between CN and US markets

**What you CANNOT do:**
- Modify `eval_factor.py` or any source file under `src/`
- Bypass the gate system
- Install new packages
- Exceed the 8-hour time budget

**DSL Quick Reference:**
- Features: close, open, high, low, volume, amount, turnover_rate, vwap, ret_1d, ret_5d, ret_20d
- Time-series: ts_mean, ts_std, ts_max, ts_min, ts_rank, ts_corr, delta, pct_change, decay_linear
- Cross-sectional: rank, zscore, demean
- Universal: abs, sign, log, sqrt, power, clamp, if_then
- Windows: 1, 2, 3, 5, 10, 20, 40, 60, 120
- Max nesting depth: 3, max formula length: 200 chars

**NEVER STOP**: Once the loop has begun, do NOT pause to ask the human. They may be asleep.
Run experiments until the 8-hour timer expires. If you run out of ideas, search papers,
re-read the journal, try more radical combinations, or explore untouched feature families.
```

- [ ] **Step 2: Verify program.md is readable and complete**

Read through and check all commands are correct.

- [ ] **Step 3: Commit**

```bash
cd $FACTOR_LAB_ROOT
git add program.md
git commit -m "feat: add program.md for Claude Code autonomous mining"
```

---

### Task 3: research_journal.md — Initial Template

**Files:**
- Create: `research_journal.md` (project root)

- [ ] **Step 1: Write initial template**

```markdown
# Factor Research Journal

## Current Understanding

No experiments run yet. Starting from scratch.

**CN market**: ~5500 stocks, 300+ trading days. Noisier, lower IC thresholds.
**US market**: ~500 stocks, 500+ trading days. Cleaner data, stricter gates.

## Confirmed Patterns

(None yet — will be populated by research sessions)

## Open Questions

- Which feature families yield the most independent alpha?
- Do volume-based factors work better in CN (retail-driven) vs US (institutional)?
- What is the optimal factor complexity (depth 1 vs 2 vs 3)?

## Exploration Map

```
feature        | delta | pct_chg | ts_mean | ts_std | rank | zscore | combined
---------------|-------|---------|---------|--------|------|--------|--------
close          |       |         |         |        |      |        |
volume         |       |         |         |        |      |        |
amount         |       |         |         |        |      |        |
turnover_rate  |       |         |         |        |      |        |
ret_1d         |       |         |         |        |      |        |
ret_5d         |       |         |         |        |      |        |
ret_20d        |       |         |         |        |      |        |
high           |       |         |         |        |      |        |
low            |       |         |         |        |      |        |
vwap           |       |         |         |        |      |        |
```

## Data Requests

(None yet)

## Session Log

(No sessions yet)
```

- [ ] **Step 2: Commit**

```bash
cd $FACTOR_LAB_ROOT
git add research_journal.md
git commit -m "feat: add research_journal.md template for cross-session memory"
```

---

### Task 4: End-to-End Integration Test

Run the full autoresearch workflow manually to verify everything works together.

- [ ] **Step 1: Test the full experiment cycle — CN market**

```bash
cd $FACTOR_LAB_ROOT

# IS evaluation
uv run python eval_factor.py --market cn --formula "rank(delta(volume, 5) / ts_mean(volume, 20))" > run.log 2>&1
echo "=== IS Results ==="
cat run.log

# OOS check (if gates passed)
grep -q "^gates:.*PASS" run.log && \
  uv run python eval_factor.py --market cn \
    --formula "rank(delta(volume, 5) / ts_mean(volume, 20))" \
    --oos-check > oos.log 2>&1 && \
  echo "=== OOS ==="  && cat oos.log
```

- [ ] **Step 2: Test the full experiment cycle — US market**

```bash
cd $FACTOR_LAB_ROOT

uv run python eval_factor.py --market us --formula "rank(pct_change(volume, 5))" > run.log 2>&1
echo "=== IS Results ==="
cat run.log
```

- [ ] **Step 3: Test correlation gate — run two similar formulas**

```bash
cd $FACTOR_LAB_ROOT

# First: promote a factor
uv run python eval_factor.py --market cn \
  --formula "rank(delta(volume, 5))" \
  --name "vol_delta_5" --hypothesis "test" --direction long --promote 2>&1

# Second: run a similar factor — should show non-zero max_corr
uv run python eval_factor.py --market cn \
  --formula "rank(delta(volume, 10))" 2>&1 | grep "max_corr"
```

- [ ] **Step 4: Test composite evaluation**

```bash
cd $FACTOR_LAB_ROOT
uv run python eval_factor.py --eval-composite --market cn 2>&1
```

- [ ] **Step 5: Verify experiments.jsonl can be written**

```bash
cd $FACTOR_LAB_ROOT
echo '{"ts":"2026-03-20T00:00:00","n":0,"market":"cn","formula":"test","status":"smoke_test"}' > experiments.jsonl
cat experiments.jsonl
```

- [ ] **Step 6: Clean up test data if needed**

Remove test factor from registry if it shouldn't persist:
```bash
cd $FACTOR_LAB_ROOT && python -c "
import duckdb
con = duckdb.connect('data/factor_lab.duckdb')
con.execute(\"DELETE FROM factor_registry WHERE name='test_volume_delta' OR name='vol_delta_5'\")
con.close()
print('cleaned up test factors')
"
```

- [ ] **Step 7: Final commit**

```bash
cd $FACTOR_LAB_ROOT
git add experiments.jsonl research_journal.md
git commit -m "feat: autoresearch mode ready — eval_factor.py + program.md + journal"
```



> **Note:** Data caching is built into `eval_factor.py` (Task 1) via `_cached_load()` with pickle. First call takes ~10-30s, subsequent calls < 0.5s. Cache expires after 24 hours. Cache files at `data/.cache/*.pkl` should be gitignored.

---

### Task 5: generate_factor_report.py — Report Section for Pipeline Reports

Generates a markdown section summarizing overnight mining results. This section gets appended to the existing US/CN daily reports.

**Files:**
- Create: `scripts/generate_factor_report.py`
- Read: `experiments.jsonl`, `data/factor_lab.duckdb` (factor_registry), `research_journal.md`

- [ ] **Step 1: Write generate_factor_report.py**

```python
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


def load_experiments(target_date: str) -> list[dict]:
    """Load experiments from JSONL for a given date."""
    if not EXPERIMENTS_FILE.exists():
        return []
    experiments = []
    for line in EXPERIMENTS_FILE.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            ts = entry.get("ts", "")
            if ts.startswith(target_date):
                experiments.append(entry)
        except json.JSONDecodeError:
            continue
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
```

- [ ] **Step 2: Test with mock data**

```bash
cd $FACTOR_LAB_ROOT

# Generate report for today (uses whatever experiments.jsonl + registry has)
uv run python scripts/generate_factor_report.py --date 2026-03-20
```

Expected: markdown section output to stdout (or "No Factor Lab data" if no experiments today).

- [ ] **Step 3: Test append mode with a temp file**

```bash
cd $FACTOR_LAB_ROOT

echo "# Test Report" > /tmp/test_report.md
echo "Some existing content." >> /tmp/test_report.md
uv run python scripts/generate_factor_report.py --date 2026-03-20 --append-to /tmp/test_report.md
cat /tmp/test_report.md
```

Expected: original content + Factor Lab section appended.

- [ ] **Step 4: Commit**

```bash
cd $FACTOR_LAB_ROOT
git add scripts/generate_factor_report.py
git commit -m "feat: add generate_factor_report.py for pipeline report injection"
```

---

### Task 6: Inject Factor Lab Section into Pipeline Reports

Add a single call to each pipeline's report generation script to append the Factor Lab section.

**Files:**
- Modify: `$QUANT_US_ROOT/scripts/run_agents.sh` (~line 513)
- Modify: `$QUANT_CN_ROOT/scripts/run_agents.sh` (~line 420)

- [ ] **Step 1: Read the exact injection points in both files**

Read the relevant sections to find the exact line to add the call.

US: Find where `ZH_REPORT` is finalized (after `cp` to final path, before email).
CN: Find where report is finalized (after merge agent writes, before email check).

- [ ] **Step 2: Add Factor Lab section to US pipeline**

In `$QUANT_US_ROOT/scripts/run_agents.sh`, after the report is finalized (the line that copies/creates `$ZH_REPORT`), add:

```bash
# Append Factor Lab experiment report section
echo "  Appending Factor Lab section..."
cd $FACTOR_LAB_ROOT && \
  uv run python scripts/generate_factor_report.py \
    --date "$DATE" \
    --append-to "$ZH_REPORT" 2>/dev/null || true
cd "$SCRIPT_DIR/.."
```

The `|| true` ensures the pipeline continues even if factor-lab has no data or errors.

- [ ] **Step 3: Add Factor Lab section to CN pipeline**

In `$QUANT_CN_ROOT/scripts/run_agents.sh`, after the report is finalized, add the same call:

```bash
# Append Factor Lab experiment report section
echo "  Appending Factor Lab section..."
cd $FACTOR_LAB_ROOT && \
  uv run python scripts/generate_factor_report.py \
    --date "$DATE" \
    --append-to "$REPORT_FILE" 2>/dev/null || true
cd "$SCRIPT_DIR/.."
```

- [ ] **Step 4: Test US injection with a dry run**

```bash
# Create a mock report to test injection
DATE=$(date +%Y-%m-%d)
MOCK_REPORT="/tmp/test_us_report.md"
echo "# US Daily Report $DATE" > "$MOCK_REPORT"
echo "## Market Summary" >> "$MOCK_REPORT"
echo "Test content..." >> "$MOCK_REPORT"

cd $FACTOR_LAB_ROOT && \
  uv run python scripts/generate_factor_report.py \
    --date "$DATE" \
    --append-to "$MOCK_REPORT" 2>&1

cat "$MOCK_REPORT"
```

- [ ] **Step 5: Commit changes to both pipelines**

```bash
cd $QUANT_US_ROOT
git add scripts/run_agents.sh
git commit -m "feat: append Factor Lab section to daily reports"

cd $QUANT_CN_ROOT
git add scripts/run_agents.sh
git commit -m "feat: append Factor Lab section to daily reports"
```

- [ ] **Step 6: Add .gitignore entries for cache**

```bash
cd $FACTOR_LAB_ROOT
echo "data/.cache/" >> .gitignore
echo "*.pkl" >> .gitignore
git add .gitignore
git commit -m "chore: gitignore pickle cache files"
```
