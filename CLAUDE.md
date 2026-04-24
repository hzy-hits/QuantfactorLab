# Factor Lab — Development Guide

## What This Project Does

AI-assisted quantitative factor mining. Agents propose factor hypotheses → system backtests with strict anti-overfit controls → validated factors auto-promote into the US/CN quant pipelines.

**Read `spec.md` first for full architecture.**

## Project Locations

```
This project:     $FACTOR_LAB_ROOT   (auto-detected from src/paths.py)
US pipeline:      $QUANT_US_ROOT     (sibling repo quant-research-v1)
CN pipeline:      $QUANT_CN_ROOT     (sibling repo quant-research-cn)
US data (read):   $QUANT_US_ROOT/data/quant.duckdb
CN data (read):   $QUANT_CN_ROOT/data/quant_cn.duckdb
Own data:         $FACTOR_LAB_ROOT/data/factor_lab.duckdb
```

## Core Philosophy (NON-NEGOTIABLE)

1. **Agent proposes, system validates.** Agent cannot bypass OOS gates or inflate metrics.
2. **Every factor must have economic logic.** No formula-only submissions — hypothesis required.
3. **OOS results are PASS/FAIL only.** Agent never sees OOS metric values. No reverse-engineering.
4. **Budget is hard-capped.** 50 experiments per session. No exceptions.
5. **Factors must be independent.** Correlation > 0.7 with existing factor → auto-reject.
6. **GPU is optional.** All code must work on CPU. cuDF/cuML are accelerators, not dependencies.

## Tech Stack

- **Python 3.11+** via uv
- **DuckDB** for all data storage (consistent with pipelines)
- **pandas / polars** for data manipulation (cuDF optional GPU backend)
- **numpy / scipy** for statistics (cuPy optional)
- **xgboost** for non-linear factor combination (GPU mode optional)
- **Claude API** for agent loop (via `claude -p` or anthropic SDK)

## Key Design Decisions

### DSL, Not Arbitrary Code
Agent writes factor formulas in a constrained DSL (see spec.md §3.1). This prevents:
- Overfitting via complex if-else trees
- Security risks from arbitrary code execution
- Unreproducible factors

### Walk-Forward, Not Full-Sample
All backtests use expanding-window walk-forward (spec.md §3.2). Never show full-sample results. IS/OOS split is enforced at the system level.

### IC_IR-Weighted Combination
Multiple promoted factors are combined via IC_IR weighting (not equal weight, not optimized). This is the sweet spot between naive and overfit.

### One Vote in Convergence
All promoted factors combine into ONE `lab_composite` signal source. Not N separate votes. This prevents Factor Lab from overwhelming the existing pipeline signals.

## Pipeline Integration

Promoted factors inject into existing pipelines via:

```sql
-- CN: analytics table
INSERT INTO analytics (ts_code, as_of, module, metric, value, detail)
VALUES (?, ?, 'lab_factor', 'lab_composite', ?, ?);

-- US: analysis_daily table
INSERT INTO analysis_daily (symbol, date, module_name, ...)
VALUES (?, ?, 'lab_factor', ...);
```

Then `notable.rs` / `classify.py` treat `lab_factor` as one independent convergence source.

## Database Schema

See spec.md §3.5 for full schema. Key tables:
- `factor_registry` — all discovered factors + status
- `factor_daily` — daily factor values per symbol
- `experiment_log` — audit trail of all agent experiments

## Common Commands

```bash
# Run one agent session (50 experiments)
./scripts/run_session.sh

# Compute promoted factors for today
uv run python scripts/daily_compute.py

# Check factor health / auto-retire
uv run python scripts/retire_check.py

# Run tests
uv run pytest tests/
```

## Anti-Overfit Rules for Development

When writing code for this project:
- NEVER expose OOS metric values to the agent prompt
- NEVER allow the agent to modify backtest parameters
- NEVER skip the correlation dedup check
- NEVER increase the experiment budget above 50 without explicit approval
- Walk-forward folds must have minimum 6 months test period
- Bootstrap stability tests must use minimum 1000 resamples

## Known Constraints

- US pipeline has ~500 trading days of data (since ~2024). Factor strategies needing longer lookback may not have enough IS data.
- CN pipeline has ~300 trading days. Even more constrained.
- Finnhub free tier limits fundamental data freshness.
- CBOE options data only available for US (CN has limited 300ETF options only).
- No intraday data — all factors are daily frequency.

## File Naming Conventions

- Source files: `snake_case.py`
- Test files: `test_<module>.py`
- Factor names: `snake_case` (e.g., `volume_price_divergence`)
- Factor IDs: SHA256 hash of formula string (deterministic dedup)
