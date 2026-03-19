# Factor Lab — Autoresearch Mode Design

## Overview

Transform factor-lab from a 50-experiment capped agent loop into an autonomous overnight research system, inspired by autoresearch's "Claude Code IS the agent" pattern. Claude Code acts as a senior quant researcher: generates hypotheses with economic logic, evaluates via CLI tools, maintains a research journal across sessions, searches academic literature, and runs for 8 hours unattended.

**Budget note**: The CLAUDE.md 50-experiment cap applies to the programmatic agent loop (`src/agent/loop.py`). Autoresearch mode replaces that loop entirely — Claude Code is the agent, `eval_factor.py` is the tool. The budget is time-based (8 hours), not experiment-count-based.

**OOS trust model**: Claude Code has filesystem access and could read OOS metric values from source code. We accept this: the real guard against overfitting is the requirement for economic logic + correlation dedup + walk-forward methodology, not information hiding. The original PASS/FAIL design was for a stateless API agent.

## Phase 1 Scope (This Implementation)

### New Files

| File | Purpose |
|------|---------|
| `eval_factor.py` | CLI tool: DSL formula → backtest → gates → OOS → promote |
| `program.md` | Claude Code autonomous mining instructions |
| `research_journal.md` | LLM-maintained cross-session research log |
| `experiments.jsonl` | Machine-readable experiment log (also writes to DuckDB `experiment_log`) |

### Bug Fixes

| Fix | Location | Detail |
|-----|----------|--------|
| ~~IS/OOS boundary leak~~ | ~~`src/backtest/walk_forward.py`~~ | Already fixed (lines 263-274 exclude last 5 trading days) |
| Correlation gate bypass | `src/backtest/gates.py` + `eval_factor.py` | Load promoted factors, compute correlation, pass to `check_gates()` |

---

## eval_factor.py — CLI Tool Design

Single entry point wrapping existing modules: DSL parser → compute engine → walk-forward backtest → gates → optional OOS → optional promote.

### Modes

```bash
# Mode 1: IS evaluation (default)
uv run python eval_factor.py \
  --market cn \
  --formula "rank(delta(volume, 5) / ts_mean(volume, 20))" \
  --name "volume_surge" \
  --hypothesis "Volume surge signals institutional activity" \
  --direction long

# Mode 2: OOS check (only after IS gates PASS)
uv run python eval_factor.py \
  --market cn \
  --formula "rank(delta(volume, 5) / ts_mean(volume, 20))" \
  --oos-check

# Mode 3: Promote (only after OOS PASS)
uv run python eval_factor.py \
  --market cn \
  --formula "rank(delta(volume, 5) / ts_mean(volume, 20))" \
  --name "volume_surge" \
  --hypothesis "Volume surge signals institutional activity" \
  --direction long \
  --promote

# Mode 4: Show registry (for context)
uv run python eval_factor.py --show-registry --market cn

# Mode 5: Evaluate composite
uv run python eval_factor.py --eval-composite --market cn
```

### Output Format (grep-friendly, autoresearch-style)

```
---
market:           cn
formula:          rank(delta(volume, 5) / ts_mean(volume, 20))
is_ic:            0.032
is_ic_ir:         0.45
is_sharpe:        1.23
is_turnover:      0.28
is_monotonicity:  0.85
max_corr:         0.31
max_corr_with:    momentum_5d
gates:            PASS
gate_detail:      ic=PASS icir=PASS turnover=PASS mono=PASS corr=PASS
```

For OOS mode:
```
---
oos_result:       PASS
```

For composite mode:
```
---
composite_ic_ir:  0.52
composite_sharpe: 1.85
n_factors:        7
```

### Error Handling

- DSL parse error → exit code 1, stderr: `PARSE_ERROR: unexpected token 'foo' at position 12`
- Compute error (missing feature, etc.) → exit code 2, stderr: `COMPUTE_ERROR: feature 'rsi_14' not available for market cn`
- Backtest error → exit code 3, stderr: `BACKTEST_ERROR: insufficient data (need 252 days, have 180)`
- Gate failure is NOT an error (exit code 0, gates: FAIL)

### Internal Flow

```python
def main():
    args = parse_args()

    # 1. Parse DSL
    ast = parse(args.formula)                        # src/dsl/parser.py:parse()

    # 2. Load price data
    df = load_prices(args.market)                    # new function, consolidates
                                                     # loop.py:_load_prices() logic
                                                     # using MARKET_CONFIGS for
                                                     # CN (ts_code/trade_date) vs
                                                     # US (symbol/date) column names

    # 3. Compute factor values
    sym_col, date_col = MARKET_CONFIGS[args.market]
    values = compute_factor(ast, df,                 # src/dsl/compute.py:compute_factor()
                            sym_col=sym_col,         # canonical engine with feature
                            date_col=date_col)       # aliases + computed features
    # returns DataFrame [sym_col, date_col, "factor_value"]

    # 4. Compute forward returns
    fwd = compute_forward_returns(df, sym_col, date_col)

    # 5. Walk-forward backtest
    bt = walk_forward_backtest(values, fwd,          # src/backtest/walk_forward.py
                               market=args.market)

    # 6. Check gates (with correlation against promoted factors)
    promoted_values = load_promoted_factor_values(args.market)  # from DuckDB registry
    gate_result = check_gates(bt, args.market,                  # src/backtest/gates.py
                              existing_factors=promoted_values,
                              candidate_values=values[["factor_value"]],
                              candidate_dates=values[[date_col]])

    # 7. Print results (grep-friendly)
    print_results(bt, gate_result)

    # 8. Optional: OOS check
    if args.oos_check and gate_result.passed:
        oos_pass = run_oos_check(values, fwd,        # src/backtest/walk_forward.py:run_oos_check()
                                 market=args.market)  # returns bool only
        print(f"oos_result:       {'PASS' if oos_pass else 'FAIL'}")

    # 9. Optional: Promote to DuckDB registry
    if args.promote:
        promote_factor(args, bt)                     # adapts src/mining/daily_pipeline.py:_promote_factor()
                                                     # writes to factor_registry table with actual schema:
                                                     # factor_id (SHA256 of formula), market, name,
                                                     # hypothesis, formula, direction, ic_7d/14d/30d, etc.
```

### Data Loading Strategy

Feature loading is currently scattered. `eval_factor.py` consolidates:

1. **Price data**: Loaded from pipeline DuckDB files (CN: `quant_cn.duckdb`, US: `quant.duckdb`). Uses column mapping from `src/agent/loop.py:MARKET_CONFIGS` (lines 51-72).
2. **Feature resolution**: Handled by `src/dsl/compute.py:_FEATURE_ALIASES` and `_COMPUTED_FEATURES`. Supports `close`, `volume`, `ret_1d`, `ret_5d`, etc. without a separate features module.
3. **Fundamental features**: `pe_ttm`, `pb`, `market_cap` etc. loaded from `src/evaluate/factors.py:compute_all_factors()` when available. If not available for a market, factor formulas using them get `COMPUTE_ERROR`.

### Market Column Configuration

Reuses existing config pattern from `src/agent/loop.py`:
```python
MARKET_CONFIGS = {
    "cn": {"sym_col": "ts_code", "date_col": "trade_date", "price_col": "close", "vol_col": "vol"},
    "us": {"sym_col": "symbol", "date_col": "date", "price_col": "adj_close", "vol_col": "volume"},
}
```

---

## program.md — Claude Code Instructions

### Structure

1. **Setup**: Read spec.md, FACTORS.md, research_journal.md. Check data availability. Start timer.
2. **Research Loop** (until 8 hours elapsed):
   - Read research_journal.md for current research state
   - Choose research mode based on journal insights
   - Execute experiments
   - Update journal
3. **Research Modes**:
   - **Discover**: Generate novel hypotheses from economic logic or literature
   - **Evolve**: Mutate/combine near-miss factors from past experiments
   - **Literature**: Search papers for new factor families
   - **Composite**: Optimize the promoted factor combination
4. **Per-experiment flow**:
   - Generate formula with hypothesis
   - `uv run python eval_factor.py --market {cn|us} --formula "..." > run.log 2>&1`
   - `grep "^is_ic:\|^gates:\|^max_corr:" run.log`
   - If gates PASS → run with `--oos-check`
   - If OOS PASS → run with `--promote`
   - Append to experiments.jsonl
   - Update research_journal.md (every ~10 experiments)
5. **NEVER STOP** until 8-hour timer expires
6. **Error handling**: Read stderr on non-zero exit, fix if trivial, skip if fundamental, move on
7. **Data requests**: Write structured requests to journal for human review
8. **Paper search**: Use WebSearch for academic factor ideas when exploring new families

### Key Instructions

- Alternate between CN and US markets
- After every 10 experiments, update research_journal.md with patterns observed
- When stuck, search papers: `WebSearch("quantitative factor [topic] alpha")`
- Track exploration coverage: which feature families have been explored
- Prefer factors with economic logic over statistical flukes
- When a factor nearly passes (IC close to threshold), try variations before moving on
- Record ALL experiments in experiments.jsonl (including failures — they're informative)

### Time Budget

```
0-2h:   Discovery — broad exploration, diverse hypotheses, literature search
2-5h:   Evolution — refine near-misses, combine good factors, test variations
5-7h:   Composite — optimize factor combination, test invariants
7-8h:   Wrap-up — update journal with session summary, list open questions
```

---

## research_journal.md — Cross-Session Memory

Maintained by Claude Code, persists across sessions.

### Structure

```markdown
# Factor Research Journal

## Current Understanding
[LLM's evolving theory of what drives alpha in each market]

## Confirmed Patterns
[Validated findings with evidence]

## Open Questions
[Unresolved puzzles from experiments]

## Exploration Map
[Which feature x operator families have been explored, which are untouched]

## Data Requests
[Structured requests for new data sources, with justification]

## Session Log
### Session N — YYYY-MM-DD
- Experiments run: X
- Promoted: Y
- Key finding: ...
- Updated understanding: ...
```

---

## Correlation Gate Fix

### Problem

`check_gates()` in `src/agent/loop.py:647` called without `existing_factors`, `candidate_values`, `candidate_dates` parameters. The correlation gate in `src/backtest/gates.py` auto-passes with value=0.0 when these aren't provided.

### Fix

In `eval_factor.py`, before calling `check_gates()`:
1. Query DuckDB `factor_registry` for all promoted factors in the target market
2. For each promoted factor, re-compute daily values using `src/dsl/compute.py:compute_factor()` with the promoted formula
3. Pass these as `existing_factors` dict to `check_gates()`
4. If no promoted factors exist, correlation gate auto-passes (correct for first factor)

**Performance note**: Re-computing promoted factor values on every call is expensive with many promoted factors. Cache strategy: compute once at session start, refresh every 50 experiments. For Phase 1, re-compute each time (simplicity over speed, typically < 15 promoted factors).

---

## experiments.jsonl — Machine-Readable Log

One JSON object per line, append-only. Also written to DuckDB `experiment_log` table for queryable history.

```json
{
  "ts": "2026-03-20T01:23:45",
  "session": "2026-03-20-overnight",
  "n": 1,
  "market": "cn",
  "name": "volume_surge",
  "hypothesis": "Volume surge signals institutional activity",
  "formula": "rank(delta(volume, 5) / ts_mean(volume, 20))",
  "direction": "long",
  "source": "discover",
  "is_ic": 0.032,
  "is_ic_ir": 0.45,
  "is_turnover": 0.28,
  "is_monotonicity": 0.85,
  "max_corr": 0.31,
  "gates": "PASS",
  "oos": "PASS",
  "status": "promoted",
  "error": null
}
```

---

## Composite Evaluation

`eval_factor.py --eval-composite --market cn` computes:

1. Load all promoted factors for the market from `factor_registry`
2. Re-compute daily factor values using `src/dsl/compute.py`
3. Combine via IC_IR-weighted average (adapts logic from `src/mining/export_to_pipeline.py:export()` lines 192-213)
4. Compute composite IC, IC_IR, Sharpe on IS period
5. Output metrics

This enables Phase 3 (composite optimization) where Claude tries adding/removing factors to improve the composite metric.

---

## Report Injection into Pipeline Reports

Overnight session results are included as a section in the existing daily pipeline reports (US 07:00 CST, CN 09:00 CST).

### Mechanism: Post-processing append

A shared script `generate_factor_report.py` reads `experiments.jsonl` + DuckDB `factor_registry` and produces a markdown section. This section is appended to the pipeline report markdown **after** agents generate the report but **before** the email is sent.

### New Files

| File | Location | Purpose |
|------|----------|---------|
| `generate_factor_report.py` | `factor-lab/scripts/` | Read experiments.jsonl + registry → output markdown section |

### Modifications to Existing Pipelines

| File | Change |
|------|--------|
| `quant-research-v1/scripts/run_agents.sh` ~line 513 | Add call to generate + append factor-lab section |
| `quant-research-cn/scripts/run_agents.sh` ~line 420 | Same call |

### Report Section Format

```markdown
---

## Factor Lab 因子实验报告

**Session**: 2026-03-20 overnight | **实验数**: 247 | **通过 IS Gates**: 31 | **OOS 通过**: 8 | **新 Promoted**: 5

### 新发现因子

| 名称 | 公式 | 市场 | IC | IC_IR | 假设 |
|------|------|------|-----|-------|------|
| margin_squeeze | `rank(-delta(margin_bal,5)) * rank(ret_5d)` | CN | 0.054 | 0.62 | 融资余额下降+价格上涨 |

### Composite 状态
- CN: IC_IR=0.52 (7 factors) | US: IC_IR=0.48 (5 factors)

### 研究笔记
> (Excerpted from research_journal.md — key findings from this session)
```

### generate_factor_report.py Interface

```bash
# Generate markdown section for a specific date/session
uv run python scripts/generate_factor_report.py --date 2026-03-20 --market cn
# Outputs markdown to stdout

# Append to an existing report file
uv run python scripts/generate_factor_report.py --date 2026-03-20 --market cn --append-to /path/to/report.md
```

### Pipeline Integration — Already Wired

Promoted factors automatically flow into both pipelines via existing infrastructure:
- `export_to_pipeline.py` reads `factor_registry` → writes to pipeline DBs
- US `run_full.sh` step 1.5 calls `export_to_pipeline --market us`
- CN `daily_pipeline.sh` step 2.5 calls `export_to_pipeline --market cn`
- Both pipelines consume `lab_factor` as an independent signal source

No additional code needed for factor values to enter the pipelines. Only the report section is new.

### Cron Safety

- Overnight session window: **21:00 → 05:00 CST** (8 hours)
- `eval_factor.py` reads pipeline DBs with `read_only=True` — no write lock conflicts
- Only writes to `factor_lab.duckdb` (instant INSERT operations)
- 04:00 `daily_factors.sh` writes to same DB but collision is transient and safe (DuckDB handles concurrent readers + single writer)

---

## Phase 2 (Future)

Not implemented now, but designed for:

| Component | Purpose |
|-----------|---------|
| `gp_engine.py` | Genetic programming with GPU-parallel evaluation |
| `--risk-decompose` flag | Factor exposure to known risk factors (size, value, momentum) |
| `invariant_detector.py` | Cross-regime stability analysis of factor combinations |
| `factor_graph.py` | Correlation graph, structural hole detection |
| `surrogate.py` | XGBoost surrogate model for pre-screening candidates |

---

## Success Criteria

After one overnight session (8 hours):
- 200+ experiments evaluated (mix of CN and US)
- 10+ factors promoted (with economic logic and OOS validation)
- research_journal.md contains meaningful research insights
- Exploration covers 3+ independent factor families
- All promoted factors pass correlation gate against existing registry
- Next morning's pipeline reports include Factor Lab section
- Promoted factors participate in pipeline signal scoring
