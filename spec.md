# Factor Lab — Specification

## 1. Overview

AI-assisted quantitative factor mining and evaluation platform. Agents generate factor hypotheses with economic logic, the system backtests them with strict anti-overfit controls, and validated factors are automatically promoted into the production quant pipelines (US + CN).

**Core Principle**: Agent is both generator (proposes hypotheses) and discriminator (evaluates results). The system enforces statistical rigor that the agent cannot bypass.

## 2. Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Factor Lab                        │
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
│  │  Agent    │──▶│  Factor  │──▶│  Walk-Forward    │ │
│  │  Loop     │   │  DSL     │   │  Backtest Engine │ │
│  │          │   │  Parser  │   │  (cuDF + numpy)  │ │
│  └────┬─────┘   └──────────┘   └────────┬─────────┘ │
│       │                                  │           │
│       │         ┌──────────────┐         │           │
│       ◀─────────│  Evaluator   │◀────────┘           │
│                 │  IC/Sharpe   │                      │
│                 │  Monotonicity│                      │
│                 │  OOS Gate    │                      │
│                 └──────┬───────┘                      │
│                        │                              │
│                 ┌──────▼───────┐                      │
│                 │  Registry    │─── promote ──▶ Pipeline
│                 │  (DuckDB)    │                      │
│                 └──────────────┘                      │
└─────────────────────────────────────────────────────┘
```

## 3. Components

### 3.1 Factor DSL (Domain-Specific Language)

Constrained expression language for factor definitions. Agent writes factors in DSL, not arbitrary Python.

**Operators** (transforms):
```
rank(x)              — cross-sectional percentile rank [0, 1]
zscore(x)            — cross-sectional z-score, clamped [-3, 3]
delta(x, n)          — x[t] - x[t-n]
pct_change(x, n)     — (x[t] / x[t-n]) - 1
decay_linear(x, n)   — linearly-weighted moving average (recent = heavier)
ts_mean(x, n)        — trailing n-bar mean
ts_std(x, n)         — trailing n-bar standard deviation
ts_max(x, n)         — trailing n-bar max
ts_min(x, n)         — trailing n-bar min
ts_rank(x, n)        — trailing n-bar percentile rank of current value
ts_corr(x, y, n)     — trailing n-bar rolling correlation
ts_cov(x, y, n)      — trailing n-bar rolling covariance
sign(x)              — -1, 0, or +1
abs(x)               — absolute value
log(x)               — natural log (guarded: log(max(x, 1e-9)))
power(x, p)          — x^p (p restricted to 0.5, 2, 3)
clamp(x, lo, hi)     — clamp to range
if_then(cond, a, b)  — conditional: cond > 0 → a, else → b
```

**Features** (inputs):
```
# Price
close, open, high, low, vwap
ret_1d, ret_5d, ret_20d, ret_60d

# Volume
volume, amount, turnover_rate

# Technical (pre-computed)
rsi_14, bb_position, ma20_dist, sma20, sma60

# Fundamental (if available)
pe_ttm, pb, market_cap, circ_market_cap

# Flow (A-share specific)
net_mf_amount, large_net_in, margin_balance, margin_delta_5d

# Market-level
index_ret_1d, index_ret_5d, sector_ret_5d
```

**Constraints**:
- Max nesting depth: 3 (e.g., `rank(delta(ts_mean(close, 5), 10))` = depth 3)
- Max expression length: 200 characters
- No loops, no recursion, no external data access
- Lookback windows: [1, 2, 3, 5, 10, 20, 40, 60, 120] only

**Example factors**:
```yaml
- name: volume_price_divergence
  hypothesis: "Rising price on declining volume = unsustainable, expect reversal"
  formula: rank(ret_5d) * rank(-delta(volume, 5) / ts_mean(volume, 20))
  direction: short  # high score = short signal

- name: margin_momentum_confirm
  hypothesis: "Margin balance increase confirms price momentum"
  formula: rank(pct_change(margin_balance, 5)) * rank(ret_5d)
  direction: long

- name: volatility_compression_breakout
  hypothesis: "Low volatility followed by volume surge = breakout imminent"
  formula: rank(-ts_std(ret_1d, 20)) * rank(volume / ts_mean(volume, 20))
  direction: long
```

### 3.2 Walk-Forward Backtest Engine

**Methodology**: Expanding-window walk-forward with fixed OOS holdout.

```
Data: 2022-01-01 ─────────────────────────────────── 2026-03-17
      │◄── IS (agent can see) ──►│◄── OOS (agent cannot see) ──►│
      │         ~3.5 years        │        ~6 months              │
      │                           │                                │
      │ Walk-forward within IS:   │ Final validation:              │
      │ ├─ train: 2022-01~2024-06│ Agent sees PASS/FAIL only,     │
      │ ├─ test:  2024-07~2024-12│ never the OOS metric values    │
      │ ├─ train: 2022-01~2024-12│                                │
      │ └─ test:  2025-01~2025-06│                                │
```

**Metrics computed per walk-forward fold**:
- IC (Information Coefficient): rank correlation of factor vs 5D forward return
- IC_IR: IC mean / IC std (stability)
- Long-short return: top quintile - bottom quintile
- Quintile monotonicity: Spearman correlation of quintile rank vs quintile return
- Turnover: daily portfolio turnover rate
- Max drawdown of long-short portfolio

### 3.3 Evaluator & Anti-Overfit Gates

**5-Gate System** (ALL must pass):

| Gate | Threshold | Rationale |
|------|-----------|-----------|
| IC > 0.02 | Minimum predictive power | |
| IC_IR > 0.3 | Signal must be stable across time | |
| Turnover < 40%/month | Tradeable, not noise-chasing | |
| Quintile monotonicity > 0.8 | Factor must rank stocks correctly, not just extremes | |
| Factor corr < 0.7 vs all existing | Must add new information | |

**OOS Gate**:
- Agent sees: "OOS: PASS" or "OOS: FAIL"
- Agent does NOT see: OOS IC value, OOS Sharpe, OOS returns
- This prevents the agent from reverse-engineering the OOS holdout

**Budget Control**:
- Max 50 experiments per session (prevents multiple-testing abuse)
- At 50 experiments × 5% significance → ~2.5 false positives (manageable)
- Session budget resets daily

### 3.4 Agent Loop

```python
# Pseudocode for one session
budget = 50
discovered = []

for i in range(budget):
    # 1. Agent generates hypothesis + formula
    response = claude(f"""
        You are a quantitative researcher.

        Market data available: {FEATURE_LIST}
        DSL operators: {OPERATOR_LIST}

        Previously tried (this session):
        {tried_factors_summary}

        Previously discovered (all time):
        {registry_summary}

        Generate a NEW factor hypothesis.

        Output format:
        HYPOTHESIS: <1-2 sentences, economic logic>
        FORMULA: <DSL expression>
        DIRECTION: <long or short>
        EXPECTED_DECAY: <fast/medium/slow>
    """)

    # 2. Parse DSL, validate syntax
    factor = parse_dsl(response.formula)

    # 3. Compute factor values (GPU-accelerated)
    values = compute_factor(factor, price_data)  # cuDF

    # 4. Walk-forward backtest
    is_metrics = walk_forward_backtest(values, forward_returns)

    # 5. Show IS results to agent (NOT OOS)
    # 6. If IS passes gates → run OOS
    if passes_is_gates(is_metrics):
        oos_pass = run_oos_check(values, oos_data)  # bool only
        if oos_pass:
            discovered.append(factor)
            register_factor(factor, is_metrics)

    # 7. Agent sees results, iterates
    tried_factors_summary.append({
        "formula": response.formula,
        "is_ic": is_metrics.ic,
        "is_icir": is_metrics.ic_ir,
        "passed_is": passes_is_gates(is_metrics),
        "oos_result": "PASS" if oos_pass else "FAIL" if passes_is_gates(is_metrics) else "N/A",
    })
```

### 3.5 Factor Registry (DuckDB)

```sql
CREATE TABLE factor_registry (
    factor_id    VARCHAR PRIMARY KEY,  -- hash of formula
    name         VARCHAR NOT NULL,
    hypothesis   VARCHAR NOT NULL,
    formula      VARCHAR NOT NULL,
    direction    VARCHAR NOT NULL,     -- 'long' or 'short'
    discovered   TIMESTAMP NOT NULL,

    -- IS metrics
    is_ic        DOUBLE,
    is_ic_ir     DOUBLE,
    is_sharpe    DOUBLE,
    is_turnover  DOUBLE,
    is_monotonicity DOUBLE,

    -- OOS status
    oos_pass     BOOLEAN,

    -- Correlation with existing factors
    max_corr_factor VARCHAR,
    max_corr_value  DOUBLE,

    -- Status
    status       VARCHAR DEFAULT 'candidate',  -- candidate / promoted / retired
    promoted_at  TIMESTAMP,
    retired_at   TIMESTAMP,
    retire_reason VARCHAR,
);

CREATE TABLE factor_daily (
    factor_id VARCHAR NOT NULL,
    ts_code   VARCHAR NOT NULL,
    date      DATE NOT NULL,
    value     DOUBLE,
    PRIMARY KEY (factor_id, ts_code, date)
);

CREATE TABLE experiment_log (
    session_id   VARCHAR NOT NULL,
    experiment_n INTEGER NOT NULL,
    timestamp    TIMESTAMP NOT NULL,
    hypothesis   VARCHAR,
    formula      VARCHAR,
    is_ic        DOUBLE,
    is_ic_ir     DOUBLE,
    passed_is    BOOLEAN,
    oos_result   VARCHAR,  -- 'PASS', 'FAIL', 'N/A'
    PRIMARY KEY (session_id, experiment_n)
);
```

### 3.6 Pipeline Promotion

When a factor achieves `status = 'promoted'`:

1. Daily cron computes factor values for the full universe
2. Values written to `factor_daily` table
3. IC_IR-weighted combination of all promoted factors → `lab_composite` score
4. `lab_composite` injected into pipeline as one signal source:
   - **CN**: `analytics` table, module='lab_factor'
   - **US**: `analysis_daily` table, module_name='lab_factor'
5. `notable.rs` / `classify.py` treat it as an independent convergence vote

**Auto-retirement**: If a promoted factor's rolling 60-day IC drops below 0.01 for 20 consecutive days, it is auto-retired and removed from the composite.

## 4. GPU Acceleration (5070 Ti, 16GB GDDR7)

| Component | Library | GPU Benefit |
|-----------|---------|-------------|
| Factor value computation | cuDF | 800 sym × 500 days × 200 factors in 2s vs 3min |
| Cross-sectional rank/zscore | cuDF | Native GPU groupby |
| Walk-forward IC computation | cuDF + cuPy | Rank correlation on GPU |
| Bootstrap stability test | cuPy | 10000 resamples in 10s vs 20min |
| XGBoost factor interaction | xgboost gpu_hist | 10-50x vs CPU |
| Covariance / PCA | cuML | 800×800 matrix in milliseconds |

**Fallback**: All code must also work on CPU-only (pandas/numpy/sklearn). GPU is an accelerator, not a requirement.

## 5. Data Sources

Reads from existing pipeline databases (read-only):
- **US**: `$QUANT_US_ROOT/data/quant.duckdb`
- **CN**: `$QUANT_CN_ROOT/data/quant_cn.duckdb`

Own database:
- **Factor Lab**: `<project-root>/data/factor_lab.duckdb`

## 6. Project Structure

```
factor-lab/
├── CLAUDE.md              # Dev guidance
├── spec.md                # This file
├── pyproject.toml          # Dependencies
├── data/
│   └── factor_lab.duckdb   # Registry + experiment logs
├── src/
│   ├── dsl/
│   │   ├── parser.py       # DSL → AST → executable
│   │   ├── operators.py    # Operator implementations (pandas + cuDF)
│   │   └── features.py     # Feature loading from pipeline DBs
│   ├── backtest/
│   │   ├── walk_forward.py # Walk-forward engine
│   │   ├── metrics.py      # IC, Sharpe, monotonicity, turnover
│   │   └── bootstrap.py    # GPU Monte Carlo stability
│   ├── evaluate/
│   │   ├── gates.py        # 5-gate system
│   │   ├── oos.py          # OOS holdout (returns bool only)
│   │   └── correlation.py  # Factor dedup vs registry
│   ├── agent/
│   │   ├── loop.py         # Main agent experiment loop
│   │   ├── prompts.py      # Prompt templates
│   │   └── budget.py       # Experiment budget tracking
│   ├── registry/
│   │   ├── db.py           # DuckDB schema + CRUD
│   │   └── promote.py      # Factor → pipeline injection
│   └── gpu/
│       ├── backend.py      # cuDF/cuML auto-detection + fallback
│       └── xgboost_combine.py  # Non-linear factor combination
├── scripts/
│   ├── run_session.sh      # Run one agent session (50 experiments)
│   ├── daily_compute.py    # Compute promoted factors for pipeline
│   └── retire_check.py     # Auto-retire degraded factors
└── tests/
    ├── test_dsl.py
    ├── test_backtest.py
    └── test_gates.py
```

## 7. Non-Goals (Explicit)

- No live trading or execution
- No portfolio optimization (that's downstream)
- No deep learning factor models (not enough data, overfit risk)
- No alternative data ingestion (use existing pipeline data only)
- No distributed computing (single machine, single GPU)
