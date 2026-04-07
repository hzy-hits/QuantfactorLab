# QuantfactorLab

AI-driven quantitative alpha factor discovery platform built around one asymmetry: **agents are good at generating hypotheses and bad at judging themselves**. QuantfactorLab lets LLMs search for factor ideas inside a constrained DSL, then forces every candidate through walk-forward validation, hidden holdouts, and anti-overfit gates before anything reaches production.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white" />
  <img alt="DuckDB" src="https://img.shields.io/badge/DuckDB-OLAP-fef000?logo=duckdb" />
  <img alt="Claude" src="https://img.shields.io/badge/Agent-Claude%20%2F%20Codex-6b46c1" />
  <img alt="License" src="https://img.shields.io/badge/License-MIT-blue" />
</p>

## Thesis

- **Constrained search beats arbitrary code**: agents propose formulas, not scripts
- **The holdout stays hidden**: out-of-sample validation returns only `PASS` or `FAIL`
- **Promotion is earned, not narrated**: only factors that clear hard statistical gates flow into production pipelines

## Architecture

```
Agent Loop (Claude / Codex)
  │  proposes DSL formula + economic hypothesis
  ▼
DSL Parser (Pratt parser)
  │  validates syntax, depth ≤ 3, length ≤ 200
  ▼
Factor Computation Engine
  │  computes daily factor values from prices (GPU optional)
  ▼
Walk-Forward Backtest
  │  expanding-window IS/OOS split
  ▼
5-Gate Evaluator
  │  IC, IC_IR, turnover, monotonicity, correlation
  │  all gates must pass before OOS check
  ▼
OOS Validation
  │  binary PASS/FAIL only — agent never sees OOS metrics
  ▼
Factor Registry (DuckDB)
  │  tracks status: candidate → promoted → retired
  ▼
Pipeline Integration
  promoted factors → lab_composite signal → US & CN production pipelines
```

### Key Design Decisions

- **Constrained DSL, not arbitrary code.** Agents write factors using ~35 operators (time-series, cross-sectional, utility) in a sandboxed expression language. No loops, recursion, or external data — prevents overfitting via complex control flow.
- **OOS results are boolean only.** The agent sees "PASS" or "FAIL" for out-of-sample validation, never the actual metric values. This prevents reverse-engineering the holdout set.
- **Hard budget cap.** 50 experiments per session limits multiple-testing inflation.
- **Walk-forward expanding window.** Train from start of IS period, rolling test window — never full-sample results.
- **Single composite signal.** Multiple promoted factors combine into one IC_IR-weighted `lab_composite` source, preventing Factor Lab from overwhelming existing pipeline signals.
- **Auto-retirement.** Rolling 60-day IC < 0.01 triggers automatic factor retirement.

### 5-Gate Validation System

| Gate | CN Threshold | US Threshold | Purpose |
|------|-------------|-------------|---------|
| IC (abs) | > 0.01 | > 0.02 | Minimum predictive power |
| IC_IR | > 0.2 | > 0.3 | Signal stability across time |
| Turnover | < 50%/month | < 40%/month | Tradeable, not noise-chasing |
| Monotonicity | disabled | > 0.7 | Factor ranks stocks correctly |
| Max Correlation | < 0.6 | < 0.6 | Adds new information vs existing factors |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ (~8,400 LOC across 37 modules) |
| Storage | DuckDB (factor registry, experiment logs, read-only access to pipeline DBs) |
| Analytics | pandas, polars, numpy, scipy, scikit-learn |
| GPU (optional) | cuDF / cuML / cuPy — 800-symbol backtest: 2s GPU vs 3min CPU |
| LLM | Claude API (agent synthesis), Codex (factor mining) |
| DSL | Custom Pratt parser with ~35 operators |
| Tooling | uv (package manager), pytest, cron scheduling |

### DSL Operators

**Time-series** (per-stock, along time): `ts_mean`, `ts_std`, `ts_max`, `ts_min`, `ts_rank`, `ts_corr`, `ts_skew`, `ts_kurt`, `delta`, `pct_change`, `shift`, `decay_linear`, `decay_exp`, ...

**Cross-sectional** (per-date, across stocks): `rank`, `zscore`, `demean`, `neutralize`

**Utility**: `abs`, `sign`, `log`, `sqrt`, `power`, `clamp`, `if_then`, arithmetic (`+`, `-`, `*`, `/`)

**Example factors:**
```
rank(delta(ts_mean(close, 5), 10))                              # momentum divergence
rank(-ts_std(ret_1d, 20)) * rank(volume / ts_mean(volume, 20))  # vol compression breakout
rank(pct_change(margin_balance, 5)) * rank(ret_5d)              # margin momentum (CN)
rank(-ts_corr(high-low, close, 60))                             # range-price decorrelation
```

## How to Run

### Setup

```bash
cd QuantfactorLab
uv sync
mkdir -p data/.cache
```

### Evaluate a Single Factor

```bash
# In-sample evaluation
uv run python eval_factor.py --market cn --formula "rank(delta(volume, 5))"

# With OOS check
uv run python eval_factor.py --market cn --formula "rank(delta(volume, 5))" --oos-check

# Promote a validated factor
uv run python eval_factor.py --market cn --formula "rank(delta(volume, 5))" \
  --name "volume_delta_5" --hypothesis "volume momentum" --direction long --promote
```

### Run Agent Loop (Automated Discovery)

```bash
python -m src.agent.loop --market cn --budget 50 --output reports/session.md
```

### Daily Pipeline

```bash
python -m src.mining.daily_pipeline --market cn --max-factors 500
python -m src.mining.daily_pipeline --market us --max-factors 500
```

### View Factor Registry

```bash
uv run python eval_factor.py --show-registry --market cn
uv run python eval_factor.py --show-registry --market us
```

### Backtest Strategy

```bash
python scripts/run_strategy.py --market cn
python scripts/run_strategy.py --market us --today   # today's picks
```

### Scheduled Automation (Cron)

```bash
bash scripts/autoresearch.sh      # 3 daily sessions (02:00, 10:00, 14:00 CST)
bash scripts/daily_factors.sh     # Daily pipeline (06:00 CST)
```

### Run Tests

```bash
uv run pytest
```

## Project Structure

```
QuantfactorLab/
├── eval_factor.py                # Main CLI entry point
├── spec.md                       # Technical specification
├── FACTORS.md                    # DSL operators & features reference
├── research_journal.md           # Session logs and confirmed patterns
├── experiments.jsonl              # Experiment audit trail
├── src/
│   ├── dsl/                      # Pratt parser, AST evaluator, operators
│   ├── backtest/                 # Walk-forward engine + 5-gate system
│   ├── evaluate/                 # IC, quintile, turnover, correlation, signal analysis
│   ├── agent/                    # LLM loop (Claude/Codex backends)
│   ├── mining/                   # Batch mining + daily pipeline + export
│   ├── strategy/                 # Rolling best-factor strategy
│   ├── paper/                    # Paper trading tracker
│   ├── market_data.py            # Price loading with caching
│   └── paths.py                  # Database path configuration
├── scripts/                      # Automation (sessions, daily, weekly, strategy)
├── data/
│   ├── factor_lab.duckdb         # Factor registry + experiment logs
│   └── .cache/                   # Cached prices and forward returns
└── reports/                      # Generated session and evaluation reports
```

## License

MIT
