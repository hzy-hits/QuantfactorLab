# Factor Lab — Implementation Plan v2

*Updated based on Codex review feedback.*

## Revised MVP Scope

Codex 指出 4 天全量不现实。**缩小 MVP 到最核心链路**:

```
MVP (3天):
  Phase 1.5: turnover + rolling IC               (0.5天)
  Phase 2:   DSL parser + compute + 2-fold WF     (1.5天)
  Phase 3:   Semi-auto agent loop (手动触发)       (1天)

Later:
  Registry + promotion + pipeline injection
  GPU acceleration
  Sector neutralization
```

---

## Phase 1.5: Eval 增强 + Bug Fix (0.5天)

### Bug fixes from Codex review

1. **quintile.py**: 检测 qcut 实际分组数，<5 组时跳过该日
2. **factors.py**: RSI avg_loss≈0 返回 NaN 而非 100
3. **correlation.py**: pairwise 有效日计算，不整天丢弃
4. **ic.py**: IC_IR 标注为"启发式，非显著性检验"（5D 重叠）

### New: turnover.py

```python
def compute_turnover(factor_values, dates, sym_col, top_pct=0.2):
    """日间 top 组换手率。"""
    # 每天取 top 20% 股票列表
    # 计算 |今天 ∩ 昨天| / |top组|
    # 输出: daily_turnover, avg_monthly (×21)
```

### New: rolling_ic.py

```python
def compute_rolling_ic(ic_series, window=60):
    """60日滚动 IC 均值 + 趋势检测。"""
    # rolling mean of daily IC
    # linear regression slope → IC 在衰减还是增强
```

---

## Phase 2: DSL + Backtest (1.5天)

### 2.1 DSL Parser — 修订

**Codex 反馈: 必须加 infix 算术运算符。**

```
支持的表达式:
  rank(delta(close, 5))                    # 纯函数式
  rank(ret_5d) * rank(-volume_ratio)       # infix * 和 unary -
  if_then(rsi_14 - 30, close, 0)           # infix - 做比较

语法 (Pratt parser):
  expr     = unary ((+|-|*|/) unary)*
  unary    = (-) unary | primary
  primary  = NUMBER | IDENT | IDENT "(" args ")"
  args     = expr ("," expr)*
```

Pratt parser 比递归下降更适合处理 operator precedence。~150 行。

**安全约束 (不变)**:
- 白名单函数
- max depth 4 (含 infix，Codex 建议从 3 提到 4)
- max length 200
- 白名单 lookback windows

### 2.2 Factor Compute Engine

```python
def compute_factor(ast, prices_df, sym_col, date_col):
    """
    AST → factor values.

    Key design:
    - 时序算子: groupby(sym).apply()
    - 截面算子: groupby(date).apply()
    - Infix: numpy 向量运算
    """
```

### 2.3 Walk-Forward — 修订

**Codex 反馈: 500天数据默认 2-fold，不是 3-fold。**

```python
def walk_forward(factor_values, fwd_returns,
                 oos_start="2025-10-01",  # 最近 ~5 个月
                 n_folds=2,                # Codex: 2-fold 更稳健
                 min_train=120,
                 min_test=60):             # 每折至少 60 天测试
    """
    Anchored expanding window:

    Fold 1: train 2024-03~2025-03 (250d), test 2025-04~2025-09 (120d)
    Fold 2: train 2024-03~2025-06 (310d), test 2025-07~2025-09 (60d)

    OOS: 2025-10~2026-03 (~120d) — PASS/FAIL only
    """
```

**加简单交易成本**:
```python
# 每次换手扣 0.3% (A股印花税+佣金) 或 0.1% (US)
cost_adjusted_return = raw_return - turnover * cost_per_trade
```

### 2.4 Gates — 修订

**Codex 反馈: CN/US 分别设阈值。**

```python
GATES = {
    "cn": {
        "ic_min": 0.01,        # CN baseline 最高 IC=0.008，降低门槛
        "ic_ir_min": 0.2,      # 相应降低
        "turnover_max": 0.50,  # A股换手自然更高
        "monotonicity_min": 0.6,
        "corr_max": 0.7,
    },
    "us": {
        "ic_min": 0.02,
        "ic_ir_min": 0.3,
        "turnover_max": 0.40,
        "monotonicity_min": 0.7,
        "corr_max": 0.7,
    },
}
```

---

## Phase 3: Agent Loop — MVP (1天)

### 修订: OOS 查询限制

**Codex 反馈: 每次实验都反馈 OOS PASS/FAIL = 变相泄露。**

修改:
```
旧: 每个因子 IS 过关 → 立即查 OOS → 告诉 agent PASS/FAIL
新: 全部 50 次实验跑完 → 取 IS 排名 top 3 → 只对 top 3 查 OOS
    Agent 只在 session 结束时看到 3 个 OOS 结果
    无法通过多次小修改来推断 OOS 分布
```

### 修订: 用 Anthropic SDK

**Codex 反馈: claude -p 不适合 50 步循环。**

```python
from anthropic import Anthropic

client = Anthropic()

def ask_agent(prompt: str) -> str:
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",  # 因子生成用 sonnet 够了
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text
```

### MVP Agent Loop

```python
def run_session(market: str, budget: int = 50):
    prices = load_prices(market)
    fwd_rets = compute_forward_returns(prices)

    experiments = []
    for i in range(budget):
        # 1. Agent proposes (sees IS results of previous experiments)
        response = ask_agent(build_prompt(experiments, market))

        # 2. Parse + compute + evaluate IS only
        ast = parse_dsl(response.formula)
        values = compute_factor(ast, prices)
        is_metrics = walk_forward_is_only(values, fwd_rets)
        gates = check_gates(is_metrics, market)

        experiments.append({...is_metrics, gates...})

    # 3. Session end: OOS check top 3 IS performers
    top3 = sorted(experiments, key=lambda e: e["is_ic"])[-3:]
    for exp in top3:
        exp["oos_pass"] = run_oos(exp["values"], fwd_rets)

    # 4. Report
    print_session_summary(experiments, top3)
```

---

## Schema 修订 (from Codex spec review)

```sql
CREATE TABLE factor_registry (
    factor_id    VARCHAR PRIMARY KEY,
    market       VARCHAR NOT NULL,      -- 'cn' or 'us' (NEW)
    name         VARCHAR NOT NULL,
    hypothesis   VARCHAR NOT NULL,
    formula      VARCHAR NOT NULL,
    direction    VARCHAR NOT NULL,
    fwd_horizon  INTEGER DEFAULT 5,     -- NEW
    discovered   TIMESTAMP NOT NULL,

    -- IS metrics (market-specific)
    is_ic        DOUBLE,
    is_ic_ir     DOUBLE,
    is_turnover  DOUBLE,
    is_monotonicity DOUBLE,

    -- OOS
    oos_pass     BOOLEAN,

    -- Correlation
    max_corr_factor VARCHAR,
    max_corr_value  DOUBLE,

    -- Lifecycle: candidate → promoted → watchlist → retired (NEW: watchlist)
    status       VARCHAR DEFAULT 'candidate',
    promoted_at  TIMESTAMP,
    watchlist_at TIMESTAMP,             -- NEW
    retired_at   TIMESTAMP,
    retire_reason VARCHAR,
);
```

**Lifecycle — 30 天短寿命因子模型**:

因子寿命预期 ~30 天。不是"找一个因子用几年"，而是"因子工厂持续产出"。

```
candidate → promoted → watchlist → retired
                ↑           │
                └───────────┘ (recover if IC rebounds)

promoted → watchlist: rolling 20d IC < 0.005 for 5 days (快速降级)
watchlist → retired:  5 天内 IC 没恢复 > 0.01 (快速退休)
watchlist → promoted: 5 天内 IC 恢复 > 0.01 (允许回来)

从上线到退休: 最快 10 天，典型 20-30 天，超过 60 天算长寿

运营节奏:
  每周: Agent 跑 1-2 个 session，发现 2-3 个新因子
  每天: 活跃因子 5-10 个，retire_check 自动淘汰衰减的
  每月: 回顾存活率、平均寿命、IC 衰减曲线
```

**因子库规模**: 见 FACTORS.md (91 个经典因子 + 35 个算子)
- MVP (P0): 34 个基础因子 + 35 算子
- P1: +32 (资金面+估值+微观结构)
- P2: +55 (时间效应+滚动窗口变体)


**Pipeline injection 修订 (Codex 建议)**:
```
Factor Lab 只写自己的 DB (factor_lab.duckdb)
Pipeline 的 cron 脚本从 factor_lab.duckdb 读取 promoted 因子值
写入自己的 analytics/analysis_daily 表
= 单向数据流，无并发写冲突
```

---

## Revised Timeline

| 任务 | 时间 | 产出 |
|------|------|------|
| Phase 1.5: eval bugfix + turnover + rolling IC | 0.5天 | 完善的评估工具 |
| Phase 2.1: Pratt parser + safety | 0.5天 | DSL → AST |
| Phase 2.2: compute engine | 0.5天 | AST → factor values |
| Phase 2.3: 2-fold WF + costs + gates | 0.5天 | IS metrics + OOS gate |
| Phase 3: agent loop MVP | 1天 | 跑通一个 session |
| **Total MVP** | **3天** | 能跑 agent 因子挖掘 session |
| Later: registry + promotion + GPU | +2天 | 完整生命周期 |
