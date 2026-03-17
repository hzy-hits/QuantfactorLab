# Factor Lab — Implementation Plan

## Phase 1.5: Eval 增强 (1天)

### 目标
完善评估框架，为 Phase 2/3 的因子验证提供完整工具链。

### 交付物

**1. turnover.py — 因子换手率分析**
```python
def compute_turnover(factor_values, dates, sym_col, top_pct=0.2):
    """
    每天取因子值 top 20% 的股票组合，计算日间换手率。
    turnover = 1 - |今天top组 ∩ 昨天top组| / |top组|

    输出:
    - daily_turnover: 每天的换手率
    - avg_turnover: 日均换手率
    - annualized_turnover: 年化换手率 (daily * 250)
    - 月度 turnover 时序图数据
    """
```

**2. rolling_ic.py — IC 稳定性时序分析**
```python
def compute_rolling_ic(factor_values, forward_returns, dates, window=60):
    """
    滚动60天窗口的 IC 均值。

    输出:
    - rolling_ic_series: 每天一个60天滚动IC值
    - ic_decay: IC 是否随时间衰减 (线性回归斜率)
    - regime_transition_ic: regime 切换时 IC 变化
    """
```

**3. report.py 增强 — 生成完整 markdown + 简单文本图表**
```python
# 在报告中加入:
# - IC 时序图 (ASCII sparkline 或保存为 matplotlib PNG)
# - 分组收益柱状图
# - 因子相关性热力图
```

### 实现细节

turnover 的计算：
```sql
-- 每天取 top 20% 组的股票列表
WITH daily_top AS (
    SELECT date, ts_code,
           PERCENT_RANK() OVER (PARTITION BY date ORDER BY factor_value DESC) AS prank
    FROM factor_data
),
top_group AS (
    SELECT date, ts_code FROM daily_top WHERE prank >= 0.8
)
-- 日间重叠率
SELECT t1.date,
       1.0 - COUNT(t2.ts_code)::FLOAT / COUNT(t1.ts_code) AS turnover
FROM top_group t1
LEFT JOIN top_group t2 ON t1.ts_code = t2.ts_code
    AND t2.date = (SELECT MAX(date) FROM top_group WHERE date < t1.date)
GROUP BY t1.date
```

---

## Phase 2: DSL Parser + Backtest Engine (2-3天)

### 目标
让 Agent 能用受限 DSL 写因子公式，系统自动计算因子值并 walk-forward 回测。

### 2.1 DSL Parser (dsl/parser.py)

**架构**: DSL string → AST → 执行计划 → pandas/numpy 计算

```python
# 输入
"rank(delta(ts_mean(close, 5), 10))"

# 解析为 AST
FunctionCall("rank", [
    FunctionCall("delta", [
        FunctionCall("ts_mean", [
            Feature("close"),
            Literal(5)
        ]),
        Literal(10)
    ])
])

# 执行
for each date:
    for each stock:
        1. ts_mean(close, 5) → 5天均价
        2. delta(result, 10) → 跟10天前的差
    3. rank(result) → 全市场排名
```

**Parser 实现方案**: 递归下降解析器 (不用外部库)
```python
class Token:
    LPAREN = "("
    RPAREN = ")"
    COMMA = ","
    NUMBER = "NUMBER"
    IDENT = "IDENT"
    EOF = "EOF"

class Lexer:
    """Tokenize DSL string."""

class Parser:
    """Parse tokens into AST."""
    def parse_expr(self) -> ASTNode
    def parse_function_call(self) -> FunctionCall
    def parse_atom(self) -> Union[Feature, Literal]

class ASTNode: ...
class FunctionCall(ASTNode): name, args
class Feature(ASTNode): name
class Literal(ASTNode): value
```

**Operator 实现 (dsl/operators.py)**:
```python
OPERATORS = {
    # 时序算子: (stock_series, *params) → stock_series
    "ts_mean":  lambda s, n: s.rolling(n).mean(),
    "ts_std":   lambda s, n: s.rolling(n).std(),
    "ts_max":   lambda s, n: s.rolling(n).max(),
    "ts_min":   lambda s, n: s.rolling(n).min(),
    "ts_rank":  lambda s, n: s.rolling(n).apply(lambda x: pd.Series(x).rank().iloc[-1] / len(x)),
    "delta":    lambda s, n: s - s.shift(n),
    "pct_change": lambda s, n: s / s.shift(n) - 1,
    "decay_linear": lambda s, n: s.rolling(n).apply(lambda x: np.average(x, weights=range(1, len(x)+1))),

    # 截面算子: (cross_section_series) → cross_section_series
    "rank":     lambda df_col: df_col.rank(pct=True),
    "zscore":   lambda df_col: (df_col - df_col.mean()) / (df_col.std() + 1e-9),

    # 通用算子
    "abs":      lambda s: s.abs(),
    "sign":     lambda s: np.sign(s),
    "log":      lambda s: np.log(np.maximum(s, 1e-9)),
    "power":    lambda s, p: s ** p,
    "clamp":    lambda s, lo, hi: s.clip(lo, hi),
    "if_then":  lambda cond, a, b: np.where(cond > 0, a, b),
}
```

**安全性约束**:
- Allowed functions: 白名单制，只有 OPERATORS 中定义的
- Max depth: 递归解析时计数，>3 拒绝
- Max length: 解析前检查字符串长度 >200 拒绝
- No import/exec/eval: AST 中只有 FunctionCall/Feature/Literal
- Allowed lookbacks: [1,2,3,5,10,20,40,60,120]，其他值拒绝

### 2.2 Factor Computation Engine (dsl/compute.py)

```python
def compute_factor(ast: ASTNode, price_data: pd.DataFrame,
                   sym_col: str, date_col: str) -> pd.DataFrame:
    """
    Execute AST against price data.

    Strategy:
    1. Time-series operators: apply per-stock (groupby sym, apply)
    2. Cross-section operators: apply per-date (groupby date, apply)
    3. Nested: inside-out evaluation

    Returns: DataFrame with [sym_col, date_col, "factor_value"]
    """
```

**关键设计决策**: 时序 vs 截面算子的区分
```
ts_mean(close, 20)  → 每只股票独立计算 → groupby(sym).rolling(20).mean()
rank(x)             → 每天所有股票一起排 → groupby(date).rank()

嵌套: rank(ts_mean(close, 20))
  步骤1: 每只股票算20天均价 (时序)
  步骤2: 每天对所有均价排名 (截面)
```

### 2.3 Walk-Forward Backtest (backtest/walk_forward.py)

```python
def walk_forward_backtest(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    is_end: str = "2025-09-01",     # IS/OOS 分割点
    n_folds: int = 3,               # walk-forward 折数
    min_train_days: int = 120,      # 最少训练天数
    fwd_horizon: int = 5,           # 预测周期
) -> BacktestResult:
    """
    Expanding-window walk-forward.

    Fold 1: train 2024-03 ~ 2024-12, test 2025-01 ~ 2025-03
    Fold 2: train 2024-03 ~ 2025-03, test 2025-04 ~ 2025-06
    Fold 3: train 2024-03 ~ 2025-06, test 2025-07 ~ 2025-09

    Returns: IS metrics (per-fold + average), OOS pass/fail
    """

class BacktestResult:
    fold_metrics: list[FoldMetrics]  # per-fold IC, Sharpe, etc.
    avg_is_ic: float
    avg_is_icir: float
    avg_is_sharpe: float
    avg_turnover: float
    avg_monotonicity: float
    oos_pass: bool                   # True/False only, no OOS metric values
```

### 2.4 Gates (evaluate/gates.py)

```python
def check_gates(result: BacktestResult, registry_factors: list) -> GateResult:
    """
    5-gate system. ALL must pass for factor to be considered valid.

    Gate 1: avg_is_ic > 0.02
    Gate 2: avg_is_icir > 0.3
    Gate 3: avg_turnover < 0.40 (40% monthly)
    Gate 4: avg_monotonicity > 0.7
    Gate 5: max correlation with registry < 0.7
    """

class GateResult:
    passed: bool
    gate_details: dict[str, {"passed": bool, "value": float, "threshold": float}]
```

---

## Phase 3: Agent Loop (2-3天)

### 目标
让 Claude 自主进行因子研究会话：提假说 → 写公式 → 看结果 → 迭代。

### 3.1 Prompt 设计 (agent/prompts.py)

**System Prompt**:
```
You are a quantitative researcher at a systematic fund.
Your job: discover new alpha factors from price/volume data.

RULES:
1. Every factor MUST have an economic hypothesis (WHY should this work?)
2. Write factors in DSL syntax (see below)
3. You have {budget_remaining} experiments left this session
4. Previously tried factors and their results are shown below
5. You CANNOT see OOS results — only PASS/FAIL
6. Target: IC > 0.02, IC_IR > 0.3, monotonicity > 0.7

DSL OPERATORS: {operator_list}
FEATURES: {feature_list}
MAX DEPTH: 3
LOOKBACK WINDOWS: [1,2,3,5,10,20,40,60,120]

MARKET CONTEXT:
- Current regime distribution: {regime_dist}
- Best existing factors: {top_factors}
- Known gaps: {gaps}

OUTPUT FORMAT:
HYPOTHESIS: <1-2 sentences>
FORMULA: <DSL expression>
DIRECTION: <long|short>
REASONING: <why this should have alpha>
```

**Result Feedback Prompt**:
```
Experiment {n}/{budget}:
  Formula: {formula}
  IS Results:
    IC = {ic}, IC_IR = {icir}
    Quintile returns: {quintiles}
    Turnover: {turnover}%
    Monotonicity: {mono}
  Gate results: {gate_details}
  OOS: {PASS|FAIL|N/A}

  Previous experiments this session:
  {experiment_history}

Analyze why this factor {worked|failed} and propose your next experiment.
```

### 3.2 Session Controller (agent/loop.py)

```python
class FactorSession:
    budget: int = 50
    experiments: list[Experiment]
    discovered: list[Factor]

    def run(self):
        """Main loop."""
        for i in range(self.budget):
            # 1. Get agent's next hypothesis
            response = self.ask_agent(self.build_prompt())

            # 2. Parse DSL
            try:
                ast = parse_dsl(response.formula)
            except DSLError as e:
                self.record_error(i, str(e))
                continue

            # 3. Compute factor values
            values = compute_factor(ast, self.price_data)

            # 4. Walk-forward backtest
            result = walk_forward_backtest(values, self.fwd_returns)

            # 5. Check gates
            gates = check_gates(result, self.registry)

            # 6. OOS check (if IS passes)
            oos_pass = None
            if gates.passed:
                oos_pass = self.run_oos(values)
                if oos_pass:
                    self.discovered.append(Factor(
                        name=response.name,
                        hypothesis=response.hypothesis,
                        formula=response.formula,
                        metrics=result,
                    ))

            # 7. Record and continue
            self.experiments.append(Experiment(
                n=i, formula=response.formula,
                hypothesis=response.hypothesis,
                is_metrics=result, gates=gates,
                oos_result=oos_pass,
            ))

    def ask_agent(self, prompt: str) -> AgentResponse:
        """Call Claude API for next factor."""
        # Use claude -p or anthropic SDK
        pass

    def build_prompt(self) -> str:
        """Build prompt with context and history."""
        pass
```

### 3.3 Registry Integration (registry/db.py)

```python
class FactorRegistry:
    def __init__(self, db_path: str):
        self.con = duckdb.connect(db_path)
        self.init_schema()

    def register(self, factor: Factor): ...
    def promote(self, factor_id: str): ...
    def retire(self, factor_id: str, reason: str): ...
    def get_active(self) -> list[Factor]: ...
    def compute_daily(self, price_data, date) -> pd.DataFrame: ...
```

### 3.4 Pipeline Injection (registry/promote.py)

```python
def inject_into_pipeline(registry: FactorRegistry, market: str, date: str):
    """
    Compute lab_composite from all promoted factors,
    write to pipeline's analytics/analysis_daily table.

    lab_composite = IC_IR-weighted average of promoted factor values
    """
    active = registry.get_active()
    if not active:
        return

    # Compute each factor's value for today
    values = {}
    total_weight = 0
    for f in active:
        v = compute_factor(parse_dsl(f.formula), price_data)
        values[f.id] = v
        total_weight += f.ic_ir

    # IC_IR-weighted combination
    composite = sum(f.ic_ir / total_weight * values[f.id] for f in active)

    # Write to pipeline DB
    write_to_pipeline(composite, market, date)
```

---

## 时间线

| Phase | 内容 | 估计时间 | 依赖 |
|-------|------|---------|------|
| 1.5 | turnover + rolling IC | 0.5天 | Phase 1 ✅ |
| 2.1 | DSL parser | 0.5天 | 无 |
| 2.2 | Factor computation engine | 0.5天 | 2.1 |
| 2.3 | Walk-forward backtest | 0.5天 | Phase 1 评估函数 |
| 2.4 | Gates | 0.5天 | 2.3 |
| 3.1 | Prompt design | 0.5天 | 2.1-2.4 |
| 3.2 | Session controller | 0.5天 | 3.1 |
| 3.3 | Registry | 0.5天 | 无 |
| 3.4 | Pipeline injection | 0.5天 | 3.3 |
| **总计** | | **~4天** | |

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| DSL 太受限，agent 表达不了好因子 | 先用 Alpha101 的经典因子测试 DSL 覆盖度 |
| 500天数据太少，walk-forward 折数不够 | 最少2折，每折至少60天测试期 |
| Agent 陷入局部最优 | 在 prompt 中加 "diversify" 指令 + 显示因子相关性 |
| GPU 加速的 cuDF 安装困难 | 全部 CPU fallback，GPU 是 Phase 5 |
| OOS holdout 太短 (~6个月) | 先用，等数据积累到2年再扩展 |
