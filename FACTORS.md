# Factor Lab — 因子库 & 算子清单

*参考: WorldQuant Alpha101, Qlib Alpha158, 常用量化因子*

---

## 一、DSL 算子 (Operators)

### 时序算子 (per-stock, along time axis)

| 算子 | 参数 | 含义 | 示例 |
|------|------|------|------|
| `ts_mean(x, n)` | series, window | N日均值 | `ts_mean(close, 20)` = 20日均价 |
| `ts_std(x, n)` | series, window | N日标准差 | `ts_std(ret_1d, 20)` = 20日波动率 |
| `ts_max(x, n)` | series, window | N日最高 | `ts_max(high, 20)` = 20日最高价 |
| `ts_min(x, n)` | series, window | N日最低 | `ts_min(low, 20)` = 20日最低价 |
| `ts_sum(x, n)` | series, window | N日累加 | `ts_sum(volume, 5)` = 5日总量 |
| `ts_rank(x, n)` | series, window | 当前值在N日内的排名% | `ts_rank(close, 60)` = 60日内位置 |
| `ts_argmax(x, n)` | series, window | N日内最高值的位置 | `ts_argmax(high, 20)` = 最高价出现在几天前 |
| `ts_argmin(x, n)` | series, window | N日内最低值的位置 | `ts_argmin(low, 20)` = 最低价出现在几天前 |
| `ts_corr(x, y, n)` | 2 series, window | N日滚动相关性 | `ts_corr(close, volume, 20)` = 量价相关性 |
| `ts_cov(x, y, n)` | 2 series, window | N日滚动协方差 | `ts_cov(ret_1d, volume, 10)` |
| `ts_skew(x, n)` | series, window | N日偏度 | `ts_skew(ret_1d, 20)` = 收益分布偏斜 |
| `ts_kurt(x, n)` | series, window | N日峰度 | `ts_kurt(ret_1d, 20)` = 尾部风险 |
| `delta(x, n)` | series, offset | x[t] - x[t-n] | `delta(close, 5)` = 5日价格变化 |
| `pct_change(x, n)` | series, offset | x[t]/x[t-n] - 1 | `pct_change(volume, 5)` = 5日量变化% |
| `shift(x, n)` | series, offset | x[t-n] (滞后) | `shift(close, 1)` = 昨日收盘价 |
| `decay_linear(x, n)` | series, window | 线性衰减加权均值 | `decay_linear(volume, 10)` = 近期量权重更大 |
| `decay_exp(x, n)` | series, half_life | 指数衰减加权均值 | `decay_exp(ret_1d, 5)` = EWMA收益 |
| `ts_product(x, n)` | series, window | N日连乘 | `ts_product(1+ret_1d, 5)-1` = 5日累计收益 |
| `ts_count(cond, n)` | bool series, window | N日内条件为真的天数 | `ts_count(ret_1d > 0, 20)` = 20日内上涨天数 |

### 截面算子 (cross-sectional, across all stocks on same day)

| 算子 | 参数 | 含义 | 示例 |
|------|------|------|------|
| `rank(x)` | series | 当日全市场排名 [0,1] | `rank(volume)` = 成交量排名 |
| `zscore(x)` | series | 当日全市场标准化 | `zscore(ret_5d)` = 相对涨幅 |
| `quantile(x, n)` | series, n_groups | 分N组，返回组号 | `quantile(pe_ttm, 5)` = PE分5组 |
| `demean(x)` | series | 减去截面均值 | `demean(ret_5d)` = 超额收益 |
| `neutralize(x, group)` | series, group_col | 行业中性化 | `neutralize(ret_5d, industry)` |

### 通用算子

| 算子 | 含义 |
|------|------|
| `abs(x)` | 绝对值 |
| `sign(x)` | 符号 (-1/0/+1) |
| `log(x)` | 自然对数 (guarded) |
| `power(x, p)` | x^p |
| `sqrt(x)` | 平方根 |
| `clamp(x, lo, hi)` | 截断到区间 |
| `max(x, y)` | 两值取大 |
| `min(x, y)` | 两值取小 |
| `if_then(cond, a, b)` | 条件: cond>0 → a, else → b |
| `+ - * /` | 四则运算 (infix) |
| `- (unary)` | 取负 |

**总计: ~30 个算子**

---

## 二、Features (输入特征)

### 价格类 (8个)

| Feature | 含义 |
|---------|------|
| `open` | 开盘价 |
| `high` | 最高价 |
| `low` | 最低价 |
| `close` | 收盘价 |
| `vwap` | 成交均价 (amount/volume) |
| `ret_1d` | 日收益率 |
| `ret_5d` | 5日收益率 |
| `ret_20d` | 20日收益率 |

### 量能类 (4个)

| Feature | 含义 |
|---------|------|
| `volume` | 成交量 (股) |
| `amount` | 成交额 (元) |
| `turnover_rate` | 换手率 |
| `volume_ratio` | 量比 (vol/avg_vol_20) |

### 技术指标类 (6个, 预计算)

| Feature | 含义 |
|---------|------|
| `rsi_14` | RSI 14期 |
| `bb_position` | 布林带位置 [0,1] |
| `ma20_dist` | 距20日均线% |
| `ma60_dist` | 距60日均线% |
| `atr_14` | 14日ATR |
| `obv` | OBV能量潮 |

### 基本面类 (6个)

| Feature | 含义 | 市场 |
|---------|------|------|
| `pe_ttm` | 滚动市盈率 | US+CN |
| `pb` | 市净率 | US+CN |
| `market_cap` | 总市值 | US+CN |
| `circ_market_cap` | 流通市值 | CN |
| `ps_ttm` | 滚动市销率 | US+CN |
| `dividend_yield` | 股息率 | US |

### 资金面类 (6个, A股特有)

| Feature | 含义 | 市场 |
|---------|------|------|
| `net_mf_amount` | 大单净流入 | CN |
| `large_net_in` | 超大单净流入 | CN |
| `margin_balance` | 融资余额 | CN |
| `margin_delta_5d` | 融资5日变化 | CN |
| `block_premium` | 大宗交易溢折价 | CN |
| `holder_change` | 股东增减持 | CN |

### 市场环境类 (4个)

| Feature | 含义 |
|---------|------|
| `index_ret_1d` | 基准指数日收益 |
| `index_ret_5d` | 基准指数5日收益 |
| `sector_ret_5d` | 所属行业5日收益 |
| `vix` | VIX (US only) |

**总计: ~34 个 features**

---

## 三、经典因子公式库 (按类别)

### A. K线形态类 (Qlib KBAR, 9个)

```yaml
KMID:   (close - open) / open                    # K线实体方向
KLEN:   (high - low) / open                       # K线长度
KMID2:  (close - open) / (high - low + 1e-12)    # 实体占比
KUP:    (high - max(open, close)) / open          # 上影线
KUP2:   (high - max(open, close)) / (high - low + 1e-12)
KLOW:   (min(open, close) - low) / open           # 下影线
KLOW2:  (min(open, close) - low) / (high - low + 1e-12)
KSFT:   (2*close - high - low) / open             # K线偏移
KSFT2:  (2*close - high - low) / (high - low + 1e-12)
```

### B. 动量/反转类 (20个)

```yaml
# 短期反转
REV_1D:   -ret_1d                                  # 1日反转
REV_5D:   -ret_5d                                  # 5日反转
REV_20D:  -ret_20d                                 # 20日反转

# 中期动量
MOM_60D:  pct_change(close, 60)                    # 60日动量
MOM_120D: pct_change(close, 120)                   # 半年动量

# 加速度
ACCEL:    ret_5d / shift(ret_5d, 5)                # 动量加速度
ACCEL2:   ret_5d - shift(ret_5d, 5)                # 动量加速变化

# 相对强度
RS_5_20:  ret_5d - ret_20d                         # 短期vs中期
RS_5_60:  ret_5d - pct_change(close, 60)           # 短期vs长期

# 路径依赖
MAX_DRAWDOWN_20: (close - ts_max(high, 20)) / ts_max(high, 20)
DIST_HIGH_52W:   (close - ts_max(high, 250)) / ts_max(high, 250)
DIST_LOW_52W:    (close - ts_min(low, 250)) / ts_min(low, 250)

# 均值回归
REVERSION_SIGNED: (50 - rsi_14) / 50 * 0.35 + (0.5 - bb_position) * 0.35 + (-ma20_dist/10) * 0.30

# 趋势强度
ADX:      ts_std(ret_1d, 14) / ts_mean(abs(ret_1d), 14)    # 简化ADX
TREND_STRENGTH: abs(ret_20d) / ts_std(ret_1d, 20) / sqrt(20)  # t-stat of trend
```

### C. 量价关系类 (20个)

```yaml
# 量价相关
VP_CORR_5:    ts_corr(close, volume, 5)            # 5日量价相关
VP_CORR_20:   ts_corr(close, volume, 20)           # 20日量价相关
VP_CORR_60:   ts_corr(close, volume, 60)           # 60日量价相关

# 量价背离
VP_DIV:       rank(ret_5d) * rank(-pct_change(volume, 5))  # 涨但缩量

# OBV (能量潮)
OBV_SLOPE:    pct_change(ts_sum(if_then(ret_1d, volume, -volume), 20), 5)

# 成交量变化
VOL_SURGE:    volume / ts_mean(volume, 20)          # 量比
VOL_SHRINK:   -volume / ts_mean(volume, 20)         # 缩量程度
VOL_STD:      ts_std(volume, 20) / ts_mean(volume, 20)  # 量的波动性

# VWAP相关
VWAP_DIST:    (close - vwap) / vwap                 # 收盘vs均价
VWAP_TREND:   pct_change(vwap, 5)                   # VWAP趋势

# 换手率
TURN_ACCEL:   turnover_rate / ts_mean(turnover_rate, 20)
TURN_HIGH:    turnover_rate / ts_max(turnover_rate, 60)

# 高低价量
HIGH_VOL:     ts_corr(high, volume, 10)             # 高价时放量?
LOW_VOL:      ts_corr(low, volume, 10)              # 低价时放量?

# 资金集中度
AMIHUD:       abs(ret_1d) / (amount + 1e-9)         # Amihud非流动性
```

### D. 波动率类 (15个)

```yaml
# 已实现波动率
RVOL_5:       ts_std(ret_1d, 5)                     # 5日波动率
RVOL_20:      ts_std(ret_1d, 20)                    # 20日波动率
RVOL_60:      ts_std(ret_1d, 60)                    # 60日波动率

# 波动率变化
VOL_CHANGE:   ts_std(ret_1d, 5) / ts_std(ret_1d, 20)  # 短期vs中期波动
SQUEEZE:      ts_std(ret_1d, 20) / ts_std(ret_1d, 60)  # 波动率压缩

# Garman-Klass
GK_VOL:       sqrt(0.5*(log(high/low))^2 - (2*log(2)-1)*(log(close/open))^2)

# Parkinson
PARK_VOL:     log(high/low) / (2*sqrt(log(2)))

# 下行波动
DOWNVOL:      ts_std(min(ret_1d, 0), 20)            # 只算下跌的波动
UPDOWN_RATIO: ts_std(max(ret_1d,0), 20) / (ts_std(min(ret_1d,0), 20) + 1e-9)

# 尾部风险
TAIL_5:       ts_min(ret_1d, 20)                    # 20日最大单日跌幅
SKEW_20:      ts_skew(ret_1d, 20)                   # 收益偏度
KURT_20:      ts_kurt(ret_1d, 20)                   # 收益峰度

# ATR
ATR_14:       ts_mean(high - low, 14)               # 14日ATR
ATR_NORM:     ts_mean(high - low, 14) / close       # 归一化ATR
```

### E. 估值/基本面类 (10个)

```yaml
EP:           1 / pe_ttm                            # 盈利收益率
BP:           1 / pb                                # 账面价值率
SP:           1 / ps_ttm                            # 销售收益率
SIZE:         log(market_cap)                       # 对数市值
SIZE_NL:      log(market_cap)^3                     # 市值非线性

# 相对估值
PE_RANK:      rank(-pe_ttm)                         # PE从低到高排名
PB_RANK:      rank(-pb)                             # PB从低到高排名

# 估值+动量交叉
VALUE_MOM:    rank(-pe_ttm) * rank(ret_20d)         # 低估值+上涨
GARP:         rank(-pe_ttm) * rank(ret_60d)         # 合理价格成长
```

### F. 资金面类 (A股特有, 12个)

```yaml
# 大单
BIGORDER_NET: rank(net_mf_amount / amount)          # 大单净流入占比
BIGORDER_5D:  rank(ts_sum(net_mf_amount, 5))        # 5日累计大单
BIGORDER_ACC: pct_change(ts_sum(net_mf_amount, 5), 5)  # 大单加速

# 融资
MARGIN_RATIO: margin_balance / (circ_market_cap + 1e-9)  # 融资占流通比
MARGIN_DELTA: rank(pct_change(margin_balance, 5))   # 融资变化排名
MARGIN_MOM:   pct_change(margin_balance, 20)        # 融资20日动量

# 大宗交易
BLOCK_PREM:   rank(block_premium)                   # 大宗溢价排名

# 股东行为
HOLDER_NET:   rank(holder_change)                   # 增减持排名

# 组合信号
SMART_MONEY:  rank(net_mf_amount) * rank(-turnover_rate)  # 大单买+缩量=聪明钱
PANIC_SELL:   rank(-net_mf_amount) * rank(turnover_rate)  # 大单卖+放量=恐慌
MARGIN_CONFIRM: rank(pct_change(margin_balance, 5)) * rank(ret_5d)  # 融资确认方向
```

### G. 时间/日历类 (5个)

```yaml
DAY_OF_WEEK:  day_of_week                           # 周几效应
MONTH_END:    if_then(days_to_month_end < 3, 1, 0)  # 月末效应
QUARTER_END:  if_then(days_to_quarter_end < 5, 1, 0)  # 季末效应
TURN_OF_YEAR: if_then(month == 1, 1, 0)             # 一月效应
HOLIDAY_PRE:  if_then(days_to_holiday < 2, 1, 0)    # 节前效应(A股)
```

### H. 市场微观结构类 (10个)

```yaml
# 价格位置
RSV:          (close - ts_min(low, 20)) / (ts_max(high, 20) - ts_min(low, 20) + 1e-12)
IMAX_20:      ts_argmax(high, 20) / 20              # 最高价在20日内位置
IMIN_20:      ts_argmin(low, 20) / 20               # 最低价在20日内位置
IMXD_20:      (ts_argmax(high, 20) - ts_argmin(low, 20)) / 20  # 高低差位置

# 涨跌天数
UP_RATIO_20:  ts_count(ret_1d > 0, 20) / 20         # 20日上涨天数占比
DOWN_RATIO_20: ts_count(ret_1d < 0, 20) / 20        # 20日下跌天数占比
UP_DOWN_DIFF: (ts_count(ret_1d > 0, 20) - ts_count(ret_1d < 0, 20)) / 20

# 正负收益不对称
POS_SUM_20:   ts_sum(max(ret_1d, 0), 20)            # 20日正收益之和
NEG_SUM_20:   ts_sum(min(ret_1d, 0), 20)            # 20日负收益之和
PN_RATIO:     ts_sum(max(ret_1d,0), 20) / (-ts_sum(min(ret_1d,0), 20) + 1e-9)
```

---

## 四、总计

| 类别 | 算子数 | 因子数 |
|------|--------|--------|
| 时序算子 | 19 | - |
| 截面算子 | 5 | - |
| 通用算子 | 11+ | - |
| K线形态 | - | 9 |
| 动量/反转 | - | 15 |
| 量价关系 | - | 15 |
| 波动率 | - | 15 |
| 估值/基本面 | - | 10 |
| 资金面(A股) | - | 12 |
| 时间/日历 | - | 5 |
| 微观结构 | - | 10 |
| **总计** | **~35** | **~91** |

加上 Qlib 的 PRICE 40个 + VOLUME 5个 滚动窗口变体，可以轻松扩展到 **150+**。

---

## 五、Agent 使用策略

Agent 不需要记住 150 个因子。它需要：

1. **知道有哪些类别**（8 大类）
2. **知道每类的代表性因子**（每类 2-3 个经典的）
3. **知道 DSL 算子**（用算子自由组合新因子）
4. **知道当前市场 regime**（决定探索哪类因子）

```
Agent 在 trending 态 → 重点探索动量类 + 量价确认类
Agent 在 MR 态 → 重点探索反转类 + 波动率类
Agent 在 noisy 态 → 重点探索微观结构类 + 时间效应类
```

---

## 六、实现优先级

```
P0 (MVP): K线9 + 动量反转10 + 量价10 + 波动率5 = ~34 个基础因子
P1: 资金面12 (A股) + 估值10 + 微观结构10 = ~32 个
P2: 时间效应5 + Qlib滚动窗口变体50+ = 55+
```

MVP 只需要 P0 的 34 个因子 + 35 个算子，agent 就有足够的搜索空间了。
