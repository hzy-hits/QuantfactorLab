# CN Batch 2 Mining Report (Illiquidity-Safe)

Generated: 2026-03-18
Data: 2024-03-13 to 2026-03-17
Universe: 5,569 A-share symbols
Trading days: 486
Forward horizon: 5D

Notes:
- Evaluated 25 factors on A-share data.
- The user-provided list contained 24 formulas, so `momentum_quality` was added as factor 25.
- Excluded Amihud-style illiquidity constructions such as `abs(ret_1d) / volume`, `abs(ret_1d) / amount`, or equivalent return-over-liquidity ratios.
- One `ConstantInputWarning` was raised inside daily IC calculation on dates where a factor cross-section became constant; the batch still completed.

## Factor Set

```text
vp_div_3d          = rank(-ts_corr(close, volume, 3))
vp_div_10d         = rank(-ts_corr(close, volume, 10))
vp_div_change      = rank(ts_corr(close, volume, 5) - ts_corr(close, volume, 20))
vol_price_asym     = rank(ts_corr(close, volume, 10)) * rank(-ret_5d)
shrink_surge_3d    = rank(-ts_std(volume, 10) / ts_mean(volume, 10)) * rank(volume / ts_mean(volume, 3))
shrink_surge_hl    = rank(-ts_std(high - low, 20) / ts_mean(high - low, 20)) * rank((high - low) / ts_mean(high - low, 5))
quiet_breakout     = rank(-ts_std(close, 20) / ts_mean(close, 20)) * rank(abs(ret_1d))
rev_5d_vol_confirm = rank(-ret_5d) * rank(volume / ts_mean(volume, 5))
rev_10d            = rank(-pct_change(close, 10))
rev_gap_fill       = rank(-(close - open) / close) * rank(-ret_5d)
oversold_bounce    = rank(ts_min(low, 10) / close - 1) * rank(ret_1d)
hammer             = rank((close - low) / (high - low + 0.01)) * rank(-ret_5d)
engulfing          = rank((close - open) / (high - low + 0.01) - shift((close - open) / (high - low + 0.01), 1))
doji_reversal      = rank(-abs(close - open) / (high - low + 0.01)) * rank(abs(ret_5d))
upper_wick_sell    = rank((high - close) / (high - low + 0.01)) * rank(ret_5d)
vol_squeeze_5_20   = rank(-ts_std(ret_1d, 5) / (ts_std(ret_1d, 20) + 0.001))
vol_squeeze_10_60  = rank(-ts_std(ret_1d, 10) / (ts_std(ret_1d, 60) + 0.001))
realized_vs_range  = rank(-ts_std(ret_1d, 20) / ((ts_max(high, 20) - ts_min(low, 20)) / close + 0.001))
higher_lows        = rank(ts_min(low, 5) / ts_min(low, 20) - 1)
lower_highs        = rank(-(ts_max(high, 5) / ts_max(high, 20) - 1))
range_contraction  = rank(-(ts_max(high, 5) - ts_min(low, 5)) / (ts_max(high, 20) - ts_min(low, 20) + 0.001))
amount_surge       = rank(amount / ts_mean(amount, 20))
amount_trend       = rank(ts_mean(amount, 5) / ts_mean(amount, 20))
smart_money_rev    = rank(amount / ts_mean(amount, 20)) * rank(-ret_5d)
momentum_quality   = rank(ret_20d) * rank(-ts_std(ret_1d, 20))
```

## Top Signals

- `vp_div_10d`: IC `0.0282`, IR `0.189`
- `shrink_surge_hl`: IC `0.0263`, IR `0.223`
- `oversold_bounce`: IC `0.0233`, IR `0.132`
- `smart_money_rev`: IC `0.0211`, IR `0.150`
- `range_contraction`: IC `0.0205`, IR `0.167`

## Full Ranking

| Factor | IC | IC_IR | Q5-Q1% | Mono | Status |
|--------|----:|------:|-------:|-----:|:------:|
| vp_div_10d | 0.0282 | 0.189 | 0.041 | 0.10 | ✅ |
| shrink_surge_hl | 0.0263 | 0.223 | 0.187 | 0.70 | ✅ |
| oversold_bounce | 0.0233 | 0.132 | -0.164 | -0.90 | ✅ |
| smart_money_rev | 0.0211 | 0.150 | 0.114 | 0.70 | ✅ |
| range_contraction | 0.0205 | 0.167 | 0.013 | 0.50 | ✅ |
| momentum_quality | 0.0203 | 0.096 | -0.123 | -0.60 | ❌ |
| rev_5d_vol_confirm | 0.0175 | 0.119 | 0.002 | 0.60 | ✅ |
| amount_surge | -0.0174 | -0.131 | 0.094 | 0.20 | ✅ |
| rev_gap_fill | 0.0172 | 0.092 | -0.022 | -0.60 | ❌ |
| rev_10d | 0.0157 | 0.079 | -0.155 | -0.90 | ❌ |
| vol_squeeze_10_60 | 0.0155 | 0.109 | -0.097 | -0.30 | ✅ |
| vol_squeeze_5_20 | 0.0150 | 0.127 | -0.083 | -1.00 | ✅ |
| hammer | 0.0141 | 0.090 | -0.016 | -0.40 | ❌ |
| doji_reversal | -0.0140 | -0.118 | 0.002 | 0.50 | ✅ |
| shrink_surge_3d | 0.0137 | 0.131 | -0.040 | -0.70 | ✅ |
| amount_trend | -0.0134 | -0.099 | 0.152 | 0.60 | ❌ |
| lower_highs | 0.0084 | 0.046 | 0.245 | 0.30 | ❌ |
| vp_div_3d | 0.0078 | 0.073 | -0.120 | -0.90 | ❌ |
| quiet_breakout | 0.0077 | 0.057 | -0.105 | -1.00 | ❌ |
| engulfing | -0.0064 | -0.034 | 0.027 | 0.10 | ❌ |
| vp_div_change | -0.0061 | -0.053 | -0.061 | -0.90 | ❌ |
| vol_price_asym | 0.0051 | 0.032 | 0.006 | 0.30 | ❌ |
| higher_lows | -0.0033 | -0.018 | 0.982 | 1.00 | ❌ |
| realized_vs_range | 0.0012 | 0.010 | -0.201 | -0.40 | ❌ |
| upper_wick_sell | 0.0006 | 0.004 | 0.104 | 1.00 | ❌ |
