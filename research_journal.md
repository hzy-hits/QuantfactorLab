# Factor Research Journal

## Current Understanding

**CN market**: ~5500 stocks, 300+ trading days. Available features: close, open, high, low, volume, amount, vwap (derived). NO fundamentals, NO flow data in current DB.
**US market**: ~500 stocks, 500+ trading days. Available features: close, open, high, low, volume. NO amount, NO fundamentals.

**Gate thresholds**:
- CN: IC>0.01(abs), IC_IR>0.2, turnover<0.50, mono>0.6(abs), corr<0.7
- US: IC>0.02(abs), IC_IR>0.3, turnover<0.40, mono>0.7(abs), corr<0.7

## Confirmed Patterns

### Works (promoted)
1. **Volume/amount stability (CV)** at 20d window: `rank(-ts_std(X, 20) / (ts_mean(X, 20) + eps))` — Stable volume/amount stocks have better forward returns. Works in both CN and US. Economic logic: consistent institutional ownership, less noise.
2. **Range-volume decorrelation**: `rank(-ts_corr(high-low, volume, 20))` — Stocks where range and volume are decoupled perform better. CN only.
3. **Trend quality (high-low corr * low vol)**: `rank(ts_corr(high, low, N)) * rank(-ts_std(ret_1d, N))` — Coherent trending + low volatility = quality momentum. US only (N=40 best). CN lacks monotonicity for this pattern.
4. **Volume stability at 60d**: Same CV pattern with longer window. US only.
5. **Range-price decorrelation**: `rank(-ts_corr(high-low, close, 60))` — Stocks with orderly advances (range narrows as price rises) perform better. US only. IC=0.049, IC_IR=0.390.
6. **Return-volume divergence**: `rank(-ts_corr(ret_1d, delta(volume,1), 40))` — Volume increases on down-days signal accumulation. US only. IC=0.028, IC_IR=0.342.
7. **Bullish close quality**: `rank(ts_mean(close/high, 40)) * rank(-ts_std(ret_1d, 40))` — Stocks consistently closing near highs with low vol = institutional accumulation. US only. IC=0.056, IC_IR=0.396.
8. **Smart trend composite**: `rank(-ts_corr(ret_1d, delta(volume,1), 20)) * rank(ts_corr(high, low, 40))` — Smart money accumulation + coherent trending = institutional positioning signal. US only. IC=0.043, IC_IR=0.499.

### Strong IS signal but fails monotonicity (CN-specific problem)
- `rank(-ts_mean(high/low - 1, 10))`: IC=0.056, IC_IR=0.302, **mono=0.000**
- `rank(-(ts_max(high,5)-ts_min(low,5))/close)`: IC=0.062, IC_IR=0.370, **mono=0.000**
- `rank(pct_change(close,60)) * rank(-ts_std(ret_1d,60))`: IC=0.071, IC_IR=0.269, **mono=0.000**
- `rank(-ts_corr(close,volume,20))`: IC=0.035, IC_IR=0.271, **mono varies**
- `rank(-ts_skew(ret_1d,20)) * rank(-ts_std(ret_1d,20))`: IC=0.050, IC_IR=0.342, **mono=0.000**

### Strong IS signal but fails OOS (CN)
- `rank(-pct_change(close, 120))`: IC=0.048, IC_IR=0.229, mono=0.900. **OOS FAIL**
- `rank(-pct_change(close,120)/(ts_std(ret_1d,120)+0.001))`: IC=0.040, IC_IR=0.215, mono=1.000. **OOS FAIL**
- `rank(-ts_corr(close, delta(close,5), 20))`: IC=0.026, IC_IR=0.229, mono=1.000. **OOS FAIL**

### Strong in US but not CN
- `rank(ts_corr(high, low, 60))`: US IC=0.059, IC_IR=0.614, but US mono=-0.15; CN IC≈0
- `rank(ts_corr(high, low, 40)) * rank(-ts_std(ret_1d, 40))`: US passes all, CN fails mono

## Key Insights

### Monotonicity is the hardest gate in CN
- Many factors have IC=0.03-0.07 and IC_IR>0.3 but **zero quintile monotonicity**
- Only 20d stability/CV measures and 120d+ reversal produce adequate monotonicity
- Hypothesis: CN A-share cross-section has non-linear factor-return relationships (extremes matter but middle quintiles are random)

### What structures produce monotonicity?
- **Relationship measures** (correlation, CV ratio) > level measures (std, mean)
- **20d window** is special for monotonicity — 10d and 40d fail
- **Long-horizon reversal** (120d) has excellent monotonicity but doesn't survive OOS
- Simple `rank(feature)` generally lacks monotonicity in CN

### US vs CN
- US has stronger factor-return relationships (higher IC, IC_IR, better monotonicity)
- Volume stability works in both markets — universal signal
- High-low correlation coherence (trend quality) is US-specific
- CN OOS is very difficult — 3 factors passed IS gates but failed OOS

### Return autocorrelation — independent but weak
- `rank(-ts_corr(ret_1d, shift(ret_1d,1), 40))`: max_corr=0.12 with existing factors (very independent!)
- IC=0.028, IC_IR=0.377 in CN but mono=0.35 (below threshold)
- Worth revisiting with better monotonicity-inducing transforms

## Open Questions

- Why does the 20d window specifically produce monotonicity when 10d and 40d don't?
- Can non-linear transforms (quantile bucketing, conditional splits) fix mono=0 factors?
- Would sector neutralization improve monotonicity?
- Is the CN OOS failure systematic (regime change) or random?
- Can the autocorrelation signal be made monotonic via conditional structures?

## Exploration Map

```
feature        | delta | pct_chg | ts_mean | ts_std | rank | zscore | ts_corr | combined
---------------|-------|---------|---------|--------|------|--------|---------|--------
close          |   .   |  120d!  |   .     |  20d.  |  .   |        | w/vol:+ | w/delta
volume         |   .   |    .    |   .     | CV:+++ |  .   |        | w/rng:+ | *low_vol
amount         |   .   |    .    |   .     | CV:+++ |  .   |        |   .     |
high-low       |   .   |    .    | +++!    | CV:+   |  .   |        | w/vol:+ |
open           |       |         |         |        |      |        | w/vol:. |
vwap           |       |         |         | .      |      |        |         |
ret_1d         |       |         |         | .      |      |        | autocor |
high           |       |         |         |        |      |        | w/low:++|
low            |       |         |         |        |      |        |         |
```
Legend: . = tested weak, + = decent IC, ++ = strong IC, +++ = promoted, ! = strong but mono fail

## Data Requests

1. **CN fundamentals (pe_ttm, pb, market_cap)**: Not in daily_cn DB. Would enable value/size factors.
2. **CN flow data (net_mf_amount, margin_balance)**: Not available. Would enable smart money factors.
3. **US amount/turnover data**: Not in prices_daily. Would enable more volume-price analysis.
4. **US bug: eval_factor.py `_load_prices_raw` creates duplicate 'close' column** because `SELECT *` gets both `close` and `adj_close`, then renames adj_close→close. Fixed manually via cache patch this session, but needs code fix for persistence.

## Session Log

### Session 1 (2026-03-20, 21:00-23:30 CST)
- **Experiments**: ~70 (CN: ~40, US: ~30)
- **Promoted**: 9 total
  - CN (2): `amount_stability_20`, `range_vol_decorr_20`
  - US (7): `volume_stability_20`, `volume_stability_60`, `trend_quality_40`, `range_price_decorr_60`, `ret_vol_divergence_40`, `bullish_close_quality_40`, `smart_trend_20_40`
- **IS pass, OOS fail**: 3 (all CN — 120d reversal variants, close-delta corr)
- **IS pass, corr fail**: ~5 (volume variants too close to each other)
- **Key discoveries**:
  - Volume/amount stability CV is universal cross-market alpha
  - High-low correlation * low volatility = "trend quality" (US only)
  - Range-price decorrelation predicts orderly advances (US only)
  - Return-volume divergence signals smart money accumulation (US only)
  - Price-level correlations (high-low, close-open) have extreme IC_IR (0.4-0.6) in US but fail monotonicity
- **Key frustration**: CN has many strong-IC factors (0.05-0.07) that all fail monotonicity
- **US cache fix**: Patched duplicate close column in .cache/us_prices.pkl
- **Final composite status**:
  - CN: IC=0.022, IC_IR=0.189, 8 factors
  - US: IC=0.052, IC_IR=0.451, 17 factors (massive improvement from IC=0.033, IC_IR=0.358)
- **Key factors by weight in US composite**: smart_trend_20_40 (0.09), bullish_close_quality_40 (0.07), volume_stability_20 (0.07), trend_quality_40 (0.07), range_price_decorr_60 (0.07)
