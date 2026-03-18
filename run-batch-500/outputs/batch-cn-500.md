The batch run completed successfully. `--max-factors 500` produced `377` unique formulas, and all `377` evaluated with `0` errors. The report is [batch_cn_500.md](/home/ivena/coding/python/factor-lab/reports/batch_cn_500.md#L1). During evaluation there were some `ConstantInputWarning` messages, but they did not stop the run.

Top 10 from [batch_cn_500.md](/home/ivena/coding/python/factor-lab/reports/batch_cn_500.md#L10):

| # | Formula | IC | IC_IR | Q5-Q1% | Mono |
|---|---|---:|---:|---:|---:|
| 1 | `rank(-ts_corr(close, volume, 20)) * rank(-ret_5d)` | 0.0321 | 0.192 | 0.025 | 0.30 |
| 2 | `rank(delta(volume, 10))` | -0.0260 | -0.221 | -0.174 | -1.00 |
| 3 | `rank(-delta(volume, 10))` | 0.0260 | 0.221 | 0.174 | 1.00 |
| 4 | `rank(-ts_corr(close, volume, 10)) * rank(-ret_5d)` | 0.0296 | 0.187 | 0.024 | 0.40 |
| 5 | `rank(-ts_corr(close, volume, 10))` | 0.0282 | 0.189 | 0.041 | 0.10 |
| 6 | `rank(ts_corr(close, volume, 10))` | -0.0282 | -0.189 | -0.040 | -0.10 |
| 7 | `rank(volume / ts_min(volume, 10) - 1)` | -0.0245 | -0.208 | -0.016 | -0.70 |
| 8 | `rank(-ts_corr(close, volume, 5))` | 0.0246 | 0.205 | 0.040 | 0.00 |
| 9 | `rank(ts_corr(close, volume, 5))` | -0.0246 | -0.205 | -0.040 | 0.00 |
| 10 | `rank(-ts_corr(close, volume, 5)) * rank(-ret_5d)` | 0.0281 | 0.179 | 0.040 | 0.40 |

The main pattern is clear: the top ranks are dominated by price-volume correlation divergence signals and short-horizon volume change/reversal signals. The best raw IC is rank #1, but the strongest quintile spread is rank #3, `rank(-delta(volume, 10))`, with `Q5-Q1 = 0.174`. Per the gate analysis in [batch_cn_500.md](/home/ivena/coding/python/factor-lab/reports/batch_cn_500.md#L44), ranks `#2`, `#3`, and `#7` are the only top-10 entries that pass all three gates (`IC`, `IR`, `Mono`); `#8` and `#9` pass `IC` and `IR` but fail monotonicity. Overall distribution is decent breadth but not exceptional strength: `90` factors have `IC > 0.02`, `261` have `IC > 0.01`, and `0` exceed `IC_IR > 0.3` [batch_cn_500.md](/home/ivena/coding/python/factor-lab/reports/batch_cn_500.md#L77).