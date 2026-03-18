# Batch Factor Mining — CN

Generated: 2026-03-18 09:47
Factors tested: 377
Valid results: 377
Errors: 0

## Top 30 Factors

| # | IC | IC_IR | Q5-Q1% | Mono | Formula |
|---|-----|-------|--------|------|---------|
| 1 | 0.0321 | 0.192 | 0.025 | 0.30 | `rank(-ts_corr(close, volume, 20)) * rank(-ret_5d)` |
| 2 | -0.0260 | -0.221 | -0.174 | -1.00 | `rank(delta(volume, 10))` |
| 3 | 0.0260 | 0.221 | 0.174 | 1.00 | `rank(-delta(volume, 10))` |
| 4 | 0.0296 | 0.187 | 0.024 | 0.40 | `rank(-ts_corr(close, volume, 10)) * rank(-ret_5d)` |
| 5 | 0.0282 | 0.189 | 0.041 | 0.10 | `rank(-ts_corr(close, volume, 10))` |
| 6 | -0.0282 | -0.189 | -0.040 | -0.10 | `rank(ts_corr(close, volume, 10))` |
| 7 | -0.0245 | -0.208 | -0.016 | -0.70 | `rank(volume / ts_min(volume, 10) - 1)` |
| 8 | 0.0246 | 0.205 | 0.040 | 0.00 | `rank(-ts_corr(close, volume, 5))` |
| 9 | -0.0246 | -0.205 | -0.040 | 0.00 | `rank(ts_corr(close, volume, 5))` |
| 10 | 0.0281 | 0.179 | 0.040 | 0.40 | `rank(-ts_corr(close, volume, 5)) * rank(-ret_5d)` |
| 11 | -0.0217 | -0.195 | -0.047 | -0.20 | `rank(delta(volume, 60))` |
| 12 | 0.0217 | 0.195 | 0.047 | 0.20 | `rank(-delta(volume, 60))` |
| 13 | -0.0276 | -0.153 | -0.152 | -0.90 | `rank(delta(ret_5d, 20))` |
| 14 | 0.0276 | 0.153 | 0.153 | 0.90 | `rank(-delta(ret_5d, 20))` |
| 15 | -0.0257 | -0.140 | -0.180 | -0.90 | `rank(delta(ret_5d, 10))` |
| 16 | 0.0257 | 0.140 | 0.180 | 0.90 | `rank(-delta(ret_5d, 10))` |
| 17 | 0.0191 | 0.176 | 0.102 | 0.80 | `rank(-ts_std(volume, 20) / ts_mean(volume, 20)) * rank(volum` |
| 18 | 0.0218 | 0.151 | 0.122 | 0.30 | `rank(-ret_5d) * rank(volume / ts_mean(volume, 20))` |
| 19 | 0.0228 | 0.133 | -0.099 | 0.00 | `rank(-ts_corr(close, volume, 20))` |
| 20 | -0.0228 | -0.133 | 0.100 | 0.00 | `rank(ts_corr(close, volume, 20))` |
| 21 | -0.0219 | -0.133 | -0.124 | -0.90 | `rank(high / ts_min(high, 5) - 1)` |
| 22 | 0.0202 | 0.126 | 0.004 | 0.10 | `rank(-ret_5d) * rank(-volume / ts_mean(volume, 10))` |
| 23 | 0.0188 | 0.134 | 0.005 | 0.40 | `rank(ts_mean(volume, 60))` |
| 24 | -0.0217 | -0.116 | -0.144 | -0.70 | `rank(delta(ret_1d, 20))` |
| 25 | 0.0217 | 0.116 | 0.144 | 0.70 | `rank(-delta(ret_1d, 20))` |
| 26 | -0.0209 | -0.120 | -0.038 | -0.30 | `rank(open / ts_min(open, 5) - 1)` |
| 27 | -0.0171 | -0.142 | -0.082 | -0.70 | `rank(delta(volume, 5))` |
| 28 | 0.0171 | 0.142 | 0.082 | 0.70 | `rank(-delta(volume, 5))` |
| 29 | 0.0193 | 0.123 | 0.009 | 0.20 | `rank(-ret_5d) * rank(-volume / ts_mean(volume, 5))` |
| 30 | -0.0208 | -0.113 | -0.059 | -0.20 | `rank(delta(ret_5d, 60))` |

## Gate Analysis (top 30)

  1. **d2_20_364**: ✅IC ❌IR ❌Mono — VP divergence + decline (20d)
  2. **d1_volume_10_73**: ✅IC ✅IR ✅Mono — N-day change (volume, 10d)
  3. **d1_volume_10_101**: ✅IC ✅IR ✅Mono — N-day reversal (volume, 10d)
  4. **d2_10_363**: ✅IC ❌IR ❌Mono — VP divergence + decline (10d)
  5. **vpcorr_10**: ✅IC ❌IR ❌Mono — VP divergence 10d
  6. **vpcorr_neg_10**: ✅IC ❌IR ❌Mono — VP alignment 10d
  7. **d1_volume_10_325**: ✅IC ✅IR ✅Mono — distance from N-day low (volume, 10d)
  8. **vpcorr_5**: ✅IC ✅IR ❌Mono — VP divergence 5d
  9. **vpcorr_neg_5**: ✅IC ✅IR ❌Mono — VP alignment 5d
  10. **d2_5_362**: ✅IC ❌IR ❌Mono — VP divergence + decline (5d)
  11. **d1_volume_60_75**: ✅IC ❌IR ❌Mono — N-day change (volume, 60d)
  12. **d1_volume_60_103**: ✅IC ❌IR ❌Mono — N-day reversal (volume, 60d)
  13. **d1_ret_5d_20_82**: ✅IC ❌IR ✅Mono — N-day change (ret_5d, 20d)
  14. **d1_ret_5d_20_110**: ✅IC ❌IR ✅Mono — N-day reversal (ret_5d, 20d)
  15. **d1_ret_5d_10_81**: ✅IC ❌IR ✅Mono — N-day change (ret_5d, 10d)
  16. **d1_ret_5d_10_109**: ✅IC ❌IR ✅Mono — N-day reversal (ret_5d, 10d)
  17. **d2_20_372**: ✅IC ❌IR ✅Mono — shrink then surge (20d)
  18. **d2_20_356**: ✅IC ❌IR ❌Mono — reversal + volume spike (20d)
  19. **vpcorr_20**: ✅IC ❌IR ❌Mono — VP divergence 20d
  20. **vpcorr_neg_20**: ✅IC ❌IR ❌Mono — VP alignment 20d
  21. **d1_high_5_312**: ✅IC ❌IR ✅Mono — distance from N-day low (high, 5d)
  22. **d2_10_351**: ✅IC ❌IR ❌Mono — reversal + shrinking vol (10d)
  23. **d1_volume_60_187**: ✅IC ❌IR ❌Mono — N-day moving average (volume, 60d)
  24. **d1_ret_1d_20_78**: ✅IC ❌IR ✅Mono — N-day change (ret_1d, 20d)
  25. **d1_ret_1d_20_106**: ✅IC ❌IR ✅Mono — N-day reversal (ret_1d, 20d)
  26. **d1_open_5_320**: ✅IC ❌IR ❌Mono — distance from N-day low (open, 5d)
  27. **d1_volume_5_72**: ✅IC ❌IR ✅Mono — N-day change (volume, 5d)
  28. **d1_volume_5_100**: ✅IC ❌IR ✅Mono — N-day reversal (volume, 5d)
  29. **d2_5_350**: ✅IC ❌IR ❌Mono — reversal + shrinking vol (5d)
  30. **d1_ret_5d_60_83**: ✅IC ❌IR ❌Mono — N-day change (ret_5d, 60d)

## Distribution
  IC > 0.02: 90
  IC > 0.01: 261
  IC_IR > 0.3: 0
  Monotonicity > 0.7: 171