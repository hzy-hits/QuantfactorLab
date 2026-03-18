# Batch Factor Mining — US

Generated: 2026-03-18 09:46
Factors tested: 377
Valid results: 377
Errors: 0

## Top 30 Factors

| # | IC | IC_IR | Q5-Q1% | Mono | Formula |
|---|-----|-------|--------|------|---------|
| 1 | 0.0713 | 0.954 | 0.538 | 0.80 | `rank(volume)` |
| 2 | -0.0713 | -0.954 | -0.536 | -0.80 | `rank(-volume)` |
| 3 | 0.0693 | 0.926 | 0.443 | 0.80 | `rank(ts_mean(volume, 5))` |
| 4 | 0.0673 | 0.895 | 0.404 | 0.80 | `rank(ts_mean(volume, 10))` |
| 5 | 0.0647 | 0.856 | 0.653 | 1.00 | `rank(ts_mean(volume, 20))` |
| 6 | 0.0637 | 0.831 | 1.035 | 0.70 | `rank(ts_mean(volume, 60))` |
| 7 | 0.0588 | 0.841 | 0.486 | 0.80 | `rank(ts_std(volume, 5))` |
| 8 | -0.0588 | -0.841 | -0.487 | -0.80 | `rank(-ts_std(volume, 5))` |
| 9 | 0.0560 | 0.798 | 0.537 | 1.00 | `rank(ts_std(volume, 10))` |
| 10 | -0.0560 | -0.798 | -0.539 | -1.00 | `rank(-ts_std(volume, 10))` |
| 11 | 0.0485 | 0.672 | 0.776 | 1.00 | `rank(ts_std(volume, 20))` |
| 12 | -0.0485 | -0.672 | -0.776 | -1.00 | `rank(-ts_std(volume, 20))` |
| 13 | 0.0380 | 0.477 | 1.106 | 1.00 | `rank(ts_std(volume, 60))` |
| 14 | -0.0380 | -0.477 | -1.103 | -1.00 | `rank(-ts_std(volume, 60))` |
| 15 | 0.0388 | 0.331 | 0.010 | 0.40 | `rank(ret_5d) * rank(volume / ts_mean(volume, 60))` |
| 16 | 0.0367 | 0.339 | -0.098 | 0.00 | `rank(ret_5d) * rank(volume / ts_mean(volume, 20))` |
| 17 | 0.0362 | 0.282 | 0.031 | 0.00 | `rank((close - ts_min(low, 60)) / (ts_max(high, 60) - ts_min(` |
| 18 | 0.0296 | 0.296 | -0.352 | 0.20 | `rank(-ts_corr(close, volume, 40))` |
| 19 | -0.0296 | -0.296 | 0.354 | -0.20 | `rank(ts_corr(close, volume, 40))` |
| 20 | 0.0195 | 0.298 | -0.025 | 0.00 | `rank(-ts_corr(close, volume, 3))` |
| 21 | -0.0195 | -0.298 | 0.024 | 0.00 | `rank(ts_corr(close, volume, 3))` |
| 22 | 0.0280 | 0.196 | 0.220 | 0.10 | `rank(delta(close, 60))` |
| 23 | 0.0280 | 0.196 | 0.237 | 0.10 | `rank(delta(open, 60))` |
| 24 | -0.0280 | -0.196 | -0.222 | -0.10 | `rank(-delta(close, 60))` |
| 25 | -0.0280 | -0.196 | -0.237 | -0.10 | `rank(-delta(open, 60))` |
| 26 | 0.0276 | 0.194 | 0.233 | 0.10 | `rank(delta(high, 60))` |
| 27 | -0.0276 | -0.194 | -0.233 | -0.10 | `rank(-delta(high, 60))` |
| 28 | 0.0274 | 0.192 | 0.237 | 0.10 | `rank(delta(low, 60))` |
| 29 | -0.0274 | -0.192 | -0.235 | -0.10 | `rank(-delta(low, 60))` |
| 30 | 0.0224 | 0.159 | 0.244 | 0.40 | `rank(ts_mean(ret_5d, 60))` |

## Gate Analysis (top 30)

  1. **d1_volume_5_16**: ✅IC ✅IR ✅Mono — raw cross-sectional rank (volume, 5d)
  2. **d1_volume_5_44**: ✅IC ✅IR ✅Mono — inverted rank (volume, 5d)
  3. **d1_volume_5_184**: ✅IC ✅IR ✅Mono — N-day moving average (volume, 5d)
  4. **d1_volume_10_185**: ✅IC ✅IR ✅Mono — N-day moving average (volume, 10d)
  5. **d1_volume_20_186**: ✅IC ✅IR ✅Mono — N-day moving average (volume, 20d)
  6. **d1_volume_60_187**: ✅IC ✅IR ✅Mono — N-day moving average (volume, 60d)
  7. **d1_volume_5_212**: ✅IC ✅IR ✅Mono — N-day volatility (volume, 5d)
  8. **d1_volume_5_240**: ✅IC ✅IR ✅Mono — N-day low volatility (volume, 5d)
  9. **d1_volume_10_213**: ✅IC ✅IR ✅Mono — N-day volatility (volume, 10d)
  10. **d1_volume_10_241**: ✅IC ✅IR ✅Mono — N-day low volatility (volume, 10d)
  11. **d1_volume_20_214**: ✅IC ✅IR ✅Mono — N-day volatility (volume, 20d)
  12. **d1_volume_20_242**: ✅IC ✅IR ✅Mono — N-day low volatility (volume, 20d)
  13. **d1_volume_60_215**: ✅IC ✅IR ✅Mono — N-day volatility (volume, 60d)
  14. **d1_volume_60_243**: ✅IC ✅IR ✅Mono — N-day low volatility (volume, 60d)
  15. **d2_60_361**: ✅IC ✅IR ❌Mono — momentum + volume confirm (60d)
  16. **d2_20_360**: ✅IC ✅IR ❌Mono — momentum + volume confirm (20d)
  17. **rsv_60**: ✅IC ✅IR ❌Mono — RSV 60d
  18. **vpcorr_40**: ✅IC ✅IR ❌Mono — VP divergence 40d
  19. **vpcorr_neg_40**: ✅IC ✅IR ❌Mono — VP alignment 40d
  20. **vpcorr_3**: ❌IC ✅IR ❌Mono — VP divergence 3d
  21. **vpcorr_neg_3**: ❌IC ✅IR ❌Mono — VP alignment 3d
  22. **d1_close_60_59**: ✅IC ❌IR ❌Mono — N-day change (close, 60d)
  23. **d1_open_60_71**: ✅IC ❌IR ❌Mono — N-day change (open, 60d)
  24. **d1_close_60_87**: ✅IC ❌IR ❌Mono — N-day reversal (close, 60d)
  25. **d1_open_60_99**: ✅IC ❌IR ❌Mono — N-day reversal (open, 60d)
  26. **d1_high_60_63**: ✅IC ❌IR ❌Mono — N-day change (high, 60d)
  27. **d1_high_60_91**: ✅IC ❌IR ❌Mono — N-day reversal (high, 60d)
  28. **d1_low_60_67**: ✅IC ❌IR ❌Mono — N-day change (low, 60d)
  29. **d1_low_60_95**: ✅IC ❌IR ❌Mono — N-day reversal (low, 60d)
  30. **d1_ret_5d_60_195**: ✅IC ❌IR ❌Mono — N-day moving average (ret_5d, 60d)

## Distribution
  IC > 0.02: 178
  IC > 0.01: 259
  IC_IR > 0.3: 72
  Monotonicity > 0.7: 118