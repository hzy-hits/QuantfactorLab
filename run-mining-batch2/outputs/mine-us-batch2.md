Ran the batch exactly as provided against `/home/ivena/coding/python/quant-research-v1/data/quant.duckdb`. All 25 factors evaluated without execution errors.

```text
=== US BATCH 2 RANKING ===
Factor                          IC    IC_IR   Q5-Q1%   Mono
momentum_quality            0.0556    0.299   -0.699  -1.00 ✅
shrink_surge_3d             0.0442    0.546   -0.223  -0.40 ✅
smart_money_rev             0.0419    0.389   -0.534  -0.60 ✅
upper_wick_sell             0.0411    0.298   -0.048  -0.10 ✅
oversold_bounce             0.0399    0.251   -0.736  -1.00 ✅
quiet_breakout              0.0358    0.392   -0.451  -0.90 ✅
amount_surge                0.0333    0.420   -0.127   0.00 ✅
doji_reversal              -0.0325   -0.232    0.435   0.90 ✅
lower_highs                -0.0298   -0.216    1.779   0.90 ✅
shrink_surge_hl             0.0293    0.405   -0.470  -0.90 ✅
rev_5d_vol_confirm          0.0235    0.241   -0.040  -0.40 ✅
vp_div_10d                  0.0228    0.261   -0.387  -0.10 ✅
amount_trend                0.0201    0.261   -0.039  -0.30 ✅
vp_div_3d                   0.0195    0.298   -0.025   0.00 ✅
rev_gap_fill                0.0156    0.132   -0.154  -0.40 ✅
hammer                     -0.0122   -0.095   -0.301  -0.70 ❌
higher_lows                -0.0105   -0.086    0.292   0.80 ❌
vp_div_change              -0.0101   -0.158   -0.096  -0.70 ✅
vol_price_asym             -0.0079   -0.088   -0.186  -0.50 ❌
engulfing                  -0.0073   -0.077   -0.104  -0.90 ❌
vol_squeeze_5_20            0.0067    0.089    0.014   0.50 ❌
realized_vs_range          -0.0063   -0.081    0.353   0.40 ❌
vol_squeeze_10_60          -0.0060   -0.074   -0.087  -0.10 ❌
rev_10d                    -0.0034   -0.026    0.339   0.00 ❌
range_contraction          -0.0004   -0.006    0.336   0.30 ❌
```

Main takeaways: `momentum_quality` had the strongest IC, `shrink_surge_3d` had the best IR, and `lower_highs` had the strongest quintile spread. A lot of the positive-IC factors still show negative `Q5-Q1`, so before keeping them as long-top-quintile signals, their direction should be checked or inverted. If you want, I can turn this into a clean winners/losers shortlist for Batch 3.