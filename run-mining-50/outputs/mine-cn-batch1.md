I ran the session against a local snapshot of the CN DuckDB file because the original database was locked by another process. One formula failed to parse: `downside_vol` with `Expected RPAREN at position 44, got NUMBER ('0')`.

Top signals by absolute IC were `range_20d`, `vol_price_corr_5`, `vol_shrink_reversal`/`vol_reversal`, and `value_reversal`. Full ranking:

```text
=== CN RANKING ===
Factor                          IC    IC_IR   Q5-Q1%   Mono
range_20d                  -0.0267   -0.112    0.230   0.80 ✅
vol_price_corr_5            0.0246    0.205    0.040   0.00 ✅
vol_shrink_reversal         0.0241    0.149   -0.005  -0.20 ✅
vol_reversal                0.0241    0.149   -0.005  -0.20 ✅
value_reversal              0.0230    0.117   -0.158  -0.90 ✅
vol_price_corr_20           0.0228    0.133   -0.099   0.00 ✅
shrink_then_surge           0.0191    0.176    0.102   0.80 ✅
dist_low_20                -0.0188   -0.096    0.328   1.00 ❌
rsv_60                     -0.0172   -0.085    0.185   0.50 ❌
rev_20d                     0.0163    0.077   -0.230  -0.90 ❌
vol_compression             0.0127    0.084   -0.009  -0.10 ❌
mom_accel                  -0.0115   -0.062   -0.132  -0.30 ❌
vol_trend                  -0.0113   -0.089    0.148   0.60 ❌
ret_ratio_5_20             -0.0100   -0.101   -0.019  -0.40 ❌
kbar_lower_shadow          -0.0089   -0.053    0.011   0.30 ❌
rev_1d                      0.0085    0.044   -0.017  -0.60 ❌
dist_high_20                0.0084    0.041   -0.010   0.00 ❌
kbar_shift                 -0.0069   -0.040   -0.026  -0.10 ❌
amihud                      0.0053    0.046   -0.198  -0.90 ❌
kbar_upper_shadow           0.0050    0.029    0.044   0.10 ❌
breakout_confirm           -0.0035   -0.021    0.047   0.10 ❌
kbar_body                  -0.0033   -0.018    0.060   0.30 ❌
rsv_20                     -0.0020   -0.011    0.312   0.90 ❌
vol_price_corr_60           0.0014    0.007   -0.318  -0.90 ❌
```

If you want, I can do a second pass that fixes `downside_vol`, removes the duplicate `vol_reversal` formula, and writes the results to CSV/Markdown.