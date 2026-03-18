The script completed successfully on the US equity dataset in `/home/ivena/coding/python/quant-research-v1/data/quant.duckdb`. All 25 formulas ran; no factor errors.

```text
Factor                          IC    IC_IR   Q5-Q1%   Mono   Flag
amihud                      0.0683    0.557   -0.177  -0.70   pass
range_20d                  -0.0580   -0.296    0.797   1.00   pass
breakout_confirm            0.0537    0.413   -0.400  -0.90   pass
shrink_then_surge           0.0525    0.564   -0.506  -1.00   pass
dist_high_20                0.0465    0.312   -0.631  -1.00   pass
dist_low_20                -0.0396   -0.252    0.459   0.70   pass
rsv_60                      0.0362    0.282    0.031   0.00   pass
kbar_upper_shadow           0.0361    0.243   -0.259  -0.10   pass
kbar_shift                 -0.0354   -0.237   -0.024   0.00   pass
kbar_lower_shadow          -0.0344   -0.229   -0.196  -0.20   pass
vol_price_corr_60           0.0322    0.295   -0.138  -0.20   pass
kbar_body                  -0.0275   -0.198    0.009   0.40   pass
vol_price_corr_5            0.0251    0.340   -0.060  -0.50   pass
vol_price_corr_20           0.0203    0.223   -0.452  -0.90   pass
vol_trend                   0.0187    0.261   -0.071  -0.10   pass
vol_shrink_reversal        -0.0120   -0.130   -0.072  -0.10   pass
vol_reversal               -0.0120   -0.130   -0.072  -0.10   pass
rev_20d                    -0.0109   -0.080    0.432   0.40   fail
vol_compression            -0.0104   -0.138   -0.281  -0.30   pass
momentum_5d                 0.0072    0.060    0.053   0.40   fail
rsv_20                      0.0059    0.048   -0.139  -0.20   fail
rev_1d                      0.0044    0.035    0.198   0.30   fail
ret_ratio_5_20              0.0020    0.034   -0.065  -0.10   fail
value_reversal              0.0020    0.016    0.083   0.00   fail
mom_accel                  -0.0003   -0.002    0.286   0.10   fail
```

Notable points: `amihud` had the highest IC, `range_20d` had the strongest positive Q5-Q1 spread, and several high-IC factors (`breakout_confirm`, `shrink_then_surge`, `dist_high_20`) showed negative long-short spreads, which suggests their effective trade direction may be inverted relative to the raw rank.