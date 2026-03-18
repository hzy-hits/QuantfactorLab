The batch run completed successfully with exit code `0` and wrote [batch_us_500.md](/home/ivena/coding/python/factor-lab/reports/batch_us_500.md#L1). One caveat: the script generated `377` unique formulas rather than `500`, so the ranking is over `377` valid factors with `0` errors.

The top 10 in [batch_us_500.md](/home/ivena/coding/python/factor-lab/reports/batch_us_500.md#L8) are dominated by volume-level and volume-volatility signals:

1. `rank(volume)` — IC `0.0713`, IR `0.954`, Q5-Q1 `0.538`, Mono `0.80`
2. `rank(-volume)` — exact inverse of #1
3. `rank(ts_mean(volume, 5))` — IC `0.0693`, IR `0.926`, Q5-Q1 `0.443`, Mono `0.80`
4. `rank(ts_mean(volume, 10))` — IC `0.0673`, IR `0.895`, Q5-Q1 `0.404`, Mono `0.80`
5. `rank(ts_mean(volume, 20))` — IC `0.0647`, IR `0.856`, Q5-Q1 `0.653`, Mono `1.00`
6. `rank(ts_mean(volume, 60))` — IC `0.0637`, IR `0.831`, Q5-Q1 `1.035`, Mono `0.70`
7. `rank(ts_std(volume, 5))` — IC `0.0588`, IR `0.841`, Q5-Q1 `0.486`, Mono `0.80`
8. `rank(-ts_std(volume, 5))` — inverse of #7
9. `rank(ts_std(volume, 10))` — IC `0.0560`, IR `0.798`, Q5-Q1 `0.537`, Mono `1.00`
10. `rank(-ts_std(volume, 10))` — inverse of #9

The main takeaway is that this run found a very concentrated signal family: simple cross-sectional volume ranks, moving averages of volume, and rolling volume volatility dominate the leaderboard. Several entries are just sign-flipped mirrors of the same effect, so the genuinely distinct winners are fewer than ten.