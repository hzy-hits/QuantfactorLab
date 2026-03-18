Ran the A-share batch against `/home/ivena/coding/rust/quant-research-cn/data/quant_cn.duckdb` and saved the full result set in [mine-cn-batch2-safe.md](/home/ivena/coding/python/factor-lab/run-mining-batch2/outputs/mine-cn-batch2-safe.md#L1). The sample covered 2024-03-13 to 2026-03-17, 5,569 symbols, 486 trading days. Your list had 24 factors, so I added `momentum_quality` as the 25th. No Amihud-style return-over-volume/amount formula was used.

Best signals by absolute IC:
- `vp_div_10d`: IC `0.0282`, IR `0.189`
- `shrink_surge_hl`: IC `0.0263`, IR `0.223`
- `oversold_bounce`: IC `0.0233`, IR `0.132`
- `smart_money_rev`: IC `0.0211`, IR `0.150`
- `range_contraction`: IC `0.0205`, IR `0.167`

11 factors cleared the session heuristic (`|IC| > 0.01` and `|IR| > 0.1`). The top-ranked section is in [mine-cn-batch2-safe.md](/home/ivena/coding/python/factor-lab/run-mining-batch2/outputs/mine-cn-batch2-safe.md#L45), and the full 25-factor ranking starts at [mine-cn-batch2-safe.md](/home/ivena/coding/python/factor-lab/run-mining-batch2/outputs/mine-cn-batch2-safe.md#L53). A `ConstantInputWarning` appeared on some daily cross-sections, but the batch completed successfully.