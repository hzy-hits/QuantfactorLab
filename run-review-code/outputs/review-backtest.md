**Findings**

1. High: the IS/OOS boundary leaks OOS realized returns into the IS selection metrics. [walk_forward.py](/home/ivena/coding/python/factor-lab/src/backtest/walk_forward.py#L263), [walk_forward.py](/home/ivena/coding/python/factor-lab/src/backtest/walk_forward.py#L375), [forward_returns.py](/home/ivena/coding/python/factor-lab/src/evaluate/forward_returns.py#L18)  
`walk_forward_backtest()` defines IS as anchor dates `< 2025-10-01`, but `fwd_5d` is built with `LEAD(close, 5)`. That means IS rows dated `2025-09-24` through `2025-09-30` are still scored using returns realized on or after `2025-10-01`. So `run_oos_check()` may return only `bool`, but the top-3 ranking can already be contaminated by early OOS labels.

2. Medium: cost-adjusted Sharpe is not deducting turnover costs on the correct daily series. [walk_forward.py](/home/ivena/coding/python/factor-lab/src/backtest/walk_forward.py#L193)  
The code computes `avg_turnover` once per fold, then subtracts `avg_turnover * cost_per_trade` from every day’s long-short return. That is not the same as subtracting `turnover_t * cost_per_trade` day by day, and it changes the Sharpe whenever turnover varies over time. It also imputes the same cost onto the first day even though that day’s turnover is `NaN`.

3. Medium: the correlation gate is not correctly comparing against existing factors end-to-end. [loop.py](/home/ivena/coding/python/factor-lab/src/agent/loop.py#L552), [loop.py](/home/ivena/coding/python/factor-lab/src/agent/loop.py#L638), [gates.py](/home/ivena/coding/python/factor-lab/src/backtest/gates.py#L43), [gates.py](/home/ivena/coding/python/factor-lab/src/backtest/gates.py#L136)  
In the actual loop, `check_gates()` is called without `existing_factors`, so the correlation gate always passes. Even if it were wired up, `_cross_sectional_corr()` only gets value series plus `candidate_dates`; it does not align by symbol/date, so different row order or coverage can miscompare factors or fail.

4. Low: the splitter works for the 500-day case, but the generic implementation is looser than the API implies. [walk_forward.py](/home/ivena/coding/python/factor-lab/src/backtest/walk_forward.py#L295), [walk_forward.py](/home/ivena/coding/python/factor-lab/src/backtest/walk_forward.py#L307)  
`min_test_days` is not enforced strictly; the last fold is accepted down to `min_test_days // 2`, and floor-division chunking can drop one remainder day when the available IS test days are odd.

**Direct Answers**

1. Walk-forward split: for exactly 500 in-sample trading dates before `2025-10-01`, the math works. It produces `120` initial train days and `380` available test days, split into two `190`-day test folds:
   - fold 1: train indices `0..119`, test `120..309`
   - fold 2: train indices `0..309`, test `310..499`  
   If those 500 dates are the 500 business days ending `2025-09-30`, that is:
   - train `2023-11-01` to `2024-04-16`, test `2024-04-17` to `2025-01-07`
   - train `2023-11-01` to `2025-01-07`, test `2025-01-08` to `2025-09-30`

2. IC computation: yes. Per fold, it computes daily cross-sectional Spearman rank correlation on the fold’s test dates only, then averages those daily ICs.

3. Cost-adjusted Sharpe: no. Cost is deducted using fold-average turnover, not per-day turnover, so the Sharpe is mismeasured.

4. Gates: the thresholds are broadly reasonable heuristic cutoffs, with CN being fairly lenient. The correlation threshold `0.7` is standard enough. The correlation gate implementation itself is not correct operationally for the reasons above.

5. OOS check: `run_oos_check()` really does return only `bool`, so there is no direct metric leakage through that API. The bigger problem is indirect leakage: IS metrics already include some post-`2025-10-01` realized returns.

6. Look-ahead bias: not obvious from same-day factor construction in these two files, but yes, there is boundary leakage through the `fwd_5d` horizon at fold and OOS cutoffs.

I verified the 500-IS-day split with a small synthetic run; there do not appear to be repo tests covering this path.