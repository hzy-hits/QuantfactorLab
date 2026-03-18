**Results**
1. DSL parsing passed for all 5 test expressions.
2. Factor computation worked once the SQL date literal was corrected:
   `Factor computed: 41098 rows, 5489 symbols`
3. Walk-forward backtest completed end-to-end:
   `avg_ic=-0.0254`, `avg_icir=-0.135`
   Folds:
   `2024-03-13~2024-09-04 -> 2024-09-05~2025-05-12: IC=-0.0356`
   `2024-03-13~2025-05-12 -> 2025-05-13~2025-12-31: IC=-0.0151`
   Gates:
   `passed=False`
   Failing gates were `ic_ir` and `monotonicity`.

**Traceback**
The only failure was in step 2 as written, before Factor Lab code ran. DuckDB requires single quotes for string literals:

```text
Traceback (most recent call last):
  File "<string>", line 8, in <module>
    prices = con.execute('SELECT ts_code, trade_date, close, vol as volume FROM prices WHERE trade_date >= "2026-01-01" ORDER BY ts_code, trade_date').fetchdf()
             ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_duckdb.BinderException: Binder Error: Referenced column "2026-01-01" not found in FROM clause!
Candidate bindings: "low"

LINE 1: ...date, close, vol as volume FROM prices WHERE trade_date >= "2026-01-01" ORDER BY ts_code, trade_date
                                                                      ^
```

I reran that step with `trade_date >= '2026-01-01'`, and it succeeded. No repository code changes were needed.

**Note**
The fold dates stopping before `2026-01-01` are expected. [walk_forward.py](/home/ivena/coding/python/factor-lab/src/backtest/walk_forward.py#L230) defines this routine as IS-only, and [walk_forward.py](/home/ivena/coding/python/factor-lab/src/backtest/walk_forward.py#L263) explicitly filters to dates `< oos_start` before building folds.