**Findings**
1. High: The DSL does not enforce a feature whitelist. Any bare identifier becomes a `Feature` in [parser.py:229](/home/ivena/coding/python/factor-lab/src/dsl/parser.py#L229), and `_resolve_feature()` accepts any exact DataFrame column in [compute.py:80](/home/ivena/coding/python/factor-lab/src/dsl/compute.py#L80). If the frame contains labels like `fwd_5d`, `parse("fwd_5d")` returns them directly. That is a real safety/leakage hole.

2. High: Time-series results are silently wrong on unsorted input. `_apply_ts()` rolls in existing row order in [compute.py:197](/home/ivena/coding/python/factor-lab/src/dsl/compute.py#L197), while `compute_factor()` only documents sorting and never enforces it in [compute.py:271](/home/ivena/coding/python/factor-lab/src/dsl/compute.py#L271). On a shuffled toy frame, `delta(close, 1)` changed from the correct `[nan, 1, 1, nan, -1, -1]` to `[nan, -2, 1, nan, 2, -1]`.

3. High: `ts_count` is implemented incorrectly. Its docstring says “count of truthy (> 0) values,” but the code just does a rolling sum in [operators.py:81](/home/ivena/coding/python/factor-lab/src/dsl/operators.py#L81). For `[2, -1, 3, 4]` with `n=3`, it returns `[nan, nan, 4, 6]` instead of `[nan, nan, 2, 2]`.

4. High: Function arity is not validated. `_validate()` only checks TS functions have at least 2 args and a literal window in [parser.py:285](/home/ivena/coding/python/factor-lab/src/dsl/parser.py#L285). That lets `rank()` crash with `IndexError` in [compute.py:236](/home/ivena/coding/python/factor-lab/src/dsl/compute.py#L236), `rank(close, 1)` silently ignore its extra arg, and `ts_mean(close, volume, 1)` raise `TypeError` from [compute.py:214](/home/ivena/coding/python/factor-lab/src/dsl/compute.py#L214).

5. Medium: Computed features do not handle missing base columns gracefully. `_COMPUTED_FEATURES` hardcodes lowercase `"close"` / `"volume"` in [compute.py:48](/home/ivena/coding/python/factor-lab/src/dsl/compute.py#L48), so `ret_1d` with only `Close` or `volume_ratio` with only `Volume` raises raw `KeyError` instead of `DSLParseError`. The `vwap` fallback in [compute.py:84](/home/ivena/coding/python/factor-lab/src/dsl/compute.py#L84) also ignores aliases.

6. Medium: Universal dispatch breaks valid literal-based formulas. `_apply_univ()` scalarizes any constant-valued Series in [compute.py:247](/home/ivena/coding/python/factor-lab/src/dsl/compute.py#L247), but `op_max`, `op_min`, and `if_then` expect Series in [operators.py:189](/home/ivena/coding/python/factor-lab/src/dsl/operators.py#L189). `max(close, 0)` raises `TypeError`, and `if_then(rank(close), 1, -1)` raises `AttributeError`.

7. Medium: The parser whitelist and runtime registry disagree. `quantile` and `neutralize` are allowed in [parser.py:36](/home/ivena/coding/python/factor-lab/src/dsl/parser.py#L36) but missing from [operators.py:211](/home/ivena/coding/python/factor-lab/src/dsl/operators.py#L211), so `parse()` accepts them and evaluation fails later.

8. Medium: The tokenizer silently skips invalid characters. `_tokenize()` uses regex `finditer` without checking gaps in [parser.py:128](/home/ivena/coding/python/factor-lab/src/dsl/parser.py#L128), so `parse("rank(close)$")` succeeds as if the `$` were not there. That does not bypass the function whitelist directly, but it weakens the syntax boundary.

9. Low: The depth check only counts nested `FunctionCall`s in [parser.py:255](/home/ivena/coding/python/factor-lab/src/dsl/parser.py#L255). If the goal was total AST depth, it is easy to evade with unary/binary nesting; the 200-char cap limits abuse, but the check is narrower than its name suggests.

**Checklist**
- The Pratt parser is correct for `rank(delta(close, 5)) * rank(-pct_change(volume, 5))`. It parses as `BinOp('*', rank(delta(...)), rank(UnaryOp('-', pct_change(...))))`, so precedence and unary-minus handling are fine.
- The 31 implemented operator functions in [operators.py](/home/ivena/coding/python/factor-lab/src/dsl/operators.py) did not crash in my empty/all-NaN smoke pass. The real correctness bug there is `ts_count`; the common crash cases come from dispatch/arity handling in [compute.py](/home/ivena/coding/python/factor-lab/src/dsl/compute.py).
- `groupby(sym).transform()` is fine for single-series rolling ops if rows are already ordered within each symbol. `ts_corr` / `ts_cov` do not use `transform`, but they have the same ordering dependency.
- Missing direct features are handled gracefully with `DSLParseError`; derived features are not.

I didn’t find a `tests/` tree for this DSL, so this review is from code inspection plus direct smoke runs.