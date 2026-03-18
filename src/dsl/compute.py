"""
Factor computation engine — executes an AST against a price DataFrame.

Usage
-----
>>> from src.dsl.parser import parse
>>> from src.dsl.compute import compute_factor
>>> ast = parse("rank(delta(close, 5))")
>>> result = compute_factor(ast, prices_df)
# result: DataFrame with [sym_col, date_col, "factor_value"]
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.dsl.parser import (
    ASTNode,
    BinOp,
    DSLParseError,
    Feature,
    FunctionCall,
    Literal,
    UnaryOp,
)
from src.dsl.operators import OPERATOR_REGISTRY


# ---------------------------------------------------------------------------
# Feature resolver
# ---------------------------------------------------------------------------

# Canonical feature name  →  list of possible column names (first match wins).
_FEATURE_ALIASES: dict[str, list[str]] = {
    "close":         ["close", "Close", "adj_close"],
    "open":          ["open", "Open"],
    "high":          ["high", "High"],
    "low":           ["low", "Low"],
    "volume":        ["volume", "vol", "Volume"],
    "amount":        ["amount", "Amount", "turnover"],
    "vwap":          ["vwap", "VWAP"],
    "turnover_rate": ["turnover_rate", "turnover"],
}

# Features that can be computed on the fly from base columns.
_COMPUTED_FEATURES: dict[str, Callable[[pd.DataFrame, str, str], pd.Series]] = {
    "ret_1d": lambda df, sym, dt: df.groupby(sym)["close"].pct_change(1),
    "ret_5d": lambda df, sym, dt: df.groupby(sym)["close"].pct_change(5),
    "ret_20d": lambda df, sym, dt: df.groupby(sym)["close"].pct_change(20),
    "volume_ratio": lambda df, sym, dt: (
        df["volume"] / df.groupby(sym)["volume"].transform(
            lambda s: s.rolling(20, min_periods=1).mean()
        )
    ),
}


def _resolve_feature(
    name: str,
    prices_df: pd.DataFrame,
    sym_col: str,
    date_col: str,
) -> pd.Series:
    """Return a pd.Series aligned to *prices_df* for the given feature name."""

    # 1. Check aliases (direct column lookup)
    candidates = _FEATURE_ALIASES.get(name)
    if candidates:
        for col in candidates:
            if col in prices_df.columns:
                return prices_df[col].copy()

    # 2. Computed features
    if name in _COMPUTED_FEATURES:
        return _COMPUTED_FEATURES[name](prices_df, sym_col, date_col)

    # 3. Exact column match (for arbitrary features like rsi_14, pe_ttm, etc.)
    if name in prices_df.columns:
        return prices_df[name].copy()

    # 4. Try vwap = amount / volume if vwap requested but not present
    if name == "vwap":
        if "amount" in prices_df.columns and "volume" in prices_df.columns:
            vol = prices_df["volume"].replace(0, np.nan)
            return prices_df["amount"] / vol

    raise DSLParseError(
        f"Feature {name!r} not found in DataFrame columns: "
        f"{sorted(prices_df.columns.tolist())}"
    )


# ---------------------------------------------------------------------------
# AST evaluator
# ---------------------------------------------------------------------------

class _Evaluator:
    """
    Walks the AST and produces a pd.Series of factor values,
    aligned to the index of *prices_df*.

    Strategy:
      - Leaf Feature → column lookup / compute
      - Leaf Literal  → scalar (broadcast when needed)
      - FunctionCall  → dispatch via OPERATOR_REGISTRY
        * ts operators → groupby(sym_col).transform(...)
        * cs operators → groupby(date_col).transform(...)
        * univ operators → element-wise
      - BinOp → numpy vectorised arithmetic
      - UnaryOp → negation
    """

    def __init__(
        self,
        prices_df: pd.DataFrame,
        sym_col: str,
        date_col: str,
    ) -> None:
        self.df = prices_df
        self.sym_col = sym_col
        self.date_col = date_col

        # Cache resolved features to avoid repeated computation.
        self._cache: dict[str, pd.Series] = {}

    # -- public entry --------------------------------------------------------

    def evaluate(self, node: ASTNode) -> pd.Series:
        if isinstance(node, Literal):
            return self._eval_literal(node)
        if isinstance(node, Feature):
            return self._eval_feature(node)
        if isinstance(node, UnaryOp):
            return self._eval_unary(node)
        if isinstance(node, BinOp):
            return self._eval_binop(node)
        if isinstance(node, FunctionCall):
            return self._eval_call(node)
        raise DSLParseError(f"Cannot evaluate node type: {type(node)}")

    # -- leaf nodes ----------------------------------------------------------

    def _eval_literal(self, node: Literal) -> pd.Series:
        s = pd.Series(node.value, index=self.df.index, dtype=float)
        s._is_literal = True  # mark for _apply_univ scalarization
        return s

    def _eval_feature(self, node: Feature) -> pd.Series:
        name = node.name
        if name not in self._cache:
            self._cache[name] = _resolve_feature(
                name, self.df, self.sym_col, self.date_col
            )
        return self._cache[name].copy()

    # -- operators -----------------------------------------------------------

    def _eval_unary(self, node: UnaryOp) -> pd.Series:
        child = self.evaluate(node.operand)
        if node.op == "-":
            return -child
        raise DSLParseError(f"Unknown unary operator: {node.op!r}")

    def _eval_binop(self, node: BinOp) -> pd.Series:
        left = self.evaluate(node.left)
        right = self.evaluate(node.right)
        if node.op == "+":
            return left + right
        if node.op == "-":
            return left - right
        if node.op == "*":
            return left * right
        if node.op == "/":
            return left / right.replace(0, np.nan)
        raise DSLParseError(f"Unknown binary operator: {node.op!r}")

    def _eval_call(self, node: FunctionCall) -> pd.Series:
        entry = OPERATOR_REGISTRY.get(node.name)
        if entry is None:
            raise DSLParseError(f"No implementation for function: {node.name!r}")

        func, op_type = entry

        # Evaluate all argument sub-trees first.
        evaluated_args = [self.evaluate(a) for a in node.args]

        if op_type == "ts":
            return self._apply_ts(func, evaluated_args, node.name)
        elif op_type == "cs":
            return self._apply_cs(func, evaluated_args, node.name)
        else:  # "univ"
            return self._apply_univ(func, evaluated_args, node.name)

    # -- dispatch helpers ----------------------------------------------------

    def _apply_ts(
        self,
        func: Callable,
        args: list[pd.Series],
        name: str,
    ) -> pd.Series:
        """
        Time-series operator: apply *func* per stock (groupby sym_col).

        The last argument is always the window (int scalar extracted from the
        constant Literal that the parser already validated).
        """
        window = int(args[-1].iloc[0])  # constant series → scalar
        series_args = args[:-1]

        sym = self.df[self.sym_col]

        if len(series_args) == 1:
            # Single-series ts operator: ts_mean(x, n), delta(x, n), etc.
            x = series_args[0]
            return x.groupby(sym, group_keys=False).transform(
                lambda s: func(s, window)
            )
        elif len(series_args) == 2:
            # Two-series ts operator: ts_corr(x, y, n), ts_cov(x, y, n)
            x, y = series_args
            # Must group both in lockstep
            result = pd.Series(np.nan, index=self.df.index, dtype=float)
            for key, idx in x.groupby(sym).groups.items():
                sx = x.loc[idx]
                sy = y.loc[idx]
                result.loc[idx] = func(sx, sy, window).values
            return result
        else:
            raise DSLParseError(
                f"Time-series function {name!r} got unexpected number of "
                f"series arguments: {len(series_args)}"
            )

    def _apply_cs(
        self,
        func: Callable,
        args: list[pd.Series],
        name: str,
    ) -> pd.Series:
        """Cross-sectional operator: apply *func* per date (groupby date_col)."""
        x = args[0]
        date = self.df[self.date_col]
        return x.groupby(date, group_keys=False).transform(func)

    def _apply_univ(
        self,
        func: Callable,
        args: list[pd.Series],
        name: str,
    ) -> pd.Series:
        """Universal (element-wise) operator."""
        # Only Literal-sourced series should be scalarized.
        # A time-series result with NaN warm-up and one non-NaN value is NOT a literal.
        # We mark literal-sourced series with a _is_literal attribute in _eval.
        call_args = []
        for a in args:
            if isinstance(a, pd.Series) and getattr(a, '_is_literal', False):
                val = a.dropna().unique()
                if len(val) == 1:
                    call_args.append(val[0])
                else:
                    call_args.append(a)
            else:
                call_args.append(a)
        return func(*call_args)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_factor(
    ast: ASTNode,
    prices_df: pd.DataFrame,
    sym_col: str = "ts_code",
    date_col: str = "trade_date",
) -> pd.DataFrame:
    """
    Execute an AST against price data.

    Parameters
    ----------
    ast : ASTNode
        Parsed DSL expression.
    prices_df : pd.DataFrame
        Must contain at least *sym_col*, *date_col*, and whatever features the
        expression references.  Should be sorted by [sym_col, date_col].
    sym_col : str
        Column identifying the stock / instrument.
    date_col : str
        Column identifying the trading date.

    Returns
    -------
    pd.DataFrame
        Columns: [sym_col, date_col, "factor_value"].
    """
    if sym_col not in prices_df.columns:
        raise DSLParseError(f"Symbol column {sym_col!r} not found in DataFrame")
    if date_col not in prices_df.columns:
        raise DSLParseError(f"Date column {date_col!r} not found in DataFrame")

    prices_df = prices_df.sort_values([sym_col, date_col]).reset_index(drop=True)

    evaluator = _Evaluator(prices_df, sym_col, date_col)
    factor_values = evaluator.evaluate(ast)

    result = prices_df[[sym_col, date_col]].copy()
    result["factor_value"] = factor_values
    return result
