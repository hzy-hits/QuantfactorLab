#!/usr/bin/env python3
"""Compare signal construction methods under one walk-forward protocol.

Methods:
1. Single-factor top picks: evaluate each promoted factor individually and
   summarize the top-K performers.
2. IC_IR-weighted composite: weighted sum of promoted factors.
3. Voting composite: dynamic Q1/Q5 voting with rolling realized return lookup.

Usage:
    uv run python -m src.evaluate.compare_signal_builders --market us
    uv run python -m src.evaluate.compare_signal_builders --market all --top-k 5
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import duckdb
import numpy as np
import pandas as pd

from eval_factor import FACTOR_LAB_DB, MARKET_CONFIGS, compute_forward_returns, load_prices
from src.backtest.walk_forward import walk_forward_backtest
from src.dsl.compute import compute_factor
from src.dsl.parser import parse
from src.mining.export_to_pipeline import _resolve_direction


DEFAULT_START_DATE = "2024-03-25"
DEFAULT_LOOKBACK_DAYS = 120
DEFAULT_MIN_HISTORY_DATES = 20


@dataclass
class StrategyMetrics:
    avg_ic: float
    avg_ic_ir: float
    avg_sharpe: float
    avg_turnover: float
    avg_monotonicity: float
    n_folds: int
    n_rows: int
    first_date: str | None
    last_date: str | None


def _load_promoted_factors(market: str) -> list[dict]:
    con = duckdb.connect(FACTOR_LAB_DB, read_only=True)
    try:
        rows = con.execute("""
            SELECT factor_id, name, formula, direction, composite_score,
                   ic_7d, ic_14d, ic_30d, ic_ir_7d
            FROM factor_registry
            WHERE market=? AND status='promoted'
            ORDER BY COALESCE(composite_score, 0) DESC, factor_id
        """, [market]).fetchall()
    finally:
        con.close()

    promoted = []
    for row in rows:
        promoted.append({
            "factor_id": row[0],
            "name": row[1] or row[0],
            "formula": row[2],
            "direction": row[3],
            "composite_score": float(row[4] or 0.0),
            "ic_7d": float(row[5] or 0.0),
            "ic_14d": float(row[6] or 0.0),
            "ic_30d": float(row[7] or 0.0),
            "ic_ir_7d": float(row[8] or 0.0),
        })
    return promoted


def _backtest_metrics(
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    market: str,
    sym_col: str,
    date_col: str,
) -> StrategyMetrics:
    bt = walk_forward_backtest(
        factor_df,
        forward_returns,
        sym_col=sym_col,
        date_col=date_col,
        oos_start=MARKET_CONFIGS[market]["oos_start"],
        cost_per_trade=MARKET_CONFIGS[market]["cost_per_trade"],
    )
    first_date = None
    last_date = None
    if not factor_df.empty:
        first_date = str(pd.to_datetime(factor_df[date_col]).min().date())
        last_date = str(pd.to_datetime(factor_df[date_col]).max().date())
    return StrategyMetrics(
        avg_ic=round(float(bt.avg_ic), 4),
        avg_ic_ir=round(float(bt.avg_ic_ir), 3),
        avg_sharpe=round(float(bt.avg_sharpe), 3),
        avg_turnover=round(float(bt.avg_turnover), 4),
        avg_monotonicity=round(float(bt.avg_monotonicity), 3),
        n_folds=len(bt.fold_metrics),
        n_rows=int(len(factor_df)),
        first_date=first_date,
        last_date=last_date,
    )


def _compute_promoted_factor_data(
    market: str,
    promoted: list[dict],
    prices: pd.DataFrame,
    start_date: str,
    sym_col: str,
    date_col: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, pd.Series]]]:
    factor_frames: dict[str, pd.DataFrame] = {}
    factor_daily: dict[str, dict[str, pd.Series]] = {}

    for factor in promoted:
        factor_id = factor["factor_id"]
        factor_df = compute_factor(parse(factor["formula"]), prices, sym_col=sym_col, date_col=date_col)
        factor_df = factor_df[[sym_col, date_col, "factor_value"]].copy()

        resolved_direction = _resolve_direction(
            factor["direction"],
            factor["ic_7d"],
            factor["ic_14d"],
            factor["ic_30d"],
        )
        if resolved_direction == "short":
            factor_df["factor_value"] = -factor_df["factor_value"]

        factor_df[date_col] = pd.to_datetime(factor_df[date_col]).dt.strftime("%Y-%m-%d")
        factor_frames[factor_id] = factor_df[factor_df[date_col] >= start_date].copy()
        factor_daily[factor_id] = {
            dt: group.set_index(sym_col)["factor_value"].dropna()
            for dt, group in factor_df.groupby(date_col)
        }

    return factor_frames, factor_daily


def _evaluate_single_factor_top_picks(
    market: str,
    promoted: list[dict],
    factor_frames: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    sym_col: str,
    date_col: str,
    top_k: int,
) -> dict:
    results = []
    promoted_by_id = {factor["factor_id"]: factor for factor in promoted}

    for factor_id, factor_df in factor_frames.items():
        if factor_df.empty:
            continue
        metrics = _backtest_metrics(factor_df, forward_returns, market, sym_col, date_col)
        results.append({
            "factor_id": factor_id,
            "name": promoted_by_id[factor_id]["name"],
            "formula": promoted_by_id[factor_id]["formula"],
            **asdict(metrics),
        })

    results.sort(key=lambda row: row["avg_sharpe"], reverse=True)
    top = results[:top_k]
    if not top:
        return {"metrics": None, "top_factors": []}

    summary = StrategyMetrics(
        avg_ic=round(float(np.mean([row["avg_ic"] for row in top])), 4),
        avg_ic_ir=round(float(np.mean([row["avg_ic_ir"] for row in top])), 3),
        avg_sharpe=round(float(np.mean([row["avg_sharpe"] for row in top])), 3),
        avg_turnover=round(float(np.mean([row["avg_turnover"] for row in top])), 4),
        avg_monotonicity=round(float(np.mean([row["avg_monotonicity"] for row in top])), 3),
        n_folds=int(round(np.mean([row["n_folds"] for row in top]))),
        n_rows=int(round(np.mean([row["n_rows"] for row in top]))),
        first_date=top[0]["first_date"],
        last_date=top[0]["last_date"],
    )
    return {
        "metrics": asdict(summary),
        "top_factors": top,
    }


def _evaluate_weighted_composite(
    market: str,
    promoted: list[dict],
    factor_frames: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    sym_col: str,
    date_col: str,
) -> dict:
    weighted_sum: pd.Series | None = None
    weights: dict[str, float] = {}

    for factor in promoted:
        factor_id = factor["factor_id"]
        factor_df = factor_frames.get(factor_id)
        if factor_df is None or factor_df.empty:
            continue
        series = factor_df.set_index([sym_col, date_col])["factor_value"]
        weight = max(abs(float(factor.get("ic_ir_7d", 0.0))), 0.01)
        weights[factor_id] = weight
        weighted_component = series * weight
        weighted_sum = weighted_component if weighted_sum is None else weighted_sum.add(weighted_component, fill_value=0.0)

    if weighted_sum is None or not weights:
        return {"metrics": None, "weights": {}}

    total = sum(weights.values())
    norm_weights = {factor_id: weight / total for factor_id, weight in weights.items()}
    weighted_sum = weighted_sum / total
    composite_df = weighted_sum.rename("factor_value").reset_index()
    metrics = _backtest_metrics(composite_df, forward_returns, market, sym_col, date_col)
    return {"metrics": asdict(metrics), "weights": norm_weights}


def _prepare_voting_quintiles(
    factor_daily: dict[str, dict[str, pd.Series]],
    fwd_map: dict[str, pd.Series],
) -> tuple[dict[str, dict[str, pd.Series]], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    quintiles_by_factor: dict[str, dict[str, pd.Series]] = {}
    q1_mean_by_factor: dict[str, dict[str, float]] = {}
    q5_mean_by_factor: dict[str, dict[str, float]] = {}

    for factor_id, daily_map in factor_daily.items():
        factor_quintiles: dict[str, pd.Series] = {}
        q1_means: dict[str, float] = {}
        q5_means: dict[str, float] = {}

        for trade_date, values in daily_map.items():
            clean = values.dropna()
            if len(clean) < 25:
                continue
            try:
                quintiles = pd.qcut(clean, 5, labels=False, duplicates="drop") + 1
            except ValueError:
                continue
            if pd.Series(quintiles).nunique() < 5:
                continue

            factor_quintiles[trade_date] = quintiles.astype(int)

            fwd_series = fwd_map.get(trade_date)
            if fwd_series is None:
                continue
            merged = pd.DataFrame({"quintile": quintiles}).join(
                fwd_series.rename("fwd_5d"),
                how="inner",
            ).dropna()
            if merged.empty:
                continue
            means = merged.groupby("quintile")["fwd_5d"].mean()
            if pd.notna(means.get(1)):
                q1_means[trade_date] = float(means.get(1))
            if pd.notna(means.get(5)):
                q5_means[trade_date] = float(means.get(5))

        quintiles_by_factor[factor_id] = factor_quintiles
        q1_mean_by_factor[factor_id] = q1_means
        q5_mean_by_factor[factor_id] = q5_means

    return quintiles_by_factor, q1_mean_by_factor, q5_mean_by_factor


def _evaluate_voting_composite(
    market: str,
    factor_daily: dict[str, dict[str, pd.Series]],
    forward_returns: pd.DataFrame,
    sym_col: str,
    date_col: str,
    start_date: str,
    lookback_days: int,
    min_history_dates: int,
) -> dict:
    fwd_map = {
        trade_date: group.set_index(sym_col)["fwd_5d"]
        for trade_date, group in forward_returns.groupby(date_col)
    }
    all_dates = sorted(d for d in forward_returns[date_col].unique() if d >= start_date)
    quintiles_by_factor, q1_mean_by_factor, q5_mean_by_factor = _prepare_voting_quintiles(factor_daily, fwd_map)

    rows: list[dict] = []
    for idx, trade_date in enumerate(all_dates):
        history_dates = all_dates[max(0, idx - lookback_days):idx]
        today_quintiles: dict[str, pd.Series] = {}
        today_factor_values: dict[str, pd.Series] = {}
        best_quintiles: dict[str, int] = {}
        worst_quintiles: dict[str, int] = {}
        all_symbols: set[str] = set()

        for factor_id, quintile_map in quintiles_by_factor.items():
            quintiles = quintile_map.get(trade_date)
            if quintiles is None:
                continue

            raw_values = factor_daily[factor_id].get(trade_date)
            if raw_values is not None:
                today_factor_values[factor_id] = raw_values

            hist_q1 = [q1_mean_by_factor[factor_id][dt] for dt in history_dates if dt in q1_mean_by_factor[factor_id]]
            hist_q5 = [q5_mean_by_factor[factor_id][dt] for dt in history_dates if dt in q5_mean_by_factor[factor_id]]

            best_q = 5
            if len(hist_q1) >= min_history_dates and len(hist_q5) >= min_history_dates:
                best_q = 1 if float(np.mean(hist_q1)) > float(np.mean(hist_q5)) else 5

            today_quintiles[factor_id] = quintiles
            best_quintiles[factor_id] = best_q
            worst_quintiles[factor_id] = 1 if best_q == 5 else 5
            all_symbols.update(quintiles.index)

        votes: dict[str, float] = {}
        for symbol in all_symbols:
            best_votes = 0
            worst_votes = 0
            total_votes = 0
            for factor_id, quintiles in today_quintiles.items():
                if symbol not in quintiles.index:
                    continue
                total_votes += 1
                quintile = int(quintiles[symbol])
                if quintile == best_quintiles[factor_id]:
                    best_votes += 1
                elif quintile == worst_quintiles[factor_id]:
                    worst_votes += 1
            if total_votes == 0:
                continue
            votes[symbol] = (best_votes - worst_votes) / total_votes

        try:
            factor_ids = list(today_quintiles.keys())
            if factor_ids:
                common_symbols = sorted(set.intersection(*[
                    set(today_factor_values[factor_id].index)
                    for factor_id in factor_ids
                    if factor_id in today_factor_values
                ]))
            else:
                common_symbols = []
            if len(common_symbols) >= 100:
                mat = np.column_stack([
                    today_factor_values[factor_id].loc[common_symbols].values
                    for factor_id in factor_ids
                ])
                mask = ~np.any(np.isnan(mat), axis=1)
                clean_symbols = [common_symbols[i] for i in range(len(common_symbols)) if mask[i]]
                mat_clean = mat[mask]
                if len(mat_clean) >= 100:
                    mat_std = (mat_clean - mat_clean.mean(axis=0)) / (mat_clean.std(axis=0) + 1e-12)
                    u, s, _ = np.linalg.svd(mat_std, full_matrices=False)
                    pc_scores = u * s
                    eigenvalues = s ** 2 / len(mat_std)
                    weighted = pc_scores / (np.sqrt(eigenvalues) + 1e-6)
                    mahal = np.sqrt(np.sum(weighted ** 2, axis=1))
                    threshold = np.percentile(mahal, 80)
                    outliers = {
                        clean_symbols[i]
                        for i in range(len(clean_symbols))
                        if mahal[i] > threshold
                    }
                    for symbol in outliers:
                        if symbol in votes:
                            votes[symbol] *= 0.5
        except Exception:
            pass

        for symbol, value in votes.items():
            rows.append({
                sym_col: symbol,
                date_col: trade_date,
                "factor_value": value,
            })

    if not rows:
        return {"metrics": None}

    composite_df = pd.DataFrame(rows)
    metrics = _backtest_metrics(composite_df, forward_returns, market, sym_col, date_col)
    return {"metrics": asdict(metrics)}


def compare_market(
    market: str,
    start_date: str,
    top_k: int,
    lookback_days: int,
    min_history_dates: int,
) -> dict:
    cfg = MARKET_CONFIGS[market]
    sym_col = cfg["sym_col"]
    date_col = cfg["date_col"]

    prices = load_prices(market).copy()
    forward_returns = compute_forward_returns(market).copy()
    prices[date_col] = pd.to_datetime(prices[date_col]).dt.strftime("%Y-%m-%d")
    forward_returns[date_col] = pd.to_datetime(forward_returns[date_col]).dt.strftime("%Y-%m-%d")
    forward_returns = forward_returns[forward_returns[date_col] >= start_date].copy()

    promoted = _load_promoted_factors(market)
    factor_frames, factor_daily = _compute_promoted_factor_data(
        market,
        promoted,
        prices,
        start_date,
        sym_col,
        date_col,
    )

    single = _evaluate_single_factor_top_picks(
        market,
        promoted,
        factor_frames,
        forward_returns,
        sym_col,
        date_col,
        top_k,
    )
    weighted = _evaluate_weighted_composite(
        market,
        promoted,
        factor_frames,
        forward_returns,
        sym_col,
        date_col,
    )
    voting = _evaluate_voting_composite(
        market,
        factor_daily,
        forward_returns,
        sym_col,
        date_col,
        start_date,
        lookback_days,
        min_history_dates,
    )

    return {
        "market": market,
        "start_date": start_date,
        "promoted_factors": len(promoted),
        "single_factor_top_picks": single,
        "weighted_composite": weighted,
        "voting_composite": voting,
    }


def _print_report(report: dict) -> None:
    print("---")
    print(f"market: {report['market']}")
    print(f"start_date: {report['start_date']}")
    print(f"promoted_factors: {report['promoted_factors']}")

    single = report["single_factor_top_picks"]
    if single["metrics"] is not None:
        print("single_factor_top_picks:")
        print(f"  avg_top_sharpe: {single['metrics']['avg_sharpe']:.3f}")
        print(f"  avg_top_ic: {single['metrics']['avg_ic']:.4f}")
        print(f"  avg_top_ic_ir: {single['metrics']['avg_ic_ir']:.3f}")
        top_names = ", ".join(
            f"{row['name']}({row['avg_sharpe']:.3f})"
            for row in single["top_factors"]
        )
        print(f"  top_factors: {top_names}")
    else:
        print("single_factor_top_picks: none")

    weighted = report["weighted_composite"]["metrics"]
    if weighted is not None:
        print("weighted_composite:")
        print(f"  sharpe: {weighted['avg_sharpe']:.3f}")
        print(f"  ic: {weighted['avg_ic']:.4f}")
        print(f"  ic_ir: {weighted['avg_ic_ir']:.3f}")
        print(f"  turnover: {weighted['avg_turnover']:.4f}")
    else:
        print("weighted_composite: none")

    voting = report["voting_composite"]["metrics"]
    if voting is not None:
        print("voting_composite:")
        print(f"  sharpe: {voting['avg_sharpe']:.3f}")
        print(f"  ic: {voting['avg_ic']:.4f}")
        print(f"  ic_ir: {voting['avg_ic_ir']:.3f}")
        print(f"  turnover: {voting['avg_turnover']:.4f}")
    else:
        print("voting_composite: none")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare signal construction methods")
    parser.add_argument("--market", choices=["cn", "us", "all"], required=True)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--min-history-dates", type=int, default=DEFAULT_MIN_HISTORY_DATES)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text")
    args = parser.parse_args()

    markets = ["cn", "us"] if args.market == "all" else [args.market]
    reports = [
        compare_market(
            market=market,
            start_date=args.start_date,
            top_k=args.top_k,
            lookback_days=args.lookback_days,
            min_history_dates=args.min_history_dates,
        )
        for market in markets
    ]

    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
    else:
        for report in reports:
            _print_report(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
