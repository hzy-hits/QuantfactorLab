#!/usr/bin/env python3
"""
Daily Factor Pipeline — automated factor lifecycle management.

Flow:
  1. Mine: generate 500 factors, evaluate IC, keep top 30 candidates
  2. Backtest: multi-horizon (7D/14D/30D) walk-forward on top 30
  3. Select: pick best 3 factors for today's report
  4. Health check: retire decaying promoted factors
  5. Output: write to factor_lab.duckdb for pipeline consumption

Usage:
    python -m src.mining.daily_pipeline --market cn
    python -m src.mining.daily_pipeline --market us
    python -m src.mining.daily_pipeline --market cn --skip-mine  # only health check + select

Cron (06:00 CST, before morning pipeline):
    0 6 * * 1-5 cd $FACTOR_LAB_ROOT && python3 -m src.mining.daily_pipeline --market cn >> logs/daily_cn.log 2>&1
"""
import sys
import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd
import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dsl.parser import parse, DSLParseError
from src.dsl.compute import compute_factor
from src.evaluate.forward_returns import compute_forward_returns
from src.evaluate.ic import compute_ic_series, ic_summary
from src.evaluate.quintile import compute_quintile_returns
from src.mining.batch_mine import generate_factor_formulas, CONFIGS
from src.paths import (
    FACTOR_LAB_DB,
    QUANT_CN_DB,
    QUANT_CN_REPORT_DB,
    QUANT_US_DB,
    QUANT_US_REPORT_DB,
)

# ── Config ────────────────────────────────────────────────────────────────────

CANDIDATE_POOL_SIZE = 30
PROMOTE_COUNT = 3
MAX_PROMOTED = 30  # max active promoted factors at any time

HORIZONS = [7, 14, 30]  # multi-horizon backtest windows (trading days)

# Gate thresholds adjusted for multiple testing:
# With 500 factors, naive 5% FDR → ~25 false positives.
# Bonferroni-like adjustment: raise IC threshold so t-stat > 3
# t ≈ IC * sqrt(n_eff), n_eff ≈ 80 (400 days / 5D overlap)
# IC > 3/sqrt(80) ≈ 0.034 for strict significance
# We use a moderate threshold: stricter than before, not full Bonferroni
GATE_THRESHOLDS = {
    "cn": {"ic_min": 0.015, "ir_min": 0.15, "mono_min": 0.0},   # mono disabled: A-share non-monotonic
    "us": {"ic_min": 0.02, "ir_min": 0.2, "mono_min": 0.6},     # unchanged
}

# Health check thresholds
HEALTH_WATCH_IC = 0.005    # IC below this → watchlist
HEALTH_WATCH_DAYS = 5      # consecutive days
HEALTH_RETIRE_DAYS = 5     # days on watchlist before retire
HEALTH_RECOVER_IC = 0.01   # IC above this → recover from watchlist

HORIZON_WEIGHTS = {7: 0.5, 14: 0.3, 30: 0.2}
SIGREG_REDUNDANCY_SOFT = 0.55
SIGREG_REDUNDANCY_HARD = 0.85
SIGREG_HEALTH_WATCH_SCORE = 0.45
SIGREG_HEALTH_RECOVER_SCORE = 0.65
SIGREG_PENALTY_WEIGHTS = {
    "redundancy": 0.45,
    "diversity": 0.20,
    "health": 0.35,
}
N_EFF_SHRINKAGE_FLOOR = 0.60
RUN_AS_OF: str | None = None


def current_as_of() -> str:
    return RUN_AS_OF or date.today().isoformat()


def init_db():
    """Create factor_lab.duckdb tables if not exist."""
    FACTOR_LAB_DB.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(FACTOR_LAB_DB))
    con.execute("""
        CREATE TABLE IF NOT EXISTS factor_registry (
            factor_id VARCHAR PRIMARY KEY,
            market VARCHAR NOT NULL,
            name VARCHAR,
            hypothesis VARCHAR,
            formula VARCHAR NOT NULL,
            direction VARCHAR DEFAULT 'long',
            discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            ic_7d DOUBLE, ic_14d DOUBLE, ic_30d DOUBLE,
            ic_ir_7d DOUBLE, ic_ir_14d DOUBLE, ic_ir_30d DOUBLE,
            mono_7d DOUBLE, mono_14d DOUBLE, mono_30d DOUBLE,
            q5_q1_7d DOUBLE, q5_q1_14d DOUBLE, q5_q1_30d DOUBLE,

            composite_score DOUBLE,
            status VARCHAR DEFAULT 'candidate',
            promoted_at TIMESTAMP,
            watchlist_at TIMESTAMP,
            retired_at TIMESTAMP,
            retire_reason VARCHAR,

            last_health_check TIMESTAMP,
            rolling_ic_20d DOUBLE,
            health_watch_count INTEGER DEFAULT 0
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS daily_candidates (
            date DATE NOT NULL,
            market VARCHAR NOT NULL,
            factor_id VARCHAR NOT NULL,
            formula VARCHAR NOT NULL,
            ic DOUBLE, ic_ir DOUBLE, mono DOUBLE, q5_q1 DOUBLE,
            rank INTEGER,
            PRIMARY KEY (date, market, factor_id)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS factor_daily_values (
            factor_id VARCHAR NOT NULL,
            ts_code VARCHAR NOT NULL,
            date DATE NOT NULL,
            value DOUBLE,
            PRIMARY KEY (factor_id, ts_code, date)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS health_log (
            date DATE NOT NULL,
            factor_id VARCHAR NOT NULL,
            rolling_ic_20d DOUBLE,
            status_before VARCHAR,
            status_after VARCHAR,
            PRIMARY KEY (date, factor_id)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS factor_weights (
            as_of DATE NOT NULL,
            market VARCHAR NOT NULL,
            factor_id VARCHAR NOT NULL,
            weight DOUBLE NOT NULL,
            source VARCHAR DEFAULT 'agent_regime',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (as_of, market, factor_id)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            as_of DATE NOT NULL,
            market VARCHAR NOT NULL,
            stage VARCHAR NOT NULL,
            candidate_count INTEGER NOT NULL,
            note VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (as_of, market, stage)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS candidate_feedback (
            date DATE NOT NULL,
            market VARCHAR NOT NULL,
            factor_id VARCHAR NOT NULL,
            feedback_score DOUBLE,
            feedback_multiplier DOUBLE,
            missed_alpha_overlap DOUBLE,
            stale_overlap DOUBLE,
            false_positive_overlap DOUBLE,
            capture_overlap DOUBLE,
            detail_json VARCHAR,
            PRIMARY KEY (date, market, factor_id)
        )
    """)
    con.execute("""
        UPDATE factor_registry
        SET direction = CASE
            WHEN (0.5 * COALESCE(ic_7d, 0.0) + 0.3 * COALESCE(ic_14d, 0.0) + 0.2 * COALESCE(ic_30d, 0.0)) < 0
                THEN 'short'
            ELSE 'long'
        END
        WHERE direction IS NULL
           OR direction NOT IN ('long', 'short')
           OR (
                direction = 'long' AND
                (0.5 * COALESCE(ic_7d, 0.0) + 0.3 * COALESCE(ic_14d, 0.0) + 0.2 * COALESCE(ic_30d, 0.0)) < 0
           )
           OR (
                direction = 'short' AND
                (0.5 * COALESCE(ic_7d, 0.0) + 0.3 * COALESCE(ic_14d, 0.0) + 0.2 * COALESCE(ic_30d, 0.0)) > 0
           )
    """)
    con.close()


def _stable_factor_id(market: str, formula: str) -> str:
    """Deterministic factor id so registry/export names stay stable across runs."""
    digest = hashlib.sha1(formula.encode("utf-8")).hexdigest()[:12]
    return f"{market}_{digest}"


def _infer_direction_from_ic(ic: float) -> str:
    return "short" if ic < 0 else "long"


def _infer_direction_from_horizons(horizon_metrics: dict[int, dict], fallback_ic: float) -> str:
    weighted_ic = sum(
        HORIZON_WEIGHTS.get(h, 0.0) * float(metrics.get("ic", 0.0))
        for h, metrics in horizon_metrics.items()
    )
    if abs(weighted_ic) < 1e-12:
        weighted_ic = fallback_ic
    return _infer_direction_from_ic(weighted_ic)


def _is_blacklisted_formula(formula: str) -> bool:
    """Filter out direct size/liquidity proxies that masquerade as alpha."""
    normalized = formula.replace(" ", "")
    blocked_patterns = [
        "rank(volume)",
        "rank(-volume)",
        "rank(ts_mean(volume,",
        "rank(-ts_mean(volume,",
        "rank(ts_std(volume,",
        "rank(-ts_std(volume,",
        "rank(volume/ts_mean(volume,",
        "rank(-volume/ts_mean(volume,",
        "rank(amount)",
        "rank(-amount)",
        "rank(ts_mean(amount,",
        "rank(-ts_mean(amount,",
        "rank(ts_std(amount,",
        "rank(-ts_std(amount,",
        "abs(ret_1d)/(volume",
        "abs(ret_1d)/(amount",
    ]
    return any(pattern in normalized for pattern in blocked_patterns)


def _cross_sectional_corr(a: pd.Series, b: pd.Series) -> float:
    """Average daily cross-sectional rank correlation, matching the formal gate."""
    common = a.index.intersection(b.index)
    if len(common) <= 100:
        return 0.0

    aligned = pd.DataFrame({
        "a": a.loc[common].values,
        "b": b.loc[common].values,
    }, index=common).dropna()
    if len(aligned) <= 100:
        return 0.0

    corrs = []
    for _, group in aligned.groupby(level="trade_date"):
        if len(group) < 10:
            continue
        corr = group["a"].corr(group["b"], method="spearman")
        if pd.notna(corr):
            corrs.append(abs(float(corr)))
    return float(np.mean(corrs)) if corrs else 0.0


def _candidate_metric_values(candidate: dict) -> list[float]:
    return [
        candidate.get("ic_7d", 0.0), candidate.get("ic_14d", 0.0), candidate.get("ic_30d", 0.0),
        candidate.get("ic_ir_7d", 0.0), candidate.get("ic_ir_14d", 0.0), candidate.get("ic_ir_30d", 0.0),
        candidate.get("mono_7d", 0.0), candidate.get("mono_14d", 0.0), candidate.get("mono_30d", 0.0),
        candidate.get("q5_q1_7d", 0.0), candidate.get("q5_q1_14d", 0.0), candidate.get("q5_q1_30d", 0.0),
    ]


def _latest_factor_snapshot(factor_df: pd.DataFrame) -> pd.Series | None:
    if factor_df.empty:
        return None

    latest_date = factor_df["trade_date"].max()
    latest = factor_df[factor_df["trade_date"] == latest_date][["ts_code", "factor_value"]].dropna()
    if latest.empty:
        return None

    latest = latest.drop_duplicates(subset=["ts_code"], keep="last")
    snapshot = latest.set_index("ts_code")["factor_value"]
    return snapshot if len(snapshot) >= 25 else None


def _load_promoted_factor_snapshots(market: str, prices: pd.DataFrame) -> dict[str, pd.Series]:
    con = duckdb.connect(FACTOR_LAB_DB, read_only=True)
    try:
        rows = con.execute("""
            SELECT factor_id, formula
            FROM factor_registry
            WHERE market=? AND status='promoted'
        """, [market]).fetchall()
    finally:
        con.close()

    snapshots: dict[str, pd.Series] = {}
    for factor_id, formula in rows:
        try:
            factor_df = compute_factor(parse(formula), prices, sym_col="ts_code", date_col="trade_date")
            snapshot = _latest_factor_snapshot(factor_df)
            if snapshot is not None:
                snapshots[factor_id] = snapshot
        except Exception:
            continue
    return snapshots


def _effective_count_shrinkage(
    n_effective: float | None,
    n_total: float | None,
    *,
    floor: float = N_EFF_SHRINKAGE_FLOOR,
) -> float:
    try:
        total = max(float(n_total or 0.0), 0.0)
    except (TypeError, ValueError):
        total = 0.0
    if total <= 1.0:
        return 1.0

    try:
        effective = float(n_effective or total)
    except (TypeError, ValueError):
        effective = total
    effective = min(max(effective, 1.0), total)

    shrink = float(np.sqrt(effective / total))
    return round(float(np.clip(shrink, floor, 1.0)), 4)


def _apply_sigreg_penalties(candidates: list[dict], market: str, prices: pd.DataFrame) -> list[dict]:
    if not candidates:
        return candidates

    from src.evaluate.sigreg import (
        factor_diversity_score,
        ic_health_test,
        multi_collinearity_check,
    )

    promoted_snapshots = _load_promoted_factor_snapshots(market, prices)
    base_diversity = factor_diversity_score(promoted_snapshots) if len(promoted_snapshots) >= 3 else {
        "diversity_score": 1.0,
        "cluster_warning": False,
        "n_effective": len(promoted_snapshots),
        "n_total": len(promoted_snapshots),
    }

    print(
        "  SigReg penalty layer: "
        f"{len(promoted_snapshots)} promoted refs, "
        f"base_diversity={base_diversity.get('diversity_score', 1.0):.3f}"
    )

    penalized = []
    for candidate in candidates:
        latest_values = candidate.get("_latest_values")
        ic_series = np.asarray(candidate.pop("_ic_series", []), dtype=float)
        existing_snapshots = {
            fid: snap for fid, snap in promoted_snapshots.items() if fid != candidate["factor_id"]
        }

        redundancy = {"r_squared": 0.0, "is_redundant": False, "top_contributors": []}
        if latest_values is not None and len(existing_snapshots) >= 3:
            redundancy = multi_collinearity_check(
                latest_values,
                existing_snapshots,
                threshold=SIGREG_REDUNDANCY_HARD,
            )
        redundancy_penalty = float(np.clip(
            (float(redundancy.get("r_squared", 0.0)) - SIGREG_REDUNDANCY_SOFT)
            / max(SIGREG_REDUNDANCY_HARD - SIGREG_REDUNDANCY_SOFT, 1e-9),
            0.0,
            1.0,
        ))
        if redundancy.get("is_redundant"):
            redundancy_penalty = max(redundancy_penalty, 0.8)

        diversity_before = (
            factor_diversity_score(existing_snapshots)
            if len(existing_snapshots) >= 3
            else base_diversity
        )
        diversity_after = diversity_before
        diversity_penalty = 0.0
        if latest_values is not None:
            trial = dict(existing_snapshots)
            trial[candidate["factor_id"]] = latest_values
            diversity_after = factor_diversity_score(trial)
            before_score = float(diversity_before.get("diversity_score", 1.0))
            after_score = float(diversity_after.get("diversity_score", 1.0))
            diversity_drop = max(0.0, before_score - after_score)
            diversity_penalty = float(np.clip(diversity_drop / 0.25, 0.0, 1.0))
            if diversity_after.get("cluster_warning") and after_score < before_score:
                diversity_penalty = min(1.0, diversity_penalty + 0.15)

        health = {"health_score": 0.5, "regime_change_detected": False}
        health_penalty = 0.0
        clean_ic = ic_series[~np.isnan(ic_series)]
        if len(clean_ic) >= 20:
            health = ic_health_test(
                clean_ic,
                window=min(60, max(20, len(clean_ic) // 2)),
            )
            health_penalty = float(np.clip(
                (0.60 - float(health.get("health_score", 0.5))) / 0.60,
                0.0,
                1.0,
            ))
            if health.get("regime_change_detected"):
                health_penalty = min(1.0, health_penalty + 0.20)

        total_penalty = (
            SIGREG_PENALTY_WEIGHTS["redundancy"] * redundancy_penalty
            + SIGREG_PENALTY_WEIGHTS["diversity"] * diversity_penalty
            + SIGREG_PENALTY_WEIGHTS["health"] * health_penalty
        )
        multiplier = max(0.15, 1.0 - total_penalty)
        n_eff = float(diversity_after.get("n_effective", len(existing_snapshots) + 1))
        n_total = float(diversity_after.get("n_total", len(existing_snapshots) + 1))
        n_eff_shrinkage = _effective_count_shrinkage(n_eff, n_total)

        candidate["sigreg_penalty"] = round(total_penalty, 3)
        candidate["sigreg_multiplier"] = round(multiplier, 3)
        candidate["sigreg_n_effective"] = round(n_eff, 2)
        candidate["sigreg_n_total"] = round(n_total, 2)
        candidate["sigreg_n_eff_shrinkage"] = n_eff_shrinkage
        candidate["rank_score"] = round(candidate["composite_score"] * multiplier * n_eff_shrinkage, 6)
        candidate["sigreg_redundancy_r2"] = float(redundancy.get("r_squared", 0.0))
        candidate["sigreg_diversity_score"] = float(diversity_after.get("diversity_score", 1.0))
        candidate["sigreg_health_score"] = float(health.get("health_score", 0.5))
        candidate["sigreg_regime_change"] = bool(health.get("regime_change_detected", False))
        penalized.append(candidate)

    penalized.sort(key=lambda r: r.get("rank_score", r["composite_score"]), reverse=True)

    for candidate in penalized[:5]:
        print(
            "    "
            f"{candidate['name']}: raw={candidate['composite_score']:.4f}, "
            f"rank={candidate.get('rank_score', candidate['composite_score']):.4f}, "
            f"R2={candidate.get('sigreg_redundancy_r2', 0.0):.3f}, "
            f"div={candidate.get('sigreg_diversity_score', 1.0):.3f}, "
            f"health={candidate.get('sigreg_health_score', 0.5):.2f}, "
            f"n_eff={candidate.get('sigreg_n_effective', 1.0):.1f}/"
            f"{candidate.get('sigreg_n_total', 1.0):.1f}, "
            f"shrink={candidate.get('sigreg_n_eff_shrinkage', 1.0):.3f}"
            + (" regime_change" if candidate.get("sigreg_regime_change") else "")
        )

    return penalized


def _load_report_feedback(market: str) -> dict[str, dict[str, float]]:
    if market == "cn":
        candidates = [QUANT_CN_REPORT_DB, QUANT_CN_DB]
    else:
        candidates = [QUANT_US_DB, QUANT_US_REPORT_DB]

    rows = []
    source_table = None
    for pipeline_db in candidates:
        if not pipeline_db.exists():
            continue
        try:
            con = duckdb.connect(str(pipeline_db), read_only=True)
        except Exception as exc:
            print(f"  Report feedback unavailable ({market}, {pipeline_db.name}): {exc}")
            continue

        try:
            rows = con.execute("""
                SELECT symbol, label, feedback_action, feedback_weight
                FROM algorithm_postmortem
                WHERE evaluation_date >= CURRENT_DATE - INTERVAL '45 days'
                  AND feedback_action IS NOT NULL
                  AND feedback_weight IS NOT NULL
            """).fetchall()
            if rows:
                source_table = "algorithm_postmortem"
                break
        except duckdb.Error:
            rows = []

        try:
            rows = con.execute("""
                SELECT symbol, label, factor_feedback_action, factor_feedback_weight
                FROM alpha_postmortem
                WHERE evaluation_date >= CURRENT_DATE - INTERVAL '45 days'
                  AND factor_feedback_action IS NOT NULL
                  AND factor_feedback_weight IS NOT NULL
            """).fetchall()
            if rows:
                source_table = "alpha_postmortem"
                break
        except duckdb.Error:
            rows = []
        finally:
            try:
                con.close()
            except Exception:
                pass

    if not rows:
        return {}

    buckets: dict[str, dict[str, float]] = {
        "missed_alpha": {},
        "stale": {},
        "false_positive": {},
        "captured": {},
    }
    for symbol, label, action, weight in rows:
        try:
            w = float(weight)
        except (TypeError, ValueError):
            continue
        if not symbol or w <= 0:
            continue
        if label == "missed_alpha" or action == "boost_recall":
            buckets["missed_alpha"][symbol] = buckets["missed_alpha"].get(symbol, 0.0) + w
        elif (
            label
            in {
                "alpha_already_paid",
                "good_signal_bad_timing",
                "stale_chase",
                "right_but_no_fill",
            }
            or action == "penalize_stale_chase"
        ):
            buckets["stale"][symbol] = buckets["stale"].get(symbol, 0.0) + w
        elif (
            label in {"false_positive", "false_positive_executable", "wrong_way_executable"}
            or action == "penalize_false_positive"
        ):
            buckets["false_positive"][symbol] = buckets["false_positive"].get(symbol, 0.0) + w
        elif (
            label in {"captured", "won_and_executable"}
            or action in {"reward_capture", "reward_executable_capture"}
        ):
            buckets["captured"][symbol] = buckets["captured"].get(symbol, 0.0) + w

    total = sum(len(v) for v in buckets.values())
    if total:
        print(
            f"  Report feedback overlay ({source_table or 'unknown'}): "
            f"missed={len(buckets['missed_alpha'])}, "
            f"stale={len(buckets['stale'])}, "
            f"false_positive={len(buckets['false_positive'])}, "
            f"captured={len(buckets['captured'])}"
        )
    return buckets


def _feedback_overlap(symbols: list[str], weights: dict[str, float]) -> float:
    if not symbols or not weights:
        return 0.0
    denom = max(1.0, len(symbols) / 5.0)
    return round(sum(weights.get(sym, 0.0) for sym in symbols) / denom, 4)


def _basket_symbols(latest_values: pd.Series, direction: str) -> list[str]:
    if latest_values is None or len(latest_values) == 0:
        return []
    clean = latest_values.dropna()
    if clean.empty:
        return []

    basket_n = max(10, min(50, int(len(clean) * 0.03)))
    ranked = clean.sort_values(ascending=(direction == "short"))
    return list(ranked.head(basket_n).index)


def _apply_report_feedback(candidates: list[dict], market: str) -> list[dict]:
    feedback = _load_report_feedback(market)
    if not feedback:
        return candidates

    adjusted = []
    for candidate in candidates:
        latest_values = candidate.get("_latest_values")
        basket = _basket_symbols(latest_values, candidate.get("direction", "long"))
        missed_overlap = _feedback_overlap(basket, feedback.get("missed_alpha", {}))
        stale_overlap = _feedback_overlap(basket, feedback.get("stale", {}))
        false_overlap = _feedback_overlap(basket, feedback.get("false_positive", {}))
        capture_overlap = _feedback_overlap(basket, feedback.get("captured", {}))

        feedback_score = round(
            1.30 * missed_overlap
            + 0.35 * capture_overlap
            - 0.75 * stale_overlap
            - 1.00 * false_overlap,
            4,
        )
        raw_multiplier = float(np.clip(1.0 + 0.25 * feedback_score, 0.85, 1.20))
        n_eff_shrinkage = float(candidate.get("sigreg_n_eff_shrinkage", 1.0) or 1.0)
        multiplier = float(np.clip(1.0 + (raw_multiplier - 1.0) * n_eff_shrinkage, 0.85, 1.20))
        candidate["report_feedback_score"] = feedback_score
        candidate["report_feedback_multiplier_raw"] = round(raw_multiplier, 4)
        candidate["report_feedback_multiplier"] = round(multiplier, 4)
        candidate["report_feedback_detail"] = {
            "missed_alpha_overlap": missed_overlap,
            "stale_chase_overlap": stale_overlap,
            "stale_overlap": stale_overlap,
            "false_positive_overlap": false_overlap,
            "captured_overlap": capture_overlap,
            "capture_overlap": capture_overlap,
            "basket_n": len(basket),
            "n_eff_shrinkage": round(n_eff_shrinkage, 4),
        }
        base_rank = candidate.get("rank_score", candidate["composite_score"])
        candidate["rank_score"] = round(base_rank * multiplier, 6)
        adjusted.append(candidate)

    adjusted.sort(key=lambda r: r.get("rank_score", r["composite_score"]), reverse=True)
    return adjusted


def _refresh_promoted_factor(con: duckdb.DuckDBPyConnection, factor_id: str, candidate: dict) -> None:
    con.execute("""
        UPDATE factor_registry SET
            name=?, hypothesis=?, composite_score=?, direction=?,
            ic_7d=?, ic_14d=?, ic_30d=?,
            ic_ir_7d=?, ic_ir_14d=?, ic_ir_30d=?,
            mono_7d=?, mono_14d=?, mono_30d=?,
            q5_q1_7d=?, q5_q1_14d=?, q5_q1_30d=?
        WHERE factor_id=?
    """, [
        candidate["name"], candidate["hypothesis"], candidate["composite_score"], candidate["direction"],
        *_candidate_metric_values(candidate),
        factor_id,
    ])


def _promote_factor(
    con: duckdb.DuckDBPyConnection,
    market: str,
    candidate: dict,
    existing_factor_id: str | None = None,
) -> None:
    if existing_factor_id:
        con.execute("""
            UPDATE factor_registry SET
                name=?, hypothesis=?, formula=?, direction=?, composite_score=?,
                status='promoted', promoted_at=CURRENT_TIMESTAMP,
                watchlist_at=NULL, retired_at=NULL, retire_reason=NULL,
                health_watch_count=0,
                ic_7d=?, ic_14d=?, ic_30d=?,
                ic_ir_7d=?, ic_ir_14d=?, ic_ir_30d=?,
                mono_7d=?, mono_14d=?, mono_30d=?,
                q5_q1_7d=?, q5_q1_14d=?, q5_q1_30d=?
            WHERE factor_id=?
        """, [
            candidate["name"], candidate["hypothesis"], candidate["formula"],
            candidate["direction"], candidate["composite_score"],
            *_candidate_metric_values(candidate),
            existing_factor_id,
        ])
        return

    con.execute("""
        INSERT INTO factor_registry
            (factor_id, market, name, hypothesis, formula, direction, composite_score,
             status, promoted_at,
             ic_7d, ic_14d, ic_30d, ic_ir_7d, ic_ir_14d, ic_ir_30d,
             mono_7d, mono_14d, mono_30d, q5_q1_7d, q5_q1_14d, q5_q1_30d)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'promoted', CURRENT_TIMESTAMP,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        candidate["factor_id"], market, candidate["name"], candidate["hypothesis"],
        candidate["formula"], candidate["direction"], candidate["composite_score"],
        *_candidate_metric_values(candidate),
    ])


def _save_factor_weights(market: str, weights: dict[str, float]) -> None:
    if not weights:
        return

    init_db()
    con = duckdb.connect(FACTOR_LAB_DB)
    as_of = current_as_of()
    for factor_id, weight in weights.items():
        con.execute("""
            INSERT OR REPLACE INTO factor_weights (as_of, market, factor_id, weight, source)
            VALUES (?, ?, ?, ?, 'agent_regime')
        """, [as_of, market, factor_id, float(weight)])
    con.close()


def _log_pipeline_run(market: str, stage: str, candidate_count: int, note: str = "") -> None:
    init_db()
    con = duckdb.connect(FACTOR_LAB_DB)
    con.execute("""
        INSERT OR REPLACE INTO pipeline_runs (as_of, market, stage, candidate_count, note)
        VALUES (?, ?, ?, ?, ?)
    """, [current_as_of(), market, stage, int(candidate_count), note])
    con.close()


def step1_mine(market: str, max_factors: int = 500) -> list[dict]:
    """Mine factors: generate formulas, compute IC, return top candidates."""
    cfg = CONFIGS[market]
    print(f"\n[1/4] Mining {max_factors} factors for {market.upper()}...")

    con = duckdb.connect(cfg["db_path"], read_only=True)
    prices = con.execute(cfg["sql"]).fetchdf()
    con.close()

    # Universe filter: keep only top N stocks by market_cap each day
    top_n = cfg.get("universe_top_n")
    if top_n and "market_cap" in prices.columns:
        prices["_mcap_rank"] = prices.groupby("trade_date")["market_cap"].rank(
            ascending=False, method="first", na_option="bottom"
        )
        prices = prices[prices["_mcap_rank"] <= top_n].drop(columns=["_mcap_rank"]).reset_index(drop=True)

    fwd = compute_forward_returns(
        cfg["db_path"], cfg["table"], cfg["date_col"],
        cfg["close_col"], cfg["sym_col"] if market == "cn" else "symbol"
    )
    if market == "us":
        fwd = fwd.rename(columns={"symbol": "ts_code", "date": "trade_date"})

    formulas = generate_factor_formulas(max_factors)
    print(f"  Generated {len(formulas)} unique formulas")

    results = []
    errors = 0
    for i, (name, formula, hypothesis) in enumerate(formulas):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(formulas)} ({len(results)} valid)")
        try:
            ast = parse(formula)
            factor_df = compute_factor(ast, prices, sym_col="ts_code", date_col="trade_date")
            merged = factor_df.merge(
                fwd[["ts_code", "trade_date", "fwd_5d"]],
                on=["ts_code", "trade_date"], how="inner"
            ).dropna(subset=["fwd_5d", "factor_value"])

            if len(merged) < 500:
                continue

            ic_stats = ic_summary(compute_ic_series(
                merged["factor_value"], merged["fwd_5d"], merged["trade_date"]
            ))
            q = compute_quintile_returns(
                merged["factor_value"], merged["fwd_5d"], merged["trade_date"]
            )

            # Score: reward IC×IR alignment with monotonicity
            ic_mono_sign = 1 if (ic_stats["ic_mean"] * q["monotonicity"] >= 0) else -1
            score = abs(ic_stats["ic_mean"]) * abs(ic_stats["ic_ir"]) * ic_mono_sign

            results.append({
                "name": name, "formula": formula, "hypothesis": hypothesis,
                "ic": ic_stats["ic_mean"], "ic_ir": ic_stats["ic_ir"],
                "q5_q1": q["long_short_pct"], "mono": q["monotonicity"],
                "score": score,
                "direction": _infer_direction_from_ic(ic_stats["ic_mean"]),
                "factor_id": _stable_factor_id(market, formula),
                "_values": merged.set_index(["ts_code", "trade_date"])["factor_value"],  # for correlation
            })
        except Exception:
            errors += 1

    results = [r for r in results if not _is_blacklisted_formula(r["formula"])]

    # Filter by gates
    gates = GATE_THRESHOLDS[market]
    ic_pass = [r for r in results if abs(r["ic"]) >= gates["ic_min"]]
    ir_pass = [r for r in ic_pass if abs(r["ic_ir"]) >= gates["ir_min"]]
    passed = [r for r in ir_pass if abs(r["mono"]) >= gates["mono_min"]]
    print(
        "  Gate attrition: "
        f"valid={len(results)} -> ic={len(ic_pass)} -> ic+ir={len(ir_pass)} -> ic+ir+mono={len(passed)}"
    )

    # Sort by score
    passed.sort(key=lambda r: r["score"], reverse=True)

    # Bootstrap significance test on gate-passed factors (Rust by default)
    try:
        from src.evaluate.gpu_bootstrap import batch_bootstrap
        print(f"  Running bootstrap significance on {len(passed)} gate-passed factors...")
        passed = batch_bootstrap(passed, prices, fwd, n_bootstrap=100_000, backend="rust")
        before_bootstrap = len(passed)
        passed = [r for r in passed if r.get("bootstrap_significant", False)]
        print(f"  Bootstrap filter: {before_bootstrap} → {len(passed)} significant (p<0.01)")
    except Exception as e:
        print(f"  Bootstrap skipped: {e}")

    # Correlation dedup: greedily keep factors with corr < 0.7 vs already-kept
    kept = []
    for candidate in passed:
        if len(kept) >= CANDIDATE_POOL_SIZE:
            break
        is_redundant = False
        cand_vals = candidate.get("_values")
        if cand_vals is not None:
            for existing in kept:
                exist_vals = existing.get("_values")
                if exist_vals is not None:
                    # Align and compute rank correlation
                    common = cand_vals.index.intersection(exist_vals.index)
                    if len(common) > 100:
                        corr = _cross_sectional_corr(cand_vals, exist_vals)
                        if abs(corr) > 0.6:
                            is_redundant = True
                            break
        if not is_redundant:
            kept.append(candidate)

    # Clean up _values (large, not needed downstream)
    for c in kept:
        c.pop("_values", None)
    for r in results:
        r.pop("_values", None)

    candidates = kept
    print(f"  Results: {len(results)} valid, {len(passed)} passed gates, {len(candidates)} after corr dedup")
    return candidates


def step2_multi_horizon_backtest(candidates: list[dict], market: str) -> list[dict]:
    """Run multi-horizon backtest (7D/14D/30D) on candidates."""
    print(f"\n[2/4] Multi-horizon backtest on {len(candidates)} candidates...")

    cfg = CONFIGS[market]
    con = duckdb.connect(cfg["db_path"], read_only=True)
    prices = con.execute(cfg["sql"]).fetchdf()
    con.close()

    # Universe filter
    top_n = cfg.get("universe_top_n")
    if top_n and "market_cap" in prices.columns:
        prices["_mcap_rank"] = prices.groupby("trade_date")["market_cap"].rank(
            ascending=False, method="first", na_option="bottom"
        )
        prices = prices[prices["_mcap_rank"] <= top_n].drop(columns=["_mcap_rank"]).reset_index(drop=True)

    # Compute forward returns for all horizons
    fwd_all = {}
    for h in HORIZONS:
        fwd_h = compute_forward_returns(
            cfg["db_path"], cfg["table"], cfg["date_col"],
            cfg["close_col"], cfg["sym_col"] if market == "cn" else "symbol",
            horizons=[h]
        )
        if market == "us":
            fwd_h = fwd_h.rename(columns={"symbol": "ts_code", "date": "trade_date"})
        fwd_all[h] = fwd_h

    enriched = []
    shortest_horizon = min(HORIZONS)
    for c in candidates:
        try:
            ast = parse(c["formula"])
            factor_df = compute_factor(ast, prices, sym_col="ts_code", date_col="trade_date")
            latest_values = _latest_factor_snapshot(factor_df)
            reference_ic_series = np.array([], dtype=float)

            horizon_metrics = {}
            for h in HORIZONS:
                fwd_col = f"fwd_{h}d"
                merged = factor_df.merge(
                    fwd_all[h][["ts_code", "trade_date", fwd_col]],
                    on=["ts_code", "trade_date"], how="inner"
                ).dropna(subset=[fwd_col, "factor_value"])

                if len(merged) < 200:
                    continue

                ic_df = compute_ic_series(
                    merged["factor_value"], merged[fwd_col], merged["trade_date"]
                )
                ic_stats = ic_summary(ic_df)
                q = compute_quintile_returns(
                    merged["factor_value"], merged[fwd_col], merged["trade_date"]
                )
                horizon_metrics[h] = {
                    "ic": ic_stats["ic_mean"], "ic_ir": ic_stats["ic_ir"],
                    "mono": q["monotonicity"], "q5_q1": q["long_short_pct"],
                }
                if h == shortest_horizon and len(ic_df) > 0:
                    reference_ic_series = ic_df["ic"].to_numpy(dtype=float)

            if len(horizon_metrics) < 2:
                continue

            # Composite score across horizons (weight recent more)
            composite = sum(
                HORIZON_WEIGHTS.get(h, 0) * abs(m["ic"]) * abs(m["ic_ir"])
                for h, m in horizon_metrics.items()
                if m["ic"] * m.get("mono", 0) >= 0  # penalize IC/mono disagreement
            )

            c.update({
                f"ic_{h}d": horizon_metrics.get(h, {}).get("ic", 0) for h in HORIZONS
            })
            c.update({
                f"ic_ir_{h}d": horizon_metrics.get(h, {}).get("ic_ir", 0) for h in HORIZONS
            })
            c.update({
                f"mono_{h}d": horizon_metrics.get(h, {}).get("mono", 0) for h in HORIZONS
            })
            c.update({
                f"q5_q1_{h}d": horizon_metrics.get(h, {}).get("q5_q1", 0) for h in HORIZONS
            })
            c["composite_score"] = composite
            c["direction"] = _infer_direction_from_horizons(horizon_metrics, c["ic"])
            c["_latest_values"] = latest_values
            c["_ic_series"] = reference_ic_series
            enriched.append(c)

        except Exception as e:
            print(f"  {c['name']}: backtest error - {e}")

    enriched = _apply_sigreg_penalties(enriched, market, prices)
    enriched = _apply_report_feedback(enriched, market)
    print(f"  Enriched: {len(enriched)} candidates with multi-horizon metrics")
    return enriched


def step3_select_and_promote(candidates: list[dict], market: str):
    """Refresh active factors, then promote top-ranked new factors into open slots."""
    print(f"\n[Promote] Refreshing active factors and promoting replacements...")

    init_db()
    con = duckdb.connect(FACTOR_LAB_DB)
    today = current_as_of()

    con.execute("DELETE FROM daily_candidates WHERE date=? AND market=?", [today, market])
    con.execute("DELETE FROM candidate_feedback WHERE date=? AND market=?", [today, market])

    # Save all candidates to daily_candidates
    for i, c in enumerate(candidates):
        con.execute("""
            INSERT OR REPLACE INTO daily_candidates (date, market, factor_id, formula, ic, ic_ir, mono, q5_q1, rank)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [today, market, c["factor_id"], c["formula"], c["ic"], c["ic_ir"], c["mono"], c["q5_q1"], i + 1])
        if "report_feedback_score" in c:
            con.execute("""
                INSERT OR REPLACE INTO candidate_feedback (
                    date, market, factor_id, feedback_score, feedback_multiplier,
                    missed_alpha_overlap, stale_overlap, false_positive_overlap,
                    capture_overlap, detail_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                today,
                market,
                c["factor_id"],
                c.get("report_feedback_score"),
                c.get("report_feedback_multiplier"),
                (c.get("report_feedback_detail") or {}).get("missed_alpha_overlap"),
                (c.get("report_feedback_detail") or {}).get("stale_overlap"),
                (c.get("report_feedback_detail") or {}).get("false_positive_overlap"),
                (c.get("report_feedback_detail") or {}).get("capture_overlap"),
                json.dumps(c.get("report_feedback_detail") or {}, ensure_ascii=True),
            ])

    existing_rows = con.execute("""
        SELECT factor_id, formula, status
        FROM factor_registry
        WHERE market=?
    """, [market]).fetchall()
    existing_by_formula = {
        formula: {"factor_id": factor_id, "status": status}
        for factor_id, formula, status in existing_rows
    }

    refreshed = 0
    for c in candidates:
        existing = existing_by_formula.get(c["formula"])
        if existing and existing["status"] == "promoted":
            _refresh_promoted_factor(con, existing["factor_id"], c)
            refreshed += 1

    existing_promoted = con.execute(
        "SELECT COUNT(*) FROM factor_registry WHERE market=? AND status='promoted'", [market]
    ).fetchone()[0]

    promote_slots = max(0, MAX_PROMOTED - existing_promoted)
    promotions_allowed = min(PROMOTE_COUNT, promote_slots)
    promoted = 0
    for c in candidates:
        if promoted >= promotions_allowed:
            break

        existing = existing_by_formula.get(c["formula"])
        if existing and existing["status"] == "promoted":
            continue

        _promote_factor(
            con,
            market,
            c,
            existing_factor_id=existing["factor_id"] if existing else None,
        )
        promoted += 1

        if existing:
            existing["status"] = "promoted"
            print(
                f"  Re-promoted: {c['name']} "
                f"(raw={c['composite_score']:.4f}, rank={c.get('rank_score', c['composite_score']):.4f}, "
                f"direction={c['direction']})"
            )
        else:
            print(f"  Promoted: {c['name']} — {c['formula'][:60]}")
            print(
                f"    IC 7/14/30d: {c.get('ic_7d',0):.4f} / "
                f"{c.get('ic_14d',0):.4f} / {c.get('ic_30d',0):.4f} ({c['direction']})"
            )
            print(
                f"    rank={c.get('rank_score', c['composite_score']):.4f} "
                f"raw={c['composite_score']:.4f} "
                f"sigreg_penalty={c.get('sigreg_penalty', 0.0):.3f}"
            )

    con.close()
    print(f"  Refreshed {refreshed} active factors")
    print(f"  Promoted {promoted} new factors ({existing_promoted} active before fill, {promote_slots} slots available)")


def step4_health_check(market: str):
    """Check promoted factors, move decaying ones to watchlist/retired."""
    print(f"\n[Health] Checking {market.upper()} promoted factors...")

    init_db()
    con = duckdb.connect(FACTOR_LAB_DB)
    cfg = CONFIGS[market]
    today = current_as_of()
    from src.evaluate.sigreg import ic_health_test

    # Get all promoted + watchlist factors
    active = con.execute("""
        SELECT factor_id, formula, status, health_watch_count
        FROM factor_registry
        WHERE market=? AND status IN ('promoted', 'watchlist')
    """, [market]).fetchall()

    if not active:
        print("  No active factors to check")
        con.close()
        return

    # Load recent prices for IC computation
    db_con = duckdb.connect(cfg["db_path"], read_only=True)
    prices = db_con.execute(cfg["sql"]).fetchdf()
    db_con.close()

    # Universe filter
    top_n = cfg.get("universe_top_n")
    if top_n and "market_cap" in prices.columns:
        prices["_mcap_rank"] = prices.groupby("trade_date")["market_cap"].rank(
            ascending=False, method="first", na_option="bottom"
        )
        prices = prices[prices["_mcap_rank"] <= top_n].drop(columns=["_mcap_rank"]).reset_index(drop=True)

    fwd = compute_forward_returns(
        cfg["db_path"], cfg["table"], cfg["date_col"],
        cfg["close_col"], cfg["sym_col"] if market == "cn" else "symbol",
        horizons=[5]
    )
    if market == "us":
        fwd = fwd.rename(columns={"symbol": "ts_code", "date": "trade_date"})

    for factor_id, formula, status, watch_count in active:
        try:
            ast = parse(formula)
            factor_df = compute_factor(ast, prices, sym_col="ts_code", date_col="trade_date")

            # Last 20 trading days IC
            merged = factor_df.merge(
                fwd[["ts_code", "trade_date", "fwd_5d"]],
                on=["ts_code", "trade_date"], how="inner"
            ).dropna(subset=["fwd_5d", "factor_value"])

            # Get last 20 days
            recent_dates = sorted(merged["trade_date"].unique())[-20:]
            recent = merged[merged["trade_date"].isin(recent_dates)]

            if len(recent) < 100:
                rolling_ic = 0.0
            else:
                ic_series = compute_ic_series(
                    recent["factor_value"], recent["fwd_5d"], recent["trade_date"]
                )
                rolling_ic = float(ic_series["ic"].mean()) if len(ic_series) > 0 else 0.0

            full_ic_series = compute_ic_series(
                merged["factor_value"], merged["fwd_5d"], merged["trade_date"]
            )
            health = {"health_score": 0.5, "regime_change_detected": False}
            if len(full_ic_series) >= 20:
                health = ic_health_test(
                    full_ic_series["ic"].to_numpy(dtype=float),
                    window=min(60, max(20, len(full_ic_series) // 2)),
                )
            health_score = float(health.get("health_score", 0.5))
            regime_change = bool(health.get("regime_change_detected", False))

            # Lifecycle transitions
            new_status = status
            new_watch = watch_count
            if status == "promoted":
                # Use abs(IC) for health check: short factors have negative IC by design
                reasons = []
                if abs(rolling_ic) < HEALTH_WATCH_IC:
                    reasons.append("IC too low")
                if health_score < SIGREG_HEALTH_WATCH_SCORE:
                    reasons.append(f"SigReg health={health_score:.2f}")
                if regime_change:
                    reasons.append("regime change")
                is_unhealthy = bool(reasons)
                if is_unhealthy:
                    new_watch = watch_count + 1
                    if new_watch >= HEALTH_WATCH_DAYS:
                        new_status = "watchlist"
                        print(
                            f"  ⚠️ {factor_id}: promoted → watchlist "
                            f"(IC={rolling_ic:.4f}, health={health_score:.2f}, "
                            f"{', '.join(reasons)}, {new_watch}d)"
                        )
                    else:
                        print(
                            f"  📉 {factor_id}: IC={rolling_ic:.4f}, health={health_score:.2f} "
                            f"({' ; '.join(reasons)}) "
                            f"({new_watch}/{HEALTH_WATCH_DAYS} watch days)"
                        )
                else:
                    new_watch = 0  # reset counter
                    print(f"  ✅ {factor_id}: IC={rolling_ic:.4f}, health={health_score:.2f} healthy")

            elif status == "watchlist":
                if abs(rolling_ic) >= HEALTH_RECOVER_IC and health_score >= SIGREG_HEALTH_RECOVER_SCORE and not regime_change:
                    new_status = "promoted"
                    new_watch = 0
                    print(
                        f"  🔄 {factor_id}: watchlist → promoted "
                        f"(IC recovered to {rolling_ic:.4f}, health={health_score:.2f})"
                    )
                else:
                    new_watch = watch_count + 1
                    if new_watch >= HEALTH_WATCH_DAYS + HEALTH_RETIRE_DAYS:
                        new_status = "retired"
                        print(
                            f"  ❌ {factor_id}: watchlist → retired "
                            f"(IC={rolling_ic:.4f}, health={health_score:.2f}, no recovery)"
                        )
                    else:
                        remaining = HEALTH_WATCH_DAYS + HEALTH_RETIRE_DAYS - new_watch
                        print(
                            f"  ⏳ {factor_id}: watchlist IC={rolling_ic:.4f}, health={health_score:.2f} "
                            f"({remaining}d to retire)"
                        )

            # Update
            update_fields = {
                "promoted": ("promoted_at", None),
                "watchlist": ("watchlist_at", "CURRENT_TIMESTAMP"),
                "retired": ("retired_at", "CURRENT_TIMESTAMP"),
            }

            con.execute("""
                UPDATE factor_registry SET
                    status=?, rolling_ic_20d=?, health_watch_count=?,
                    last_health_check=CURRENT_TIMESTAMP,
                    watchlist_at = CASE WHEN ?='watchlist' AND status!='watchlist' THEN CURRENT_TIMESTAMP ELSE watchlist_at END,
                    retired_at = CASE WHEN ?='retired' THEN CURRENT_TIMESTAMP ELSE retired_at END,
                    retire_reason = CASE WHEN ?='retired' THEN 'IC decay' ELSE retire_reason END
                WHERE factor_id=?
            """, [new_status, rolling_ic, new_watch,
                  new_status, new_status, new_status, factor_id])

            # Log
            con.execute("""
                INSERT OR REPLACE INTO health_log (date, factor_id, rolling_ic_20d, status_before, status_after)
                VALUES (?, ?, ?, ?, ?)
            """, [today, factor_id, rolling_ic, status, new_status])

        except Exception as e:
            print(f"  ❗ {factor_id}: health check error - {e}")

    con.close()


def run(
    market: str,
    skip_mine: bool = False,
    max_factors: int = 500,
    use_agent: bool = True,
    as_of: str | None = None,
):
    """Run the full daily factor pipeline."""
    global RUN_AS_OF
    RUN_AS_OF = as_of

    print(f"{'='*60}")
    print(f"  Daily Factor Pipeline — {market.upper()}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  As-of: {current_as_of()}")
    print(f"{'='*60}")

    init_db()

    enriched = []

    if not skip_mine:
        # Step 1: Mine
        candidates = step1_mine(market, max_factors)

        if candidates:
            # Step 2: Multi-horizon backtest
            enriched = step2_multi_horizon_backtest(candidates, market)

            if enriched:
                # Step 2.5: Agent quality review (optional, fails gracefully)
                if use_agent:
                    try:
                        from src.mining.agent_review import agent_quality_review
                        print("\n[2.5/5] Agent quality review...")
                        enriched = agent_quality_review(enriched, market)
                        if not enriched:
                            print("\n[Agent] Agent review rejected all candidates")
                            _log_pipeline_run(market, "agent_empty", 0, "agent review rejected all candidates")
                    except Exception as e:
                        print(f"\n[2.5/5] Agent review skipped: {e}")

            else:
                print("\n[Backtest] No candidates survived multi-horizon validation")
                _log_pipeline_run(market, "backtest_empty", 0, "no candidates survived multi-horizon validation")
        else:
            print("\n[Mine] No candidates survived mining/bootstrap filters")
            _log_pipeline_run(market, "mine_empty", 0, "no candidates survived mining/bootstrap filters")

    # Health check runs before promotion so same-day retirements free slots immediately.
    step4_health_check(market)

    if enriched:
        step3_select_and_promote(enriched, market)

    # Step 5: Agent regime-aware weight selection (optional)
    if use_agent:
        try:
            from src.mining.agent_review import agent_regime_selection, generate_factor_commentary
            con = duckdb.connect(FACTOR_LAB_DB, read_only=True)
            promoted = con.execute("""
                SELECT factor_id, formula, name, hypothesis, status,
                       ic_7d, ic_14d, ic_30d
                FROM factor_registry WHERE market=? AND status='promoted'
            """, [market]).fetchall()
            con.close()

            if promoted:
                promoted_dicts = [
                    {"factor_id": r[0], "formula": r[1], "name": r[2],
                     "hypothesis": r[3], "status": r[4], "ic_7d": r[5]}
                    for r in promoted
                ]
                print(f"\n[5/5] Agent regime-aware selection ({len(promoted_dicts)} promoted factors)...")
                weights = agent_regime_selection(promoted_dicts, market)
                _save_factor_weights(market, weights)
                for fid, w in weights.items():
                    print(f"  {fid}: {w*100:.1f}%")

                # Generate commentary
                commentary = generate_factor_commentary(
                    [{"name": d["name"], "formula": d["formula"],
                      "ic": d.get("ic_7d", 0), "hypothesis": d["hypothesis"]}
                     for d in promoted_dicts[:3]],
                    market
                )
                if commentary:
                    print(f"\n  Factor commentary: {commentary[:200]}...")

                    # Save commentary to file for pipeline to read
                    commentary_path = Path(f"data/{market}_factor_commentary.txt")
                    commentary_path.write_text(commentary, encoding="utf-8")
        except Exception as e:
            print(f"\n[5/5] Agent selection skipped: {e}")

    # Summary
    con = duckdb.connect(FACTOR_LAB_DB, read_only=True)
    promoted = con.execute(
        "SELECT COUNT(*) FROM factor_registry WHERE market=? AND status='promoted'", [market]
    ).fetchone()[0]
    watchlist = con.execute(
        "SELECT COUNT(*) FROM factor_registry WHERE market=? AND status='watchlist'", [market]
    ).fetchone()[0]
    retired = con.execute(
        "SELECT COUNT(*) FROM factor_registry WHERE market=? AND status='retired'", [market]
    ).fetchone()[0]
    con.close()

    print(f"\n{'='*60}")
    print(f"  Summary: {promoted} promoted, {watchlist} watchlist, {retired} retired")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Daily factor pipeline")
    parser.add_argument("--market", choices=["cn", "us"], required=True)
    parser.add_argument("--skip-mine", action="store_true", help="Skip mining, only health check")
    parser.add_argument("--max-factors", type=int, default=500)
    parser.add_argument("--no-agent", action="store_true", help="Skip agent review/selection")
    parser.add_argument("--date", "--as-of", dest="as_of", type=str, default=None, help="As-of date YYYY-MM-DD")
    args = parser.parse_args()
    run(args.market, args.skip_mine, args.max_factors, use_agent=not args.no_agent, as_of=args.as_of)


if __name__ == "__main__":
    main()
