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
    0 6 * * 1-5 cd /home/ivena/coding/python/factor-lab && python3 -m src.mining.daily_pipeline --market cn >> logs/daily_cn.log 2>&1
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

# ── Config ────────────────────────────────────────────────────────────────────

FACTOR_LAB_DB = "/home/ivena/coding/python/factor-lab/data/factor_lab.duckdb"
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


def init_db():
    """Create factor_lab.duckdb tables if not exist."""
    Path(FACTOR_LAB_DB).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(FACTOR_LAB_DB)
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
    as_of = date.today().isoformat()
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
    """, [date.today().isoformat(), market, stage, int(candidate_count), note])
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
    for c in candidates:
        try:
            ast = parse(c["formula"])
            factor_df = compute_factor(ast, prices, sym_col="ts_code", date_col="trade_date")

            horizon_metrics = {}
            for h in HORIZONS:
                fwd_col = f"fwd_{h}d"
                merged = factor_df.merge(
                    fwd_all[h][["ts_code", "trade_date", fwd_col]],
                    on=["ts_code", "trade_date"], how="inner"
                ).dropna(subset=[fwd_col, "factor_value"])

                if len(merged) < 200:
                    continue

                ic_stats = ic_summary(compute_ic_series(
                    merged["factor_value"], merged[fwd_col], merged["trade_date"]
                ))
                q = compute_quintile_returns(
                    merged["factor_value"], merged[fwd_col], merged["trade_date"]
                )
                horizon_metrics[h] = {
                    "ic": ic_stats["ic_mean"], "ic_ir": ic_stats["ic_ir"],
                    "mono": q["monotonicity"], "q5_q1": q["long_short_pct"],
                }

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
            enriched.append(c)

        except Exception as e:
            print(f"  {c['name']}: backtest error - {e}")

    enriched.sort(key=lambda r: r["composite_score"], reverse=True)
    print(f"  Enriched: {len(enriched)} candidates with multi-horizon metrics")
    return enriched


def step3_select_and_promote(candidates: list[dict], market: str):
    """Refresh active factors, then promote top-ranked new factors into open slots."""
    print(f"\n[Promote] Refreshing active factors and promoting replacements...")

    init_db()
    con = duckdb.connect(FACTOR_LAB_DB)
    today = date.today().isoformat()

    # Save all candidates to daily_candidates
    for i, c in enumerate(candidates):
        con.execute("""
            INSERT OR REPLACE INTO daily_candidates (date, market, factor_id, formula, ic, ic_ir, mono, q5_q1, rank)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [today, market, c["factor_id"], c["formula"], c["ic"], c["ic_ir"], c["mono"], c["q5_q1"], i + 1])

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
            print(f"  Re-promoted: {c['name']} (composite={c['composite_score']:.4f}, direction={c['direction']})")
        else:
            print(f"  Promoted: {c['name']} — {c['formula'][:60]}")
            print(
                f"    IC 7/14/30d: {c.get('ic_7d',0):.4f} / "
                f"{c.get('ic_14d',0):.4f} / {c.get('ic_30d',0):.4f} ({c['direction']})"
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
    today = date.today().isoformat()

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

            # Lifecycle transitions
            new_status = status
            if status == "promoted":
                if abs(rolling_ic) < HEALTH_WATCH_IC:
                    new_watch = watch_count + 1
                    if new_watch >= HEALTH_WATCH_DAYS:
                        new_status = "watchlist"
                        print(f"  ⚠️ {factor_id}: promoted → watchlist (IC={rolling_ic:.4f}, {new_watch}d below threshold)")
                    else:
                        print(f"  📉 {factor_id}: IC={rolling_ic:.4f} ({new_watch}/{HEALTH_WATCH_DAYS} watch days)")
                else:
                    new_watch = 0  # reset counter
                    print(f"  ✅ {factor_id}: IC={rolling_ic:.4f} healthy")

            elif status == "watchlist":
                if abs(rolling_ic) >= HEALTH_RECOVER_IC:
                    new_status = "promoted"
                    new_watch = 0
                    print(f"  🔄 {factor_id}: watchlist → promoted (IC recovered to {rolling_ic:.4f})")
                else:
                    new_watch = watch_count + 1
                    if new_watch >= HEALTH_WATCH_DAYS + HEALTH_RETIRE_DAYS:
                        new_status = "retired"
                        print(f"  ❌ {factor_id}: watchlist → retired (IC={rolling_ic:.4f}, no recovery)")
                    else:
                        remaining = HEALTH_WATCH_DAYS + HEALTH_RETIRE_DAYS - new_watch
                        print(f"  ⏳ {factor_id}: watchlist IC={rolling_ic:.4f} ({remaining}d to retire)")

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
            """, [new_status, rolling_ic, new_watch if 'new_watch' in dir() else watch_count,
                  new_status, new_status, new_status, factor_id])

            # Log
            con.execute("""
                INSERT OR REPLACE INTO health_log (date, factor_id, rolling_ic_20d, status_before, status_after)
                VALUES (?, ?, ?, ?, ?)
            """, [today, factor_id, rolling_ic, status, new_status])

        except Exception as e:
            print(f"  ❗ {factor_id}: health check error - {e}")

    con.close()


def run(market: str, skip_mine: bool = False, max_factors: int = 500, use_agent: bool = True):
    """Run the full daily factor pipeline."""
    print(f"{'='*60}")
    print(f"  Daily Factor Pipeline — {market.upper()}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
                    commentary_path.write_text(commentary)
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
    args = parser.parse_args()
    run(args.market, args.skip_mine, args.max_factors, use_agent=not args.no_agent)


if __name__ == "__main__":
    main()
