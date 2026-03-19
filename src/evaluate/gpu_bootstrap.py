"""
Bootstrap significance test for factor IC.

Default backend is the Rust bootstrap binary for production pipeline use.
Python/CuPy remains available as a fallback or for local experimentation.
The test works on the daily IC series with a circular moving block bootstrap,
which preserves short-range serial dependence from overlapping forward returns.

Usage:
    result = bootstrap_significance(factor_values, forward_returns, dates, n_bootstrap=100000)
    print(result["p_value"])  # < 0.01 = statistically significant
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cupy as cp
    # Test if GPU actually works (new architectures may not have prebuilt kernels)
    cp.array([1.0]) + cp.array([1.0])
    GPU_AVAILABLE = True
except Exception:
    cp = np  # fallback to numpy
    GPU_AVAILABLE = False


REPO_ROOT = Path(__file__).resolve().parents[2]
RUST_BOOTSTRAP_DIR = REPO_ROOT / "rust-bootstrap"
RUST_BOOTSTRAP_MANIFEST = RUST_BOOTSTRAP_DIR / "Cargo.toml"
RUST_BOOTSTRAP_TARGET_DIR = Path(
    os.environ.get("FACTOR_LAB_RUST_BOOTSTRAP_TARGET_DIR", "/tmp/factor-lab-rust-bootstrap-target")
)
RUST_BOOTSTRAP_BIN = RUST_BOOTSTRAP_TARGET_DIR / "release" / "rust-bootstrap"
_RUST_BOOTSTRAP_STATE: dict[str, bool] = {"checked": False, "available": False}


def _daily_ic_series(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    dates: pd.Series,
) -> np.ndarray:
    """Build the daily cross-sectional IC series used by both backends."""
    df = pd.DataFrame({
        "date": dates.values,
        "factor": factor_values.values,
        "fwd": forward_returns.values,
    }).dropna()

    daily_ics = []
    for _, group in df.groupby("date", sort=True):
        if len(group) < 10:
            continue
        ic = _spearman_ic(group["factor"].to_numpy(), group["fwd"].to_numpy())
        if not np.isnan(ic):
            daily_ics.append(ic)

    return np.asarray(daily_ics, dtype=np.float64)


def _python_bootstrap_from_daily_ics(
    daily_ics: np.ndarray,
    n_bootstrap: int = 100_000,
    seed: int = 42,
    block_size: int | None = None,
) -> dict:
    """Python/CuPy backend using the centered daily IC series."""
    n_days = len(daily_ics)
    if n_days < 10:
        return {
            "real_ic": 0.0, "p_value": 1.0, "null_mean": 0.0, "null_std": 0.0,
            "percentile": 50.0, "significant_01": False, "significant_05": False,
            "n_bootstrap": 0, "block_size": 0, "gpu": False, "backend": "python",
        }

    real_ic = float(np.nanmean(daily_ics))
    centered_ics = daily_ics - real_ic

    if block_size is None:
        block_size = max(5, int(round(np.sqrt(n_days))))
    block_size = min(n_days, max(1, int(block_size)))
    n_blocks = (n_days + block_size - 1) // block_size

    xp = cp if GPU_AVAILABLE else np
    rng = cp.random.default_rng(seed) if GPU_AVAILABLE else np.random.default_rng(seed)
    centered = xp.asarray(centered_ics, dtype=xp.float32)

    starts = rng.integers(0, n_days, size=(n_bootstrap, n_blocks))
    starts = starts.astype(xp.int32, copy=False)
    offsets = xp.arange(block_size, dtype=xp.int32)
    boot_indices = (starts[..., None] + offsets).reshape(n_bootstrap, n_blocks * block_size) % n_days
    boot_indices = boot_indices[:, :n_days]
    null_ics = xp.mean(centered[boot_indices], axis=1)

    if GPU_AVAILABLE:
        null_ics_np = cp.asnumpy(null_ics)
    else:
        null_ics_np = null_ics

    p_value = float(np.mean(np.abs(null_ics_np) >= abs(real_ic)))
    percentile = float(np.mean(null_ics_np < real_ic) * 100)

    return {
        "real_ic": round(real_ic, 6),
        "p_value": round(p_value, 6),
        "null_mean": round(float(np.mean(null_ics_np)), 6),
        "null_std": round(float(np.std(null_ics_np)), 6),
        "percentile": round(percentile, 1),
        "significant_01": p_value < 0.01,
        "significant_05": p_value < 0.05,
        "n_bootstrap": n_bootstrap,
        "block_size": block_size,
        "gpu": GPU_AVAILABLE,
        "backend": "python_gpu" if GPU_AVAILABLE else "python_cpu",
    }


def _ensure_rust_bootstrap_binary() -> Path | None:
    """Build or locate the Rust bootstrap binary once per process."""
    source_files = [RUST_BOOTSTRAP_MANIFEST, *RUST_BOOTSTRAP_DIR.glob("src/**/*.rs")]
    needs_build = not RUST_BOOTSTRAP_BIN.exists()
    if not needs_build and source_files:
        latest_source_mtime = max(path.stat().st_mtime for path in source_files if path.exists())
        needs_build = RUST_BOOTSTRAP_BIN.stat().st_mtime < latest_source_mtime

    if not needs_build:
        _RUST_BOOTSTRAP_STATE["checked"] = True
        _RUST_BOOTSTRAP_STATE["available"] = True
        return RUST_BOOTSTRAP_BIN

    if _RUST_BOOTSTRAP_STATE["checked"] and not _RUST_BOOTSTRAP_STATE["available"]:
        return None

    _RUST_BOOTSTRAP_STATE["checked"] = True
    cargo = shutil.which("cargo")
    if cargo is None or not RUST_BOOTSTRAP_MANIFEST.exists():
        return None

    try:
        result = subprocess.run(
            [
                cargo,
                "build",
                "--release",
                "--manifest-path",
                str(RUST_BOOTSTRAP_MANIFEST),
                "--target-dir",
                str(RUST_BOOTSTRAP_TARGET_DIR),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(REPO_ROOT),
        )
    except Exception as exc:
        print(f"  Rust bootstrap build failed: {exc}")
        return None

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no output").strip()
        print(f"  Rust bootstrap build failed: {detail[:240]}")
        return None

    if RUST_BOOTSTRAP_BIN.exists():
        _RUST_BOOTSTRAP_STATE["available"] = True
        return RUST_BOOTSTRAP_BIN

    return None


def _rust_bootstrap_from_daily_ics(
    daily_ics: np.ndarray,
    n_bootstrap: int = 100_000,
    seed: int = 42,
    block_size: int | None = None,
) -> dict | None:
    """Run the Rust bootstrap binary on a precomputed daily IC series."""
    if len(daily_ics) < 10:
        return {
            "real_ic": 0.0, "p_value": 1.0, "null_mean": 0.0, "null_std": 0.0,
            "percentile": 50.0, "significant_01": False, "significant_05": False,
            "n_bootstrap": 0, "block_size": 0, "gpu": False, "backend": "rust",
        }

    binary = _ensure_rust_bootstrap_binary()
    if binary is None:
        return None

    cmd = [str(binary), "--n", str(n_bootstrap), "--seed", str(seed)]
    if block_size is not None:
        cmd.extend(["--block-size", str(block_size)])

    payload = " ".join(f"{value:.12g}" for value in daily_ics)
    try:
        result = subprocess.run(
            cmd,
            input=payload,
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(REPO_ROOT),
        )
    except Exception as exc:
        print(f"  Rust bootstrap execution failed: {exc}")
        return None

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no output").strip()
        print(f"  Rust bootstrap execution failed: {detail[:240]}")
        return None

    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(f"  Rust bootstrap JSON parse failed: {exc}")
        return None

    parsed["gpu"] = False
    parsed["backend"] = "rust"
    return parsed


def bootstrap_significance(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    dates: pd.Series,
    n_bootstrap: int = 100_000,
    seed: int = 42,
    block_size: int | None = None,
    backend: str = "rust",
) -> dict:
    """
    Moving block bootstrap test for factor IC significance.

    Procedure:
    1. Compute real IC (daily cross-sectional Spearman, averaged)
    2. Center the daily IC series under the null hypothesis mean=0
    3. Circular block-bootstrap the centered series n_bootstrap times
    4. p-value = fraction of null means more extreme than the real mean IC

    Returns:
        {
            "real_ic": float,
            "p_value": float,       # < 0.01 = significant at 99%
            "null_mean": float,
            "null_std": float,
            "percentile": float,    # where real IC sits in null distribution
            "significant_01": bool, # p < 0.01
            "significant_05": bool, # p < 0.05
            "n_bootstrap": int,
            "block_size": int,
            "gpu": bool,
            "backend": str,
        }
    """
    daily_ics = _daily_ic_series(factor_values, forward_returns, dates)
    if len(daily_ics) < 10:
        return {
            "real_ic": 0.0, "p_value": 1.0, "null_mean": 0.0, "null_std": 0.0,
            "percentile": 50.0, "significant_01": False, "significant_05": False,
            "n_bootstrap": 0, "block_size": 0, "gpu": False, "backend": backend,
        }

    backend = backend.lower()
    if backend == "rust":
        result = _rust_bootstrap_from_daily_ics(
            daily_ics,
            n_bootstrap=n_bootstrap,
            seed=seed,
            block_size=block_size,
        )
        if result is not None:
            return result

    return _python_bootstrap_from_daily_ics(
        daily_ics,
        n_bootstrap=n_bootstrap,
        seed=seed,
        block_size=block_size,
    )


def batch_bootstrap(
    candidates: list[dict],
    prices_df: pd.DataFrame,
    fwd_df: pd.DataFrame,
    sym_col: str = "ts_code",
    date_col: str = "trade_date",
    n_bootstrap: int = 100_000,
    backend: str = "rust",
) -> list[dict]:
    """
    Run bootstrap significance test on a batch of factor candidates.
    Adds 'bootstrap_p' and 'bootstrap_significant' to each candidate.
    """
    import sys
    sys.path.insert(0, str(pd.__file__).rsplit("/", 3)[0])

    from src.dsl.parser import parse
    from src.dsl.compute import compute_factor

    tested = 0
    significant = 0
    used_backends: set[str] = set()

    for c in candidates:
        try:
            ast = parse(c["formula"])
            factor_df = compute_factor(ast, prices_df, sym_col=sym_col, date_col=date_col)

            merged = factor_df.merge(
                fwd_df[[sym_col, date_col, "fwd_5d"]],
                on=[sym_col, date_col], how="inner"
            ).dropna(subset=["fwd_5d", "factor_value"])

            if len(merged) < 500:
                c["bootstrap_p"] = 1.0
                c["bootstrap_significant"] = False
                continue

            result = bootstrap_significance(
                merged["factor_value"], merged["fwd_5d"], merged[date_col],
                n_bootstrap=n_bootstrap,
                backend=backend,
            )

            c["bootstrap_p"] = result["p_value"]
            c["bootstrap_significant"] = result["significant_01"]
            c["bootstrap_percentile"] = result["percentile"]
            c["bootstrap_backend"] = result.get("backend", backend)
            used_backends.add(c["bootstrap_backend"])

            tested += 1
            if result["significant_01"]:
                significant += 1

        except Exception as e:
            c["bootstrap_p"] = 1.0
            c["bootstrap_significant"] = False

    backend_desc = ",".join(sorted(used_backends)) if used_backends else backend
    print(f"  Bootstrap: {tested} tested, {significant} significant (p<0.01), backend={backend_desc}")
    return candidates


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Fast Spearman IC for two arrays."""
    n = len(x)
    if n < 3:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry))
    if denom == 0:
        return 0.0
    return float(np.sum(rx * ry) / denom)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks with tie handling."""
    n = len(x)
    ranks = np.empty(n, dtype=np.float64)
    idx = np.argsort(x, kind="mergesort")
    sorted_x = x[idx]

    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_x[end] == sorted_x[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[idx[start:end]] = avg_rank
        start = end

    return ranks
