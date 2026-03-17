"""Compute factor values from raw prices for evaluation.

These are standalone implementations — they don't depend on the pipeline's
analytics tables. This lets us evaluate factors over the full price history
(~500 days) rather than just the days where the pipeline ran.
"""
import numpy as np
import pandas as pd


def compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI-14 for a single stock's close series."""
    rsi = np.full(len(closes), np.nan)
    if len(closes) < period + 1:
        return rsi
    delta = np.diff(closes)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])

    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss < 1e-10:
            rsi[i + 1] = np.nan
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def compute_bb_position(closes: np.ndarray, period: int = 20) -> np.ndarray:
    """Bollinger Band position: (close - lower) / (upper - lower)."""
    bb = np.full(len(closes), np.nan)
    for i in range(period, len(closes)):
        window = closes[i - period + 1:i + 1]
        sma = np.mean(window)
        std = np.std(window)
        if std < 1e-10:
            bb[i] = 0.5
        else:
            upper = sma + 2 * std
            lower = sma - 2 * std
            bb[i] = np.clip((closes[i] - lower) / (upper - lower), 0, 1)
    return bb


def compute_ma_distance(closes: np.ndarray, period: int = 20) -> np.ndarray:
    """(close - SMA) / SMA as percentage."""
    dist = np.full(len(closes), np.nan)
    for i in range(period, len(closes)):
        sma = np.mean(closes[i - period + 1:i + 1])
        if sma > 1e-10:
            dist[i] = (closes[i] - sma) / sma
    return dist


def compute_regime(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Per-bar regime from lag-1 autocorrelation. 0=trending, 1=MR, 2=noisy."""
    regime = np.full(len(returns) + 1, 2.0)  # +1 because returns is diff of closes
    for i in range(window, len(returns)):
        r = returns[i - window:i]
        if len(r) < 4:
            continue
        ac = np.corrcoef(r[:-1], r[1:])[0, 1]
        if np.isnan(ac):
            regime[i + 1] = 2.0
        elif ac > 0.15:
            regime[i + 1] = 0.0  # trending
        elif ac < -0.10:
            regime[i + 1] = 1.0  # mean_reverting
        else:
            regime[i + 1] = 2.0  # noisy
    return regime


def compute_volume_ratio(volumes: np.ndarray, period: int = 20) -> np.ndarray:
    """Today's volume / 20D average volume."""
    vr = np.full(len(volumes), np.nan)
    for i in range(period, len(volumes)):
        avg = np.mean(volumes[i - period:i])
        if avg > 0:
            vr[i] = volumes[i] / avg
    return vr


def compute_squeeze(returns: np.ndarray, short: int = 20, long: int = 60) -> np.ndarray:
    """Volatility squeeze: std(ret, short) / std(ret, long). <0.7 = compressed."""
    sq = np.full(len(returns) + 1, np.nan)
    for i in range(long, len(returns)):
        s_short = np.std(returns[i - short:i])
        s_long = np.std(returns[i - long:i])
        if s_long > 1e-10:
            sq[i + 1] = s_short / s_long
    return sq


def compute_all_factors(prices_df: pd.DataFrame,
                        sym_col: str = "ts_code",
                        date_col: str = "trade_date",
                        close_col: str = "close",
                        vol_col: str = "vol") -> pd.DataFrame:
    """
    Compute all evaluation factors from raw prices.
    Returns DataFrame with one row per (symbol, date) and factor columns.
    """
    results = []

    for sym, group in prices_df.groupby(sym_col):
        g = group.sort_values(date_col).reset_index(drop=True)
        closes = g[close_col].values.astype(float)
        volumes = g[vol_col].values.astype(float) if vol_col in g.columns else np.ones(len(closes))
        dates = g[date_col].values

        if len(closes) < 60:
            continue

        log_rets = np.diff(np.log(np.maximum(closes, 1e-9)))

        rsi = compute_rsi(closes)
        bb = compute_bb_position(closes)
        ma20 = compute_ma_distance(closes, 20)
        regime = compute_regime(log_rets)
        vol_ratio = compute_volume_ratio(volumes)
        squeeze = compute_squeeze(log_rets)

        # Signed reversion score [-1, +1]
        # Positive = oversold (expect up), Negative = overbought (expect down)
        rsi_signal = np.nan_to_num((50.0 - rsi) / 50.0)       # +1 at RSI=0, -1 at RSI=100
        bb_signal = np.nan_to_num((0.5 - bb) * 2.0)            # +1 at lower band, -1 at upper
        ma_signal = np.nan_to_num(-ma20 / 0.10)                # positive when below MA
        reversion = np.clip(0.35 * rsi_signal + 0.35 * bb_signal + 0.30 * ma_signal, -1, 1)

        # 5D return (for momentum)
        ret_5d = np.full(len(closes), np.nan)
        for i in range(5, len(closes)):
            ret_5d[i] = closes[i] / closes[i - 5] - 1

        for i in range(60, len(closes)):
            results.append({
                sym_col: sym,
                date_col: dates[i],
                "rsi_14": rsi[i],
                "bb_position": bb[i],
                "ma20_dist": ma20[i],
                "reversion_score": reversion[i],
                "regime": regime[i],
                "volume_ratio": vol_ratio[i] if vol_ratio[i] is not None else np.nan,
                "squeeze_ratio": squeeze[i] if i < len(squeeze) else np.nan,
                "ret_5d": ret_5d[i],
            })

    return pd.DataFrame(results)
