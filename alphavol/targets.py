"""Daily variance measures: 5-minute realized variance (BTC) and Garman–Klass (SPX, NIFTY).

Fixes vs. legacy (analysis/target_quality.md, DECISIONS.md D13-D14):
* SPX/NIFTY use Garman–Klass computed here from daily OHLC. The committed legacy files stored
  log squared close-to-close returns under the name ``log_RV``.
* No log(0) floors and no row dropping: invalid days become NaN and are handled downstream.
* BTC daily RV records the number of 5-minute bars, so incomplete days are visible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alphavol.logging_utils import get_logger

log = get_logger(__name__)
_GK_K = 2.0 * np.log(2.0) - 1.0


def _require_utc_index(df: pd.DataFrame, what: str) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{what}: expected a DatetimeIndex, got {type(df.index).__name__}")
    if df.index.tz is None or str(df.index.tz) != "UTC":
        raise ValueError(f"{what}: index must be tz-aware UTC, got tz={df.index.tz}")
    if df.index.has_duplicates:
        raise ValueError(f"{what}: duplicate timestamps ({int(df.index.duplicated().sum())})")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{what}: index not sorted")


def daily_realized_variance(bars: pd.DataFrame, *, min_bars: int, flag_below: int,
                            bars_per_day: int = 288) -> pd.DataFrame:
    """Sum of squared 5-minute log returns per UTC day.

    Each bar is indexed by its open time. The return of a bar is ``log(close_t / close_{t-1})``
    using the previous *available* bar, so variance over an exchange outage is captured by one
    return spanning the gap. A bar is assigned to the UTC day of its open time.

    Returns a DataFrame indexed by UTC day with columns ``variance`` (NaN when fewer than
    ``min_bars`` bars), ``n_bars`` and ``incomplete`` (n_bars < flag_below).
    """
    _require_utc_index(bars, "5-minute bars")
    close = bars["close"].astype(float)
    if (close <= 0).any():
        raise ValueError(f"non-positive close prices: {int((close <= 0).sum())} bars")
    r = np.log(close).diff()
    day = bars.index.floor("D")
    grouped = pd.DataFrame({"r2": r ** 2, "day": day}).groupby("day")
    out = pd.DataFrame({"variance": grouped["r2"].sum(min_count=1), "n_bars": grouped.size()})
    too_few = out["n_bars"] < min_bars
    if too_few.any():
        log.warning("%d day(s) with < %d bars set to NaN: %s", int(too_few.sum()), min_bars,
                    [str(d.date()) for d in out.index[too_few]][:10])
    out.loc[too_few, "variance"] = np.nan
    out["incomplete"] = out["n_bars"] < flag_below
    if (out["n_bars"] > bars_per_day).any():
        raise ValueError("more bars than expected in a day: check bar frequency")
    out.index.name = "date"
    return out


def garman_klass(ohlc: pd.DataFrame) -> pd.Series:
    """Garman–Klass (1980) daily variance: 0.5·ln(H/L)² − (2ln2−1)·ln(C/O)².

    Rows with non-positive prices, high < low, or open/close outside [low, high] are invalid and
    return NaN (logged). Zero variance (high == low) is returned as NaN, since log variance is
    undefined; see DECISIONS.md D13.
    """
    _require_utc_index(ohlc, "daily OHLC")
    o, h, l, c = (ohlc[k].astype(float) for k in ("open", "high", "low", "close"))
    invalid = ((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0) | (h < l)
               | (o > h) | (o < l) | (c > h) | (c < l)) | ohlc[["open", "high", "low", "close"]].isna().any(axis=1)
    gk = 0.5 * np.log(h / l) ** 2 - _GK_K * np.log(c / o) ** 2
    zero = (gk <= 0) & ~invalid
    if invalid.any():
        log.warning("Garman-Klass: %d invalid OHLC rows set to NaN", int(invalid.sum()))
    if zero.any():
        log.warning("Garman-Klass: %d zero-variance rows (high == low) set to NaN", int(zero.sum()))
    gk = gk.where(~(invalid | zero))
    gk.name = "variance"
    return gk


def log_variance(variance: pd.Series) -> pd.Series:
    """Natural log of a strictly positive variance; non-positive or NaN → NaN (never floored)."""
    v = variance.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(v.where(v > 0))
    out.name = "log_variance"
    return out
