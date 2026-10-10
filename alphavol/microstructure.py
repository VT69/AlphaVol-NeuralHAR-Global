"""Causal daily microstructure features from BTC 5-minute klines (DECISIONS.md D16, D22).

All features for UTC day t use only bars that open on or before day t. VPIN's bucket size uses
only volume from days before t (the legacy version used the whole-sample mean).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from alphavol.logging_utils import get_logger

log = get_logger(__name__)
REQUIRED = ("close", "volume", "quote_asset_volume", "taker_buy_base_vol")


def _check(bars: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in bars.columns]
    if missing:
        raise KeyError(f"5-minute bars missing columns {missing}")
    if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
        raise ValueError("5-minute bars must be sorted with unique timestamps")


def order_flow_imbalance(bars: pd.DataFrame) -> pd.Series:
    """|Σ taker-buy − Σ taker-sell| / Σ volume per UTC day, using aggressor-side volume."""
    _check(bars)
    day = bars.index.floor("D")
    buy = bars["taker_buy_base_vol"].astype(float)
    vol = bars["volume"].astype(float)
    g = pd.DataFrame({"buy": buy, "vol": vol}).groupby(day).sum()
    ofi = (2 * g["buy"] - g["vol"]).abs() / g["vol"].where(g["vol"] > 0)
    return ofi.rename("ofi_abs")


def amihud(bars: pd.DataFrame) -> pd.Series:
    """Mean over the day's bars of |r| / quote volume (USDT), scaled by 1e9. Bars with zero
    quote volume are excluded."""
    _check(bars)
    r = np.log(bars["close"].astype(float)).diff().abs()
    qv = bars["quote_asset_volume"].astype(float)
    x = (r / qv.where(qv > 0)) * 1e9
    return x.groupby(bars.index.floor("D")).mean().rename("amihud")


def roll_spread(bars: pd.DataFrame) -> pd.Series:
    """Roll (1984) effective spread from within-day first-order return autocovariance."""
    _check(bars)
    r = np.log(bars["close"].astype(float)).diff()
    day = bars.index.floor("D")

    def _roll(x: pd.Series) -> float:
        x = x.dropna().to_numpy()
        if len(x) < 3:
            return np.nan
        c = np.cov(x[1:], x[:-1])[0, 1]
        return float(2.0 * np.sqrt(-c)) if c < 0 else 0.0

    return r.groupby(day).apply(_roll).rename("roll")


def vpin(bars: pd.DataFrame, *, n_buckets: int = 50, trailing_days: int = 30,
         min_trailing_days: int = 7, sigma_window: int = 250, sigma_min: int = 50) -> pd.Series:
    """Bulk-volume-classification VPIN, sampled at the last bucket completed in each UTC day.

    The bucket size for day t is V*_t = mean daily volume over the previous ``trailing_days`` days
    (at least ``min_trailing_days``), divided by ``n_buckets``. Bar volumes are assigned whole to
    buckets in order. Price change per bucket is the log change between consecutive bucket-end
    closes; its scale is a rolling std over the last ``sigma_window`` buckets (inclusive).
    """
    _check(bars)
    day = bars.index.floor("D")
    daily_vol = bars["volume"].astype(float).groupby(day).sum()
    vstar = (daily_vol.shift(1).rolling(trailing_days, min_periods=min_trailing_days).mean() / n_buckets)
    bar_vstar = vstar.reindex(day).to_numpy()
    valid = np.isfinite(bar_vstar) & (bar_vstar > 0)
    b = bars.loc[valid]
    if b.empty:
        return pd.Series(dtype=float, name="vpin")
    nv = b["volume"].to_numpy(float) / bar_vstar[valid]
    bucket = np.floor(np.cumsum(nv)).astype(np.int64)
    g = pd.DataFrame({"close": b["close"].to_numpy(float), "vol": b["volume"].to_numpy(float),
                      "ts": b.index, "bucket": bucket}).groupby("bucket")
    bk = pd.DataFrame({"close": g["close"].last(), "vol": g["vol"].sum(), "end": g["ts"].last()})
    bk = bk.iloc[:-1]  # the final bucket may be incomplete
    dp = np.log(bk["close"]).diff()
    sigma = dp.rolling(sigma_window, min_periods=sigma_min).std()
    z = dp / sigma
    buy = bk["vol"] * norm.cdf(z)
    imbalance = (2 * buy - bk["vol"]).abs()
    v = imbalance.rolling(n_buckets, min_periods=n_buckets).sum() / bk["vol"].rolling(n_buckets, min_periods=n_buckets).sum()
    v.index = pd.DatetimeIndex(bk["end"]).floor("D")
    return v.groupby(level=0).last().rename("vpin")


def daily_microstructure(bars: pd.DataFrame) -> pd.DataFrame:
    """All four BTC microstructure features per UTC day (lag 0: computed from day-t bars)."""
    out = pd.concat([order_flow_imbalance(bars), vpin(bars), amihud(bars), roll_spread(bars)], axis=1)
    out.index.name = "date"
    return out
