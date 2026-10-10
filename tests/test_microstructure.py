"""Microstructure features: hand cases and causality (perturbing the future leaves the past unchanged)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from alphavol.microstructure import amihud, daily_microstructure, order_flow_imbalance, roll_spread, vpin


def _bars(n_days=60, seed=0):
    rng = np.random.default_rng(seed)
    n = 288 * n_days
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    vol = rng.uniform(5, 15, n)
    return pd.DataFrame({"close": close, "volume": vol, "quote_asset_volume": vol * close,
                         "taker_buy_base_vol": vol * rng.uniform(0.3, 0.7, n)}, index=idx)


def test_ofi_hand_case():
    idx = pd.date_range("2024-01-01", periods=2, freq="5min", tz="UTC")
    b = pd.DataFrame({"close": [1.0, 1.0], "volume": [10.0, 10.0], "quote_asset_volume": [10.0, 10.0],
                      "taker_buy_base_vol": [8.0, 6.0]}, index=idx)
    # buy 14, sell 6 -> |14-6|/20 = 0.4
    assert math.isclose(order_flow_imbalance(b).iloc[0], 0.4)


def test_roll_zero_for_positive_autocovariance_and_positive_for_bounce():
    idx = pd.date_range("2024-01-01", periods=7, freq="5min", tz="UTC")
    bounce = pd.DataFrame({"close": [100, 101, 100, 101, 100, 101, 100.0], "volume": 1.0,
                           "quote_asset_volume": 100.0, "taker_buy_base_vol": 0.5}, index=idx)
    # returns 0.01, 0.02, 0.03, ... are positively autocorrelated -> Roll spread 0 by definition
    trend = bounce.assign(close=100 * np.exp(np.cumsum([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])))
    assert roll_spread(bounce).iloc[0] > 0
    assert roll_spread(trend).iloc[0] == 0.0


def test_amihud_hand_case():
    idx = pd.date_range("2024-01-01", periods=2, freq="5min", tz="UTC")
    b = pd.DataFrame({"close": [100.0, 110.0], "volume": 1.0, "quote_asset_volume": [1e6, 2e6],
                      "taker_buy_base_vol": 0.5}, index=idx)
    assert math.isclose(amihud(b).iloc[0], abs(math.log(1.1)) / 2e6 * 1e9)


def test_vpin_is_bounded_and_causal():
    b = _bars()
    base = vpin(b)
    assert base.dropna().between(0, 1).all() and base.notna().sum() > 30
    cut = pd.Timestamp("2024-02-10", tz="UTC")
    future = b.copy()
    mask = future.index >= cut
    future.loc[mask, "volume"] *= 5          # change everything from the cut onwards
    future.loc[mask, "close"] *= 1.3
    pert = vpin(future)
    before = base.index < cut - pd.Timedelta(days=1)   # last pre-cut day's final bucket may straddle
    pd.testing.assert_series_equal(base[before], pert.reindex(base.index)[before])


def test_daily_microstructure_shape():
    out = daily_microstructure(_bars(n_days=40))
    assert list(out.columns) == ["ofi_abs", "vpin", "amihud", "roll"]
    assert len(out) == 40
