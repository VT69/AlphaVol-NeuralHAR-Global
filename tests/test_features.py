"""Feature timing (no look-ahead), staleness, loud failures, and real-data target alignment."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphavol.config import load_config
from alphavol.features import RawInputs, align_lagged, build_dataset, load_raw_inputs
from alphavol.io import MissingDataError

CFG = load_config()


def _ohlc(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n, tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = c * np.exp(rng.normal(0, 0.003, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.005, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.005, n)))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)


def _fred(idx, seed=1):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(idx.min() - pd.Timedelta(days=10), idx.max(), tz="UTC")
    return pd.DataFrame({"T10Y2Y": rng.normal(0, 1, len(days)), "BAA10Y": rng.normal(2, 0.1, len(days)),
                         "VIXCLS": rng.uniform(10, 30, len(days))}, index=days)


def test_align_lagged_lag1_bday_and_no_lookahead():
    rows = pd.bdate_range("2024-03-04", periods=10, tz="UTC")            # Mon..
    s = pd.Series(np.arange(30.0), index=pd.bdate_range("2024-02-12", periods=30, tz="UTC"))
    a = align_lagged(s, rows, lag=1, unit="bday", max_staleness=5)
    assert a.loc["2024-03-04"] == s.loc["2024-03-01"]                    # Monday uses Friday
    s2 = s.copy()
    s2.loc[s2.index >= pd.Timestamp("2024-03-04", tz="UTC")] += 1000      # perturb from Monday on
    b = align_lagged(s2, rows, lag=1, unit="bday", max_staleness=5)
    assert a.loc["2024-03-04"] == b.loc["2024-03-04"]                    # Monday unaffected


def test_align_lagged_staleness_gives_nan():
    rows = pd.bdate_range("2024-03-01", periods=15, tz="UTC")
    s = pd.Series([1.0], index=pd.DatetimeIndex(["2024-02-28"], tz="UTC"))
    a = align_lagged(s, rows, lag=1, unit="bday", max_staleness=5)
    assert a.notna().sum() > 0 and a.iloc[-1] != a.iloc[-1]             # old value expires -> NaN


def test_build_spx_like_features_lagged_and_indicators():
    px = _ohlc()
    raw = RawInputs(prices=px, fred=_fred(px.index))
    res = build_dataset("spx", raw, CFG, groups=["macro", "implied_vol"])
    df = res.data
    t = df.index[50]
    prev_bday = t - pd.offsets.BDay(1)
    assert df.loc[t, "term_slope"] == raw.fred.loc[prev_bday, "T10Y2Y"]
    assert {"macro_missing", "implied_vol_missing"} <= set(df.columns)
    assert res.features["macro"] == ["term_slope", "credit_spread"]
    # changing FRED values from day t onwards must not change row t
    fred2 = raw.fred.copy()
    fred2.loc[fred2.index >= t] += 99
    df2 = build_dataset("spx", RawInputs(prices=px, fred=fred2), CFG, groups=["macro"]).data
    assert df2.loc[t, "term_slope"] == df.loc[t, "term_slope"]


def test_missing_required_input_is_loud_and_pending_is_recorded():
    px = _ohlc()
    fred = _fred(px.index).drop(columns=["BAA10Y"])
    with pytest.raises(MissingDataError):
        build_dataset("spx", RawInputs(prices=px, fred=fred), CFG, groups=["macro"])
    res = build_dataset("spx", RawInputs(prices=px, fred=fred), CFG, groups=["macro"],
                        pending={"macro.credit_spread"})
    assert res.pending == ["macro.credit_spread"] and "credit_spread" not in res.data


def test_no_zero_fill_of_exogenous_gaps():
    px = _ohlc()
    fred = _fred(px.index)
    fred.loc[fred.index[40:60], "T10Y2Y"] = np.nan             # 20-bday gap > staleness 5
    df = build_dataset("spx", RawInputs(prices=px, fred=fred), CFG, groups=["macro"]).data
    assert df["term_slope"].isna().any()
    assert (df.loc[df["term_slope"].isna(), "macro_missing"] == 1).all()
    assert not (df["term_slope"] == 0).any()


def test_regime_feature_is_gone():
    px = _ohlc()
    df = build_dataset("spx", RawInputs(prices=px, fred=_fred(px.index)), CFG, groups=["macro"]).data
    assert not any("regime" in c or c == "gmsi" for c in df.columns)


@pytest.mark.parametrize("asset", ["btc", "spx", "nifty"])
def test_real_data_every_target_is_next_trading_day(asset):
    """Assertion-based check on the committed development data (R2). Uses only the HAR group so it
    does not depend on pending inputs. build_dataset calls assert_targets_aligned internally; we
    re-check explicitly here and also check the dev boundary."""
    raw = load_raw_inputs(asset, CFG)
    res = build_dataset(asset, raw, CFG, groups=[])
    df, lv = res.data, res.log_variance_full
    nxt = pd.Series(lv.index[1:], index=lv.index[:-1])
    assert (pd.DatetimeIndex(df["target_date"]) == pd.DatetimeIndex(nxt.reindex(df.index))).all()
    assert np.array_equal(df["target"].to_numpy(), lv.reindex(df["target_date"]).to_numpy())
    assert df["target_date"].max() <= CFG.dev_end
    gaps = df["target_date"] - df.index
    assert gaps.dt.days.max() <= (1 if asset == "btc" else 5)
