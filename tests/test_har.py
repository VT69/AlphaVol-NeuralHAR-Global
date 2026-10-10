"""HAR regressors, calendar handling and target alignment."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from alphavol.har import (AlignmentError, assert_targets_aligned, har_regressors, next_day_target,
                          regularise_calendar)
from alphavol.targets import log_variance


def _series(vals, freq="D", start="2024-01-01"):
    return pd.Series(vals, index=pd.date_range(start, periods=len(vals), freq=freq, tz="UTC"), dtype=float)


def test_windows_end_at_t_and_use_log_of_mean():
    v = _series(np.arange(1, 31))
    X = har_regressors(v, w=5, m=22, min_w=5, min_m=22)
    t = v.index[25]
    assert math.isclose(X.loc[t, "RV_d"], math.log(26))                         # day t itself
    assert math.isclose(X.loc[t, "RV_w"], math.log(np.mean(np.arange(22, 27))))  # t-4..t
    assert math.isclose(X.loc[t, "RV_m"], math.log(np.mean(np.arange(5, 27))))  # t-21..t
    assert np.isnan(X["RV_m"].iloc[20]) and not np.isnan(X["RV_m"].iloc[21])


def test_target_is_next_trading_day_on_exchange_calendar():
    # trading days with a weekend gap: Fri -> Mon
    idx = pd.DatetimeIndex(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"], tz="UTC")
    lv = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    T = next_day_target(lv, "exchange")
    assert T.loc["2024-01-05", "target"] == 3.0
    assert T.loc["2024-01-05", "target_date"] == pd.Timestamp("2024-01-08", tz="UTC")
    assert np.isnan(T["target"].iloc[-1])


def test_invalid_day_costs_at_most_two_rows_not_a_cascade():
    v = _series(np.full(80, 1e-4))
    v.iloc[40] = np.nan                       # e.g. zero-range day
    X = har_regressors(v, w=5, m=22, min_w=4, min_m=18)
    T = next_day_target(log_variance(v), "exchange")
    df = X.join(T).dropna()
    lost = set(v.index[21:-1]) - set(df.index)   # rows that would exist without the invalid day
    assert lost == {v.index[39], v.index[40]}     # previous day's target and the day itself


def test_continuous_calendar_inserts_missing_days():
    v = _series([1.0, 2.0, 3.0, 4.0]).drop(pd.Timestamp("2024-01-03", tz="UTC"))
    r = regularise_calendar(v, "continuous")
    assert len(r) == 4 and np.isnan(r.loc["2024-01-03"])
    T = next_day_target(log_variance(r), "continuous")
    assert np.isnan(T.loc["2024-01-02", "target"])   # its next day is missing: no jump to 01-04


def test_assert_targets_aligned_catches_misalignment():
    lv = log_variance(_series(np.arange(1.0, 11.0)))
    T = next_day_target(lv, "exchange").dropna()
    assert_targets_aligned(T, lv)                       # passes
    bad = T.copy()
    bad.iloc[3, bad.columns.get_loc("target")] = lv.iloc[7]   # a target from 3 days ahead
    with pytest.raises(AlignmentError):
        assert_targets_aligned(bad, lv)
    # the legacy failure mode: shift applied after dropping rows
    legacy = lv.drop(lv.index[5]).shift(-1).rename("target").to_frame()
    legacy["target_date"] = lv.drop(lv.index[5]).index.to_series().shift(-1)
    with pytest.raises(AlignmentError):
        assert_targets_aligned(legacy.dropna(), lv)
