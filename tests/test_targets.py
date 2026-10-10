"""Variance measures vs hand-computed cases."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from alphavol.targets import daily_realized_variance, garman_klass, log_variance


def _bars(closes, start="2024-01-01", freq="5min"):
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC")
    return pd.DataFrame({"close": closes}, index=idx)


def test_realized_variance_hand_case():
    # two bars on day 1, then day 2 starts: returns ln(2), ln(1/2) ...
    closes = [1.0, 2.0, 1.0, 1.0]
    b = _bars(closes, start="2024-01-01 23:50", freq="5min")  # 23:50, 23:55 on day 1; 00:00, 00:05 day 2
    out = daily_realized_variance(b, min_bars=1, flag_below=1)
    d1, d2 = out.index
    assert math.isclose(out.loc[d1, "variance"], math.log(2) ** 2)          # only one return on day 1
    assert math.isclose(out.loc[d2, "variance"], math.log(0.5) ** 2 + 0.0)  # 00:00 return spans midnight
    assert out.loc[d1, "n_bars"] == 2 and out.loc[d2, "n_bars"] == 2


def test_realized_variance_min_bars_and_flags():
    closes = np.exp(np.cumsum(np.full(288 + 100, 0.001)))
    out = daily_realized_variance(_bars(closes), min_bars=144, flag_below=276)
    assert out["n_bars"].tolist() == [288, 100]
    assert np.isnan(out["variance"].iloc[1]) and bool(out["incomplete"].iloc[1])
    assert not bool(out["incomplete"].iloc[0])


def test_realized_variance_rejects_bad_input():
    b = _bars([1.0, 2.0])
    with pytest.raises(ValueError):
        daily_realized_variance(b.iloc[::-1], min_bars=1, flag_below=1)
    with pytest.raises(ValueError):
        daily_realized_variance(_bars([1.0, -1.0]), min_bars=1, flag_below=1)
    naive = b.tz_localize(None)
    with pytest.raises(ValueError):
        daily_realized_variance(naive, min_bars=1, flag_below=1)


def test_garman_klass_hand_case():
    idx = pd.date_range("2024-01-01", periods=1, tz="UTC")
    df = pd.DataFrame({"open": [100.0], "high": [110.0], "low": [95.0], "close": [105.0]}, index=idx)
    expected = 0.5 * math.log(110 / 95) ** 2 - (2 * math.log(2) - 1) * math.log(105 / 100) ** 2
    assert math.isclose(garman_klass(df).iloc[0], expected, rel_tol=1e-12)


def test_garman_klass_invalid_rows_are_nan_not_floored():
    idx = pd.date_range("2024-01-01", periods=3, tz="UTC")
    df = pd.DataFrame({"open": [100, 100, 100.0], "high": [100, 99, 101.0],
                       "low": [100, 101, 99.0], "close": [100, 100, 100.0]}, index=idx)
    gk = garman_klass(df)
    assert np.isnan(gk.iloc[0])          # high == low: zero variance
    assert np.isnan(gk.iloc[1])          # high < low: invalid
    assert gk.iloc[2] > 0
    assert np.isnan(log_variance(gk).iloc[0])  # never log(0) -> -inf or a floor


def test_realized_quarticity_hand_case():
    import math
    closes = [1.0, math.e, 1.0]                      # returns +1, -1 within one day
    b = _bars(closes)
    out = daily_realized_variance(b, min_bars=1, flag_below=1)
    # r4 sum = 2, n_bars = 3 -> rq = 3/3 * 2 = 2
    assert math.isclose(out["rq"].iloc[0], 2.0)
