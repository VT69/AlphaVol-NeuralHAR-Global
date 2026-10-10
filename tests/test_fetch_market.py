"""Market fetchers with fake clients (no network)."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from alphavol.fetch.http import FetchError
from alphavol.fetch.market import assert_coverage, binance_klines, fear_greed, fred_series

UTC = timezone.utc


class FakeClient:
    def __init__(self, fn):
        self.fn = fn

    def get(self, path="", params=None, cache_tag=None):
        return self.fn(path, params or {})


def test_binance_pages_and_types():
    def fn(path, p):
        start = p["startTime"]
        n = min(1000, (p["endTime"] + 1 - start) // 300_000)
        return [[start + i * 300_000, "1", "2", "0.5", "1.5", "10", start + i * 300_000 + 299_999, "15", 3, "6", "9", "0"]
                for i in range(n)]
    s = datetime(2026, 7, 29, tzinfo=UTC)
    df = binance_klines(s, datetime(2026, 8, 3, tzinfo=UTC), client=FakeClient(fn))
    assert len(df) == 5 * 288 and df.index.is_monotonic_increasing and df["close"].dtype == float


def test_binance_empty_is_loud():
    with pytest.raises(FetchError):
        binance_klines(datetime(2026, 7, 29, tzinfo=UTC), datetime(2026, 7, 30, tzinfo=UTC),
                       client=FakeClient(lambda p, q: []))


def test_fred_csv_parsing_and_coverage_assert():
    csv = "observation_date,BAA10Y\n2015-01-02,2.5\n2015-01-05,.\n2015-01-06,2.6\n"
    s = fred_series("BAA10Y", "2015-01-01", "2015-01-10", client=FakeClient(lambda p, q: csv))
    assert s.isna().sum() == 1 and s.iloc[0] == 2.5
    assert_coverage(s, first_on_or_before="2015-01-02")
    with pytest.raises(FetchError):
        assert_coverage(s, first_on_or_before="2014-12-31")


def test_fear_greed_parsing():
    body = {"data": [{"value": "40", "timestamp": "1704067200"}, {"value": "60", "timestamp": "1704153600"}]}
    s = fear_greed(client=FakeClient(lambda p, q: body))
    assert list(s) == [40.0, 60.0] and str(s.index[0].date()) == "2024-01-01"
