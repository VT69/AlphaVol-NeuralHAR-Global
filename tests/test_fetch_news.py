"""News adapters with fake clients: splitting on caps, paging, truncation logging, store, D25 rule."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from alphavol.fetch.news import (AlphaVantage, ArticleStore, FetchLog, GdeltDoc, Guardian, NYT, assess_density,
                                 month_windows, probe_windows)

UTC = timezone.utc


class FakeClient:
    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def get(self, path="", params=None):
        self.calls.append(dict(params or {}))
        return self.fn(path, params or {})


def _gdelt_fn(n_per_hour):
    def fn(path, p):
        s = datetime.strptime(p["startdatetime"], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        e = datetime.strptime(p["enddatetime"], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        hours = int((e - s).total_seconds() // 3600) or 1
        n = min(250, n_per_hour * hours)
        arts = [{"url": f"u{s:%Y%m%d%H}-{i}", "title": "t", "seendate": f"{s:%Y%m%dT%H%M%S}Z",
                 "domain": "d", "language": "English", "sourcecountry": "US"} for i in range(n)]
        return {"articles": arts}
    return fn


def test_gdelt_splits_capped_windows_until_under_cap():
    g = GdeltDoc(FakeClient(_gdelt_fn(20)))                    # 20/h -> 480/day > cap
    log = FetchLog()
    s = datetime(2024, 1, 1, tzinfo=UTC)
    arts = g.articles("spx", s, s + timedelta(days=1), log)
    assert len({a["url"] for a in arts}) == 480 and not log.truncations


def test_gdelt_records_truncation_at_min_window():
    g = GdeltDoc(FakeClient(_gdelt_fn(1000)))                  # cap hit even per hour
    log = FetchLog()
    s = datetime(2024, 1, 1, tzinfo=UTC)
    g.articles("spx", s, s + timedelta(hours=2), log)
    assert len(log.truncations) == 2


def test_guardian_pages_through_all_results():
    def fn(path, p):
        total, page, size = 450, p["page"], p["page-size"]
        n = max(0, min(size, total - (page - 1) * size))
        res = [{"webPublicationDate": "2024-01-02T10:00:00Z", "webTitle": f"t{page}-{i}", "webUrl": f"u{page}-{i}"}
               for i in range(n)]
        return {"response": {"status": "ok", "total": total, "pages": -(-total // size), "results": res}}
    log = FetchLog()
    arts = Guardian(FakeClient(fn)).articles("spx", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC), log)
    assert len(arts) == 450 and log.requests == 3


def test_nyt_splits_when_hits_exceed_1000():
    def fn(path, p):
        days = (datetime.strptime(p["end_date"], "%Y%m%d") - datetime.strptime(p["begin_date"], "%Y%m%d")).days + 1
        hits = 60 * days
        docs = [{"pub_date": "2024-01-02T00:00:00Z", "headline": {"main": "h"}, "web_url": f"{p['begin_date']}-{p['page']}-{i}"}
                for i in range(min(10, max(0, hits - 10 * p["page"])))]
        return {"status": "OK", "response": {"docs": docs, "meta": {"hits": hits}}}
    c = FakeClient(fn)
    log = FetchLog()
    arts = NYT(c).articles("spx", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC), log)
    assert len(arts) == 60 * 31 and not log.truncations


def test_alphavantage_splits_on_cap():
    def fn(path, p):
        s = datetime.strptime(p["time_from"], "%Y%m%dT%H%M")
        e = datetime.strptime(p["time_to"], "%Y%m%dT%H%M")
        n = min(1000, int(100 * (e - s).total_seconds() / 86400))
        return {"feed": [{"time_published": f"{s:%Y%m%dT%H%M%S}", "title": "x", "url": f"{s:%Y%m%d%H%M}-{i}"} for i in range(n)]}
    log = FetchLog()
    arts = AlphaVantage(FakeClient(fn)).articles("btc", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 31, tzinfo=UTC), log)
    assert len(arts) >= 2900 and not log.truncations


def test_store_accumulates_and_dedups(tmp_path):
    st = ArticleStore(tmp_path)
    row = {"source": "gdelt", "asset": "spx", "published_at": pd.Timestamp("2024-01-05", tz="UTC"), "title": "a",
           "snippet": "", "url": "u1", "domain": "", "language": "", "country": ""}
    assert st.append([row]) == 1
    assert st.append([row, {**row, "url": "u2"}]) == 1          # u1 already stored
    assert len(st.load("spx")) == 2


def test_windows_helpers():
    w = list(month_windows(datetime(2024, 1, 15, tzinfo=UTC), datetime(2024, 3, 10, tzinfo=UTC)))
    assert w[0] == (datetime(2024, 1, 15, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC)) and len(w) == 3
    pw = probe_windows(datetime(2015, 1, 1, tzinfo=UTC), datetime(2016, 1, 1, tzinfo=UTC))
    assert [x[0].month for x in pw] == [1, 4, 7, 10] and all((b - a).days == 7 for a, b in pw)


def test_assess_density_rule():
    def probe(vals):  # one value per quarter, 2015..
        starts = pd.date_range("2015-01-15", periods=len(vals), freq="QS-JAN", tz="UTC") + pd.Timedelta(days=14)
        return pd.DataFrame({"window_start": starts, "articles_per_trading_day": vals})
    ok = assess_density(probe([5] * 8 + [6] * 4), end_year=2017)
    assert ok["accepted"] and ok["start_year"] == 2015
    hole = assess_density(probe([5, 0.5] + [5] * 10), end_year=2017)  # hole in 2015 -> start 2016
    assert hole["accepted"] and hole["start_year"] == 2016
    sparse_then_dense = assess_density(probe([1] * 4 + [10] * 8), end_year=2017)   # ratio 10 > 4
    assert sparse_then_dense["start_year"] == 2016
    never = assess_density(probe([0] * 12), end_year=2017)
    assert not never["accepted"]


def test_select_sources_prefers_longest_common_period():
    from alphavol.fetch.news import select_sources
    starts = pd.date_range("2015-01-15", periods=12, freq="QS-JAN", tz="UTC") + pd.Timedelta(days=14)
    rows = []
    for a in ("btc", "spx", "nifty"):
        for i, s in enumerate(starts):
            rows.append({"source": "steady", "asset": a, "window_start": s, "articles_per_trading_day": 5.0})
            # 'late' starts in 2017 with high volume: adding it breaks the ratio test for the union
            rows.append({"source": "late", "asset": a, "window_start": s,
                         "articles_per_trading_day": 0.0 if s.year < 2017 else 50.0})
    res = select_sources(pd.DataFrame(rows), assets=("btc", "spx", "nifty"), end_year=2017)
    assert res["choice"]["sources"] == ["steady"] and res["choice"]["common_start_year"] == 2015


def test_gdelt_sample_mode_one_request_and_records_cap():
    c = FakeClient(_gdelt_fn(1000))
    log = FetchLog()
    s = datetime(2024, 1, 1, tzinfo=UTC)
    arts = GdeltDoc(c).articles("spx", s, s + timedelta(days=1), log, sample=True)
    assert len(arts) == 250 and len(c.calls) == 1 and c.calls[0]["sort"] == "hybridrel"
    assert len(log.truncations) == 1


def test_gdelt_daily_volume_sums_by_day():
    body = {"timeline": [{"series": "Article Count", "data": [
        {"date": "20240101T000000Z", "value": 3}, {"date": "20240101T120000Z", "value": 2},
        {"date": "20240102T000000Z", "value": 7}]}]}
    v = GdeltDoc(FakeClient(lambda p, q: body)).daily_volume("spx", datetime(2024, 1, 1, tzinfo=UTC),
                                                             datetime(2024, 1, 3, tzinfo=UTC))
    assert v.tolist() == [5.0, 7.0]
