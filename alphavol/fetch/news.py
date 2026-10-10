"""News source adapters, article store, and the coverage probe (DECISIONS.md D25, D26).

Each adapter offers ``count(asset, start, end)`` (article volume, used by the coverage probe)
and ``articles(asset, start, end)`` (headlines for FinBERT). Windows that hit a source's
per-request cap are split recursively; windows still capped at the minimum width are recorded
as truncated (``Truncation`` log), never silently ignored.

Queries are index- or market-level (R3). NIFTY uses Indian market terms restricted to Indian
sources, not a single-stock proxy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Protocol

import pandas as pd

from alphavol.fetch.http import CachedClient, FetchError
from alphavol.logging_utils import get_logger

log = get_logger(__name__)
UTC = timezone.utc
ARTICLE_COLUMNS = ["source", "asset", "published_at", "title", "snippet", "url", "domain", "language", "country"]


# ── Queries (fixed in advance; changing them must be logged in DECISIONS.md) ──────────────────
GDELT_QUERIES = {
    "btc": '(bitcoin OR cryptocurrency OR "crypto market") sourcelang:english',
    "spx": '("S&P 500" OR "Wall Street" OR "stock market" OR "Dow Jones" OR Nasdaq) sourcecountry:unitedstates sourcelang:english',
    "nifty": '(Sensex OR Nifty OR "Dalal Street" OR "Indian stocks" OR "Indian stock market") sourcecountry:india sourcelang:english',
}
GUARDIAN_QUERIES = {
    "btc": 'bitcoin OR cryptocurrency',
    "spx": '"S&P 500" OR "Wall Street" OR "Dow Jones" OR "stock market"',
    "nifty": 'Sensex OR Nifty OR "Indian stock market" OR "Bombay Stock Exchange"',
}
NYT_QUERIES = GUARDIAN_QUERIES
AV_PARAMS = {"btc": {"tickers": "CRYPTO:BTC"}, "spx": {"topics": "financial_markets"}}  # no NIFTY query: AV has no index-level Indian coverage


@dataclass
class Truncation:
    source: str
    asset: str
    start: str
    end: str
    returned: int


@dataclass
class FetchLog:
    truncations: list[Truncation] = field(default_factory=list)
    requests: int = 0


class NewsSource(Protocol):
    name: str

    def count(self, asset: str, start: datetime, end: datetime) -> int: ...
    def articles(self, asset: str, start: datetime, end: datetime, flog: FetchLog) -> list[dict]: ...


def _ts(s: str, fmt: str | None = None) -> pd.Timestamp:
    return pd.to_datetime(s, format=fmt, utc=True)


def _split(start: datetime, end: datetime, *, day_aligned: bool = False
           ) -> tuple[tuple[datetime, datetime], tuple[datetime, datetime]]:
    """Split [start, end) in two. Day-granular APIs (inclusive date bounds) need midnight splits,
    otherwise the boundary day is requested twice."""
    mid = start + (end - start) / 2
    if day_aligned:
        mid = mid.replace(hour=0, minute=0, second=0, microsecond=0)
        if mid <= start:
            mid = start + timedelta(days=1)
    return (start, mid), (mid, end)


# ── GDELT DOC 2.0 (no key) ─────────────────────────────────────────────────────────────
class GdeltDoc:
    name = "gdelt"
    CAP = 250
    MIN_WINDOW = timedelta(hours=1)

    def __init__(self, client: CachedClient | None = None):
        self.c = client or CachedClient("gdelt", "https://api.gdeltproject.org/api/v2/doc/doc", min_interval=5.5)

    @staticmethod
    def _fmt(d: datetime) -> str:
        return d.astimezone(UTC).strftime("%Y%m%d%H%M%S")

    def count(self, asset: str, start: datetime, end: datetime) -> int:
        body = self.c.get("", {"query": GDELT_QUERIES[asset], "mode": "timelinevolraw", "format": "json",
                               "startdatetime": self._fmt(start), "enddatetime": self._fmt(end)})
        if not isinstance(body, dict) or "timeline" not in body:
            raise FetchError(f"gdelt: unexpected timeline payload for {asset} {start}..{end}: {str(body)[:200]}")
        total = 0
        for series in body["timeline"]:
            for pt in series.get("data", []):
                total += int(pt.get("value", 0))
        return total

    def articles(self, asset: str, start: datetime, end: datetime, flog: FetchLog) -> list[dict]:
        body = self.c.get("", {"query": GDELT_QUERIES[asset], "mode": "artlist", "format": "json",
                               "maxrecords": self.CAP, "sort": "datedesc",
                               "startdatetime": self._fmt(start), "enddatetime": self._fmt(end)})
        flog.requests += 1
        if not isinstance(body, dict):
            raise FetchError(f"gdelt: unexpected artlist payload: {str(body)[:200]}")
        arts = body.get("articles", [])
        if len(arts) >= self.CAP:
            if end - start > self.MIN_WINDOW:
                a, b = _split(start, end)
                return self.articles(asset, *a, flog) + self.articles(asset, *b, flog)
            flog.truncations.append(Truncation(self.name, asset, str(start), str(end), len(arts)))
        return [{"source": self.name, "asset": asset, "published_at": _ts(x["seendate"], "%Y%m%dT%H%M%SZ"),
                 "title": x.get("title", ""), "snippet": "", "url": x.get("url", ""), "domain": x.get("domain", ""),
                 "language": x.get("language", ""), "country": x.get("sourcecountry", "")} for x in arts]


# ── The Guardian Open Platform (key: GUARDIAN_API_KEY) ───────────────────────────────────
class Guardian:
    name = "guardian"
    PAGE = 200

    def __init__(self, client: CachedClient | None = None):
        self.c = client or CachedClient("guardian", "https://content.guardianapis.com", min_interval=1.1,
                                        key_env="GUARDIAN_API_KEY", key_param="api-key")

    def _query(self, asset, start, end, page, size):
        body = self.c.get("/search", {"q": GUARDIAN_QUERIES[asset], "from-date": start.strftime("%Y-%m-%d"),
                                      "to-date": (end - timedelta(seconds=1)).strftime("%Y-%m-%d"),
                                      "page": page, "page-size": size, "order-by": "oldest",
                                      "show-fields": "trailText"})
        r = body.get("response", {}) if isinstance(body, dict) else {}
        if r.get("status") != "ok":
            raise FetchError(f"guardian: bad response for {asset} {start:%Y-%m-%d}: {str(body)[:200]}")
        return r

    def count(self, asset, start, end) -> int:
        return int(self._query(asset, start, end, 1, 1)["total"])

    def articles(self, asset, start, end, flog: FetchLog) -> list[dict]:
        out, page = [], 1
        while True:
            r = self._query(asset, start, end, page, self.PAGE)
            flog.requests += 1
            for x in r.get("results", []):
                out.append({"source": self.name, "asset": asset, "published_at": _ts(x["webPublicationDate"]),
                            "title": x.get("webTitle", ""), "snippet": (x.get("fields") or {}).get("trailText", ""),
                            "url": x.get("webUrl", ""), "domain": "theguardian.com", "language": "en", "country": "UK"})
            if page >= int(r.get("pages", 1)):
                return out
            page += 1


# ── NYT Article Search (key: NYT_API_KEY) ─────────────────────────────────────────────
class NYT:
    name = "nyt"
    PAGE, MAX_PAGES = 10, 100

    def __init__(self, client: CachedClient | None = None):
        self.c = client or CachedClient("nyt", "https://api.nytimes.com/svc/search/v2/articlesearch.json",
                                        min_interval=12.5, key_env="NYT_API_KEY", key_param="api-key")

    def _page(self, asset, start, end, page):
        body = self.c.get("", {"q": NYT_QUERIES[asset], "begin_date": start.strftime("%Y%m%d"),
                               "end_date": (end - timedelta(seconds=1)).strftime("%Y%m%d"), "page": page,
                               "sort": "oldest"})
        if not isinstance(body, dict) or body.get("status") != "OK":
            raise FetchError(f"nyt: bad response for {asset} {start:%Y-%m-%d}: {str(body)[:200]}")
        resp = body["response"]
        meta = resp.get("meta") or resp.get("metadata") or {}
        return resp.get("docs") or [], int(meta.get("hits", 0))

    def count(self, asset, start, end) -> int:
        return self._page(asset, start, end, 0)[1]

    def articles(self, asset, start, end, flog: FetchLog) -> list[dict]:
        docs, hits = self._page(asset, start, end, 0)
        flog.requests += 1
        if hits > self.PAGE * self.MAX_PAGES:
            if end - start > timedelta(days=1):
                a, b = _split(start, end, day_aligned=True)
                return self.articles(asset, *a, flog) + self.articles(asset, *b, flog)
            flog.truncations.append(Truncation(self.name, asset, str(start), str(end), hits))
        out = list(docs)
        for page in range(1, min(self.MAX_PAGES, -(-hits // self.PAGE))):
            more, _ = self._page(asset, start, end, page)
            flog.requests += 1
            out += more
        return [{"source": self.name, "asset": asset, "published_at": _ts(d["pub_date"]),
                 "title": (d.get("headline") or {}).get("main", ""), "snippet": d.get("abstract", ""),
                 "url": d.get("web_url", ""), "domain": "nytimes.com", "language": "en", "country": "US"} for d in out]


# ── Alpha Vantage NEWS_SENTIMENT (key: ALPHAVANTAGE_API_KEY) ──────────────────────────────
class AlphaVantage:
    name = "alphavantage"
    CAP = 1000

    def __init__(self, client: CachedClient | None = None):
        self.c = client or CachedClient("alphavantage", "https://www.alphavantage.co/query", min_interval=15.0,
                                        key_env="ALPHAVANTAGE_API_KEY", key_param="apikey")

    def _feed(self, asset, start, end):
        if asset not in AV_PARAMS:
            raise FetchError(f"alphavantage: no index-level query defined for {asset}")
        body = self.c.get("", {"function": "NEWS_SENTIMENT", **AV_PARAMS[asset], "sort": "EARLIEST",
                               "limit": self.CAP, "time_from": start.strftime("%Y%m%dT%H%M"),
                               "time_to": end.strftime("%Y%m%dT%H%M")})
        if not isinstance(body, dict) or "feed" not in body:
            raise FetchError(f"alphavantage: no 'feed' (rate limit or bad key?) for {asset}: {str(body)[:200]}")
        return body["feed"]

    def count(self, asset, start, end) -> int:
        return len(self._feed(asset, start, end))           # lower bound when == CAP

    def articles(self, asset, start, end, flog: FetchLog) -> list[dict]:
        feed = self._feed(asset, start, end)
        flog.requests += 1
        if len(feed) >= self.CAP:
            if end - start > timedelta(days=1):
                a, b = _split(start, end)
                return self.articles(asset, *a, flog) + self.articles(asset, *b, flog)
            flog.truncations.append(Truncation(self.name, asset, str(start), str(end), len(feed)))
        return [{"source": self.name, "asset": asset, "published_at": _ts(x["time_published"], "%Y%m%dT%H%M%S"),
                 "title": x.get("title", ""), "snippet": x.get("summary", ""), "url": x.get("url", ""),
                 "domain": x.get("source_domain", ""), "language": "en", "country": ""} for x in feed]


SOURCES = {"gdelt": GdeltDoc, "guardian": Guardian, "nyt": NYT, "alphavantage": AlphaVantage}


# ── Accumulating store (append + dedup; never overwrite) ─────────────────────────────────
class ArticleStore:
    """Monthly parquet files under ``root/<source>/<asset>/YYYY-MM.parquet``."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, source: str, asset: str, month: str) -> Path:
        return self.root / source / asset / f"{month}.parquet"

    def append(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        df = pd.DataFrame(rows, columns=ARTICLE_COLUMNS)
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
        added = 0
        for (src, asset, month), g in df.groupby([df["source"], df["asset"], df["published_at"].dt.strftime("%Y-%m")]):
            p = self.path(src, asset, month)
            old = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=ARTICLE_COLUMNS)
            merged = pd.concat([old, g], ignore_index=True)
            merged["published_at"] = pd.to_datetime(merged["published_at"], utc=True)
            merged = merged.drop_duplicates(subset=["url"], keep="first").sort_values("published_at")
            added += len(merged) - len(old)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            merged.to_parquet(tmp, index=False)
            tmp.replace(p)
        return added

    def load(self, asset: str, sources: list[str] | None = None) -> pd.DataFrame:
        files = sorted(self.root.glob(f"*/{asset}/*.parquet"))
        if sources is not None:
            files = [f for f in files if f.parts[-3] in sources]
        if not files:
            return pd.DataFrame(columns=ARTICLE_COLUMNS)
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def month_windows(start: datetime, end: datetime) -> Iterator[tuple[datetime, datetime]]:
    """[start, end) cut at calendar-month boundaries (UTC)."""
    cur = start
    while cur < end:
        nxt = (cur.replace(day=1) + timedelta(days=32)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        yield cur, min(nxt, end)
        cur = min(nxt, end)


# ── Coverage probe and the pre-specified acceptance rule (D25) ─────────────────────────────
def probe_windows(first: datetime, last: datetime, days: int = 7) -> list[tuple[datetime, datetime]]:
    """One ``days``-long window per calendar quarter starting on day 15 of the quarter's first month."""
    out, q = [], datetime(first.year, 3 * ((first.month - 1) // 3) + 1, 15, tzinfo=UTC)
    while q + timedelta(days=days) <= last:
        out.append((q, q + timedelta(days=days)))
        m = q.month + 3
        q = q.replace(year=q.year + (m > 12), month=(m - 1) % 12 + 1)
    return out


def assess_density(probe: pd.DataFrame, *, min_per_day: float = 1.0, max_year_ratio: float = 4.0,
                   end_year: int) -> dict:
    """Apply the D25 criterion to probe results for one source (or union) and one asset.

    ``probe`` columns: window_start (UTC ts), articles_per_trading_day. Returns the longest
    period P ending at ``end_year`` such that (1) every quarter in P has ≥ ``min_per_day`` and
    (2) max/min of yearly means in P ≤ ``max_year_ratio``.
    """
    p = probe.sort_values("window_start").copy()
    p["year"] = p["window_start"].dt.year
    years = sorted(p["year"].unique())
    best = None
    for start_year in years:
        sub = p[(p["year"] >= start_year) & (p["year"] <= end_year)]
        if sub.empty:
            continue
        holes = bool((sub["articles_per_trading_day"] < min_per_day).any())
        ym = sub.groupby("year")["articles_per_trading_day"].mean()
        ratio = float(ym.max() / ym.min()) if (ym.min() > 0) else float("inf")
        if not holes and ratio <= max_year_ratio:
            best = {"accepted": True, "start_year": int(start_year), "end_year": int(end_year),
                    "year_ratio": ratio, "yearly_mean": {int(k): float(v) for k, v in ym.items()}}
            break                                     # earliest start = longest period
    return best or {"accepted": False, "start_year": None, "end_year": int(end_year)}
