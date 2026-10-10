"""Market and macro fetchers for the local run (DECISIONS.md D7, D15, D26, D31).

* Binance 5-minute klines via the public market-data host (paged, cached, validated).
* Yahoo Finance daily bars via ``yfinance`` (pinned), complete sessions only.
* FRED series via the public ``fredgraph.csv`` endpoint (no key needed).
* Alternative.me Crypto Fear & Greed (attribution required wherever displayed).
All functions raise on empty or malformed results; none returns partial data silently.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from alphavol.fetch.http import CachedClient, FetchError
from alphavol.logging_utils import get_logger

log = get_logger(__name__)
UTC = timezone.utc
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
              "number_of_trades", "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"]


def binance_klines(start: datetime, end: datetime, *, symbol: str = "BTCUSDT", interval: str = "5m",
                   client: CachedClient | None = None) -> pd.DataFrame:
    """Klines with open_time in [start, end), indexed by open_time (UTC)."""
    c = client or CachedClient("binance", "https://data-api.binance.vision", min_interval=0.3)
    step_ms = {"5m": 300_000, "1h": 3_600_000, "1d": 86_400_000}[interval]
    cur, end_ms, rows = int(start.timestamp() * 1000), int(end.timestamp() * 1000), []
    while cur < end_ms:
        page = c.get("/api/v3/klines", {"symbol": symbol, "interval": interval, "startTime": cur,
                                        "endTime": end_ms - 1, "limit": 1000})
        if not isinstance(page, list):
            raise FetchError(f"binance: unexpected payload {str(page)[:200]}")
        if not page:
            break
        rows += page
        nxt = int(page[-1][0]) + step_ms
        if nxt <= cur:
            raise FetchError("binance: pagination did not advance")
        cur = nxt
    if not rows:
        raise FetchError(f"binance: no klines for {symbol} {start}..{end}")
    df = pd.DataFrame(rows, columns=KLINE_COLS).drop(columns="ignore")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates("open_time").set_index("open_time").sort_index()
    num = [c for c in df.columns if c != "close_time"]
    df[num] = df[num].astype(float)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise FetchError("binance: klines not strictly increasing")
    return df


def yahoo_daily(ticker: str, start: str, end_inclusive: str) -> pd.DataFrame:
    """Daily OHLCV for [start, end_inclusive] via yfinance; index = session date at 00:00 UTC."""
    import yfinance as yf

    end_excl = (pd.Timestamp(end_inclusive) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end_excl, interval="1d", auto_adjust=False, actions=False,
                     progress=False, threads=False)
    if df is None or df.empty:
        raise FetchError(f"yahoo: no data for {ticker} {start}..{end_inclusive}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(pd.DatetimeIndex(df.index).date).tz_localize("UTC")
    df.index.name = "date"
    if df[["open", "high", "low", "close"]].isna().any().any():
        raise FetchError(f"yahoo: missing OHLC values for {ticker}")
    return df


def fred_series(code: str, start: str, end: str, *, client: CachedClient | None = None,
                cache_tag: str | None = None) -> pd.Series:
    """FRED series via fredgraph.csv (no key). Missing values ('.') become NaN."""
    c = client or CachedClient("fred", "https://fred.stlouisfed.org/graph/fredgraph.csv", min_interval=1.0,
                               expect="text")
    text = c.get("", {"id": code, "cosd": start, "coed": end}, cache_tag=cache_tag)
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    if df.shape[1] != 2 or code not in df.columns:
        raise FetchError(f"fred: unexpected CSV for {code}: columns {list(df.columns)}")
    s = pd.Series(pd.to_numeric(df[code], errors="raise").to_numpy(),
                  index=pd.to_datetime(df.iloc[:, 0]).dt.tz_localize("UTC"), name=code)
    if s.notna().sum() == 0:
        raise FetchError(f"fred: {code} has no observations in {start}..{end}")
    return s


def assert_coverage(s: pd.Series, *, first_on_or_before: str) -> None:
    """D15: fail loudly if a series does not reach back to the required start."""
    first = s.first_valid_index()
    if first is None or first > pd.Timestamp(first_on_or_before, tz="UTC"):
        raise FetchError(f"{s.name}: first valid observation {first} is after required {first_on_or_before}")


def fear_greed(*, client: CachedClient | None = None, cache_tag: str | None = None) -> pd.Series:
    """Alternative.me Crypto Fear & Greed, full history (value 0–100) indexed by UTC date."""
    c = client or CachedClient("alternative", "https://api.alternative.me/fng/", min_interval=1.0)
    body = c.get("", {"limit": 0, "format": "json"}, cache_tag=cache_tag)
    data = body.get("data") if isinstance(body, dict) else None
    if not data:
        raise FetchError(f"alternative.me: unexpected payload {str(body)[:200]}")
    idx = pd.to_datetime(np.array([int(x["timestamp"]) for x in data]), unit="s", utc=True).normalize()
    s = pd.Series([float(x["value"]) for x in data], index=idx, name="fear_greed").sort_index()
    return s[~s.index.duplicated(keep="last")]
