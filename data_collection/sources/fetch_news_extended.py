"""
Extended News Fetcher — Multi-Source Historical Headlines
===========================================================
Sources (all free, all return actual text for FinBERT):

1. Alpha Vantage News & Sentiment API
   - Free key: https://www.alphavantage.co/support/#api-key (instant)
   - Coverage: 2020 -> now, up to 1000 articles/request
   - Assets: BTC (CRYPTO:BTC), SPX (SPY), NIFTY (HDB as proxy)
   - Bonus: includes pre-scored sentiment (cross-validate with FinBERT)

2. The Guardian Open API  
   - Free key: https://open-platform.theguardian.com/access/support-us (instant)
   - Coverage: 2010 -> now, 200/page, 5000 requests/day
   - Full article text available
   - Queries: bitcoin, S&P 500, NIFTY India

3. NewsAPI (existing key ad0e16664...)
   - Historical: last 30 days on free tier (already fetched)
   - We'll maximise it with targeted queries

Output: data/raw/sentiment/extended_news_headlines.parquet
         One row per article: date | asset | source | title | description | url

Usage:
    Add keys to .env, then:
    python data_collection/sources/fetch_news_extended.py

    Or via fetch_all.py after adding to sources_to_run.
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUT_PATH = "data/raw/sentiment/extended_news_headlines.parquet"


# ─────────────────────────────────────────────────────────────
# SOURCE 1: Alpha Vantage News & Sentiment API
# Free key at: https://www.alphavantage.co/support/#api-key
# Covers: BTC (back to ~2020), SPX, global markets
# ─────────────────────────────────────────────────────────────
def fetch_alphavantage_news(api_key: str, start_date: str = "2020-01-01") -> pd.DataFrame:
    """
    Fetches news articles from Alpha Vantage Intelligence API.
    Each article includes title, summary, source, and pre-computed
    sentiment scores (overall + ticker-specific) from their own model.
    We keep the text for FinBERT re-scoring and also store their score
    as a cross-validation signal.

    Parameters
    ----------
    api_key    : Alpha Vantage API key (free at alphavantage.co)
    start_date : ISO date string, e.g. '2020-01-01'
    """
    BASE = "https://www.alphavantage.co/query"

    # Ticker mappings (Alpha Vantage format)
    ticker_map = {
        "BTC":   "CRYPTO:BTC",
        "SPX":   "SPY",          # S&P 500 ETF — good proxy
        "NIFTY": "HDB",          # HDFC Bank ADR — NIFTY proxy (AV doesn't have ^NSEI)
    }

    all_records = []
    start_dt = pd.to_datetime(start_date, utc=True)
    now_dt   = pd.Timestamp.now(tz='UTC')

    # Walk month by month to maximize coverage
    # AV free tier: 25 requests/day. Each request = up to 1000 articles.
    # For 4 years × 3 assets = 144 monthly requests — spread across days if needed.
    # Strategy: 3 assets × 1 request each = 3 calls, gets ~1000 most recent articles per asset.
    # Re-run weekly to accumulate historical data.

    for asset, ticker in ticker_map.items():
        logger.info(f"  AlphaVantage: fetching {asset} ({ticker})...")

        params = {
            "function":  "NEWS_SENTIMENT",
            "tickers":   ticker,
            "sort":      "LATEST",
            "limit":     1000,        # max per request
            "apikey":    api_key,
        }

        try:
            resp = requests.get(BASE, params=params, timeout=30)
            data = resp.json()

            if "feed" not in data:
                logger.warning(f"  AV {asset}: no 'feed' in response. "
                               f"Check key or rate limit. Response: {str(data)[:200]}")
                continue

            articles = data["feed"]
            logger.info(f"  AV {asset}: {len(articles)} articles returned")

            for art in articles:
                # Parse publish time
                try:
                    pub_str = art.get("time_published", "")
                    # Format: "20240101T120000"
                    pub_dt  = pd.to_datetime(pub_str, format="%Y%m%dT%H%M%S", utc=True)
                except Exception:
                    pub_dt = pd.NaT

                # Alpha Vantage pre-scores: overall_sentiment_score ∈ [-1, 1]
                av_score = art.get("overall_sentiment_score", None)

                all_records.append({
                    "date":        pub_dt,
                    "asset":       asset,
                    "source":      "AlphaVantage",
                    "title":       art.get("title", ""),
                    "description": art.get("summary", ""),
                    "url":         art.get("url", ""),
                    "av_sentiment_score": float(av_score) if av_score is not None else None,
                    "av_sentiment_label": art.get("overall_sentiment_label", ""),
                })

        except Exception as e:
            logger.error(f"  AlphaVantage error for {asset}: {e}")

        time.sleep(15)  # AV free tier: 25 req/day -> be conservative

    return pd.DataFrame(all_records)


# ─────────────────────────────────────────────────────────────
# SOURCE 2: The Guardian Open Platform API
# Free key at: https://open-platform.theguardian.com/access/
# Coverage: 2010 -> now, 200/page, 5000 req/day
# Full article text available on request
# ─────────────────────────────────────────────────────────────
def fetch_guardian_news(api_key: str, start_date: str = "2019-01-01",
                         max_pages_per_query: int = 25) -> pd.DataFrame:
    """
    Fetches headlines from The Guardian API.
    200 articles/page × 25 pages × 3 queries = 15,000 articles.
    With `show-fields=headline,trailText,bodyText` we get full text.

    Parameters
    ----------
    api_key            : Guardian API key (free)
    start_date         : ISO date to start from
    max_pages_per_query: how many pages to fetch per query (200/page)
    """
    BASE = "https://content.guardianapis.com/search"

    # Search queries per asset
    queries = {
        "BTC":   "bitcoin OR cryptocurrency OR crypto market OR ethereum",
        "SPX":   "S&P 500 OR stock market OR wall street OR US stocks OR federal reserve",
        "NIFTY": "NIFTY OR sensex OR BSE OR NSE OR Indian stock market OR RBI",
    }

    all_records = []

    for asset, q in queries.items():
        logger.info(f"  Guardian: fetching {asset} | '{q[:50]}...'")
        page = 1

        while page <= max_pages_per_query:
            try:
                params = {
                    "q":             q,
                    "from-date":     start_date,
                    "order-by":      "newest",
                    "page":          page,
                    "page-size":     200,           # max per page
                    "show-fields":   "headline,trailText,bodyText,publication",
                    "api-key":       api_key,
                }
                resp = requests.get(BASE, params=params, timeout=30)
                data = resp.json()

                if data.get("response", {}).get("status") != "ok":
                    logger.warning(f"  Guardian {asset} p{page}: {str(data)[:200]}")
                    break

                results = data["response"].get("results", [])
                total_pages = data["response"].get("pages", 1)

                if not results:
                    break

                for art in results:
                    fields = art.get("fields", {})
                    pub_date = art.get("webPublicationDate", "")
                    try:
                        pub_dt = pd.to_datetime(pub_date, utc=True)
                    except Exception:
                        pub_dt = pd.NaT

                    # Prefer bodyText for FinBERT, fall back to trailText, then headline
                    body     = fields.get("bodyText", "")
                    trail    = fields.get("trailText", "")
                    headline = fields.get("headline", art.get("webTitle", ""))

                    # For FinBERT: use headline + trail (first 512 chars is enough)
                    text_for_finbert = f"{headline}. {trail}"[:512]

                    all_records.append({
                        "date":        pub_dt,
                        "asset":       asset,
                        "source":      "Guardian",
                        "title":       headline,
                        "description": trail or body[:300],
                        "text_finbert": text_for_finbert,
                        "url":         art.get("webUrl", ""),
                        "av_sentiment_score": None,
                        "av_sentiment_label": "",
                    })

                logger.info(f"    Page {page}/{total_pages}: {len(results)} articles")

                if page >= total_pages:
                    break
                page += 1
                time.sleep(0.5)   # Guardian allows 12 req/sec on free tier

            except Exception as e:
                logger.error(f"  Guardian {asset} p{page} error: {e}")
                break

    return pd.DataFrame(all_records)


# ─────────────────────────────────────────────────────────────
# SOURCE 3: Finnhub News API
# Free key at: https://finnhub.io/register
# Coverage: ~2 years, stocks + crypto, good for SPX
# ─────────────────────────────────────────────────────────────
def fetch_finnhub_news(api_key: str, days_back: int = 365) -> pd.DataFrame:
    """
    Fetches recent news from Finnhub.
    Free tier: 60 API calls/minute. Good for SPX company + market news.
    """
    BASE = "https://finnhub.io/api/v1"

    all_records = []
    end_dt   = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days_back)

    end_str   = end_dt.strftime("%Y-%m-%d")
    start_str = start_dt.strftime("%Y-%m-%d")

    # General market news categories
    categories = {
        "SPX":   ("general", None),   # category-based market news
        "NIFTY": ("general", None),
    }
    # Crypto news
    crypto_symbols = [("BTC", "BINANCE:BTCUSDT")]

    try:
        # General market news (SPX / NIFTY)
        for asset, (category, _) in categories.items():
            logger.info(f"  Finnhub: fetching {asset} general news...")
            resp = requests.get(
                f"{BASE}/news",
                params={"category": category, "minId": 0, "token": api_key},
                timeout=20
            )
            articles = resp.json() if resp.status_code == 200 else []
            for art in (articles if isinstance(articles, list) else []):
                ts = art.get("datetime", 0)
                try:
                    pub_dt = pd.to_datetime(ts, unit='s', utc=True)
                except Exception:
                    pub_dt = pd.NaT
                all_records.append({
                    "date":        pub_dt,
                    "asset":       asset,
                    "source":      "Finnhub",
                    "title":       art.get("headline", ""),
                    "description": art.get("summary", ""),
                    "text_finbert": f"{art.get('headline','')}. {art.get('summary',''[:300])}",
                    "url":         art.get("url", ""),
                    "av_sentiment_score": None,
                    "av_sentiment_label": "",
                })
            time.sleep(1)

        # Crypto news
        for asset, symbol in crypto_symbols:
            logger.info(f"  Finnhub: fetching {asset} crypto news ({symbol})...")
            resp = requests.get(
                f"{BASE}/company-news",
                params={"symbol": symbol, "from": start_str, "to": end_str, "token": api_key},
                timeout=20
            )
            articles = resp.json() if resp.status_code == 200 else []
            for art in (articles if isinstance(articles, list) else []):
                ts = art.get("datetime", 0)
                try:
                    pub_dt = pd.to_datetime(ts, unit='s', utc=True)
                except Exception:
                    pub_dt = pd.NaT
                all_records.append({
                    "date":        pub_dt,
                    "asset":       asset,
                    "source":      "Finnhub",
                    "title":       art.get("headline", ""),
                    "description": art.get("summary", ""),
                    "text_finbert": f"{art.get('headline','')}. {art.get('summary',''[:300])}",
                    "url":         art.get("url", ""),
                    "av_sentiment_score": None,
                    "av_sentiment_label": "",
                })
            time.sleep(1)

    except Exception as e:
        logger.error(f"Finnhub error: {e}")

    return pd.DataFrame(all_records)


# ─────────────────────────────────────────────────────────────
# SOURCE 4: CryptoPanic (already in fetch_sentiment.py)
# Expose a standalone fetcher here for completeness.
# Free key at: https://cryptopanic.com/developers/api/
# ─────────────────────────────────────────────────────────────
def fetch_cryptopanic_extended(api_key: str, max_pages: int = 100) -> pd.DataFrame:
    """
    Paginate through CryptoPanic for BTC/ETH headlines.
    Free tier: unlimited reads with API key.
    ~50 articles/page × 100 pages = 5,000 articles.
    """
    if not api_key or api_key == "your_api_key_here":
        logger.warning("CryptoPanic API key not set — skipping.")
        return pd.DataFrame()

    all_records = []
    base_url = f"https://cryptopanic.com/api/v1/posts/?auth_token={api_key}&public=true"

    for currency in ["BTC", "ETH"]:
        url = f"{base_url}&currencies={currency}&kind=news"
        page = 0
        while url and page < max_pages:
            try:
                resp = requests.get(url, timeout=20)
                if resp.status_code != 200:
                    break
                data = resp.json()
                for item in data.get("results", []):
                    votes  = item.get("votes", {})
                    pub_at = item.get("published_at", "")
                    try:
                        pub_dt = pd.to_datetime(pub_at, utc=True)
                    except Exception:
                        pub_dt = pd.NaT

                    # Simple panic score as extra signal
                    pos = votes.get("positive", 0)
                    neg = votes.get("negative", 0) + votes.get("toxic", 0)
                    panic_score = neg / (pos + neg + 1)

                    all_records.append({
                        "date":        pub_dt,
                        "asset":       "BTC",
                        "source":      "CryptoPanic",
                        "title":       item.get("title", ""),
                        "description": "",
                        "text_finbert": item.get("title", ""),
                        "url":         item.get("url", ""),
                        "av_sentiment_score": 1 - 2 * panic_score,  # convert to [-1, 1]
                        "av_sentiment_label": "bullish" if panic_score < 0.3 else "bearish",
                    })

                url = data.get("next")
                page += 1
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"CryptoPanic error: {e}")
                break

    return pd.DataFrame(all_records)


# ─────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────
def run(start_date: str = "2019-01-01"):
    """
    Fetch from all available sources and combine into one master headlines file.
    """
    av_key        = os.getenv("ALPHAVANTAGE_API_KEY", "")
    guardian_key  = os.getenv("GUARDIAN_API_KEY", "")
    finnhub_key   = os.getenv("FINNHUB_API_KEY", "")
    cryptopanic_key = os.getenv("CRYPTOPANIC_API_KEY", "")

    frames = []

    # Source 1: Alpha Vantage (needs key)
    if av_key and av_key != "your_api_key_here":
        logger.info("[1/4] Fetching Alpha Vantage News...")
        df_av = fetch_alphavantage_news(av_key, start_date)
        if not df_av.empty:
            logger.info(f"  Alpha Vantage: {len(df_av)} articles")
            frames.append(df_av)
    else:
        logger.warning("[1/4] ALPHAVANTAGE_API_KEY not set — get free key at alphavantage.co")

    # Source 2: The Guardian (needs key)
    if guardian_key and guardian_key != "your_api_key_here":
        logger.info("[2/4] Fetching Guardian News...")
        df_g = fetch_guardian_news(guardian_key, start_date)
        if not df_g.empty:
            logger.info(f"  Guardian: {len(df_g)} articles")
            frames.append(df_g)
    else:
        logger.warning("[2/4] GUARDIAN_API_KEY not set — get free key at open-platform.theguardian.com")

    # Source 3: Finnhub (needs key)
    if finnhub_key and finnhub_key != "your_api_key_here":
        logger.info("[3/4] Fetching Finnhub News...")
        df_fh = fetch_finnhub_news(finnhub_key)
        if not df_fh.empty:
            logger.info(f"  Finnhub: {len(df_fh)} articles")
            frames.append(df_fh)
    else:
        logger.warning("[3/4] FINNHUB_API_KEY not set — get free key at finnhub.io")

    # Source 4: CryptoPanic (needs key, good for BTC)
    if cryptopanic_key and cryptopanic_key != "your_api_key_here":
        logger.info("[4/4] Fetching CryptoPanic extended...")
        df_cp = fetch_cryptopanic_extended(cryptopanic_key)
        if not df_cp.empty:
            logger.info(f"  CryptoPanic: {len(df_cp)} articles")
            frames.append(df_cp)
    else:
        logger.warning("[4/4] CRYPTOPANIC_API_KEY not set — free at cryptopanic.com/developers/api/")

    if not frames:
        logger.error("No data fetched. Add at least one API key to .env")
        return pd.DataFrame()

    # Combine, deduplicate by URL
    df_all = pd.concat(frames, ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"], utc=True, errors='coerce')
    df_all = df_all.dropna(subset=["date", "title"])
    df_all = df_all[df_all["title"].str.strip().ne("")]
    df_all = df_all.drop_duplicates(subset=["url", "title"])
    df_all = df_all.sort_values("date")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_all.to_parquet(OUT_PATH, engine='pyarrow')

    logger.info(f"\n{'='*55}")
    logger.info(f"EXTENDED NEWS COLLECTION COMPLETE")
    logger.info(f"{'='*55}")
    logger.info(f"Total articles  : {len(df_all):,}")
    logger.info(f"Date range      : {df_all['date'].min().date()} -> {df_all['date'].max().date()}")
    logger.info(f"By source:")
    for src, grp in df_all.groupby("source"):
        logger.info(f"  {src:<20} {len(grp):>6,} articles")
    logger.info(f"By asset:")
    for asset, grp in df_all.groupby("asset"):
        logger.info(f"  {asset:<20} {len(grp):>6,} articles")
    logger.info(f"Saved -> {OUT_PATH}")

    return df_all


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2019-01-01", help="Start date for historical fetch")
    args = parser.parse_args()
    run(start_date=args.start)
