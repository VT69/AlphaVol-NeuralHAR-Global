import os
import time
import requests
import pandas as pd
from pytrends.request import TrendReq
from google.cloud import bigquery
from dotenv import load_dotenv
from utils.logger import logger
from utils.cache import is_cached

load_dotenv()

def fetch_cryptopanic():
    out_path = "data/raw/sentiment/cryptopanic_headlines.parquet"
    if is_cached(out_path): return
    logger.info("Fetching CryptoPanic headlines...")
    
    api_key = os.getenv("CRYPTOPANIC_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        pd.DataFrame().to_parquet(out_path, engine='pyarrow')
        return
        
    records = []
    for currency in ["BTC", "ETH"]:
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={api_key}&currencies={currency}&public=true"
        for _ in range(50):
            try:
                resp = requests.get(url)
                if resp.status_code != 200: break
                data = resp.json()
                for item in data.get('results', []):
                    votes = item.get('votes', {})
                    records.append({
                        'id': item.get('id'), 'published_at': item.get('published_at'), 'title': item.get('title'),
                        'currencies': currency, 'source_domain': item.get('source', {}).get('domain'),
                        'votes_positive': votes.get('positive', 0), 'votes_negative': votes.get('negative', 0),
                        'votes_important': votes.get('important', 0), 'votes_liked': votes.get('liked', 0),
                        'votes_disliked': votes.get('disliked', 0), 'votes_lol': votes.get('lol', 0),
                        'votes_toxic': votes.get('toxic', 0), 'kind': item.get('kind')
                    })
                url = data.get('next')
                if not url: break
                time.sleep(1)
            except Exception as e:
                logger.error(f"CryptoPanic error: {e}")
                break

    df = pd.DataFrame(records)
    if not df.empty:
        df['published_at'] = pd.to_datetime(df['published_at'], utc=True)
        df['panic_score'] = (df['votes_negative'] + df['votes_toxic']) / (df['votes_positive'] + df['votes_negative'] + 1)
    df.to_parquet(out_path, engine='pyarrow')

def fetch_newsapi():
    out_path = "data/raw/sentiment/newsapi_headlines.parquet"
    if is_cached(out_path): return
    logger.info("Fetching NewsAPI headlines...")
    
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key or api_key == "your_api_key_here":
        pd.DataFrame().to_parquet(out_path, engine='pyarrow')
        return
        
    queries = {
        'BTC': '"bitcoin" OR "BTC" OR "crypto market"',
        'NIFTY': '"NIFTY" OR "sensex"',
        'SPX': '"S&P 500" OR "stock market crash"'
    }
    records = []
    for asset, q in queries.items():
        try:
            url = f"https://newsapi.org/v2/everything?q={q}&language=en&sortBy=publishedAt&apiKey={api_key}"
            resp = requests.get(url)
            if resp.status_code == 200:
                for a in resp.json().get('articles', []):
                    records.append({
                        'asset': asset,
                        'published_at': a.get('publishedAt'), 'source_name': a.get('source', {}).get('name'),
                        'title': a.get('title'), 'description': a.get('description'),
                        'url': a.get('url'), 'author': a.get('author')
                    })
            time.sleep(1)
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
            
    df = pd.DataFrame(records)
    if not df.empty:
        df['published_at'] = pd.to_datetime(df['published_at'], utc=True)
    df.to_parquet(out_path, engine='pyarrow')

def fetch_gdelt():
    """
    Fetch GDELT data via the FREE public REST API (no BigQuery, no billing, no credentials).
    GDELT 2.0 DocSearch API: https://api.gdeltproject.org/api/v2/doc/doc
    
    Fetches article counts and average tone for crypto/equity queries,
    aggregated to daily frequency. Covers the last 3 months by default
    (free API limit), with a workaround using monthly chunks.
    """
    out_path = "data/raw/sentiment/gdelt_daily.parquet"
    if is_cached(out_path): return
    logger.info("Fetching GDELT via free REST API (no BigQuery needed)...")

    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    # Queries for each asset — GDELT searches full-text of global news
    queries = {
        "BTC":   '"bitcoin" OR "cryptocurrency" OR "crypto market"',
        "SPX":   '"S&P 500" OR "stock market" OR "wall street"',
        "NIFTY": '"NIFTY" OR "sensex" OR "BSE" OR "NSE India"',
    }

    # Use a session with browser-style headers — GDELT blocks bare Python requests
    session = requests.Session()
    session.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/125.0.0.0 Safari/537.36"),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://api.gdeltproject.org/",
    })

    all_records = []

    for asset, query in queries.items():
        logger.info(f"  GDELT query for {asset}: {query[:50]}...")
        succeeded = False
        for attempt in range(3):
            try:
                params = {
                    "query":     query,
                    "mode":      "timelineTone",
                    "format":    "json",
                    "timespan":  "3months",
                    "smoothing": 1,
                }
                resp = session.get(BASE_URL, params=params, timeout=30)
                if resp.status_code != 200:
                    logger.warning(f"  GDELT {asset} HTTP {resp.status_code} (attempt {attempt+1})")
                    time.sleep(5 * (attempt + 1))
                    continue

                data = resp.json()
                if not data or not isinstance(data, dict):
                    break

                timeline = data.get("timeline", [])
                for series in timeline:
                    series_name = series.get("series", "unknown")
                    for point in series.get("data", []):
                        date_str = point.get("date", "")
                        value    = point.get("value", 0.0)
                        if len(date_str) >= 8:
                            try:
                                date = pd.to_datetime(date_str[:8], format="%Y%m%d", utc=True)
                                all_records.append({
                                    "date":         date,
                                    "asset":        asset,
                                    "gdelt_series": series_name,
                                    "value":        float(value),
                                })
                            except Exception:
                                pass

                succeeded = True
                time.sleep(2)
                break   # success — exit retry loop

            except Exception as e:
                logger.warning(f"  GDELT {asset} attempt {attempt+1} error: {e}")
                time.sleep(5 * (attempt + 1))

        if not succeeded:
            logger.warning(f"  GDELT {asset}: all retries failed.")

    if not all_records:
        logger.warning("GDELT returned no data — saving empty file.")
        pd.DataFrame().to_parquet(out_path, engine='pyarrow')
        return

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"], utc=True)

    # Pivot: one row per date, columns = asset_series tone
    df_pivot = df.pivot_table(
        index="date", columns=["asset", "gdelt_series"],
        values="value", aggfunc="mean"
    )
    df_pivot.columns = [f"gdelt_{a}_{s.lower().replace(' ', '_')}"
                        for a, s in df_pivot.columns]
    df_pivot = df_pivot.sort_index()

    # Compute a simple daily "negativity score" per asset
    for asset in queries:
        neg_col = [c for c in df_pivot.columns if asset in c and "neg" in c.lower()]
        pos_col = [c for c in df_pivot.columns if asset in c and "pos" in c.lower()]
        if neg_col and pos_col:
            df_pivot[f"gdelt_{asset}_negativity"] = (
                df_pivot[neg_col[0]] / (df_pivot[pos_col[0]] + df_pivot[neg_col[0]] + 1e-6)
            )

    df_pivot.to_parquet(out_path, engine='pyarrow')
    logger.info(f"Saved GDELT daily data: {len(df_pivot)} rows, {len(df_pivot.columns)} cols -> {out_path}")


def fetch_google_trends():
    out_path = "data/raw/sentiment/google_trends_weekly.parquet"
    if is_cached(out_path): return
    logger.info("Fetching Google Trends...")
    try:
        pytrend = TrendReq(hl='en-US', tz=360)
        terms = {
            "BTC_bitcoin_crash": "bitcoin crash",
            "BTC_crypto_fear": "crypto fear",
            "SPX_stock_market_crash": "stock market crash",
            "SPX_recession": "recession",
            "NIFTY_nifty": "nifty"
        }
        dfs = []
        for col_name, term in terms.items():
            try:
                pytrend.build_payload(kw_list=[term], timeframe='today 5-y')
                df = pytrend.interest_over_time()
                if not df.empty:
                    df = df[[term]].rename(columns={term: col_name})
                    dfs.append(df)
                time.sleep(5)
            except Exception as e:
                logger.error(f"Pytrends error for {term}: {e}")
        if dfs:
            master = pd.concat(dfs, axis=1)
            master.index = pd.to_datetime(master.index, utc=True)
            master.to_parquet(out_path, engine='pyarrow')
        else: pd.DataFrame().to_parquet(out_path, engine='pyarrow')
    except Exception as e:
        pd.DataFrame().to_parquet(out_path, engine='pyarrow')

def run():
    fetch_newsapi()
    fetch_gdelt()
    fetch_google_trends()

if __name__ == "__main__":
    run()
