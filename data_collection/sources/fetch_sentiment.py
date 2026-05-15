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
    out_path = "data/raw/sentiment/gdelt_daily.parquet"
    if is_cached(out_path): return
    logger.info("Fetching GDELT via BigQuery...")
    
    credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials or credentials == "/path/to/your/gcp_credentials.json":
        pd.DataFrame().to_parquet(out_path, engine='pyarrow')
        return
        
    try:
        from google.api_core.exceptions import GoogleAPICallError, DefaultCredentialsError
        client = bigquery.Client()
        query = """
        SELECT DATE(DATE) as date, COUNT(*) as event_count, AVG(CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) as avg_tone,
          COUNTIF(CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) < 0) / COUNT(*) as negative_share,
          COUNTIF(Themes LIKE '%CRISISLEX%' OR Themes LIKE '%ECON_BANKRUPTCY%') as crisis_count,
          COUNTIF(Themes LIKE '%CRYPTO%' OR Themes LIKE '%BITCOIN%') as crypto_count,
          COUNT(DISTINCT SourceCommonName) as source_diversity
        FROM `gdelt-bq.gdeltv2.gkg` WHERE DATE(DATE) BETWEEN '2023-01-01' AND '2023-01-31' GROUP BY date ORDER BY date
        """
        df = client.query(query).to_dataframe()
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        df.to_parquet(out_path, engine='pyarrow')
    except Exception as e:
        if "billing" in str(e).lower() or "forbidden" in str(e).lower() or "credentials" in str(e).lower() or "403" in str(e):
            logger.error("🚨 GDELT BIGQUERY FAILED: Please ensure Billing is enabled on your Google Cloud Project! 🚨")
        else:
            logger.error(f"GDELT error: {e}")
        pd.DataFrame().to_parquet(out_path, engine='pyarrow')

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
