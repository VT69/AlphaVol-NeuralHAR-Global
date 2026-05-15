import os
import requests
import pandas as pd
from utils.logger import logger
from utils.cache import is_cached

def fetch_cnn_fng(out_path):
    if is_cached(out_path): return
    logger.info("Fetching CNN Fear & Greed...")
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            data = resp.json().get('fear_and_greed_historical', {})
            df = pd.DataFrame(data.get('data', []))
            if not df.empty:
                df['date'] = pd.to_datetime(df['x'], unit='ms', utc=True)
                df.rename(columns={'y': 'value', 'rating': 'value_classification'}, inplace=True)
                df.drop(columns=['x'], inplace=True)
                df.to_parquet(out_path, engine='pyarrow')
                return
    except Exception as e:
        logger.error(f"CNN scrape failed: {e}")
    pd.DataFrame().to_parquet(out_path, engine='pyarrow')

def fetch_crypto_fng(out_path):
    if is_cached(out_path): return
    logger.info("Fetching Crypto Fear & Greed...")
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=3000&format=json")
        if resp.status_code == 200:
            df = pd.DataFrame(resp.json().get('data', []))
            if not df.empty:
                df['date'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
                df['value'] = df['value'].astype(float)
                df.to_parquet(out_path, engine='pyarrow')
                return
    except Exception as e:
        logger.error(f"Crypto F&G error: {e}")
    pd.DataFrame().to_parquet(out_path, engine='pyarrow')

def fetch_vix_term_structure(out_path):
    if is_cached(out_path): return
    logger.info("Fetching VIX term structure (via YFinance proxy)...")
    try:
        import yfinance as yf
        df_yf = yf.download(['^VIX', '^VIX9D', '^VIX3M'], period='max', interval='1d', progress=False)
        
        if df_yf.empty:
            raise ValueError("YFinance returned empty dataframe")
            
        df = pd.DataFrame(index=df_yf.index)
        if isinstance(df_yf.columns, pd.MultiIndex):
            df['VIX'] = df_yf['Close']['^VIX']
            df['VIX9D'] = df_yf['Close']['^VIX9D']
            df['VX3M'] = df_yf['Close']['^VIX3M']
        else:
            df['VIX'] = df_yf['^VIX'] if '^VIX' in df_yf.columns else np.nan
        
        df.dropna(inplace=True)
        df['term_slope'] = df['VX3M'] - df['VIX9D']
        df['contango'] = (df['VX3M'] > df['VIX']).astype(int)
        
        df.index = pd.to_datetime(df.index, utc=True)
        df.to_parquet(out_path, engine='pyarrow')
    except Exception as e:
        logger.error(f"VIX term structure error: {e}")
        pd.DataFrame().to_parquet(out_path, engine='pyarrow')

def run():
    fetch_cnn_fng("data/raw/macro/fear_greed_index.parquet")
    fetch_crypto_fng("data/raw/macro/crypto_fear_greed.parquet")
    fetch_vix_term_structure("data/raw/options/spx_vix_term_structure.parquet")

if __name__ == "__main__":
    run()
