import os
import time
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
from utils.logger import logger
from utils.cache import is_cached

load_dotenv()

SERIES = {
    'VIXCLS': 'vix', 'VXNCLS': 'vxn', 'OVXCLS': 'ovx', 'GVZCLS': 'gvz',
    'BAMLH0A0HYM2': 'hy_spread', 'BAMLC0A0CM': 'ig_spread', 'TEDRATE': 'ted_spread',
    'T10Y2Y': 't10y2y', 'T10Y3M': 't10y3m', 'DPCREDIT': 'discount_window',
    'DCOILWTICO': 'wti', 'GOLDAMGBD228NLBM': 'gold', 'DTWEXBGS': 'dxy',
    'DFF': 'fed_funds', 'CPIAUCSL': 'cpi', 'UNRATE': 'unrate', 'USREC': 'recession',
    'M2SL': 'm2', 'INDIRLTLT01STM': 'india_rate', 'INDCPIALLMINMEI': 'india_cpi'
}

def fetch_fred():
    out_path = "data/raw/macro/fred_series.parquet"
    if is_cached(out_path): return
    logger.info("Fetching FRED macro data...")
    
    api_key = os.getenv("FRED_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        logger.warning("FRED_API_KEY not valid. Skipping.")
        pd.DataFrame().to_parquet(out_path, engine='pyarrow')
        return
        
    try:
        fred = Fred(api_key=api_key)
        dfs = []
        for s_id, s_name in SERIES.items():
            try:
                series = fred.get_series(s_id, observation_start='2015-01-01')
                df = pd.DataFrame(series, columns=[s_id])
                dfs.append(df)
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Failed to fetch {s_id}: {e}")
                
        if dfs:
            master_df = pd.concat(dfs, axis=1)
            master_df.index.name = 'date'
            master_df.index = pd.to_datetime(master_df.index, utc=True)
            master_df = master_df.ffill() # Forward fill monthly/weekly releases to daily frequency
            master_df.to_parquet(out_path, engine='pyarrow')
            logger.info(f"Saved FRED data to {out_path}")
        else:
            pd.DataFrame().to_parquet(out_path, engine='pyarrow')
    except Exception as e:
        logger.error(f"FRED init error: {e}")
        pd.DataFrame().to_parquet(out_path, engine='pyarrow')

def run():
    fetch_fred()

if __name__ == "__main__":
    run()
