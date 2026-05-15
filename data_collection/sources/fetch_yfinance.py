import numpy as np
import pandas as pd
import yfinance as yf
from utils.logger import logger
from utils.cache import is_cached

TICKERS = {
    '^GSPC': 'spx', '^NSEI': 'nifty', '^VIX': 'vix', 'DX-Y.NYB': 'dxy',
    '^VXN': 'vxn', 'GC=F': 'gold', '^TNX': 'tnx10y', '^IRX': 'irx3m'
}

def fetch_yfinance():
    start_date = "2015-01-01"
    for ticker, name in TICKERS.items():
        out_path = f"data/raw/ohlcv/{name}_1d.parquet"
        if is_cached(out_path): continue
            
        logger.info(f"Fetching {ticker} from yfinance...")
        try:
            df = yf.download(ticker, start=start_date, progress=False)
            if df.empty:
                pd.DataFrame().to_parquet(out_path, engine='pyarrow')
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            if 'close' in df.columns:
                df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            
            df.index = pd.to_datetime(df.index, utc=True)
            df.to_parquet(out_path, engine='pyarrow')
            logger.info(f"Saved {name} to {out_path}")
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            pd.DataFrame().to_parquet(out_path, engine='pyarrow')

def run():
    fetch_yfinance()

if __name__ == "__main__":
    run()
