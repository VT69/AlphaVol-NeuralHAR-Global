import requests
import pandas as pd
from datetime import datetime, timezone
from utils.logger import logger
from utils.cache import is_cached

BASE_URL = "https://www.deribit.com/api/v2/public"

def fetch_dvol(out_path):
    if is_cached(out_path): return
    logger.info("Fetching BTC DVOL from Deribit...")
    start_ts = int(pd.to_datetime("2021-01-01", utc=True).timestamp() * 1000)
    end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    
    params = {"currency": "BTC", "resolution": "1D", "start_timestamp": start_ts, "end_timestamp": end_ts}
    resp = requests.get(f"{BASE_URL}/get_volatility_index_data", params=params)
    
    if resp.status_code == 200:
        data = resp.json().get('result', {})
        df = pd.DataFrame(data.get('data', []), columns=['timestamp', 'open', 'high', 'low', 'close'])
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.rename(columns={'close': 'dvol_close'}, inplace=True)
            df.to_parquet(out_path, engine='pyarrow')
            logger.info(f"Saved DVOL data to {out_path}")
            return
            
    logger.warning("No DVOL data retrieved.")
    pd.DataFrame().to_parquet(out_path, engine='pyarrow')

def fetch_options_chain(out_path):
    if is_cached(out_path): return
    logger.info("Fetching BTC Options Chain from Deribit...")
    
    summary_resp = requests.get(f"{BASE_URL}/get_book_summary_by_currency", params={"currency": "BTC", "kind": "option"})
    if summary_resp.status_code == 200:
        summaries = summary_resp.json().get('result', [])
        records = []
        for s in summaries:
            records.append({
                'instrument_name': s.get('instrument_name'),
                'mark_iv': s.get('mark_iv'),
                'mark_price': s.get('mark_price'),
                'underlying_price': s.get('estimated_delivery_price'),
            })
            
        df = pd.DataFrame(records)
        df['timestamp'] = pd.Timestamp.utcnow()
        df.to_parquet(out_path, engine='pyarrow')
        logger.info(f"Saved Options Chain snapshot to {out_path}")
    else:
        pd.DataFrame().to_parquet(out_path, engine='pyarrow')

def run():
    fetch_dvol("data/raw/options/btc_deribit_dvol.parquet")
    fetch_options_chain("data/raw/options/btc_deribit_options_chain.parquet")

if __name__ == "__main__":
    run()
