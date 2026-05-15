import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from utils.logger import logger
from utils.cache import is_cached

BASE_URL = "https://api.binance.com/api/v3"

def fetch_klines(symbol, interval, out_path, start_date_str="2019-01-01"):
    if is_cached(out_path):
        return
    
    logger.info(f"Fetching {symbol} {interval} klines starting {start_date_str}...")
    start_ts = int(pd.to_datetime(start_date_str, utc=True).timestamp() * 1000)
    end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    
    all_data = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        params = {"symbol": symbol, "interval": interval, "limit": 1000, "startTime": current_ts}
        resp = requests.get(f"{BASE_URL}/klines", params=params)
        if resp.status_code != 200:
            logger.error(f"Binance API error: {resp.text}")
            break
        data = resp.json()
        if not data:
            break
            
        all_data.extend(data)
        current_ts = data[-1][0] + 1
        time.sleep(0.05)
        
    if not all_data:
        pd.DataFrame().to_parquet(out_path, engine='pyarrow')
        return
        
    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
               'quote_asset_volume', 'number_of_trades', 'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore']
    df = pd.DataFrame(all_data, columns=columns)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df.set_index('open_time', inplace=True)
    
    float_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'taker_buy_base_vol', 'taker_buy_quote_vol']
    for col in float_cols:
        df[col] = df[col].astype(float)
        
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df.drop(columns=['close_time', 'ignore'], inplace=True, errors='ignore')
    
    df.to_parquet(out_path, engine='pyarrow')
    logger.info(f"Saved {len(df)} rows to {out_path}")

def fetch_agg_trades(symbol, out_path, start_date_str="2020-01-01"):
    if is_cached(out_path):
        return
    logger.info(f"Fetching {symbol} aggTrades...")
    
    start_ts = int(pd.to_datetime(start_date_str, utc=True).timestamp() * 1000)
    end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    
    # We will limit fetching to avoid infinite loop taking days, as 4 years of tick data is TBs.
    # We fetch month by month in a realistic pipeline, here we fetch a restricted chunk for demonstration.
    # In production, loop over fromId.
    params = {"symbol": symbol, "startTime": start_ts, "endTime": min(start_ts + 3600000, end_ts), "limit": 1000}
    resp = requests.get(f"{BASE_URL}/aggTrades", params=params)
    data = resp.json() if resp.status_code == 200 else []
        
    df = pd.DataFrame(data)
    if not df.empty:
        df.rename(columns={'a': 'agg_trade_id', 'p': 'price', 'q': 'qty', 'f': 'first_trade_id', 
                           'l': 'last_trade_id', 'T': 'timestamp', 'm': 'is_buyer_maker', 'M': 'ignore'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['price'] = df['price'].astype(float)
        df['qty'] = df['qty'].astype(float)
        df.drop(columns=['ignore'], inplace=True, errors='ignore')
        
    df.to_parquet(out_path, engine='pyarrow')

def fetch_orderbook_snapshot(symbol, out_path):
    logger.info(f"Fetching {symbol} orderbook snapshot...")
    params = {"symbol": symbol, "limit": 20}
    resp = requests.get(f"{BASE_URL}/depth", params=params)
    if resp.status_code == 200:
        data = resp.json()
        bids = pd.DataFrame(data['bids'], columns=['price', 'quantity']).assign(side='bid')
        asks = pd.DataFrame(data['asks'], columns=['price', 'quantity']).assign(side='ask')
        df = pd.concat([bids, asks], ignore_index=True)
        df['price'] = df['price'].astype(float)
        df['quantity'] = df['quantity'].astype(float)
        df['timestamp'] = pd.Timestamp.utcnow()
        df.to_parquet(out_path, engine='pyarrow')
    else:
        pd.DataFrame().to_parquet(out_path, engine='pyarrow')

def run():
    fetch_klines("BTCUSDT", "5m", "data/raw/ohlcv/btc_5min.parquet")
    fetch_klines("ETHUSDT", "5m", "data/raw/ohlcv/eth_5min.parquet")
    fetch_klines("BTCUSDT", "1h", "data/raw/ohlcv/btc_1h.parquet")
    fetch_klines("BTCUSDT", "1d", "data/raw/ohlcv/btc_1d.parquet")
    fetch_klines("ETHUSDT", "1d", "data/raw/ohlcv/eth_1d.parquet")
    fetch_agg_trades("BTCUSDT", "data/raw/microstructure/btc_agg_trades.parquet", "2024-01-01")
    fetch_orderbook_snapshot("BTCUSDT", "data/raw/microstructure/btc_orderbook_snapshots.parquet")

if __name__ == "__main__":
    run()
