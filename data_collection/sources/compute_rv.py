import os
import numpy as np
import pandas as pd
from utils.logger import logger

def compute_crypto_rv(symbol, in_path, out_path):
    logger.info(f"Computing RV for {symbol}...")
    if not os.path.exists(in_path):
        return
        
    df = pd.read_parquet(in_path)
    if df.empty:
        pd.DataFrame().to_parquet(out_path, engine='pyarrow')
        return

    df['date'] = df.index.date
    df['r'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    
    daily = []
    for date, group in df.groupby('date'):
        r = group['r'].values
        n = len(r)
        rv = np.sum(r**2)
        bv = (np.pi / 2) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])) if n > 1 else 0
        j_t = max(rv - bv, 0)
        c_t = rv - j_t
        h, l, o, c = group['high'].max(), group['low'].min(), group['open'].iloc[0], group['close'].iloc[-1]
        pk = (np.log(h/l)**2) / (4 * np.log(2)) if l > 0 else 0
        gk = 0.5 * (np.log(h/l)**2) - (2*np.log(2)-1) * (np.log(c/o)**2) if (l > 0 and o > 0) else 0
        
        daily.append({
            'date': pd.to_datetime(date), 'RV': rv, 'BV': bv, 'J_t': j_t, 
            'C_t': c_t, 'PK': pk, 'GK': gk
        })
        
    res = pd.DataFrame(daily).set_index('date')
    res['log_RV'] = np.log(res['RV'].replace(0, np.nan))
    res['RV_d'] = res['log_RV'].shift(1)
    res['RV_w'] = res['log_RV'].shift(1).rolling(5).mean()
    res['RV_m'] = res['log_RV'].shift(1).rolling(22).mean()
    res['RV_ann'] = res['RV'] * 365
    res['vol_ann'] = np.sqrt(res['RV_ann'])
    
    res.to_parquet(out_path, engine='pyarrow')
    logger.info(f"Saved {symbol} RV to {out_path}")

def compute_tradfi_rv(symbol, in_path, out_path):
    logger.info(f"Computing tradfi RV for {symbol}...")
    if not os.path.exists(in_path): return
    df = pd.read_parquet(in_path)
    if df.empty:
        pd.DataFrame().to_parquet(out_path, engine='pyarrow')
        return

    # For TradFi, daily OHLCV is available, we compute PK and GK
    h, l, o, c = df['high'], df['low'], df['open'], df['close']
    res = pd.DataFrame(index=df.index)
    res['PK'] = (np.log(h/l)**2) / (4 * np.log(2))
    res['GK'] = 0.5 * (np.log(h/l)**2) - (2*np.log(2)-1) * (np.log(c/o)**2)
    res['log_ret'] = df['log_ret']
    
    # Simple proxy for Daily RV
    res['RV'] = res['log_ret']**2
    res['log_RV'] = np.log(res['RV'].replace(0, np.nan))
    res['RV_d'] = res['log_RV'].shift(1)
    res['RV_w'] = res['log_RV'].shift(1).rolling(5).mean()
    res['RV_m'] = res['log_RV'].shift(1).rolling(22).mean()
    res['RV_ann'] = res['RV'] * 252
    res['vol_ann'] = np.sqrt(res['RV_ann'].replace(0, np.nan))
    
    res.to_parquet(out_path, engine='pyarrow')

def run():
    compute_crypto_rv('BTC', 'data/raw/ohlcv/btc_5min.parquet', 'data/raw/realized_vol/btc_rv_daily.parquet')
    compute_crypto_rv('ETH', 'data/raw/ohlcv/eth_5min.parquet', 'data/raw/realized_vol/eth_rv_daily.parquet')
    compute_tradfi_rv('SPX', 'data/raw/ohlcv/spx_1d.parquet', 'data/raw/realized_vol/spx_rv_daily.parquet')
    compute_tradfi_rv('NIFTY', 'data/raw/ohlcv/nifty_1d.parquet', 'data/raw/realized_vol/nifty_rv_daily.parquet')

if __name__ == "__main__":
    run()
