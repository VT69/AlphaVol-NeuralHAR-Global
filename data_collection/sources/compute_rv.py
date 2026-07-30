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
    """
    Compute daily realized volatility proxy for TradFi assets (SPX, NIFTY).

    Primary RV estimator: Garman-Klass (1980) — optimal for daily OHLCV.
    GK = 0.5*(log(H/L))^2 - (2*ln2 - 1)*(log(C/O))^2

    This is ~7x more efficient than squared daily returns as an RV proxy
    (Garman & Klass 1980), which is important for paper Table 1 statistics.

    Also kept: Parkinson (PK), close-to-close (RV_cc).
    All stored as lower-case 'rv' for consistency with data_loader.
    """
    logger.info(f"Computing tradfi RV (Garman-Klass) for {symbol}...")
    if not os.path.exists(in_path):
        return
    df = pd.read_parquet(in_path)
    if df.empty:
        pd.DataFrame().to_parquet(out_path, engine='pyarrow')
        return

    # Ensure UTC index
    df.index = pd.to_datetime(df.index, utc=True)

    h, l, o, c = df['high'], df['low'], df['open'], df['close']

    # Garman-Klass (primary RV proxy for daily OHLCV)
    gk  = (0.5 * (np.log(h / l) ** 2)
            - (2 * np.log(2) - 1) * (np.log(c / o) ** 2)).clip(lower=0)

    # Parkinson (high-low range)
    pk  = ((np.log(h / l) ** 2) / (4 * np.log(2))).clip(lower=0)

    # Close-to-close (squared log-return, weakest estimator)
    log_ret = df.get('log_ret', np.log(c / c.shift(1)).fillna(0))
    rv_cc   = log_ret ** 2

    res = pd.DataFrame({
        'rv':      gk,          # ← primary RV (lowercase, matches data_loader)
        'GK':      gk,
        'PK':      pk,
        'RV_cc':   rv_cc,
        'log_ret': log_ret,
    }, index=df.index)

    # log(RV) and HAR lags
    rv_safe      = res['rv'].replace(0, np.nan).clip(lower=1e-12)
    res['log_RV'] = np.log(rv_safe)
    res['RV_d']   = res['log_RV'].shift(1)
    res['RV_w']   = res['log_RV'].shift(1).rolling(5,  min_periods=1).mean()
    res['RV_m']   = res['log_RV'].shift(1).rolling(22, min_periods=1).mean()

    # Annualised vol
    ann = 252
    res['RV_ann']  = res['rv'] * ann
    res['vol_ann'] = np.sqrt(res['RV_ann'].clip(lower=0))

    res.to_parquet(out_path, engine='pyarrow')
    logger.info(f"Saved {symbol} GK-RV to {out_path} ({len(res)} rows)")

def run():
    compute_crypto_rv('BTC', 'data/raw/ohlcv/btc_5min.parquet', 'data/raw/realized_vol/btc_rv_daily.parquet')
    compute_crypto_rv('ETH', 'data/raw/ohlcv/eth_5min.parquet', 'data/raw/realized_vol/eth_rv_daily.parquet')
    compute_tradfi_rv('SPX', 'data/raw/ohlcv/spx_1d.parquet', 'data/raw/realized_vol/spx_rv_daily.parquet')
    compute_tradfi_rv('NIFTY', 'data/raw/ohlcv/nifty_1d.parquet', 'data/raw/realized_vol/nifty_rv_daily.parquet')

if __name__ == "__main__":
    run()
