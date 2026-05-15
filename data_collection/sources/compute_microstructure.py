import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from utils.logger import logger

def compute_obi_and_others(klines_path, out_path):
    logger.info(f"Computing OBI from {klines_path}...")
    if not os.path.exists(klines_path): return
    df = pd.read_parquet(klines_path)
    if df.empty:
        pd.DataFrame().to_parquet(out_path, engine='pyarrow')
        return
        
    df['date'] = df.index.date
    df['buy_vol_proxy'] = df['volume'] * (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    df['sell_vol_proxy'] = df['volume'] - df['buy_vol_proxy']
    df['OBI'] = (df['buy_vol_proxy'] - df['sell_vol_proxy']) / (df['buy_vol_proxy'] + df['sell_vol_proxy'] + 1e-10)
    
    df['r'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    df['abs_r_vol'] = np.abs(df['r']) / (df['volume'] + 1e-10)
    df['r_shift'] = df['r'].shift(1)
    
    daily = []
    for date, group in df.groupby('date'):
        obi = group['OBI'].values
        illiq = np.mean(group['abs_r_vol']) * 1e6
        cov_term = np.cov(group['r'][1:], group['r_shift'][1:])[0,1] if len(group) > 2 else 0
        roll = 2 * np.sqrt(max(-cov_term, 0))
        
        n = len(group)
        rv = np.sum(group['r']**2)
        rs = (np.sqrt(n) * np.sum(group['r']**3)) / (rv**(1.5) + 1e-10)
        rk = (n * np.sum(group['r']**4)) / (rv**2 + 1e-10)
        
        daily.append({
            'date': pd.to_datetime(date), 'OBI_mean': np.mean(obi), 'OBI_sq': np.mean(obi**2),
            'OBI_std': np.std(obi), 'OBI_abs': np.mean(np.abs(obi)), 'ILLIQ': illiq, 'Roll': roll, 'RS': rs, 'RK': rk
        })
        
    pd.DataFrame(daily).set_index('date').to_parquet(out_path, engine='pyarrow')
    logger.info(f"Saved OBI daily features to {out_path}")

def compute_vpin(klines_path, out_path):
    logger.info("Computing VPIN (using 5min proxy)...")
    if not os.path.exists(klines_path): return
    df = pd.read_parquet(klines_path)
    if df.empty:
        pd.DataFrame().to_parquet(out_path, engine='pyarrow')
        return
        
    df = df.reset_index()
    if 'index' in df.columns: df.rename(columns={'index': 'timestamp'}, inplace=True)
    if 'date' in df.columns and 'timestamp' not in df.columns: df.rename(columns={'date': 'timestamp'}, inplace=True)
    if 'open_time' in df.columns and 'timestamp' not in df.columns: df.rename(columns={'open_time': 'timestamp'}, inplace=True)
    
    df.sort_values('timestamp', inplace=True)
    daily_avg_vol = df['volume'].sum() / max((df['timestamp'].max() - df['timestamp'].min()).days, 1)
    V_star = max(daily_avg_vol / 50, 1)
    n_buckets = 50
    
    df['cum_vol'] = df['volume'].cumsum()
    df['bucket'] = (df['cum_vol'] // V_star).astype(int)
    
    buckets = df.groupby('bucket').agg(dP=('close', lambda x: x.iloc[-1] - x.iloc[0]), V_total=('volume', 'sum'), timestamp=('timestamp', 'last'))
    buckets['sigma_dP'] = buckets['dP'].rolling(250, min_periods=1).std().replace(0, 1e-10)
    buckets['buy_ratio'] = norm.cdf(buckets['dP'] / buckets['sigma_dP'])
    buckets['V_buy'] = buckets['V_total'] * buckets['buy_ratio']
    buckets['V_sell'] = buckets['V_total'] - buckets['V_buy']
    buckets['VPIN'] = buckets.apply(lambda r: np.abs(r['V_buy'] - r['V_sell']), axis=1).rolling(n_buckets).sum() / (n_buckets * V_star)
    
    buckets.set_index('timestamp', inplace=True)
    buckets['VPIN'].resample('D').last().to_frame().to_parquet(out_path, engine='pyarrow')
    logger.info(f"Saved VPIN daily to {out_path}")

def run():
    compute_obi_and_others("data/raw/ohlcv/btc_5min.parquet", "data/raw/microstructure/btc_obi_daily.parquet")
    compute_vpin("data/raw/ohlcv/btc_5min.parquet", "data/raw/microstructure/btc_vpin_daily.parquet")

if __name__ == "__main__":
    run()
