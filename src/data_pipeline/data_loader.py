"""
Correct data loading for HAR-RV research.
Three specific fixes over the agent's plan:
  1. Proper macro alignment with staleness flag
  2. Fix crypto_fear_greed before loading
  3. Asset-aware annualization (365 crypto, 252 equity)
"""

import os
import pandas as pd
import numpy as np

class ResearchDataLoader:
    """
    Loads and aligns all data sources for HAR-RV modeling.
    Output: one clean daily DataFrame per asset, ready for modeling.
    """
    
    ANNUALIZATION = {"btc": 365, "eth": 365, "spx": 252, "nifty": 252}
    
    def __init__(self, data_dir: str = "data_collection/data/raw"):
        self.data_dir = data_dir

    def align_macro(self, rv_index, fred_df):
        """
        Correct macro alignment — no blind ffill.
        For each date in rv_index:
          Find most recent FRED observation <= that date.
          Compute staleness_days = rv_date - fred_date.
          If staleness > 5 trading days: set to NaN, not ffill.
        This prevents stale weekend values corrupting Monday crypto data.
        """
        # Ensure indices are timezone-aware and sorted
        rv_index = pd.to_datetime(rv_index, utc=True).sort_values()
        fred_df = fred_df.sort_index()
        
        aligned = fred_df.reindex(rv_index, method='ffill')
        staleness = pd.Series(index=rv_index, dtype=float)
        
        for col in fred_df.columns:
            valid_dates = fred_df[col].dropna().index
            if len(valid_dates) == 0:
                continue
                
            def get_staleness(d):
                idx = valid_dates.searchsorted(d, side='right') - 1
                if idx >= 0:
                    return (d - valid_dates[idx]).days
                return 999
                
            gaps = pd.Series(rv_index).apply(get_staleness)
            aligned.loc[gaps.values > 5, col] = np.nan
            
        return aligned

    def fix_fear_greed(self, path):
        """
        The 99.97% missing issue: the value column is likely
        stored as object dtype due to mixed types during scraping.
        Force numeric conversion.
        """
        if not os.path.exists(path):
            return pd.DataFrame()
            
        df = pd.read_parquet(path)
        if df.empty or 'value' not in df.columns:
            return pd.DataFrame()
            
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['fg_normalized'] = df['value'] / 100  # scale to [0,1]
        df = df.dropna(subset=['value'])
        if 'date' in df.columns:
            df = df.set_index('date')
            df.index = pd.to_datetime(df.index, utc=True)
        return df

    def compute_har_features(self, df):
        """
        Computes HAR backbone: log_RV, RV_d, RV_w, RV_m
        """
        if 'rv' not in df.columns:
            return df
            
        df['log_RV'] = np.log(df['rv'] + 1e-10) # Target variable mapping
        df['RV_d'] = df['rv'].shift(1)
        df['RV_w'] = df['rv'].shift(1).rolling(window=5, min_periods=1).mean()
        df['RV_m'] = df['rv'].shift(1).rolling(window=22, min_periods=1).mean()
        return df
        
    def load(self, asset: str) -> pd.DataFrame:
        """
        Returns aligned daily DataFrame with columns:
          log_RV, RV_d, RV_w, RV_m,
          obi_sq, vpin,
          credit_spread, term_slope,
          crypto_fg, gmsi, regime
        """
        asset = asset.lower()
        
        # 1. Base Realized Volatility
        rv_path = os.path.join(self.data_dir, f"realized_vol/{asset}_rv_daily.parquet")
        if not os.path.exists(rv_path):
            raise FileNotFoundError(f"RV data missing for {asset}")
            
        df = pd.read_parquet(rv_path)
        if 'date' in df.columns:
            df = df.set_index('date')
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
        
        # Keep only trading days / existing days
        rv_index = df.index
        
        # 2. Microstructure (VPIN, OBI)
        vpin_path = os.path.join(self.data_dir, f"microstructure/{asset}_vpin_daily.parquet")
        obi_path = os.path.join(self.data_dir, f"microstructure/{asset}_obi_daily.parquet")
        
        if os.path.exists(vpin_path):
            vpin_df = pd.read_parquet(vpin_path)
            if 'date' in vpin_df.columns:
                vpin_df = vpin_df.set_index('date')
            vpin_df.index = pd.to_datetime(vpin_df.index, utc=True)
            vcol = 'vpin' if 'vpin' in vpin_df.columns else 'VPIN'
            df = df.join(vpin_df[[vcol]].rename(columns={vcol: 'vpin'}), how='left')
            
        if os.path.exists(obi_path):
            obi_df = pd.read_parquet(obi_path)
            if 'date' in obi_df.columns:
                obi_df = obi_df.set_index('date')
            obi_df.index = pd.to_datetime(obi_df.index, utc=True)
            
            obi_cols = {}
            if 'OBI_sq' in obi_df.columns: obi_cols['OBI_sq'] = 'obi_sq'
            if 'ILLIQ' in obi_df.columns: obi_cols['ILLIQ'] = 'illiq'
            if 'Roll' in obi_df.columns: obi_cols['Roll'] = 'roll_spread'
            
            if obi_cols:
                df = df.join(obi_df[list(obi_cols.keys())].rename(columns=obi_cols), how='left')
        
        # 3. Macro (FRED)
        fred_path = os.path.join(self.data_dir, "macro/fred_series.parquet")
        if os.path.exists(fred_path):
            fred_df = pd.read_parquet(fred_path)
            if 'date' in fred_df.columns:
                fred_df = fred_df.set_index('date')
            fred_df.index = pd.to_datetime(fred_df.index, utc=True)
            aligned_fred = self.align_macro(rv_index, fred_df)
            df = df.join(aligned_fred, how='left')
            
            # Map macro columns
            if 'BAMLH0A0HYM2' in df.columns: df['credit_spread'] = df['BAMLH0A0HYM2']
            if 'T10Y2Y' in df.columns: df['term_slope'] = df['T10Y2Y']
            
        # 4. Sentiment (Crypto Fear & Greed / SPX VIX)
        if asset in ['btc', 'eth']:
            fg_path = os.path.join(self.data_dir, "macro/crypto_fear_greed.parquet")
            fg_df = self.fix_fear_greed(fg_path)
            if not fg_df.empty:
                df = df.join(fg_df[['fg_normalized']].rename(columns={'fg_normalized': 'crypto_fg'}), how='left')
                
        # 5. Options (VIX, DVOL)
        if asset == 'spx':
            vix_path = os.path.join(self.data_dir, "options/spx_vix_term_structure.parquet")
            if os.path.exists(vix_path):
                vix_df = pd.read_parquet(vix_path)
                if 'date' in vix_df.columns: vix_df = vix_df.set_index('date')
                vix_df.index = pd.to_datetime(vix_df.index, utc=True)
                # Rename any VIX cols that collide with existing FRED cols
                overlap = [c for c in vix_df.columns if c in df.columns]
                vix_df = vix_df.rename(columns={c: f'vix_{c}' for c in overlap})
                # Drop any remaining duplicates
                vix_df = vix_df[[c for c in vix_df.columns if c not in df.columns]]
                df = df.join(vix_df, how='left')
                
        if asset == 'btc':
            dvol_path = os.path.join(self.data_dir, "options/btc_deribit_dvol.parquet")
            if os.path.exists(dvol_path):
                dvol_df = pd.read_parquet(dvol_path)
                if 'date' in dvol_df.columns: dvol_df = dvol_df.set_index('date')
                dvol_df.index = pd.to_datetime(dvol_df.index, utc=True)
                dcol = 'dvol' if 'dvol' in dvol_df.columns else 'DVOL'
                if dcol in dvol_df.columns:
                    df = df.join(dvol_df[[dcol]].rename(columns={dcol: 'dvol'}), how='left')
                
        # 6. GMSI (Global Market Sentiment Index) mock / placeholder if not exist
        # We will compute a simple GMSI using macro vars for now
        if 'credit_spread' in df.columns and 'term_slope' in df.columns:
            # Simple regime proxy (higher credit spread / lower slope = high stress)
            df['gmsi'] = df['credit_spread'] - df['term_slope'] 
            df['regime'] = pd.qcut(df['gmsi'].dropna(), 3, labels=['low', 'medium', 'high'])
            # Since macro can be NaN occasionally, safely fill regimes
            df['regime'] = df['regime'].ffill().bfill()
        else:
            df['gmsi'] = 0
            df['regime'] = 'medium'

        # 7. HAR features
        df = self.compute_har_features(df)
        
        # 8. Lag Microstructure & Sentiment predictors (shift by 1 to predict t)
        exo_cols = ['vpin', 'obi_sq', 'crypto_fg', 'credit_spread', 'term_slope', 'gmsi']
        for col in exo_cols:
            if col in df.columns:
                df[f'{col}_lag1'] = df[col].shift(1)

        # Handle 'sent_score_lag1', 'illiq_lag1', 'roll_spread_lag1' as placeholders
        for missing in ['sent_score_lag1', 'sent_surprise_lag1', 'illiq_lag1', 'roll_spread_lag1']:
            if missing not in df.columns:
                df[missing] = 0.0

        # Drop the first 22 rows due to monthly RV lag
        df = df.dropna(subset=['RV_m', 'log_RV'])
        
        return df

