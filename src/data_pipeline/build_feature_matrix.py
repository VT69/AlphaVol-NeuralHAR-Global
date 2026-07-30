"""
Master Feature Matrix Builder
==============================
Assembles the final aligned Parquet per asset (BTC, SPX, NIFTY)
by joining: RV lags + OBI/VPIN + FinBERT sentiment + macro regime.

Output (one file per asset):
    data/processed/{asset}_feature_matrix.parquet

Columns:
    date | log_RV | RV_d | RV_w | RV_m |
    vpin_lag1 | obi_sq_lag1 | illiq_lag1 |
    FinBERT_score | sent_surprise |
    credit_spread_lag1 | term_slope_lag1 |
    crypto_fg_lag1 | dvol_lag1 | gmsi | regime |
    Target_RV (= log_RV shifted -1, the next-day target)

Usage:
    python src/data_pipeline/build_feature_matrix.py
    python src/data_pipeline/build_feature_matrix.py --asset btc
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import logging

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data_pipeline.data_loader import ResearchDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ASSETS = ["btc", "spx", "nifty"]

# Column sets for each asset (some features only exist for crypto)
CRYPTO_ASSETS = {"btc", "eth"}
TRADFI_ASSETS = {"spx", "nifty"}


def load_sentiment_scores(asset: str, processed_dir: str) -> pd.DataFrame:
    """
    Load FinBERT daily sentiment scores.
    Falls back to zero-filled DataFrame if not yet computed.
    """
    # Asset-specific sentiment file first, then global crypto file
    candidates = [
        os.path.join(processed_dir, f"sentiment_daily_{asset}.parquet"),
        os.path.join(processed_dir, "sentiment_daily.parquet"),
    ]
    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 512:
            try:
                df = pd.read_parquet(path)
                if df.empty:
                    continue
                if 'date' in df.columns:
                    df = df.set_index('date')
                df.index = pd.to_datetime(df.index, utc=True)
                # Standardise column name
                if 'FinBERT_Sentiment' in df.columns:
                    df = df.rename(columns={'FinBERT_Sentiment': 'FinBERT_score'})
                if 'Net_Sentiment' in df.columns:
                    df = df.rename(columns={'Net_Sentiment': 'FinBERT_score'})
                if 'FinBERT_score' in df.columns:
                    logger.info(f"Loaded sentiment for {asset} from {path} ({len(df)} rows)")
                    return df[['FinBERT_score']]
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
    logger.warning(f"No sentiment file found for {asset} — filling with zeros.")
    return pd.DataFrame()


def compute_sentiment_surprise(df: pd.DataFrame,
                                 col: str = 'FinBERT_score',
                                 window: int = 5) -> pd.DataFrame:
    """
    Sentiment Surprise = raw score − rolling mean (signal vs expectation).
    This is a key feature for the HAR-S model: surprises move markets, not levels.
    """
    if col in df.columns:
        df['sent_surprise'] = df[col] - df[col].rolling(window, min_periods=1).mean().shift(1)
    else:
        df['FinBERT_score'] = 0.0
        df['sent_surprise'] = 0.0
    return df


def build_feature_matrix(asset: str, data_dir: str, processed_dir: str) -> pd.DataFrame:
    """
    Build and save the master feature matrix for one asset.
    
    Returns the assembled DataFrame (also saves to Parquet).
    """
    logger.info(f"{'='*50}")
    logger.info(f"Building feature matrix for: {asset.upper()}")
    logger.info(f"{'='*50}")

    # ── 1. Base RV + macro + microstructure (via ResearchDataLoader) ──
    loader = ResearchDataLoader(data_dir=data_dir)
    try:
        df = loader.load(asset)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error(f"Skipping {asset} — run the data pipeline first.")
        return pd.DataFrame()

    logger.info(f"Loaded base features: {df.shape}, range: {df.index.min()} → {df.index.max()}")

    # ── 2. Sentiment scores ──
    sent_df = load_sentiment_scores(asset, processed_dir)
    if not sent_df.empty:
        df = df.join(sent_df, how='left')
    else:
        df['FinBERT_score'] = 0.0

    df['FinBERT_score'] = df['FinBERT_score'].fillna(0.0)
    df = compute_sentiment_surprise(df)

    # Lag sentiment by 1 day (we use yesterday's sentiment to predict today's RV)
    df['FinBERT_score_lag1'] = df['FinBERT_score'].shift(1)
    df['sent_surprise_lag1'] = df['sent_surprise'].shift(1)

    # ── 3. Target variable: NEXT DAY's log_RV ──
    # log_RV is already computed in loader; we shift forward by 1 for the target
    if 'log_RV' in df.columns:
        df['Target_RV'] = df['log_RV'].shift(-1)
    else:
        logger.error("log_RV column missing — check ResearchDataLoader.")
        return pd.DataFrame()

    # ── 4. Select and order final columns ──
    # Core HAR backbone (always present)
    core_cols = ['log_RV', 'RV_d', 'RV_w', 'RV_m', 'Target_RV']

    # Microstructure (crypto: OBI + VPIN; TradFi: proxy OBI from klines)
    micro_cols = [c for c in ['vpin_lag1', 'obi_sq_lag1', 'illiq_lag1', 'roll_spread_lag1']
                  if c in df.columns]

    # Sentiment
    sentiment_cols = ['FinBERT_score_lag1', 'sent_surprise_lag1']

    # Macro / regime
    macro_cols = [c for c in ['credit_spread_lag1', 'term_slope_lag1',
                               'crypto_fg_lag1', 'dvol_lag1', 'gmsi', 'regime']
                  if c in df.columns]

    all_cols = core_cols + micro_cols + sentiment_cols + macro_cols
    available = [c for c in all_cols if c in df.columns]
    df_final = df[available].copy()

    # ── 5. Drop rows without valid HAR features and target ──
    required = ['RV_d', 'RV_w', 'RV_m', 'Target_RV', 'log_RV']
    before = len(df_final)
    df_final = df_final.dropna(subset=[c for c in required if c in df_final.columns])
    logger.info(f"Dropped {before - len(df_final)} rows with NaN in core HAR columns. "
                f"Final shape: {df_final.shape}")

    # ── 6. Sanity checks ──
    logger.info(f"\nFeature Summary for {asset.upper()}:")
    logger.info(f"  Date range : {df_final.index.min()} → {df_final.index.max()}")
    logger.info(f"  N rows     : {len(df_final)}")
    logger.info(f"  Columns    : {list(df_final.columns)}")
    logger.info(f"  NaN summary:\n{df_final.isnull().sum()[df_final.isnull().sum() > 0]}")

    # Check for data leakage: Target_RV should not correlate > 0.95 with RV_d
    if 'Target_RV' in df_final.columns and 'RV_d' in df_final.columns:
        corr = df_final[['Target_RV', 'RV_d']].corr().iloc[0, 1]
        if corr > 0.98:
            logger.warning(f"POTENTIAL LEAKAGE: Target_RV / RV_d correlation = {corr:.3f}")
        else:
            logger.info(f"  Leakage check: Target_RV ~ RV_d corr = {corr:.3f} ✓")

    # ── 7. Save ──
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, f"{asset}_feature_matrix.parquet")
    df_final.to_parquet(out_path, engine='pyarrow')
    logger.info(f"Saved → {out_path}\n")

    return df_final


def main():
    parser = argparse.ArgumentParser(description="Build master feature matrices for HAR modeling")
    parser.add_argument('--asset', type=str, default=None,
                        help=f'Asset to build (one of: {ASSETS}). Default: all.')
    parser.add_argument('--data_dir', type=str, default='data_collection/data/raw',
                        help='Path to raw data directory')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                        help='Output directory for feature matrices')
    args = parser.parse_args()

    assets_to_run = [args.asset.lower()] if args.asset else ASSETS

    summaries = {}
    for asset in assets_to_run:
        df = build_feature_matrix(asset, args.data_dir, args.processed_dir)
        if not df.empty:
            summaries[asset] = {
                'rows': len(df),
                'start': str(df.index.min().date()),
                'end': str(df.index.max().date()),
                'columns': len(df.columns)
            }

    # ── Final summary table ──
    if summaries:
        print("\n" + "="*60)
        print("  FEATURE MATRIX BUILD COMPLETE")
        print("="*60)
        summary_df = pd.DataFrame(summaries).T
        print(summary_df.to_string())
        print("="*60)
        print(f"\nFiles saved to: {args.processed_dir}/")
        for asset in summaries:
            print(f"  -> {asset}_feature_matrix.parquet")
    else:
        print("\nNo feature matrices built. Check data pipeline first.")


if __name__ == "__main__":
    main()
