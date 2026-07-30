"""
Volatility Timing Backtest — Economic Significance Module
===========================================================
Paper Step 4: Compare Sharpe ratios for vol-targeting strategies
using forecasts from HAR, HAR-S, and Neural-HAR.

Strategy:
    w_t = sigma* / sigma_hat_{t+1}          (standard vol-targeting)
    w_t = f * mu_hat / sigma_hat_{t+1}^2   (Kelly-adjusted, f=0.25)

Includes:
  - 5bps slippage on position changes
  - Max leverage cap (5x)
  - Annualization: 365 for crypto, 252 for TradFi

Usage:
    from backtest.vol_targeting import run_vol_timing_backtest
    results = run_vol_timing_backtest(returns, forecasts_dict, asset='btc')
"""

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Asset-specific constants
ANNUALIZATION = {'btc': 365, 'eth': 365, 'spx': 252, 'nifty': 252}


def calculate_position_size(predicted_log_rv: float,
                              target_vol: float = 0.02,
                              kelly_fraction: float = 0.25,
                              mu_hat: float = None,
                              max_leverage: float = 5.0) -> float:
    """
    Compute vol-targeted position weight for one period.

    Parameters
    ----------
    predicted_log_rv : log(RV_hat_{t+1}) — model output
    target_vol       : target daily volatility (default 2%)
    kelly_fraction   : fraction of Kelly bet (0.25 = quarter-Kelly)
    mu_hat           : expected return estimate (optional; if None, uses vol-targeting only)
    max_leverage     : cap on abs position size

    Returns
    -------
    float : position weight (positive = long)
    """
    sigma_hat = np.sqrt(max(np.exp(predicted_log_rv), 1e-10))

    if mu_hat is not None:
        # Kelly-adjusted: w = f * mu / sigma^2
        w = kelly_fraction * mu_hat / (sigma_hat ** 2)
    else:
        # Standard vol-targeting: w = target_vol / sigma_hat
        w = target_vol / sigma_hat

    return float(np.clip(w, -max_leverage, max_leverage))


def _compute_strategy_pnl(returns: np.ndarray,
                           log_rv_forecasts: np.ndarray,
                           asset: str = 'btc',
                           target_vol: float = 0.02,
                           kelly_fraction: float = 0.25,
                           slippage_bps: float = 5.0,
                           max_leverage: float = 5.0) -> dict:
    """
    Compute PnL series and performance stats for a single forecast series.
    """
    ann_factor = ANNUALIZATION.get(asset, 252)
    n = min(len(returns), len(log_rv_forecasts))
    returns   = np.asarray(returns[:n], dtype=float)
    forecasts = np.asarray(log_rv_forecasts[:n], dtype=float)

    # Position weights (using yesterday's forecast for today's position)
    weights = np.array([
        calculate_position_size(fc, target_vol, kelly_fraction, max_leverage=max_leverage)
        for fc in forecasts
    ])

    # Slippage: proportional to absolute position change
    weight_changes = np.abs(np.diff(weights, prepend=weights[0]))
    slippage_cost  = weight_changes * (slippage_bps / 10_000)

    # Daily PnL
    pnl = weights * returns - slippage_cost

    # ── Performance metrics ──
    cum_pnl    = np.cumsum(pnl)
    ann_ret    = pnl.mean() * ann_factor
    ann_vol    = pnl.std() * np.sqrt(ann_factor)
    sharpe     = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0
    max_dd     = float(np.min(cum_pnl - np.maximum.accumulate(cum_pnl)))
    calmar     = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    hit_rate   = float(np.mean(pnl > 0))
    turnover   = float(np.mean(np.abs(np.diff(weights, prepend=weights[0]))))

    return {
        'pnl':       pnl,
        'cum_pnl':   cum_pnl,
        'weights':   weights,
        'Sharpe':    round(sharpe, 4),
        'Ann_Return': round(ann_ret * 100, 2),   # in %
        'Ann_Vol':   round(ann_vol * 100, 2),    # in %
        'MaxDD':     round(max_dd * 100, 2),     # in %
        'Calmar':    round(calmar, 4) if not np.isnan(calmar) else None,
        'HitRate':   round(hit_rate, 4),
        'Turnover':  round(turnover, 4),
        'N':         n,
    }


def run_vol_timing_backtest(returns: np.ndarray,
                             forecasts_dict: dict,
                             asset: str = 'btc',
                             target_vol: float = 0.02,
                             kelly_fraction: float = 0.25,
                             slippage_bps: float = 5.0,
                             max_leverage: float = 5.0) -> pd.DataFrame:
    """
    Run vol-targeting backtest for all models in forecasts_dict.

    Parameters
    ----------
    returns        : 1-D array of realized daily log-returns (OOS period)
    forecasts_dict : {"HAR": array_of_log_rv_forecasts, "HAR-S": ..., ...}
    asset          : 'btc', 'spx', or 'nifty' (sets annualization factor)
    target_vol     : target daily vol (default 2%)
    kelly_fraction : Kelly multiplier (default 0.25 = quarter-Kelly)
    slippage_bps   : one-way transaction cost in basis points (default 5bps)
    max_leverage   : max position size cap (default 5x)

    Returns
    -------
    DataFrame: one row per model, columns = Sharpe, Ann_Return, Ann_Vol, MaxDD, etc.
               (Paper Table 5)
    """
    logger.info(f"\nRunning Vol-Timing Backtest — Asset: {asset.upper()}")
    logger.info(f"  Target vol: {target_vol*100:.1f}%  |  Kelly f: {kelly_fraction}  "
                f"|  Slippage: {slippage_bps}bps  |  Max lev: {max_leverage}x")

    results = {}
    pnl_series = {}

    for name, forecasts in forecasts_dict.items():
        logger.info(f"  Strategy: {name}")
        stats = _compute_strategy_pnl(
            returns, forecasts, asset,
            target_vol, kelly_fraction, slippage_bps, max_leverage
        )
        pnl_series[name] = stats.pop('pnl')
        _      = stats.pop('cum_pnl')
        _      = stats.pop('weights')
        results[name] = stats

    results_df = pd.DataFrame(results).T
    results_df.index.name = 'Model'

    _print_backtest_table(results_df, asset)
    return results_df, pnl_series


def _print_backtest_table(results_df: pd.DataFrame, asset: str):
    """Print Paper Table 5."""
    print("\n" + "="*70)
    print(f"  PAPER TABLE 5 — Economic Significance (Vol-Timing Backtest)")
    print(f"  Asset: {asset.upper()}  |  5bps slippage  |  Quarter-Kelly")
    print("="*70)
    display_cols = ['Sharpe', 'Ann_Return', 'Ann_Vol', 'MaxDD', 'HitRate', 'N']
    print(results_df[[c for c in display_cols if c in results_df.columns]].to_string())
    print("="*70)
    best = results_df['Sharpe'].idxmax()
    print(f"  Best strategy: {best} (Sharpe = {results_df.loc[best, 'Sharpe']})\n")


def load_and_run(asset: str = 'btc',
                  processed_dir: str = 'data/processed',
                  forecast_dir: str = 'data/processed') -> pd.DataFrame:
    """
    Convenience wrapper: load saved forecasts and returns, run backtest.
    Call this from notebooks after training is complete.
    """
    import os

    matrix_path = os.path.join(processed_dir, f'{asset}_feature_matrix.parquet')
    df = pd.read_parquet(matrix_path).dropna(subset=['Target_RV', 'log_RV'])

    # Load actual returns from OHLCV
    ohlcv_paths = {
        'btc':   'data_collection/data/raw/ohlcv/binance_btc_1d.parquet',
        'spx':   'data_collection/data/raw/ohlcv/spx_1d.parquet',
        'nifty': 'data_collection/data/raw/ohlcv/nifty_1d.parquet'
    }
    if asset in ohlcv_paths and os.path.exists(ohlcv_paths[asset]):
        ohlcv = pd.read_parquet(ohlcv_paths[asset])
        ohlcv['ret'] = np.log(ohlcv['close'] / ohlcv['close'].shift(1))
        
        # Ensure timezone matches df
        if ohlcv.index.tz is None and df.index.tz is not None:
            ohlcv.index = ohlcv.index.tz_localize('UTC')
            
        df = df.join(ohlcv[['ret']], how='left').fillna({'ret': 0})
        returns = df['ret'].values
    else:
        logger.warning(f"OHLCV not found for {asset}, falling back to fake returns for testing.")
        returns = np.random.normal(0, 0.02, len(df))

    # Load saved OOS forecasts
    forecasts_dict = {}
    for model in ['har', 'hars', 'neural_har']:
        fpath = os.path.join(forecast_dir, f'forecasts_{model}_{asset}.npy')
        if os.path.exists(fpath):
            arr = np.load(fpath)
            label_map = {'har': 'HAR', 'hars': 'HAR-S', 'neural_har': 'Neural-HAR'}
            forecasts_dict[label_map[model]] = arr

    if not forecasts_dict:
        logger.error("No forecast files found. Train models first.")
        return pd.DataFrame()


    results_df, _ = run_vol_timing_backtest(returns, forecasts_dict, asset=asset)
    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset', default='btc', choices=['btc', 'spx', 'nifty'])
    args = parser.parse_args()
    load_and_run(args.asset)
