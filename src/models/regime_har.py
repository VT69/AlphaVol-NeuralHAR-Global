"""
Regime-Conditional HAR Model
==============================
Paper Step 3: Estimate HAR separately within Low / Medium / High GMSI regimes.

Key research question:
    Does sentiment beta (FinBERT coefficient) vary by market stress regime?
    Hypothesis: Sentiment matters MORE in low-stress (complacency) regimes,
                because in high-stress regimes, microstructure/liquidity dominates.

Usage:
    from src.models.regime_har import run_regime_analysis
    results = run_regime_analysis('data/processed/btc_feature_matrix.parquet')
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REGIMES = ['low', 'medium', 'high']


def qlike_numpy(pred_log_rv: np.ndarray, actual_log_rv: np.ndarray) -> float:
    """QLIKE loss on log-RV scale."""
    actual_var = np.exp(np.clip(actual_log_rv, -15, 15))
    pred_var   = np.exp(np.clip(pred_log_rv,   -15, 15))
    return float(np.mean(actual_var / pred_var - np.log(actual_var / pred_var) - 1))


def _build_X(df: pd.DataFrame, model_type: str = 'HAR') -> pd.DataFrame:
    """
    Build design matrix for OLS.
    model_type: 'HAR' -> only RV lags
                'HAR-S' -> RV lags + sentiment + microstructure
    """
    har_cols = ['RV_d', 'RV_w', 'RV_m']

    exo_cols_candidates = [
        'vpin_lag1', 'obi_sq_lag1', 'illiq_lag1',
        'FinBERT_score_lag1', 'sent_surprise_lag1',
        'credit_spread_lag1', 'term_slope_lag1',
    ]

    if model_type == 'HAR':
        feature_cols = har_cols
    else:  # HAR-S
        feature_cols = har_cols + [c for c in exo_cols_candidates if c in df.columns]

    X = df[feature_cols].copy()
    X = sm.add_constant(X)
    return X


def fit_har_ols(df: pd.DataFrame, model_type: str = 'HAR') -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    In-sample OLS fit on a DataFrame slice.
    Returns fitted statsmodels result.
    """
    y = df['Target_RV']
    X = _build_X(df, model_type)

    # Drop rows where any feature or target is NaN
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    return sm.OLS(y[mask], X[mask]).fit(cov_type='HAC', cov_kwds={'maxlags': 5})


def expanding_window_oos(df: pd.DataFrame, model_type: str = 'HAR',
                          min_train: int = 100) -> tuple:
    """
    Expanding-window out-of-sample forecast within a single regime slice.
    Re-estimates OLS at each step (no lookahead).

    Returns: (actuals, forecasts) as numpy arrays.
    """
    y = df['Target_RV'].values
    X = _build_X(df, model_type)

    if len(df) < min_train + 10:
        logger.warning(f"  Only {len(df)} obs — skipping expanding window, using in-sample.")
        model = fit_har_ols(df, model_type)
        preds = model.predict(X).values
        return y, preds

    forecasts = []
    actuals   = []

    for t in range(min_train, len(df)):
        X_tr, y_tr = X.iloc[:t], y[:t]
        X_te       = X.iloc[t:t+1]
        y_te       = y[t]

        mask = ~(X_tr.isnull().any(axis=1) | np.isnan(y_tr))
        if mask.sum() < 30:
            continue

        model = sm.OLS(y_tr[mask], X_tr[mask]).fit()
        X_te_filled = X_te.fillna(X_tr[mask].mean())
        pred  = model.predict(X_te_filled).values[0]
        forecasts.append(pred)
        actuals.append(y_te)

    return np.array(actuals), np.array(forecasts)


def run_regime_analysis(feature_matrix_path: str,
                         model_types: list = ('HAR', 'HAR-S'),
                         min_obs_per_regime: int = 80) -> dict:
    """
    Main entry point.
    
    Splits the feature matrix by regime (low/medium/high),
    runs both HAR and HAR-S per regime, extracts coefficients and OOS metrics.

    Returns dict:
    {
      'regime_stats': DataFrame — regime sizes and date ranges,
      'coefficients': DataFrame — beta table per regime × model (Paper Table 3),
      'oos_metrics':  DataFrame — QLIKE, MSE, MAE per regime × model,
      'models':       nested dict of fitted statsmodels objects,
      'forecasts':    nested dict of (actuals, forecasts) arrays
    }
    """
    logger.info(f"Loading feature matrix: {feature_matrix_path}")
    df = pd.read_parquet(feature_matrix_path)
    df = df.sort_index()

    if 'regime' not in df.columns:
        logger.error("'regime' column missing. Run build_feature_matrix.py first.")
        return {}

    # ── Regime Summary ──
    regime_stats = []
    for r in REGIMES:
        sub = df[df['regime'] == r]
        regime_stats.append({
            'regime': r, 'n_obs': len(sub),
            'start': str(sub.index.min().date()) if len(sub) else 'N/A',
            'end':   str(sub.index.max().date()) if len(sub) else 'N/A',
            'mean_log_RV': round(sub['log_RV'].mean(), 4) if len(sub) else np.nan,
            'std_log_RV':  round(sub['log_RV'].std(),  4) if len(sub) else np.nan,
        })

    regime_stats_df = pd.DataFrame(regime_stats).set_index('regime')
    logger.info(f"\nRegime distribution:\n{regime_stats_df.to_string()}")

    # ── Per-regime modeling ──
    models_dict    = {}
    forecasts_dict = {}
    coef_rows      = []
    metric_rows    = []

    for regime in REGIMES:
        df_r = df[df['regime'] == regime].copy()
        n = len(df_r)
        logger.info(f"\n--- Regime: {regime.upper()} (n={n}) ---")

        if n < min_obs_per_regime:
            logger.warning(f"  Insufficient observations ({n} < {min_obs_per_regime}). Skipping.")
            continue

        models_dict[regime]    = {}
        forecasts_dict[regime] = {}

        for mtype in model_types:
            logger.info(f"  Fitting {mtype}...")

            # In-sample fit (for coefficient table)
            model_is = fit_har_ols(df_r, mtype)
            models_dict[regime][mtype] = model_is

            # Key coefficients for the paper
            params = model_is.params
            pvals  = model_is.pvalues
            row = {
                'regime': regime, 'model': mtype,
                'alpha':   round(params.get('const',  np.nan), 4),
                'beta_d':  round(params.get('RV_d',   np.nan), 4),
                'beta_w':  round(params.get('RV_w',   np.nan), 4),
                'beta_m':  round(params.get('RV_m',   np.nan), 4),
                'R2_insample': round(model_is.rsquared, 4),
                'N': int(model_is.nobs),
            }
            # Sentiment & micro betas (HAR-S only)
            for col in ['FinBERT_score_lag1', 'sent_surprise_lag1', 'vpin_lag1', 'obi_sq_lag1']:
                if col in params.index:
                    row[f'beta_{col}'] = round(params[col], 4)
                    row[f'p_{col}']    = round(pvals[col], 4)
            coef_rows.append(row)

            # OOS metrics via expanding window
            min_tr = min(100, max(50, n // 3))
            actuals, forecasts = expanding_window_oos(df_r, mtype, min_train=min_tr)
            forecasts_dict[regime][mtype] = (actuals, forecasts)

            if len(actuals) > 10:
                mse   = mean_squared_error(actuals, forecasts)
                mae   = mean_absolute_error(actuals, forecasts)
                qlike = qlike_numpy(forecasts, actuals)
                metric_rows.append({
                    'regime': regime, 'model': mtype,
                    'QLIKE': round(qlike, 6),
                    'MSE':   round(mse, 6),
                    'MAE':   round(mae, 6),
                    'N_oos': len(actuals),
                })
                logger.info(f"    OOS -> QLIKE={qlike:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")

    coef_df   = pd.DataFrame(coef_rows).set_index(['regime', 'model'])
    metric_df = pd.DataFrame(metric_rows).set_index(['regime', 'model'])

    # ── Print Paper Tables ──
    _print_coefficient_table(coef_df)
    _print_metrics_table(metric_df)

    return {
        'regime_stats': regime_stats_df,
        'coefficients': coef_df,
        'oos_metrics':  metric_df,
        'models':       models_dict,
        'forecasts':    forecasts_dict,
    }


def _print_coefficient_table(coef_df: pd.DataFrame):
    """Print Table 3 for the paper: regime-conditional beta table."""
    print("\n" + "="*70)
    print("  PAPER TABLE 3 — Regime-Conditional HAR Coefficients")
    print("="*70)
    print("  H: Does sentiment beta vary across market stress regimes?")
    print("-"*70)
    print(coef_df.to_string())
    print("="*70)


def _print_metrics_table(metric_df: pd.DataFrame):
    """Print OOS metrics by regime."""
    print("\n" + "="*70)
    print("  OOS Metrics by Regime (Expanding Window)")
    print("="*70)
    print(metric_df.to_string())
    print("="*70)


# ── Run from project root ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--asset',   default='btc', choices=['btc', 'spx', 'nifty'])
    parser.add_argument('--data_dir', default='data/processed')
    args = parser.parse_args()

    matrix_path = f"{args.data_dir}/{args.asset}_feature_matrix.parquet"
    results = run_regime_analysis(matrix_path)
