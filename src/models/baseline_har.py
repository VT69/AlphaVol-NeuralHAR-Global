"""
HAR Baseline & HAR-S (Augmented) Models
=========================================
Paper Steps 1 & 2:
  HAR:   RV_t = alpha + β_d·RV_{t-1} + β_w·RV_{t-5:t-1} + β_m·RV_{t-22:t-1} + ε
  HAR-S: HAR + FinBERT_score + sent_surprise + VPIN + OBI

Both use expanding-window OOS estimation (no lookahead bias).
Loss: QLIKE (Patton 2011) + MSE + MAE.
"""
import os
import argparse
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# HAR features (always present)
HAR_COLS = ['RV_d', 'RV_w', 'RV_m']

# Exogenous features added in HAR-S (use what's available)
EXO_CANDIDATES = [
    'vpin_lag1', 'obi_sq_lag1', 'illiq_lag1',
    'FinBERT_score_lag1', 'sent_surprise_lag1',
    'credit_spread_lag1', 'term_slope_lag1',
    'crypto_fg_lag1',
]


def qlike_loss_numpy(pred_log_rv: np.ndarray, actual_log_rv: np.ndarray) -> float:
    """
    QLIKE = mean(actual_var/pred_var - log(actual_var/pred_var) - 1)
    Robust to RV proxy noise — preferred for volatility papers (Patton 2011).
    """
    actual_var = np.exp(np.clip(actual_log_rv, -15, 15))
    pred_var   = np.exp(np.clip(pred_log_rv,   -15, 15))
    return float(np.mean(actual_var / pred_var - np.log(actual_var / pred_var) - 1))


def _build_design_matrix(df: pd.DataFrame, model_type: str = 'HAR') -> pd.DataFrame:
    """Build OLS design matrix. model_type: 'HAR' or 'HAR-S'."""
    if model_type == 'HAR':
        cols = HAR_COLS
    else:
        cols = HAR_COLS + [c for c in EXO_CANDIDATES if c in df.columns]
    X = df[cols].copy()
    return sm.add_constant(X)


def expanding_window_forecast(df: pd.DataFrame, model_type: str = 'HAR',
                               min_train: int = 252) -> tuple:
    """
    Expanding-window OOS forecast — re-estimates OLS at each step.
    No lookahead: model at time t is trained only on data t-1 and earlier.

    Returns: (actuals, forecasts, final_fitted_model)
    """
    logger.info(f"Expanding window [{model_type}] | min_train={min_train}")

    X = _build_design_matrix(df, model_type)
    y = df['Target_RV']  # Already log_RV from build_feature_matrix

    forecasts, actuals = [], []
    final_model = None

    for t in range(min_train, len(df)):
        X_tr, y_tr = X.iloc[:t], y.iloc[:t]
        X_te, y_te = X.iloc[t:t+1], y.iloc[t]

        mask = ~(X_tr.isnull().any(axis=1) | y_tr.isnull())
        if mask.sum() < 50:
            continue

        model       = sm.OLS(y_tr[mask], X_tr[mask]).fit()
        # Fill any NaN in test features with training-set column mean (prevents NaN forecasts)
        X_te_filled = X_te.fillna(X_tr[mask].mean())
        pred        = model.predict(X_te_filled).values[0]
        final_model = model


        forecasts.append(pred)
        actuals.append(float(y_te))

    logger.info(f"  OOS periods: {len(actuals)}")
    return np.array(actuals), np.array(forecasts), final_model


def evaluate(actuals: np.ndarray, forecasts: np.ndarray,
             model_type: str = 'HAR') -> dict:
    """Compute QLIKE, MSE, MAE and return as dict."""
    mse   = mean_squared_error(actuals, forecasts)
    mae   = mean_absolute_error(actuals, forecasts)
    qlike = qlike_loss_numpy(forecasts, actuals)
    logger.info(f"--- {model_type} OOS Metrics ---")
    logger.info(f"  QLIKE: {qlike:.4f}  MSE: {mse:.4f}  MAE: {mae:.4f}")
    return {'model': model_type, 'QLIKE': qlike, 'MSE': mse, 'MAE': mae, 'N': len(actuals)}


def run_har_and_hars(feature_matrix_path: str, asset: str = 'btc',
                      min_train: int = 252, save_dir: str = 'data/processed') -> dict:
    """
    Main runner: fits HAR and HAR-S on the feature matrix.
    Saves OOS forecasts as .npy files.
    Returns dict with metrics, forecasts, and fitted models.

    Paper Tables 1 (HAR coefficients) and 2 (HAR vs HAR-S DM test inputs).
    """
    logger.info(f"Loading: {feature_matrix_path}")
    df = pd.read_parquet(feature_matrix_path).sort_index()
    df = df.dropna(subset=['RV_d', 'RV_w', 'RV_m', 'Target_RV'])
    logger.info(f"Dataset: {len(df)} rows | {df.index.min().date()} -> {df.index.max().date()}")

    results = {}

    for mtype in ['HAR', 'HAR-S']:
        actuals, forecasts, model = expanding_window_forecast(df, mtype, min_train)
        metrics = evaluate(actuals, forecasts, mtype)
        results[mtype] = {
            'actuals': actuals, 'forecasts': forecasts,
            'model': model, 'metrics': metrics
        }

        # Save forecasts for DM test and backtest
        os.makedirs(save_dir, exist_ok=True)
        tag = mtype.lower().replace('-', 's')   # 'har' or 'hars'
        np.save(os.path.join(save_dir, f'forecasts_{tag}_{asset}.npy'), forecasts)
        np.save(os.path.join(save_dir, f'actuals_{asset}.npy'), actuals)
        logger.info(f"  Saved forecasts -> {save_dir}/forecasts_{tag}_{asset}.npy")

    # ── Print coefficient comparison (Paper Table 1) ──
    _print_coefficient_table(results, df)
    _print_metrics_comparison(results)

    return results


def _print_coefficient_table(results: dict, df: pd.DataFrame):
    """Print HAR-S in-sample coefficient table — becomes Paper Table 1."""
    print("\n" + "="*65)
    print("  PAPER TABLE 1 — HAR-S In-Sample Coefficients (Full Sample)")
    print("="*65)
    for mtype, res in results.items():
        if res['model'] is None:
            continue
        params = res['model'].params
        pvals  = res['model'].pvalues
        print(f"\n  Model: {mtype}")
        for name, val in params.items():
            sig = "***" if pvals[name] < 0.01 else "**" if pvals[name] < 0.05 else "*" if pvals[name] < 0.10 else ""
            print(f"    {name:<28} {val:>8.4f}   (p={pvals[name]:.3f}) {sig}")
        print(f"    R² (in-sample):           {res['model'].rsquared:.4f}")
    print("="*65)


def _print_metrics_comparison(results: dict):
    """Print OOS metric comparison across models."""
    print("\n" + "="*55)
    print("  OOS Metrics Comparison (Expanding Window)")
    print("="*55)
    rows = [v['metrics'] for v in results.values()]
    print(pd.DataFrame(rows).set_index('model').to_string())
    print("="*55 + "\n")


# ── CLI entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HAR and HAR-S baseline models")
    parser.add_argument('--asset', default='btc', choices=['btc', 'spx', 'nifty'])
    parser.add_argument('--data_dir', default='data/processed')
    parser.add_argument('--min_train', type=int, default=252)
    args = parser.parse_args()

    path = os.path.join(args.data_dir, f'{args.asset}_feature_matrix.parquet')
    run_har_and_hars(path, asset=args.asset, min_train=args.min_train, save_dir=args.data_dir)
