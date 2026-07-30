"""
Diebold-Mariano Test for Forecast Comparison
=============================================
Reference: Diebold & Mariano (1995), Harvey, Leybourne & Newbold (1997) correction.
Loss functions: MSE, MAE, QLIKE (Patton 2011 — correct for volatility forecasts).

Usage:
    from eval.dm_test import dm_test, run_full_comparison
"""

import numpy as np
from scipy import stats
import pandas as pd
import warnings


def _loss_differential(actual: np.ndarray, pred_a: np.ndarray,
                        pred_b: np.ndarray, loss: str = "QLIKE") -> np.ndarray:
    """
    Compute per-period loss differential d_t = L(e_A) - L(e_B).
    Positive d_t means pred_B is better (lower loss) at time t.
    
    For volatility forecasting, QLIKE is the robust loss function.
    Reference: Patton (2011) — "Volatility forecast comparison using imperfect volatility proxies."
    """
    if loss == "MSE":
        e_a = (actual - pred_a) ** 2
        e_b = (actual - pred_b) ** 2
    elif loss == "MAE":
        e_a = np.abs(actual - pred_a)
        e_b = np.abs(actual - pred_b)
    elif loss == "QLIKE":
        # pred_a, pred_b are log(RV) forecasts; actual is log(RV)
        # QLIKE = sigma^2/h - log(sigma^2/h) - 1
        # where sigma^2 = actual variance, h = predicted variance
        actual_var = np.exp(np.clip(actual, -15, 15))
        pred_var_a = np.exp(np.clip(pred_a, -15, 15))
        pred_var_b = np.exp(np.clip(pred_b, -15, 15))
        e_a = actual_var / pred_var_a - np.log(actual_var / pred_var_a) - 1
        e_b = actual_var / pred_var_b - np.log(actual_var / pred_var_b) - 1
    else:
        raise ValueError(f"Unknown loss function: {loss}. Choose 'MSE', 'MAE', or 'QLIKE'.")
    
    return e_a - e_b   # positive = B is better


def _hlb_variance(d: np.ndarray, h: int = 1) -> float:
    """
    Harvey, Leybourne & Newbold (1997) autocorrelation-corrected variance.
    Accounts for h-step-ahead forecast autocorrelation.
    """
    T = len(d)
    gamma_0 = np.var(d, ddof=1)
    
    # Add autocovariance terms for h > 1
    gamma_sum = 0.0
    for lag in range(1, h):
        if T - lag > 0:
            gamma_sum += np.cov(d[lag:], d[:-lag])[0, 1]
    
    V_d = (gamma_0 + 2 * gamma_sum) / T
    return max(V_d, 1e-20)   # Guard against near-zero variance


def dm_test(actual: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
            h: int = 1, loss: str = "QLIKE") -> dict:
    """
    Diebold-Mariano test with HLN small-sample correction.

    Parameters
    ----------
    actual   : array of true log(RV) values
    pred_a   : array of log(RV) forecasts from Model A (baseline, e.g. HAR)
    pred_b   : array of log(RV) forecasts from Model B (challenger)
    h        : forecast horizon (1 = one-step-ahead)
    loss     : 'QLIKE' (recommended for RV), 'MSE', or 'MAE'

    Returns
    -------
    dict with keys:
        DM_stat      : HLN-corrected DM statistic
        p_value      : two-sided p-value
        significant  : True if p < 0.05
        better_model : 'B' if pred_b beats pred_a, else 'A'
        mean_loss_A  : mean loss of Model A
        mean_loss_B  : mean loss of Model B
        loss         : loss function used
        n_obs        : number of forecast periods
    """
    actual = np.asarray(actual, dtype=float).ravel()
    pred_a = np.asarray(pred_a, dtype=float).ravel()
    pred_b = np.asarray(pred_b, dtype=float).ravel()
    
    assert len(actual) == len(pred_a) == len(pred_b), \
        "actual, pred_a, pred_b must have the same length."
    
    T = len(actual)
    d = _loss_differential(actual, pred_a, pred_b, loss)
    d_bar = np.mean(d)
    
    V_d = _hlb_variance(d, h)
    
    # Raw DM statistic
    DM_raw = d_bar / np.sqrt(V_d)
    
    # HLN small-sample correction factor (Harvey et al. 1997, eq. 4)
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    DM_corrected = DM_raw * correction
    
    # Two-sided p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(DM_corrected)))
    
    # Individual mean losses
    actual_var = np.exp(np.clip(actual, -15, 15))
    
    if loss == "QLIKE":
        mean_loss_a = np.mean(actual_var / np.exp(np.clip(pred_a, -15, 15)) -
                               np.log(actual_var / np.exp(np.clip(pred_a, -15, 15))) - 1)
        mean_loss_b = np.mean(actual_var / np.exp(np.clip(pred_b, -15, 15)) -
                               np.log(actual_var / np.exp(np.clip(pred_b, -15, 15))) - 1)
    elif loss == "MSE":
        mean_loss_a = np.mean((actual - pred_a) ** 2)
        mean_loss_b = np.mean((actual - pred_b) ** 2)
    else:
        mean_loss_a = np.mean(np.abs(actual - pred_a))
        mean_loss_b = np.mean(np.abs(actual - pred_b))
    
    return {
        "DM_stat":      round(DM_corrected, 4),
        "p_value":      round(p_value, 4),
        "significant":  bool(p_value < 0.05),
        "better_model": "B" if d_bar > 0 else "A",
        "mean_loss_A":  round(mean_loss_a, 6),
        "mean_loss_B":  round(mean_loss_b, 6),
        "loss":         loss,
        "n_obs":        T,
    }


def oos_r_squared(actual: np.ndarray, pred_benchmark: np.ndarray,
                  pred_model: np.ndarray) -> float:
    """
    Campbell & Thompson (2008) Out-of-Sample R².
    OOS-R² > 0 means the model beats the benchmark.
    OOS-R² = 1 - SSE_model / SSE_benchmark
    """
    sse_bm = np.sum((actual - pred_benchmark) ** 2)
    sse_m  = np.sum((actual - pred_model) ** 2)
    if sse_bm == 0:
        return 0.0
    return float(1.0 - sse_m / sse_bm)


def run_full_comparison(actual: np.ndarray, forecasts_dict: dict,
                        benchmark_key: str = "HAR",
                        losses: list = ("QLIKE", "MSE")) -> pd.DataFrame:
    """
    Run DM test comparing all models in forecasts_dict against the benchmark.
    
    Parameters
    ----------
    actual         : true log(RV) values
    forecasts_dict : {"HAR": array, "HAR-S": array, "Neural-HAR": array, ...}
    benchmark_key  : key in forecasts_dict to use as baseline (default "HAR")
    losses         : list of loss functions to test

    Returns
    -------
    DataFrame — one row per challenger model, columns for each loss function's DM stat & p-value,
                plus OOS-R² vs benchmark.
    
    Example output (Paper Table 2):
    
    Model        | QLIKE_DM | QLIKE_p | QLIKE_sig | MSE_DM | MSE_p | OOS_R2
    -------------|----------|---------|-----------|--------|-------|-------
    HAR-S        |   2.14   |  0.032  |    ***    |  1.87  | 0.061 |  0.042
    Neural-HAR   |   3.41   |  0.001  |    ***    |  2.99  | 0.003 |  0.093
    """
    if benchmark_key not in forecasts_dict:
        raise KeyError(f"Benchmark key '{benchmark_key}' not found in forecasts_dict.")
    
    actual = np.asarray(actual, dtype=float).ravel()
    pred_bm = np.asarray(forecasts_dict[benchmark_key], dtype=float).ravel()
    
    rows = []
    for name, preds in forecasts_dict.items():
        if name == benchmark_key:
            continue
        
        pred_b = np.asarray(preds, dtype=float).ravel()
        row = {"Model": name}
        
        for loss in losses:
            result = dm_test(actual, pred_bm, pred_b, h=1, loss=loss)
            sig_stars = ("***" if result["p_value"] < 0.01 else
                         "**"  if result["p_value"] < 0.05 else
                         "*"   if result["p_value"] < 0.10 else "")
            row[f"{loss}_DM"]     = result["DM_stat"]
            row[f"{loss}_p"]      = result["p_value"]
            row[f"{loss}_sig"]    = sig_stars
            row[f"{loss}_better"] = result["better_model"]
        
        row["OOS_R2"] = round(oos_r_squared(actual, pred_bm, pred_b), 4)
        rows.append(row)
    
    df = pd.DataFrame(rows).set_index("Model")
    return df


def print_dm_table(results_df: pd.DataFrame, title: str = "Forecast Comparison (DM Test)"):
    """Pretty-print the DM results table for the paper."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(results_df.to_string())
    print(f"{'='*70}")
    print("Significance: *** p<0.01, ** p<0.05, * p<0.10")
    print("DM stat > 0 means challenger beats baseline.\n")


# ─────────────────────────────────────────────
# Quick self-test (run: python eval/dm_test.py)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    T = 500
    
    # Simulate log(RV) as AR(1)
    log_rv = np.zeros(T)
    for t in range(1, T):
        log_rv[t] = 0.05 + 0.85 * log_rv[t-1] + np.random.normal(0, 0.3)
    
    actual = log_rv[252:]          # OOS period
    
    # HAR: uses lag 1 only (weak baseline)
    pred_har    = log_rv[251:-1]
    
    # HAR-S: slightly better (add tiny noise improvement)
    pred_hars   = pred_har + np.random.normal(0, 0.05, len(actual))
    
    # Neural-HAR: clearly better
    pred_neural = 0.05 + 0.85 * pred_har + np.random.normal(0, 0.1, len(actual))
    
    forecasts = {"HAR": pred_har, "HAR-S": pred_hars, "Neural-HAR": pred_neural}
    
    results = run_full_comparison(actual, forecasts, benchmark_key="HAR")
    print_dm_table(results, title="Self-Test: HAR vs HAR-S vs Neural-HAR")
    
    # Individual test
    r = dm_test(actual, pred_har, pred_neural, loss="QLIKE")
    print(f"HAR vs Neural-HAR (QLIKE): DM={r['DM_stat']}, p={r['p_value']}, sig={r['significant']}")
