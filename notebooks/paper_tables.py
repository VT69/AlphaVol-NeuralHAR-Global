"""
Paper Tables & Figures Generator
==================================
"Forecasting Realized Volatility in Markets:
 Does Sentiment Add Information Beyond the HAR Framework?"

Assets: BTC · SPX · NIFTY50
Models: HAR · HAR-S · (Neural-HAR if trained)

Run this after all models have been estimated to produce:
  - Table 1 : Descriptive statistics of realized volatility
  - Table 2 : In-sample HAR / HAR-S coefficient table
  - Table 3 : OOS loss comparison (QLIKE, MSE, MAE)
  - Table 4 : Diebold-Mariano tests (HAR vs HAR-S)
  - Table 5 : Model Confidence Set
  - Table 6 : Regime-conditional coefficient betas
  - Table 7 : Economic significance (vol-targeting Sharpe)

  - Figure 1 : RV time series (BTC, SPX, NIFTY)
  - Figure 2 : OOS forecast vs actual (per asset)
  - Figure 3 : Sentiment β by regime (bar chart)
  - Figure 4 : Cumulative PnL — vol-timing strategies

Usage:
    cd e:/AlphaVol-NeuralHAR-Global
    python notebooks/paper_tables.py
    python notebooks/paper_tables.py --asset btc --no_plots
"""

import os
import sys
import argparse
import warnings
import logging
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR  = os.path.join(ROOT, "data", "processed")
TABLES_DIR     = os.path.join(ROOT, "data", "paper_tables")
FIGURES_DIR    = os.path.join(ROOT, "data", "paper_figures")
ASSETS         = ["btc", "spx", "nifty"]
ASSET_LABELS   = {"btc": "BTC", "spx": "SPX", "nifty": "NIFTY50"}
ANN            = {"btc": 365, "spx": 252, "nifty": 252}

os.makedirs(TABLES_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_feature_matrix(asset: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, f"{asset}_feature_matrix.parquet")
    if not os.path.exists(path):
        logger.warning(f"  Feature matrix not found: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path).sort_index()


def _load_forecasts(asset: str) -> dict:
    """Load all saved OOS forecast arrays for an asset."""
    fc = {}
    for tag, label in [("har", "HAR"), ("hars", "HAR-S"), ("neural_har", "Neural-HAR")]:
        p = os.path.join(PROCESSED_DIR, f"forecasts_{tag}_{asset}.npy")
        if os.path.exists(p):
            fc[label] = np.load(p)
    act_path = os.path.join(PROCESSED_DIR, f"actuals_{asset}.npy")
    if os.path.exists(act_path):
        fc["__actuals__"] = np.load(act_path)
    return fc


def _sig_stars(p: float) -> str:
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


def _fmt(v, decimals=4) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:.{decimals}f}"


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1 — Descriptive Statistics of Realized Volatility
# ─────────────────────────────────────────────────────────────────────────────

def table1_descriptive_stats(assets: list) -> pd.DataFrame:
    """
    Per-asset RV descriptive statistics:
    N, mean, std, min, max, autocorr(1), autocorr(5), autocorr(22), Ljung-Box p
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    rows = []
    for asset in assets:
        df = _load_feature_matrix(asset)
        if df.empty or "log_RV" not in df.columns:
            continue
        rv = df["log_RV"].dropna()
        try:
            lb_stat, lb_p = acorr_ljungbox(rv, lags=[22], return_df=False)
            lb_p_val = float(lb_p[0])
        except Exception:
            lb_p_val = np.nan

        rows.append({
            "Asset":       ASSET_LABELS.get(asset, asset.upper()),
            "N":           len(rv),
            "Start":       str(rv.index.min().date()),
            "End":         str(rv.index.max().date()),
            "Mean":        _fmt(rv.mean()),
            "Std":         _fmt(rv.std()),
            "Min":         _fmt(rv.min()),
            "Max":         _fmt(rv.max()),
            "Skew":        _fmt(rv.skew(), 3),
            "Kurt":        _fmt(rv.kurt(), 3),
            "AC(1)":       _fmt(rv.autocorr(1)),
            "AC(5)":       _fmt(rv.autocorr(5)),
            "AC(22)":      _fmt(rv.autocorr(22)),
            "LB(22)_p":   _fmt(lb_p_val, 3),
        })

    t = pd.DataFrame(rows).set_index("Asset")
    path = os.path.join(TABLES_DIR, "table1_descriptive_stats.csv")
    t.to_csv(path)

    print("\n" + "=" * 75)
    print("  TABLE 1 — Descriptive Statistics of Daily log(RV)")
    print("=" * 75)
    print(t.to_string())
    print("=" * 75)
    logger.info(f"  Saved -> {path}")
    return t


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 2 — In-Sample HAR / HAR-S Coefficients
# ─────────────────────────────────────────────────────────────────────────────

def table2_insample_coefficients(assets: list) -> pd.DataFrame:
    """
    Estimates full-sample OLS HAR and HAR-S, reports coefficients with
    HAC standard errors (Newey-West, 5 lags).
    """
    import statsmodels.api as sm

    HAR_COLS = ["RV_d", "RV_w", "RV_m"]
    EXO_CANDIDATES = [
        "vpin_lag1", "obi_sq_lag1", "illiq_lag1",
        "FinBERT_score_lag1", "sent_surprise_lag1",
        "credit_spread_lag1", "term_slope_lag1", "crypto_fg_lag1",
    ]

    all_rows = []
    for asset in assets:
        df = _load_feature_matrix(asset)
        if df.empty or "Target_RV" not in df.columns:
            continue
        df = df.dropna(subset=["RV_d", "RV_w", "RV_m", "Target_RV"])
        y  = df["Target_RV"]

        for mtype in ["HAR", "HAR-S"]:
            cols = HAR_COLS if mtype == "HAR" else \
                   HAR_COLS + [c for c in EXO_CANDIDATES if c in df.columns]
            X = sm.add_constant(df[cols])
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            result = sm.OLS(y[mask], X[mask]).fit(
                cov_type="HAC", cov_kwds={"maxlags": 5}
            )
            for pname, val in result.params.items():
                pval = result.pvalues[pname]
                all_rows.append({
                    "Asset":     ASSET_LABELS.get(asset, asset.upper()),
                    "Model":     mtype,
                    "Variable":  pname,
                    "Coef":      _fmt(val),
                    "p-value":   _fmt(pval, 3),
                    "Sig":       _sig_stars(pval),
                    "R²":        _fmt(result.rsquared),
                    "N":         int(result.nobs),
                })

    t = pd.DataFrame(all_rows)
    path = os.path.join(TABLES_DIR, "table2_insample_coefs.csv")
    t.to_csv(path, index=False)

    print("\n" + "=" * 75)
    print("  TABLE 2 — In-Sample HAR / HAR-S Coefficients (HAC SE, 5 lags)")
    print("=" * 75)
    print(t.to_string(index=False))
    print("=" * 75)
    logger.info(f"  Saved -> {path}")
    return t


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 3 — OOS Loss Metrics
# ─────────────────────────────────────────────────────────────────────────────

def table3_oos_metrics(assets: list) -> pd.DataFrame:
    """QLIKE, MSE, MAE for each model × asset from saved forecast arrays."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    def qlike(act, fc):
        av = np.exp(np.clip(act, -15, 15))
        pv = np.exp(np.clip(fc,  -15, 15))
        return float(np.mean(av / pv - np.log(av / pv) - 1))

    rows = []
    for asset in assets:
        fc_dict = _load_forecasts(asset)
        if "__actuals__" not in fc_dict:
            continue
        act = fc_dict.pop("__actuals__")
        for mname, fc in fc_dict.items():
            n = min(len(act), len(fc))
            rows.append({
                "Asset":   ASSET_LABELS.get(asset, asset.upper()),
                "Model":   mname,
                "N_OOS":   n,
                "QLIKE":   _fmt(qlike(act[:n], fc[:n])),
                "MSE":     _fmt(mean_squared_error(act[:n], fc[:n])),
                "MAE":     _fmt(mean_absolute_error(act[:n], fc[:n])),
            })

    t = pd.DataFrame(rows)
    path = os.path.join(TABLES_DIR, "table3_oos_metrics.csv")
    t.to_csv(path, index=False)

    print("\n" + "=" * 75)
    print("  TABLE 3 — Out-of-Sample Loss Metrics (Expanding Window)")
    print("=" * 75)
    print(t.to_string(index=False))
    print("=" * 75)
    logger.info(f"  Saved -> {path}")
    return t


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 4 — Diebold-Mariano Tests
# ─────────────────────────────────────────────────────────────────────────────

def table4_dm_tests(assets: list) -> pd.DataFrame:
    """DM tests: HAR-S and Neural-HAR vs HAR benchmark."""
    from eval.dm_test import run_full_comparison

    all_rows = []
    for asset in assets:
        fc_dict = _load_forecasts(asset)
        if "__actuals__" not in fc_dict:
            logger.warning(f"  No actuals for {asset} — run step_har first.")
            continue
        act = fc_dict.pop("__actuals__")
        if "HAR" not in fc_dict:
            continue

        # Align lengths
        min_n = min(len(act), *(len(v) for v in fc_dict.values()))
        act   = act[:min_n]
        fc_dict = {k: v[:min_n] for k, v in fc_dict.items()}

        try:
            dm_df = run_full_comparison(act, fc_dict, benchmark_key="HAR",
                                        losses=["QLIKE", "MSE"])
            dm_df.insert(0, "Asset", ASSET_LABELS.get(asset, asset.upper()))
            all_rows.append(dm_df.reset_index())
        except Exception as e:
            logger.error(f"  DM test error for {asset}: {e}")

    if not all_rows:
        logger.warning("  No DM results — run HAR step first.")
        return pd.DataFrame()

    t = pd.concat(all_rows).reset_index(drop=True)
    path = os.path.join(TABLES_DIR, "table4_dm_tests.csv")
    t.to_csv(path, index=False)

    print("\n" + "=" * 75)
    print("  TABLE 4 — Diebold-Mariano Test (challenger vs HAR baseline)")
    print("  Positive DM_stat -> challenger beats HAR | *** p<0.01  ** p<0.05  * p<0.10")
    print("=" * 75)
    print(t.to_string(index=False))
    print("=" * 75)
    logger.info(f"  Saved -> {path}")
    return t


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 5 — Model Confidence Set
# ─────────────────────────────────────────────────────────────────────────────

def table5_mcs(assets: list, alpha: float = 0.10, B: int = 1000) -> pd.DataFrame:
    """MCS across models for each asset."""
    from eval.mcs_test import mcs_test, print_mcs_results

    def _qlike_losses(act, fc):
        n  = min(len(act), len(fc))
        av = np.exp(np.clip(act[:n], -15, 15))
        pv = np.exp(np.clip(fc[:n],  -15, 15))
        return av / pv - np.log(av / pv) - 1

    all_rows = []
    for asset in assets:
        fc_dict = _load_forecasts(asset)
        if "__actuals__" not in fc_dict:
            continue
        act = fc_dict.pop("__actuals__")
        if len(fc_dict) < 2:
            continue

        min_n  = min(len(act), *(len(v) for v in fc_dict.values()))
        L_df   = pd.DataFrame({k: _qlike_losses(act, v)[:min_n]
                                 for k, v in fc_dict.items()})
        try:
            mcs_df = mcs_test(L_df, alpha=alpha, B=B, block_size=5, stat="TR")
            print_mcs_results(mcs_df,
                              title=f"MCS — {ASSET_LABELS.get(asset, asset.upper())}",
                              alpha=alpha)
            mcs_df.insert(0, "Asset", ASSET_LABELS.get(asset, asset.upper()))
            all_rows.append(mcs_df)
        except Exception as e:
            logger.error(f"  MCS error for {asset}: {e}")

    if not all_rows:
        return pd.DataFrame()

    t = pd.concat(all_rows).reset_index(drop=True)
    path = os.path.join(TABLES_DIR, "table5_mcs.csv")
    t.to_csv(path, index=False)
    logger.info(f"  Saved -> {path}")
    return t


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 6 — Regime-Conditional Betas
# ─────────────────────────────────────────────────────────────────────────────

def table6_regime_betas(assets: list) -> pd.DataFrame:
    """Load saved regime results from CSV."""
    path_csv = os.path.join(TABLES_DIR, "table4_regime_coefficients.csv")
    if os.path.exists(path_csv):
        t = pd.read_csv(path_csv)
        print("\n" + "=" * 75)
        print("  TABLE 6 — Regime-Conditional HAR-S Sentiment Betas")
        print("=" * 75)
        cols = [c for c in ["Asset", "regime", "model",
                             "beta_FinBERT_score_lag1", "p_FinBERT_score_lag1",
                             "beta_sent_surprise_lag1", "p_sent_surprise_lag1",
                             "R2_insample", "N"] if c in t.columns]
        print(t[cols].to_string(index=False))
        print("=" * 75)
        return t
    else:
        logger.warning(f"  Regime coef table not found — run 'python run_paper.py --step regime' first.")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 7 — Economic Significance (Backtest Sharpe)
# ─────────────────────────────────────────────────────────────────────────────

def table7_backtest(assets: list) -> pd.DataFrame:
    """Load saved backtest results from CSV."""
    path_csv = os.path.join(TABLES_DIR, "table6_backtest.csv")
    if os.path.exists(path_csv):
        t = pd.read_csv(path_csv)
        print("\n" + "=" * 75)
        print("  TABLE 7 — Economic Significance: Vol-Targeting Backtest")
        print("  (5bps slippage, Quarter-Kelly, Max 5× leverage)")
        print("=" * 75)
        print(t.to_string(index=False))
        print("=" * 75)
        return t
    else:
        logger.warning("  Backtest table not found — run 'python run_paper.py --step backtest' first.")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def figures(assets: list, no_plots: bool = False):
    if no_plots:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        plt.rcParams.update({
            "font.family": "DejaVu Sans", "axes.spines.top": False,
            "axes.spines.right": False, "axes.grid": True,
            "grid.alpha": 0.3, "figure.dpi": 150,
        })
    except ImportError:
        logger.warning("  matplotlib not available — skipping figures.")
        return

    COLORS = {"BTC": "#F7931A", "SPX": "#2196F3", "NIFTY50": "#4CAF50"}

    # ── Figure 1: RV Time Series ──────────────────────────────────────────
    fig, axes = plt.subplots(len(assets), 1, figsize=(12, 3 * len(assets)), sharex=False)
    if len(assets) == 1:
        axes = [axes]

    for ax, asset in zip(axes, assets):
        df = _load_feature_matrix(asset)
        if df.empty or "log_RV" not in df.columns:
            continue
        label = ASSET_LABELS.get(asset, asset.upper())
        color = COLORS.get(label, "#888")
        rv    = df["log_RV"].dropna()
        ax.plot(rv.index, rv.values, color=color, linewidth=0.7, alpha=0.9)
        ax.set_title(f"{label} — Daily log(RV)", fontsize=11, fontweight="bold")
        ax.set_ylabel("log(RV)", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        # Shade recessions / COVID
        ax.axvspan(pd.Timestamp("2020-02-01", tz="UTC"),
                   pd.Timestamp("2020-06-01", tz="UTC"),
                   alpha=0.12, color="red", label="COVID-19")
        ax.legend(fontsize=8)

    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, "fig1_rv_timeseries.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved -> {p}")

    # ── Figure 2: OOS Forecast vs Actual ─────────────────────────────────
    for asset in assets:
        fc_dict = _load_forecasts(asset)
        if "__actuals__" not in fc_dict:
            continue
        act = fc_dict.pop("__actuals__")
        label = ASSET_LABELS.get(asset, asset.upper())

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(act, color="#555", linewidth=0.7, label="Actual log(RV)", alpha=0.9)
        model_colors = {"HAR": "#E53935", "HAR-S": "#1E88E5", "Neural-HAR": "#43A047"}
        for mname, fc in fc_dict.items():
            n = min(len(act), len(fc))
            ax.plot(range(n), fc[:n], linewidth=0.7,
                    color=model_colors.get(mname, "#888"),
                    label=mname, alpha=0.8)
        ax.set_title(f"{label} — OOS Forecasts vs Actual log(RV)", fontsize=11, fontweight="bold")
        ax.set_xlabel("OOS Period (days)")
        ax.set_ylabel("log(RV)")
        ax.legend(fontsize=9)
        plt.tight_layout()
        p = os.path.join(FIGURES_DIR, f"fig2_oos_forecast_{asset}.png")
        plt.savefig(p, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved -> {p}")

    logger.info(f"  All figures saved -> {FIGURES_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate all paper tables and figures."
    )
    parser.add_argument("--asset", type=str, default=None,
                        choices=["btc", "spx", "nifty"])
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--mcs_B", type=int, default=1000)
    parser.add_argument("--mcs_alpha", type=float, default=0.10)
    args = parser.parse_args()

    assets = [args.asset] if args.asset else ASSETS

    print("\n" + "=" * 75)
    print("  PAPER TABLES GENERATOR")
    print("  'Forecasting Realized Volatility in Markets:'")
    print("  'Does Sentiment Add Information Beyond the HAR Framework?'")
    print(f"  Assets: {[ASSET_LABELS.get(a, a.upper()) for a in assets]}")
    print("=" * 75)

    table1_descriptive_stats(assets)
    table2_insample_coefficients(assets)
    table3_oos_metrics(assets)
    table4_dm_tests(assets)
    table5_mcs(assets, alpha=args.mcs_alpha, B=args.mcs_B)
    table6_regime_betas(assets)
    table7_backtest(assets)

    figures(assets, no_plots=args.no_plots)

    print(f"\n  Tables -> {TABLES_DIR}/")
    print(f"  Figures -> {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
