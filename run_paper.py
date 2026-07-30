"""
run_paper.py — Master End-to-End Runner
=========================================
"Forecasting Realized Volatility in Markets:
 Does Sentiment Add Information Beyond the HAR Framework?"

Assets: BTC (Crypto) · SPX (US Equities) · NIFTY50 (Indian Equities)
Models: HAR · HAR-S · Neural-HAR

Paper Pipeline:
  Step 0 — Data & feature matrices (build_feature_matrix.py)
  Step 1 — HAR & HAR-S expanding-window OOS (baseline_har.py)
  Step 2 — DM tests + OOS-R² (dm_test.py)
  Step 3 — MCS (mcs_test.py)
  Step 4 — Regime-conditional HAR (regime_har.py)
  Step 5 — Economic significance backtest (vol_targeting.py)

Usage:
    # Full run (all assets, all steps):
    python run_paper.py

    # Single asset:
    python run_paper.py --asset btc

    # Single step:
    python run_paper.py --step har
    python run_paper.py --step dm
    python run_paper.py --step regime
    python run_paper.py --step backtest

    # Skip data build (matrices already present):
    python run_paper.py --skip_build
"""

import os
import sys
import argparse
import warnings
import logging
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(ROOT, "paper_run.log"), mode="w"),
    ],
)
logger = logging.getLogger("run_paper")

ASSETS        = ["btc", "spx", "nifty"]
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
DATA_RAW_DIR  = os.path.join(ROOT, "data_collection", "data", "raw")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Feature Matrix Build
# ─────────────────────────────────────────────────────────────────────────────

def step_build(assets: list):
    """Build feature matrices from raw data."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 0 — Building Feature Matrices")
    logger.info("=" * 70)
    from src.data_pipeline.build_feature_matrix import build_feature_matrix
    summaries = {}
    for asset in assets:
        df = build_feature_matrix(asset, DATA_RAW_DIR, PROCESSED_DIR)
        if not df.empty:
            summaries[asset] = {
                "rows": len(df),
                "start": str(df.index.min().date()),
                "end":   str(df.index.max().date()),
                "cols":  len(df.columns),
            }
    if summaries:
        logger.info("\n  Feature matrix summary:")
        logger.info(pd.DataFrame(summaries).T.to_string())
    return summaries


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — HAR & HAR-S Baseline Forecasts
# ─────────────────────────────────────────────────────────────────────────────

def step_har(assets: list) -> dict:
    """Run HAR and HAR-S expanding-window OOS for each asset."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1 — HAR & HAR-S Expanding-Window OOS Forecasts")
    logger.info("=" * 70)
    from src.models.baseline_har import run_har_and_hars
    all_results = {}
    for asset in assets:
        path = os.path.join(PROCESSED_DIR, f"{asset}_feature_matrix.parquet")
        if not os.path.exists(path):
            logger.warning(f"  Feature matrix not found for {asset} — skipping.")
            continue
        logger.info(f"\n  --- Asset: {asset.upper()} ---")
        res = run_har_and_hars(path, asset=asset,
                               min_train=252, save_dir=PROCESSED_DIR)
        all_results[asset] = res
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Neural-HAR Training  (GRN + HAR prior)
# ─────────────────────────────────────────────────────────────────────────────

def step_neural(assets: list,
                epochs: int = 100,
                lr: float = 5e-4,
                hidden_dim: int = 32) -> dict:
    """Train Neural-HAR (GRN residual corrector) for each asset."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2 — Neural-HAR Training  (HAR prior + GRN residual)")
    logger.info("=" * 70)
    from src.models.train_neural_har import train_asset

    neural_results = {}
    for asset in assets:
        logger.info(f"\n  Training Neural-HAR for {asset.upper()}...")
        try:
            preds, acts = train_asset(
                asset,
                epochs=epochs,
                lr=lr,
                hidden_dim=hidden_dim,
            )
            if preds is not None:
                neural_results[asset] = {"forecasts": preds, "actuals": acts}
                logger.info(f"  Neural-HAR training complete for {asset.upper()}")
        except Exception as e:
            logger.error(f"  Neural-HAR failed for {asset}: {e}")
    return neural_results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Diebold-Mariano Tests & OOS-R²  (Paper Table 2)
# ─────────────────────────────────────────────────────────────────────────────

def step_dm(assets: list, har_results: dict) -> dict:
    """DM test: HAR vs HAR-S (and vs Neural-HAR if saved)."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2 — Diebold-Mariano Test  [Paper Table 2]")
    logger.info("=" * 70)
    from eval.dm_test import run_full_comparison, print_dm_table

    dm_results = {}
    for asset in assets:
        if asset not in har_results:
            continue
        res  = har_results[asset]
        act  = res["HAR"]["actuals"]

        forecasts = {
            "HAR":   res["HAR"]["forecasts"],
            "HAR-S": res["HAR-S"]["forecasts"],
        }

        # Load Neural-HAR forecasts if trained
        nn_path = os.path.join(PROCESSED_DIR, f"forecasts_neural_har_{asset}.npy")
        if os.path.exists(nn_path):
            n_oos = min(len(act), len(np.load(nn_path)))
            forecasts["Neural-HAR"] = np.load(nn_path)[:n_oos]
            act = act[:n_oos]

        logger.info(f"\n  Asset: {asset.upper()}  |  OOS periods: {len(act)}")
        try:
            dm_df = run_full_comparison(act, forecasts, benchmark_key="HAR",
                                        losses=["QLIKE", "MSE"])
            print_dm_table(dm_df, title=f"DM Test — {asset.upper()}")
            dm_results[asset] = dm_df
        except Exception as e:
            logger.error(f"  DM test failed for {asset}: {e}")

    return dm_results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Model Confidence Set  (Paper Table 3)
# ─────────────────────────────────────────────────────────────────────────────

def step_mcs(assets: list, har_results: dict, alpha: float = 0.10,
             B: int = 1000) -> dict:
    """MCS across HAR · HAR-S · Neural-HAR per asset."""
    logger.info("\n" + "=" * 70)
    logger.info(f"STEP 3 — Model Confidence Set  alpha={alpha}  [Paper Table 3]")
    logger.info("=" * 70)
    from eval.mcs_test import mcs_test, print_mcs_results

    mcs_results = {}
    for asset in assets:
        if asset not in har_results:
            continue
        res = har_results[asset]
        act = res["HAR"]["actuals"]

        # Build QLIKE loss matrix
        loss_dict = {}
        for mname in ["HAR", "HAR-S"]:
            fc = res[mname]["forecasts"]
            n  = min(len(act), len(fc))
            av = np.exp(np.clip(act[:n], -15, 15))
            pv = np.exp(np.clip(fc[:n],  -15, 15))
            loss_dict[mname] = av / pv - np.log(av / pv) - 1

        nn_path = os.path.join(PROCESSED_DIR, f"forecasts_neural_har_{asset}.npy")
        if os.path.exists(nn_path):
            fc_nn = np.load(nn_path)
            n = min(len(act), len(fc_nn))
            av = np.exp(np.clip(act[:n], -15, 15))
            pv = np.exp(np.clip(fc_nn[:n], -15, 15))
            loss_dict["Neural-HAR"] = av / pv - np.log(av / pv) - 1

        # Align lengths
        min_n = min(len(v) for v in loss_dict.values())
        L_df  = pd.DataFrame({k: v[:min_n] for k, v in loss_dict.items()})

        logger.info(f"\n  Asset: {asset.upper()}  |  T={len(L_df)}  |  B={B}")
        try:
            mcs_df = mcs_test(L_df, alpha=alpha, B=B, block_size=5, stat="TR")
            print_mcs_results(mcs_df, title=f"MCS — {asset.upper()}", alpha=alpha)
            mcs_results[asset] = mcs_df
        except Exception as e:
            logger.error(f"  MCS failed for {asset}: {e}")

    return mcs_results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Regime-Conditional HAR  (Paper Tables 4–5)
# ─────────────────────────────────────────────────────────────────────────────

def step_regime(assets: list) -> dict:
    """Estimate HAR/HAR-S per market regime (low/medium/high stress)."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4 — Regime-Conditional HAR  [Paper Tables 4–5]")
    logger.info("=" * 70)
    from src.models.regime_har import run_regime_analysis

    regime_results = {}
    for asset in assets:
        path = os.path.join(PROCESSED_DIR, f"{asset}_feature_matrix.parquet")
        if not os.path.exists(path):
            logger.warning(f"  Feature matrix not found for {asset} — skipping.")
            continue
        logger.info(f"\n  --- Asset: {asset.upper()} ---")
        try:
            res = run_regime_analysis(path)
            regime_results[asset] = res
        except Exception as e:
            logger.error(f"  Regime analysis failed for {asset}: {e}")

    return regime_results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Economic Significance Backtest  (Paper Table 6)
# ─────────────────────────────────────────────────────────────────────────────

def step_backtest(assets: list, har_results: dict) -> dict:
    """Vol-targeting backtest comparing all model forecasts."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5 — Economic Significance (Vol-Timing Backtest)  [Paper Table 6]")
    logger.info("=" * 70)
    from backtest.vol_targeting import run_vol_timing_backtest

    backtest_results = {}
    for asset in assets:
        if asset not in har_results:
            continue

        # Load feature matrix for returns
        matrix_path = os.path.join(PROCESSED_DIR, f"{asset}_feature_matrix.parquet")
        try:
            fm = pd.read_parquet(matrix_path).dropna(subset=["Target_RV", "log_RV"])
        except Exception as e:
            logger.error(f"  Cannot load feature matrix for {asset}: {e}")
            continue

        # Attempt to load daily returns from OHLCV
        ohlcv_map = {
            "btc":   os.path.join(DATA_RAW_DIR, "ohlcv", "binance_btc_1d.parquet"),
            "spx":   os.path.join(DATA_RAW_DIR, "ohlcv", "spx_1d.parquet"),
            "nifty": os.path.join(DATA_RAW_DIR, "ohlcv", "nifty_1d.parquet"),
        }
        ret_path = ohlcv_map.get(asset, "")
        if os.path.exists(ret_path):
            ohlcv = pd.read_parquet(ret_path)
            if ohlcv.index.tz is None and fm.index.tz is not None:
                ohlcv.index = ohlcv.index.tz_localize("UTC")
            ohlcv["ret"] = np.log(ohlcv["close"] / ohlcv["close"].shift(1))
            fm = fm.join(ohlcv[["ret"]], how="left").fillna({"ret": 0.0})
            returns = fm["ret"].values
        else:
            logger.warning(f"  OHLCV not found for {asset} — using zero returns placeholder.")
            returns = np.zeros(len(fm))

        # Build forecast dict (align OOS periods)
        res   = har_results[asset]
        n_oos = len(res["HAR"]["actuals"])
        forecasts_dict = {
            "HAR":   res["HAR"]["forecasts"][:n_oos],
            "HAR-S": res["HAR-S"]["forecasts"][:n_oos],
        }
        nn_path = os.path.join(PROCESSED_DIR, f"forecasts_neural_har_{asset}.npy")
        if os.path.exists(nn_path):
            forecasts_dict["Neural-HAR"] = np.load(nn_path)[:n_oos]

        returns_oos = returns[-n_oos:]

        logger.info(f"\n  Asset: {asset.upper()}  |  OOS N={n_oos}")
        try:
            bt_df, _ = run_vol_timing_backtest(
                returns_oos, forecasts_dict, asset=asset,
                target_vol=0.02, kelly_fraction=0.25,
                slippage_bps=5.0, max_leverage=5.0,
            )
            backtest_results[asset] = bt_df
        except Exception as e:
            logger.error(f"  Backtest failed for {asset}: {e}")

    return backtest_results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Consolidated Paper Tables Export
# ─────────────────────────────────────────────────────────────────────────────

def export_paper_tables(har_results: dict, dm_results: dict,
                         mcs_results: dict, regime_results: dict,
                         backtest_results: dict, out_dir: str):
    """Export all tables as CSV for LaTeX/Excel import."""
    os.makedirs(out_dir, exist_ok=True)

    logger.info("\n" + "=" * 70)
    logger.info(f"STEP 6 — Exporting Paper Tables -> {out_dir}")
    logger.info("=" * 70)

    # ── Table 1: HAR Descriptive Stats ────────────────────────────────────
    desc_rows = []
    for asset, res in har_results.items():
        m = res["HAR"]["metrics"]
        ms = res["HAR-S"]["metrics"]
        desc_rows.append({
            "Asset":       asset.upper(),
            "HAR_QLIKE":  m["QLIKE"],  "HAR_MSE":  m["MSE"],  "HAR_MAE":  m["MAE"],
            "HARS_QLIKE": ms["QLIKE"], "HARS_MSE": ms["MSE"], "HARS_MAE": ms["MAE"],
            "N_OOS":      m["N"],
        })
    if desc_rows:
        t1 = pd.DataFrame(desc_rows).set_index("Asset")
        t1.to_csv(os.path.join(out_dir, "table1_oos_metrics.csv"))
        logger.info(f"  Saved table1_oos_metrics.csv")
        print("\n  [TABLE 1] OOS Loss Metrics — HAR vs HAR-S")
        print(t1.to_string())

    # ── Table 2: DM Tests ─────────────────────────────────────────────────
    dm_frames = []
    for asset, df in dm_results.items():
        df2 = df.copy()
        df2.insert(0, "Asset", asset.upper())
        dm_frames.append(df2)
    if dm_frames:
        t2 = pd.concat(dm_frames)
        t2.to_csv(os.path.join(out_dir, "table2_dm_test.csv"))
        logger.info(f"  Saved table2_dm_test.csv")

    # ── Table 3: MCS ─────────────────────────────────────────────────────
    mcs_frames = []
    for asset, df in mcs_results.items():
        df2 = df.copy()
        df2.insert(0, "Asset", asset.upper())
        mcs_frames.append(df2)
    if mcs_frames:
        t3 = pd.concat(mcs_frames)
        t3.to_csv(os.path.join(out_dir, "table3_mcs.csv"))
        logger.info(f"  Saved table3_mcs.csv")

    # ── Table 4: Regime Coefficients ──────────────────────────────────────
    regime_coef_frames = []
    for asset, res in regime_results.items():
        if "coefficients" in res and not res["coefficients"].empty:
            df2 = res["coefficients"].copy().reset_index()
            df2.insert(0, "Asset", asset.upper())
            regime_coef_frames.append(df2)
    if regime_coef_frames:
        t4 = pd.concat(regime_coef_frames)
        t4.to_csv(os.path.join(out_dir, "table4_regime_coefficients.csv"), index=False)
        logger.info(f"  Saved table4_regime_coefficients.csv")

    # ── Table 5: Regime OOS Metrics ───────────────────────────────────────
    regime_oos_frames = []
    for asset, res in regime_results.items():
        if "oos_metrics" in res and not res["oos_metrics"].empty:
            df2 = res["oos_metrics"].copy().reset_index()
            df2.insert(0, "Asset", asset.upper())
            regime_oos_frames.append(df2)
    if regime_oos_frames:
        t5 = pd.concat(regime_oos_frames)
        t5.to_csv(os.path.join(out_dir, "table5_regime_oos.csv"), index=False)
        logger.info(f"  Saved table5_regime_oos.csv")

    # ── Table 6: Backtest Results ─────────────────────────────────────────
    bt_frames = []
    for asset, df in backtest_results.items():
        df2 = df.copy().reset_index()
        df2.insert(0, "Asset", asset.upper())
        bt_frames.append(df2)
    if bt_frames:
        t6 = pd.concat(bt_frames)
        t6.to_csv(os.path.join(out_dir, "table6_backtest.csv"), index=False)
        logger.info(f"  Saved table6_backtest.csv")
        print("\n  [TABLE 6] Economic Significance — Vol-Timing Backtest")
        print(t6.to_string(index=False))

    logger.info(f"\n  All tables saved -> {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# MASTER RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=("Master runner for: 'Forecasting Realized Volatility in Markets: "
                     "Does Sentiment Add Information Beyond the HAR Framework?'")
    )
    parser.add_argument("--asset",      type=str, default=None,
                        choices=["btc", "spx", "nifty"],
                        help="Single asset to run (default: all)")
    parser.add_argument("--step",       type=str, default="all",
                        choices=["all", "build", "har", "neural", "dm", "mcs", "regime", "backtest"],
                        help="Pipeline step to run")
    parser.add_argument("--skip_build", action="store_true",
                        help="Skip feature matrix build (use cached parquet)")
    parser.add_argument("--mcs_B",      type=int, default=1000,
                        help="MCS bootstrap replications (default 1000)")
    parser.add_argument("--mcs_alpha",  type=float, default=0.10,
                        help="MCS significance level (default 0.10)")
    parser.add_argument("--out_dir",    type=str,
                        default=os.path.join(ROOT, "data", "paper_tables"),
                        help="Output directory for paper table CSVs")
    args = parser.parse_args()

    assets = [args.asset] if args.asset else ASSETS

    logger.info("\n" + "=" * 70)
    logger.info("  Forecasting Realized Volatility in Markets:")
    logger.info("  Does Sentiment Add Information Beyond the HAR Framework?")
    logger.info(f"  Assets: {[a.upper() for a in assets]}  |  Step: {args.step}")
    logger.info("=" * 70)

    har_results     = {}
    dm_results      = {}
    mcs_results     = {}
    regime_results  = {}
    backtest_results = {}

    run_all = (args.step == "all")

    # ── Step 0: Build ──────────────────────────────────────────────────
    if not args.skip_build and (run_all or args.step == "build"):
        step_build(assets)

    # ── Step 1: HAR ────────────────────────────────────────────────────
    if run_all or args.step == "har":
        har_results = step_har(assets)

    # ── Step 2: Neural-HAR ─────────────────────────────────────────────
    if run_all or args.step == "neural":
        step_neural(assets)

    # ── Step 3: DM ─────────────────────────────────────────────────────
    if run_all or args.step == "dm":
        if not har_results:
            har_results = step_har(assets)
        dm_results = step_dm(assets, har_results)

    # ── Step 3: MCS ────────────────────────────────────────────────────
    if run_all or args.step == "mcs":
        if not har_results:
            har_results = step_har(assets)
        mcs_results = step_mcs(assets, har_results,
                                alpha=args.mcs_alpha, B=args.mcs_B)

    # ── Step 4: Regime ─────────────────────────────────────────────────
    if run_all or args.step == "regime":
        regime_results = step_regime(assets)

    # ── Step 5: Backtest ───────────────────────────────────────────────
    if run_all or args.step == "backtest":
        if not har_results:
            har_results = step_har(assets)
        backtest_results = step_backtest(assets, har_results)

    # ── Step 6: Export Tables ──────────────────────────────────────────
    if run_all:
        export_paper_tables(har_results, dm_results, mcs_results,
                            regime_results, backtest_results, args.out_dir)

    logger.info("\n  Pipeline complete.")


if __name__ == "__main__":
    main()
