"""Phase-1 audit: is the Neural-HAR QLIKE advantage explainable by HAR's log-space bias?

HAR is fit by OLS on log(RV); exp(forecast) estimates the conditional *median* of RV, not
the mean, and QLIKE is minimised by the conditional mean. This script applies a purely
causal smearing correction (Duan 1983) to the committed HAR forecasts:
    f_t^bc = f_t + log(mean_{s < t-1} exp(e_s))
using only residuals whose targets were realised before the forecast is issued, then
reports QLIKE on the same trailing windows used in the paper's DM/MCS tables.

Inputs : data/processed/{asset}_feature_matrix.parquet, data/processed/forecasts_har_{asset}.npy
Output : analysis/artifacts/phase1_qlike_bias_check.json
Usage  : python scripts/audit_qlike_bias_check.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "analysis" / "artifacts" / "phase1_qlike_bias_check.json"
# Trailing-window lengths and Neural-HAR / HAR mean QLIKE as committed in data/paper_tables/table3_mcs.csv
PAPER = {"btc": (275, 0.31840294870141944, 0.28658503514621697),
         "spx": (287, 3.8965808980880268, 1.5669140022558876),
         "nifty": (276, 2.3304588500528824, 1.8698423491537381)}
log = logging.getLogger("qlike_bias")


def qlike(pred_log: np.ndarray, act_log: np.ndarray) -> np.ndarray:
    """Per-period QLIKE exactly as in eval/dm_test.py (clip ±15 on log scale)."""
    a = np.exp(np.clip(act_log, -15, 15))
    p = np.exp(np.clip(pred_log, -15, 15))
    return a / p - np.log(a / p) - 1


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = {}
    for asset, (n_win, har_paper, nn_paper) in PAPER.items():
        fm = pd.read_parquet(PROC / f"{asset}_feature_matrix.parquet").sort_index()
        fm = fm.dropna(subset=["RV_d", "RV_w", "RV_m", "Target_RV"])
        fc = np.load(PROC / f"forecasts_har_{asset}.npy")
        act = fm["Target_RV"].to_numpy()[-len(fc):]
        e = act - fc
        # residual for row s is known only after Target(s)=logRV(s+1) is realised; forecast for
        # row t is issued with info through t-1 (features are lagged), so use residuals s <= t-2.
        exp_e = np.exp(e)
        csum = np.concatenate([[0.0], np.cumsum(exp_e)])
        corr = np.full(len(fc), np.nan)
        for t in range(len(fc)):
            k = t - 1  # residuals with index < t-1
            if k >= 20:
                corr[t] = np.log(csum[k] / k)
        fc_bc = fc + np.nan_to_num(corr, nan=0.0)
        w = slice(len(fc) - n_win, len(fc))
        q_raw = float(qlike(fc[w], act[w]).mean())
        q_bc = float(qlike(fc_bc[w], act[w]).mean())
        res[asset] = {
            "window_N": n_win,
            "HAR_QLIKE_reproduced": q_raw,
            "HAR_QLIKE_paper_table3_mcs": har_paper,
            "HAR_smearing_bias_corrected_QLIKE": q_bc,
            "NeuralHAR_QLIKE_paper_table3_mcs": nn_paper,
            "mean_bias_correction_log_units": float(np.nanmean(corr[w])),
            "full_oos_HAR_QLIKE": float(qlike(fc, act).mean()),
            "full_oos_HAR_bc_QLIKE": float(qlike(fc_bc, act)[21:].mean()),
        }
        log.info("%s %s", asset, json.dumps(res[asset], indent=1))
    OUT.write_text(json.dumps(res, indent=2))
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
