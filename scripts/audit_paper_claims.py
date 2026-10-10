"""Audit: numeric checks behind analysis/paper_vs_code.md.

1. Which alignment produced the committed data/paper_tables/table5_mcs.csv HAR mean loss?
   (tests end-aligned vs front-aligned candidates against the committed value)
2. Can turnover x slippage explain the Sharpe/return gap claimed in Sec. 7?
3. Does HAR-S "improve in-sample R^2" (Sec. 5) per the committed coefficient table?
4. Unresolved references in the submitted paper/main.pdf (requires `pdftotext`).

Inputs: committed CSVs, data/processed/*.npy and feature matrices, paper/main.pdf.
Output: analysis/artifacts/paper_claims_check.json
Usage : python scripts/audit_paper_claims.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TAB = ROOT / "data" / "paper_tables"
OUT = ROOT / "analysis" / "artifacts" / "paper_claims_check.json"
WINDOW = {"btc": 275, "spx": 287, "nifty": 276}
LABEL = {"btc": "BTC", "spx": "SPX", "nifty": "NIFTY50"}


def qlike(pred: np.ndarray, act: np.ndarray) -> float:
    a, p = np.exp(np.clip(act, -15, 15)), np.exp(np.clip(pred, -15, 15))
    return float(np.mean(a / p - np.log(a / p) - 1))


def main() -> None:
    out: dict = {}
    t5 = pd.read_csv(TAB / "table5_mcs.csv")
    out["table5_mcs_HAR_alignment"] = {}
    for a, n in WINDOW.items():
        har = np.load(PROC / f"forecasts_har_{a}.npy")
        act_file = np.load(PROC / f"actuals_{a}.npy")  # = HAR-S actuals (overwritten last in loop)
        fm = pd.read_parquet(PROC / f"{a}_feature_matrix.parquet").sort_index()
        fm = fm.dropna(subset=["RV_d", "RV_w", "RV_m", "Target_RV"])
        har_act = fm["Target_RV"].to_numpy()[-len(har):]
        committed = float(t5.loc[(t5.Asset == LABEL[a]) & (t5.model == "HAR"), "mean_loss"].iloc[0])
        cands = {
            "end_aligned_correct_actuals": qlike(har[-n:], har_act[-n:]),
            "front_aligned_HAR[:n]_vs_actuals_file[:n]": qlike(har[:n], act_file[:n]),
            "front_aligned_HAR[:n]_vs_HAR_actuals[:n]": qlike(har[:n], har_act[:n]),
            "HAR_tail_vs_actuals_file_tail": qlike(har[-n:], act_file[-n:]),
        }
        match = [k for k, v in cands.items() if abs(v - committed) < 1e-6]
        out["table5_mcs_HAR_alignment"][a] = {"committed": committed, "candidates": cands,
                                             "exact_match": match or None}

    t6 = pd.read_csv(TAB / "table6_backtest.csv")
    ann = {"BTC": 365, "SPX": 252, "NIFTY": 252}
    t6["annual_cost_pct"] = t6["Turnover"] * 5e-4 * t6["Asset"].map(ann) * 100
    rows = {}
    for asset, g in t6.groupby("Asset"):
        g = g.set_index("Model")
        rows[asset] = {
            "annual_cost_pct": g["annual_cost_pct"].round(3).to_dict(),
            "NeuralHAR_minus_HAR_cost_pp": round(float(g.loc["Neural-HAR", "annual_cost_pct"] - g.loc["HAR", "annual_cost_pct"]), 3),
            "NeuralHAR_minus_HAR_ann_return_pp": round(float(g.loc["Neural-HAR", "Ann_Return"] - g.loc["HAR", "Ann_Return"]), 3),
        }
    out["turnover_cost_attribution"] = rows

    t2 = pd.read_csv(TAB / "table2_insample_coefs.csv")
    r2 = t2.groupby(["Asset", "Model"]).agg(R2=("R²", "first"), N=("N", "first")).reset_index()
    out["insample_R2"] = {f"{r.Asset}|{r.Model}": {"R2": float(r.R2), "N": int(r.N)} for r in r2.itertuples()}

    if shutil.which("pdftotext"):
        txt = subprocess.run(["pdftotext", "-layout", str(ROOT / "paper" / "main.pdf"), "-"],
                             capture_output=True, text=True, check=True).stdout
        out["main_pdf_unresolved_refs"] = [l.strip() for l in txt.splitlines() if "??" in l]
    else:
        out["main_pdf_unresolved_refs"] = "NOT COMPUTED - pdftotext unavailable"
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
