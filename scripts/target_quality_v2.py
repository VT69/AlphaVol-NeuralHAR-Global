"""Step 5 (interim): target quality on the rebuilt development datasets, before vs after Step 2.

Before = legacy target/specification (from analysis/artifacts/target_quality.json, produced by
scripts/target_quality.py on the committed legacy files). After = data/processed_v2 datasets:
Garman–Klass for SPX/NIFTY, 5-min RV for BTC, HAR lags ending at t, log-of-mean windows.

Reports per asset: ACF(1, 5, 22) of the log-variance target, Ljung–Box(22), the Yule–Walker
linear-R² ceiling (22 lags), the signal-share λ, and in-sample OLS R² of the fixed HAR and of the
legacy 2-step information set on the *new* target. Descriptive, in-sample: no forecasting model
is evaluated. Output: analysis/artifacts/target_quality_v2.json, analysis/tables/T_target_quality_v2.{csv,tex}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from target_quality import signal_share, yule_walker_r2  # noqa: E402  (same estimators as Phase 2)

ART, TAB = ROOT / "analysis" / "artifacts", ROOT / "analysis" / "tables"


def r2(y: pd.Series, X: pd.DataFrame) -> float:
    d = pd.concat([y, X], axis=1).dropna()
    return float(sm.OLS(d.iloc[:, 0], sm.add_constant(d.iloc[:, 1:])).fit().rsquared)


def main() -> None:
    legacy = json.loads((ART / "target_quality.json").read_text())["series"]
    out, rows = {}, []
    for a in ("btc", "spx", "nifty"):
        df = pd.read_parquet(ROOT / "data" / "processed_v2" / f"{a}_dataset.parquet")
        y = df["RV_d"]                                    # log variance of day t on the dataset rows
        rho = acf(y.to_numpy(), nlags=23, fft=False)
        lb = acorr_ljungbox(y, lags=[22])
        lam, phi = signal_share(rho)
        X_fixed = df[["RV_d", "RV_w", "RV_m"]]
        X_legacy = X_fixed.shift(1)                         # legacy information set: lags end at t-1
        res = {"n": int(len(df)), "acf1": float(rho[1]), "acf5": float(rho[5]), "acf22": float(rho[22]),
               "ljung_box22_Q": float(lb["lb_stat"].iloc[0]), "ljung_box22_p": float(lb["lb_pvalue"].iloc[0]),
               "yw_ceiling_1step": yule_walker_r2(rho, list(range(1, 23))), "signal_share": lam,
               "insample_R2_HAR_fixed": r2(df["target"], X_fixed),
               "insample_R2_HAR_legacy_info_set": r2(df["target"], X_legacy)}
        leg = legacy[f"{a.upper()} target (feature-matrix log_RV)"]
        res["legacy"] = {"acf1": leg["acf"]["1"], "yw_ceiling_1step": leg["yw_ceiling_R2_1step_lags_t..t-21"],
                         "signal_share": leg["signal_share_lambda"], "insample_R2_HAR_as_coded": leg["ols_R2_HAR_as_coded_2step"]}
        out[a] = res
        rows.append({"Asset": a.upper(), "Target (before → after)": "5-min RV → 5-min RV" if a == "btc" else "log r² → Garman–Klass",
                     "AC(1) before": leg["acf"]["1"], "AC(1) after": res["acf1"],
                     "YW R² ceiling before": leg["yw_ceiling_R2_1step_lags_t..t-21"], "YW R² ceiling after": res["yw_ceiling_1step"],
                     "HAR R² before (legacy target+spec)": leg["ols_R2_HAR_as_coded_2step"],
                     "HAR R² after, legacy info set": res["insample_R2_HAR_legacy_info_set"],
                     "HAR R² after (fixed)": res["insample_R2_HAR_fixed"], "N after": res["n"]})
    (ART / "target_quality_v2.json").write_text(json.dumps(out, indent=2))
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "T_target_quality_v2.csv", index=False, float_format="%.4f")
    (TAB / "T_target_quality_v2.tex").write_text(t.style.format(precision=3).hide(axis="index").to_latex(
        hrules=True, caption="Target quality before and after the Step 2 fixes (in-sample, development data).",
        label="tab:target_quality_v2"))
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
