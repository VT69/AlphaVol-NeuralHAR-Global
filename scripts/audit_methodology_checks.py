"""Step 1 audit: verify training-loop behaviour and every metric against hand cases / references.

Checks
------
1. Learning rate in force during walk-forward fine-tuning (train_neural_har._train_single_arch).
2. Torch QLIKE clamp at [-10, 10]: share of predictions/targets below -10 and the gradient there.
3. QLIKE and MSE implementations vs a hand-computed case (all numpy/torch copies in the repo).
4. Diebold–Mariano (eval/dm_test.py) vs a reference (OLS t-statistic of d on a constant,
   HAC/rectangular for h>1) and the HLN factor; p-value distribution used.
5. MCS (eval/mcs_test.py) vs arch.bootstrap.MCS: agreement on synthetic data, size under
   equal predictive ability (Monte-Carlo), power when one model is worse, and on the
   committed HAR vs HAR-S losses.
6. Neural-HAR test rows vs HAR forecast rows: does end-alignment pair the same dates?
7. Reported-variant selection: committed results of the three Neural-HAR variants.

Deterministic (all RNGs seeded). Output: analysis/artifacts/methodology_checks.json
Usage: python scripts/audit_methodology_checks.py   (~3-6 min, dominated by the MCS Monte-Carlo)
"""
from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from arch.bootstrap import MCS
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.dm_test import dm_test, oos_r_squared  # noqa: E402
from eval.mcs_test import mcs_test  # noqa: E402
from src.models.baseline_har import qlike_loss_numpy  # noqa: E402
from src.models.neural_har import qlike_loss as torch_qlike  # noqa: E402
from src.models.regime_har import qlike_numpy  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = ROOT / "analysis" / "artifacts" / "methodology_checks.json"
SEED = 0
log = logging.getLogger("methodology_checks")


def check_lr_schedule() -> dict:
    """Replicates _train_single_arch: Adam + CosineAnnealingLR(T_max=50, eta_min=1e-6) stepped 50x,
    then the same optimizer is reused for walk-forward fine-tuning without a new scheduler."""
    out = {}
    for lr0 in (1e-4, 1e-3, 5e-3):  # Optuna search range is [1e-4, 5e-3]
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.Adam([p], lr=lr0)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-6)
        for _ in range(50):
            opt.step()
            sch.step()
        out[f"lr0={lr0:g}"] = {"lr_during_walk_forward_finetune": opt.param_groups[0]["lr"],
                               "ratio_to_initial": opt.param_groups[0]["lr"] / lr0}
    return out


def check_clamp() -> dict:
    out = {}
    for a in ("btc", "spx", "nifty"):
        fm = pd.read_parquet(PROC / f"{a}_feature_matrix.parquet").dropna(subset=["RV_d", "RV_w", "RV_m", "Target_RV"])
        har = np.load(PROC / f"forecasts_har_{a}.npy")
        out[a] = {"share_HAR_forecasts_below_-10": float((har < -10).mean()),
                  "share_targets_below_-10": float((fm["Target_RV"] < -10).mean())}
    # gradient wrt prediction when pred != actual (QLIKE gradient is 0 only at pred == actual)
    pred = torch.tensor([-11.0, -9.0], requires_grad=True)
    loss = torch_qlike(pred, torch.tensor([-12.0, -12.0]))
    loss.backward()
    out["grad_wrt_pred_at_[-11,-9]_actual_-12"] = pred.grad.tolist()
    out["note"] = "pred=-11 lies below the clamp: zero gradient; pred=-9 is inside: non-zero gradient"
    return out


def check_qlike_mse() -> dict:
    a_log, p_log = np.array([math.log(2.0)]), np.array([0.0])        # actual var 2, predicted var 1
    hand = 2.0 - math.log(2.0) - 1.0                                   # 0.306853
    from eval.dm_test import _loss_differential
    return {
        "hand_value": hand,
        "baseline_har.qlike_loss_numpy": qlike_loss_numpy(p_log, a_log),
        "regime_har.qlike_numpy": qlike_numpy(p_log, a_log),
        "dm_test._loss_differential(A=pred,B=actual)": float(_loss_differential(a_log, p_log, a_log, "QLIKE")[0]),
        "neural_har.qlike_loss(torch)": float(torch_qlike(torch.tensor(p_log), torch.tensor(a_log))),
        "mse_hand(log2-0)^2": math.log(2.0) ** 2,
        "dm_test MSE differential": float(_loss_differential(a_log, p_log, a_log, "MSE")[0]),
        "qlike_minimised_at_pred_equal_actual": float(qlike_loss_numpy(a_log, a_log)),
    }


def check_dm() -> dict:
    rng = np.random.default_rng(SEED)
    T = 300
    y = rng.normal(size=T)
    fa, fb = y + rng.normal(0, 1.0, T), y + rng.normal(0, 0.9, T)
    res = dm_test(y, fa, fb, h=1, loss="MSE")
    d = (y - fa) ** 2 - (y - fb) ** 2
    t_ols = sm.OLS(d, np.ones(T)).fit().tvalues[0]
    hln = math.sqrt((T + 1 - 2 * 1 + 0) / T)
    ref = t_ols * hln
    # h = 2 reference: rectangular kernel with 1 autocovariance, 1/T normalisation (DM 1995)
    res2 = dm_test(y, fa, fb, h=2, loss="MSE")
    dc = d - d.mean()
    g0, g1 = (dc @ dc) / T, (dc[1:] @ dc[:-1]) / T
    ref2 = d.mean() / math.sqrt((g0 + 2 * g1) / T) * math.sqrt((T + 1 - 4 + 2 / T) / T)
    hac = sm.OLS(d, np.ones(T)).fit(cov_type="HAC", cov_kwds={"maxlags": 1, "kernel": "uniform"}).tvalues[0]
    same = dm_test(y, fa, fa, h=1, loss="MSE")
    return {
        "h1_repo_DM": res["DM_stat"], "h1_reference_OLS_t_x_HLN": ref, "h1_abs_diff": abs(res["DM_stat"] - ref),
        "h1_repo_p_normal": res["p_value"],
        "h1_p_student_t(T-1)_as_HLN_recommend": float(2 * stats.t.sf(abs(ref), T - 1)),
        "h2_repo_DM": res2["DM_stat"], "h2_reference_rectangular_1/T": ref2,
        "h2_statsmodels_uniform_HAC_t": float(hac),
        "identical_forecasts_DM": same["DM_stat"], "identical_forecasts_p": same["p_value"],
        "sign_convention": "positive DM => model B (challenger) has lower loss",
        "oos_r2_check": {"repo": oos_r_squared(y, fa, fb),
                         "hand": 1 - np.sum((y - fb) ** 2) / np.sum((y - fa) ** 2)},
    }


def mcs_both(L: pd.DataFrame, alpha: float = 0.10, B: int = 500, seed: int = 0) -> dict:
    repo = mcs_test(L, alpha=alpha, B=B, block_size=5, stat="TR", seed=seed)
    ref = MCS(L, size=alpha, reps=B, block_size=5, method="R", bootstrap="circular", seed=seed)
    ref.compute()
    return {"repo_included": sorted(repo.loc[repo["in_MCS"], "model"].tolist()),
            "arch_included": sorted(ref.included),
            "repo_pvalues": dict(zip(repo["model"], repo["MCS_pvalue"].round(4))),
            "arch_pvalues": {k: round(float(v), 4) for k, v in ref.pvalues["Pvalue"].items()}}


def check_mcs() -> dict:
    rng = np.random.default_rng(SEED)
    out: dict = {}
    T = 300
    base = np.abs(rng.normal(0.5, 0.3, T))
    L = pd.DataFrame({"worst": base + np.abs(rng.normal(0.30, 0.1, T)),
                      "mid": base + np.abs(rng.normal(0.10, 0.1, T)),
                      "best": base + np.abs(rng.normal(0.02, 0.1, T))})
    out["clear_ordering"] = mcs_both(L)
    # Size: 3 models with equal expected loss, correlated, AR(1) dependence. Share of replications
    # in which any model is eliminated should be close to alpha = 0.10.
    n_rep, B = 400, 300
    elim_repo = elim_arch = 0
    t0 = time.time()
    for r in range(n_rep):
        rr = np.random.default_rng(1000 + r)
        e = rr.normal(size=(T, 3))
        for t in range(1, T):
            e[t] = 0.3 * e[t - 1] + e[t]
        common = np.abs(rr.normal(1, 0.5, T))[:, None]
        Ls = pd.DataFrame(common + 0.2 * e, columns=["a", "b", "c"])
        m = mcs_both(Ls, B=B, seed=r)
        elim_repo += len(m["repo_included"]) < 3
        elim_arch += len(m["arch_included"]) < 3
    out["size_equal_models"] = {"replications": n_rep, "B": B, "nominal_alpha": 0.10,
                                "repo_rejection_rate": elim_repo / n_rep,
                                "arch_rejection_rate": elim_arch / n_rep,
                                "mc_se_at_0.10": math.sqrt(0.1 * 0.9 / n_rep),
                                "seconds": round(time.time() - t0, 1)}
    # Committed HAR vs HAR-S QLIKE losses on the paper's trailing windows
    win = {"btc": 275, "spx": 287, "nifty": 276}
    out["committed_HAR_vs_HARS"] = {}
    for a, n in win.items():
        fm = pd.read_parquet(PROC / f"{a}_feature_matrix.parquet").dropna(subset=["RV_d", "RV_w", "RV_m", "Target_RV"])
        y = fm["Target_RV"].to_numpy()
        har, hars = np.load(PROC / f"forecasts_har_{a}.npy"), np.load(PROC / f"forecasts_harss_{a}.npy")
        def q(p: np.ndarray, act: np.ndarray) -> np.ndarray:
            av, pv = np.exp(np.clip(act, -15, 15)), np.exp(np.clip(p, -15, 15))
            return av / pv - np.log(av / pv) - 1
        L = pd.DataFrame({"HAR": q(har[-n:], y[-n:]), "HAR-S": q(hars[-n:], y[-n:])})
        out["committed_HAR_vs_HARS"][a] = mcs_both(L, B=1000)
    return out


def check_alignment() -> dict:
    out = {}
    for a in ("btc", "spx", "nifty"):
        fm = pd.read_parquet(PROC / f"{a}_feature_matrix.parquet").sort_index()
        # baseline_har.run_har_and_hars preparation
        har_df = fm.dropna(subset=["RV_d", "RV_w", "RV_m", "Target_RV"])
        har_dates = har_df.index[252:]                       # one forecast per t >= min_train
        # train_neural_har.train_asset preparation
        nn_df = fm.dropna(subset=["RV_d", "RV_w", "RV_m", "Target_RV"])
        n = len(nn_df)
        nn_dates = nn_df.index[int(n * 0.8) + int(n * 0.1):]
        m = len(nn_dates)
        out[a] = {"n_test": m, "end_aligned_dates_identical": bool(har_dates[-m:].equals(nn_dates))}
    return out


def check_variants() -> dict:
    out = {}
    for v in ("deep_grn_results", "transformer_results", "ensemble_results"):
        t = pd.read_csv(ROOT / "data" / v / "paper_tables" / "table2_dm_test.csv")
        t = t[t["Model"] == "Neural-HAR"].set_index("Asset")
        out[v] = {a: {"QLIKE_DM": float(t.loc[a, "QLIKE_DM"]), "QLIKE_p": float(t.loc[a, "QLIKE_p"]),
                      "MSE_DM": float(t.loc[a, "MSE_DM"])} for a in t.index}
    canonical = pd.read_csv(ROOT / "data" / "paper_tables" / "table2_dm_test.csv")
    canonical = canonical[canonical["Model"] == "Neural-HAR"].set_index("Asset")
    out["paper_reports_variant"] = [v for v in ("deep_grn_results", "transformer_results", "ensemble_results")
                                    if all(abs(out[v][a]["QLIKE_DM"] - canonical.loc[a, "QLIKE_DM"]) < 1e-9
                                           for a in canonical.index)]
    out["git_history"] = subprocess.run(
        ["git", "log", "--format=%h %ad %s", "--date=iso", "--", "src/models/train_neural_har.py", "src/models/neural_har.py"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(SEED)
    res = {"lr_schedule": check_lr_schedule(), "qlike_clamp": check_clamp(), "qlike_mse": check_qlike_mse(),
           "dm": check_dm(), "alignment": check_alignment(), "variants": check_variants()}
    log.info("running MCS checks (Monte-Carlo)...")
    res["mcs"] = check_mcs()
    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
