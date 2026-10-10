"""Quantify the quality / predictability of the realized-volatility target per asset.

For BTC (5-min RV), SPX and NIFTY (daily-bar proxy) this computes, on the target series
actually used by the models (feature-matrix ``log_RV``) and on alternative estimators
available in the committed raw files:

* sample ACF of log RV at lags 1..22 and Ljung-Box Q at lags 1, 5, 10, 22;
* the Yule-Walker ceiling on linear R^2 (population R^2 of the best linear predictor
  of y_{t+1} from p past values, computed from the sample ACF): for the 1-step
  information set {y_t..y_{t-21}} and for the set the code actually uses {y_{t-1}..y_{t-22}};
* a signal-share bound: fitting rho_obs(k) = lambda * phi^k (k = 1..22) treats the observed
  series as signal + white measurement noise; lambda = Var(signal)/Var(observed) caps the R^2
  any forecaster can attain against the noisy target;
* in-sample OLS R^2 of HAR as implemented (2-step: lags start at t-1), correctly specified
  1-step HAR, and unrestricted AR(22);
* identification of which estimator ``log_RV`` is, and OHLC data-quality counts.

Deterministic (no randomness). Outputs:
  analysis/artifacts/target_quality.json
  analysis/tables/T_target_quality.{csv,tex}, analysis/tables/T_acf_lags.{csv,tex}
  analysis/figures/target_acf.{png,svg}
Usage: python scripts/target_quality.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: E402
from statsmodels.tsa.stattools import acf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_collection" / "data" / "raw"
PROC = ROOT / "data" / "processed"
ART, TAB, FIG = (ROOT / "analysis" / d for d in ("artifacts", "tables", "figures"))
NLAGS = 22
log = logging.getLogger("target_quality")


def yule_walker_r2(rho: np.ndarray, lags: list[int]) -> float:
    """Population R^2 of the best linear predictor of y_{t+1} from {y_{t+1-k}: k in lags}.

    rho[k] is the autocorrelation at lag k (rho[0] = 1). Regressor correlation matrix
    R_ij = rho[|k_i - k_j|]; target-regressor vector r_i = rho[k_i]. R^2 = r' R^-1 r.
    """
    k = np.asarray(lags)
    R = rho[np.abs(k[:, None] - k[None, :])]
    r = rho[k]
    return float(r @ np.linalg.solve(R, r))


def signal_share(rho: np.ndarray) -> tuple[float, float]:
    """Fit rho(k) = lam * phi^k for k = 1..NLAGS; return (lam, phi) with 0<lam<=1, 0<phi<1."""
    k = np.arange(1, NLAGS + 1)
    res = least_squares(lambda p: p[0] * p[1] ** k - rho[1:NLAGS + 1], x0=[0.5, 0.9],
                        bounds=([1e-6, 1e-6], [1.0, 0.9999]))
    return float(res.x[0]), float(res.x[1])


def ols_r2(y: pd.Series, X: pd.DataFrame) -> float:
    d = pd.concat([y, X], axis=1).dropna()
    return float(sm.OLS(d.iloc[:, 0], sm.add_constant(d.iloc[:, 1:])).fit().rsquared)


def har_design(x: pd.Series, start_lag: int) -> pd.DataFrame:
    """HAR regressors whose most recent lag is x shifted by ``start_lag``."""
    s = x.shift(start_lag)
    return pd.DataFrame({"d": s, "w": s.rolling(5).mean(), "m": s.rolling(22).mean()})


def analyse(name: str, y: pd.Series) -> dict:
    """y: log-variance series on its own (trading-day) calendar, target = y.shift(-1)."""
    y = y.replace([np.inf, -np.inf], np.nan).dropna()
    rho = acf(y.to_numpy(), nlags=NLAGS + 1, fft=False)
    lb = acorr_ljungbox(y, lags=[1, 5, 10, 22])
    lam, phi = signal_share(rho)
    tgt = y.shift(-1)
    ar22 = pd.concat({f"l{k}": y.shift(k) for k in range(0, 22)}, axis=1)
    return {
        "series": name, "n": int(len(y)), "mean": float(y.mean()), "std": float(y.std()),
        "acf": {int(k): float(rho[k]) for k in range(1, NLAGS + 1)},
        "ljung_box": {int(l): {"Q": float(q), "p": float(p)} for l, q, p in
                      zip(lb.index, lb["lb_stat"], lb["lb_pvalue"])},
        "yw_ceiling_R2_1step_lags_t..t-21": yule_walker_r2(rho, list(range(1, 23))),
        "yw_ceiling_R2_as_coded_lags_t-1..t-22": yule_walker_r2(rho, list(range(2, 24))),
        "signal_share_lambda": lam, "signal_decay_phi": phi,
        "ols_R2_HAR_as_coded_2step": ols_r2(tgt, har_design(y, 1)),
        "ols_R2_HAR_correct_1step": ols_r2(tgt, har_design(y, 0)),
        "ols_R2_AR22_1step": ols_r2(tgt, ar22),
        # Only meaningful when y = log(r_t^2): log r^2 = log sigma^2 + log eps^2 and, for Gaussian
        # eps, Var(log eps^2) = pi^2/2. 1 - (pi^2/2)/Var(y) bounds the predictable variance share.
        "logchi2_signal_share_bound": float(1 - (np.pi ** 2 / 2) / y.var()),
        "frac_below_mean_minus_3sd": float((y < y.mean() - 3 * y.std()).mean()),
        "min": float(y.min()),
    }


def ohlc_quality(asset: str) -> dict:
    o = pd.read_parquet(RAW / "ohlcv" / f"{asset}_1d.parquet").sort_index()
    gk = 0.5 * np.log(o.high / o.low) ** 2 - (2 * np.log(2) - 1) * np.log(o.close / o.open) ** 2
    gaps = o.index.to_series().diff().dt.days
    return {
        "rows": int(len(o)), "high_eq_low": int((o.high == o.low).sum()),
        "open_eq_close": int((o.open == o.close).sum()),
        "open_eq_prev_close": int((o.open == o.close.shift(1)).sum()),
        "open_outside_high_low": int(((o.open > o.high) | (o.open < o.low)).sum()),
        "close_outside_high_low": int(((o.close > o.high) | (o.close < o.low)).sum()),
        "gk_le_0": int((gk <= 0).sum()), "gk_lt_1e-8": int((gk < 1e-8).sum()),
        "calendar_gaps_gt_5d": {str(d.date()): int(g) for d, g in gaps[gaps > 5].items()},
        "zero_volume_days": int((o.volume == 0).sum()),
    }


def identify_log_rv(asset: str) -> dict:
    """Which estimator is the committed log_RV? Compare with log of candidate columns."""
    r = pd.read_parquet(RAW / "realized_vol" / f"{asset}_rv_daily.parquet")
    out = {"columns": list(r.columns)}
    for c in [c for c in ["RV", "GK", "PK", "rv", "RV_cc"] if c in r]:
        with np.errstate(divide="ignore"):
            lc = np.log(r[c].where(r[c] > 0))
        m = r["log_RV"].notna() & lc.notna()
        out[f"max_abs(log_RV - log({c}))"] = float(np.abs(r.loc[m, "log_RV"] - lc[m]).max())
    if "RV" in r and "log_ret" in r:
        out["max_abs(RV - log_ret^2)"] = float(np.nanmax(np.abs(r["RV"] - r["log_ret"] ** 2)))
    if "RV" in r and "GK" in r:
        out["max_abs(RV - GK)"] = float(np.nanmax(np.abs(r["RV"] - r["GK"])))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for d in (ART, TAB, FIG):
        d.mkdir(parents=True, exist_ok=True)
    res: dict = {"series": {}, "ohlc_quality": {}, "log_rv_identity": {}}
    series = {}
    for a in ["btc", "spx", "nifty"]:
        fm = pd.read_parquet(PROC / f"{a}_feature_matrix.parquet").sort_index()
        series[f"{a.upper()} target (feature-matrix log_RV)"] = fm["log_RV"]
        r = pd.read_parquet(RAW / "realized_vol" / f"{a}_rv_daily.parquet").sort_index()
        with np.errstate(divide="ignore"):
            series[f"{a.upper()} log GK (daily bar)"] = np.log(r["GK"].where(r["GK"] > 0))
            series[f"{a.upper()} log Parkinson (daily bar)"] = np.log(r["PK"].where(r["PK"] > 0))
        res["log_rv_identity"][a] = identify_log_rv(a)
        if a != "btc":
            res["ohlc_quality"][a] = ohlc_quality(a)
    for name, y in series.items():
        res["series"][name] = analyse(name, y)
        log.info("%-45s AC1=%.3f lambda=%.3f YW1=%.3f HARcoded=%.3f HARok=%.3f", name,
                 res["series"][name]["acf"][1], res["series"][name]["signal_share_lambda"],
                 res["series"][name]["yw_ceiling_R2_1step_lags_t..t-21"],
                 res["series"][name]["ols_R2_HAR_as_coded_2step"], res["series"][name]["ols_R2_HAR_correct_1step"])
    (ART / "target_quality.json").write_text(json.dumps(res, indent=2))

    rows = []
    for name, s in res["series"].items():
        rows.append({"Series": name, "N": s["n"], "AC(1)": s["acf"][1], "AC(5)": s["acf"][5],
                     "AC(22)": s["acf"][22], "LB(22) Q": s["ljung_box"][22]["Q"],
                     "LB(22) p": s["ljung_box"][22]["p"],
                     "Signal share lambda": s["signal_share_lambda"],
                     "YW R2 ceiling (1-step)": s["yw_ceiling_R2_1step_lags_t..t-21"],
                     "YW R2 ceiling (as coded)": s["yw_ceiling_R2_as_coded_lags_t-1..t-22"],
                     "R2 HAR as coded": s["ols_R2_HAR_as_coded_2step"],
                     "R2 HAR 1-step": s["ols_R2_HAR_correct_1step"], "R2 AR(22)": s["ols_R2_AR22_1step"]})
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "T_target_quality.csv", index=False, float_format="%.4f")
    (TAB / "T_target_quality.tex").write_text(t.to_latex(index=False, float_format="%.3f", escape=True,
                                                         caption="Persistence and predictability ceilings of the log-RV target.",
                                                         label="tab:target_quality"))
    acf_t = pd.DataFrame({n: s["acf"] for n, s in res["series"].items()})
    acf_t.index.name = "lag"
    acf_t.to_csv(TAB / "T_acf_lags.csv", float_format="%.4f")
    (TAB / "T_acf_lags.tex").write_text(acf_t.T.to_latex(float_format="%.2f", escape=True,
                                                          caption="Sample ACF of daily log variance, lags 1--22.",
                                                          label="tab:acf_lags"))

    # Figure: ACF for model targets, plus BTC daily-bar GK as the same-asset control
    colors = {"BTC": "#0072B2", "SPX": "#D55E00", "NIFTY": "#009E73"}  # Okabe-Ito
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    k = np.arange(1, NLAGS + 1)
    for name, s in res["series"].items():
        a = name.split()[0]
        if "target" in name:
            ax.plot(k, [s["acf"][i] for i in k], "-o", ms=3, color=colors[a], label=f"{a} model target")
        elif "log GK" in name:
            ax.plot(k, [s["acf"][i] for i in k], "--s", ms=3, color=colors[a], alpha=0.8,
                    label=f"{a} daily-bar Garman-Klass (unused alternative)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Lag (trading days)")
    ax.set_ylabel("Autocorrelation of log variance")
    ax.set_xticks([1, 5, 10, 15, 22])
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")
    ax.set_ylim(-0.02, 0.85)
    ax.set_title("Persistence of the volatility target: solid = used by models, dashed = Garman-Klass", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"target_acf.{ext}", dpi=300)
    log.info("wrote target_quality.json, tables, figures")


if __name__ == "__main__":
    main()
