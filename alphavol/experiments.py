"""Pre-registered experiment runner (analysis/PREREGISTRATION.md, configs/experiments.yaml).

Pipeline per asset: derive per-row features → choose sample → tune (initial window only) → OOS
forecasts for every registered model (5 seeds for neural) → losses, DM tests with Holm, MCS,
bootstrap CIs, minimum detectable effects. Configurations not in the YAML are refused unless
tagged exploratory. Every run is appended to analysis/EXPERIMENT_LOG.md by the CLI.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from alphavol.config import REPO_ROOT
from alphavol.evaluation import run_oos
from alphavol.logging_utils import get_logger
from alphavol.metrics import dm_test, holm, mse_log, model_confidence_set, newey_west_lag, qlike
from alphavol.models.garch import GARCH
from alphavol.models.gbm import GBM
from alphavol.models.linear import HARQ, HistoricalMean, LinearLog, LinearQLIKE, RandomWalk
from alphavol.models.neural import NeuralHAR
from alphavol.seeding import seed_everything
from alphavol.tuning import tune

log = get_logger(__name__)
EXP_CONFIG = REPO_ROOT / "configs" / "experiments.yaml"


def load_grid(path: Path = EXP_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


# ── Per-row derived features (no estimated parameters) ──────────────────────────────────
def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "sent_n_articles" in d:
        d["log1p_sent_n_articles"] = np.log1p(d["sent_n_articles"])
    if "sent_gdelt_volume" in d:
        d["log1p_sent_gdelt_volume"] = np.log1p(d["sent_gdelt_volume"])
    for iv in ("vix", "india_vix"):
        if iv in d:
            d["iv_logvar"] = 2 * np.log(d[iv] / 100.0) - np.log(252.0)
    if "amihud" in d:
        d["log_amihud"] = np.log(d["amihud"].where(d["amihud"] > 0))
    return d


def resolve_features(expr: str, asset: str, grid: dict, columns: pd.Index) -> list[str]:
    """'HAR+SENT', 'ALL', 'ALL-SENT' → column list (only columns present; order preserved)."""
    sets = grid["feature_sets"]
    groups = grid["groups_by_asset"][asset]
    if expr == "ALL":
        names = ["HAR", "SENT", "ATTN", *groups]
    elif expr == "ALL-SENT":
        names = ["HAR", *groups]
    else:
        names = expr.split("+")
    cols = []
    for n in names:
        for c in sets[n]:
            if c in columns and c not in cols:
                cols.append(c)
    return cols


def sample_start(df: pd.DataFrame, common_start_year: int | None) -> pd.Timestamp:
    first = df.index.min() + pd.Timedelta(days=30)
    if common_start_year is None:
        return first
    return max(first, pd.Timestamp(f"{common_start_year}-01-01", tz="UTC"))


def oos_start(df: pd.DataFrame, start: pd.Timestamp, years: int) -> pd.Timestamp:
    return start + pd.DateOffset(years=years)


# ── Model factories ─────────────────────────────────────────────────────────────────────
def _refit_every(grid: dict, family: str, calendar: str) -> int:
    r = grid["refit_every"]
    key = {"linear": "linear", "linear_qlike": "har_qlike", "garch": "garch", "gbm": "gbm", "neural": "neural",
           "naive_rw": "linear", "naive_mean": "linear", "harq": "linear"}[family]
    v = r[key]
    return int(v if isinstance(v, int) else v[calendar])


def factory(spec: dict, feats: list[str], params: dict, seed: int):
    fam = spec["family"]
    if fam == "naive_rw":
        return lambda: RandomWalk()
    if fam == "naive_mean":
        return lambda: HistoricalMean()
    if fam == "linear":
        return lambda: LinearLog(feats, bias=spec.get("bias", "smearing"))
    if fam == "linear_qlike":
        return lambda: LinearQLIKE(feats)
    if fam == "harq":
        return lambda: HARQ()
    if fam == "garch":
        return lambda: GARCH()
    if fam == "gbm":
        return lambda: GBM(feats, loss=spec["loss"], params=params, seed=seed)
    if fam == "neural":
        return lambda: NeuralHAR(feats, arch=spec["arch"], loss=spec["loss"], seed=seed, **params)
    raise ValueError(f"unknown family {fam}")


def _tune_key(spec: dict) -> str | None:
    if spec["family"] == "gbm":
        return f"gbm-{spec['loss']}"
    if spec["family"] == "neural":
        return f"{spec['arch']}-{spec['loss']}"
    return None


def _search(trial, key: str, feats: list[str], max_epochs: int):
    if key.startswith("gbm"):
        params = {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                  "max_iter": trial.suggest_categorical("max_iter", [100, 200, 400]),
                  "max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes", [7, 15, 31]),
                  "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [20, 40, 80]),
                  "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True)}
        return GBM(feats, loss=key.split("-")[1], params=params, seed=0)
    arch, loss = key.split("-")
    return NeuralHAR(feats, arch=arch, loss=loss, seed=0, epochs=max_epochs,
                     hidden=trial.suggest_categorical("hidden", [16, 32, 64]),
                     layers=trial.suggest_int("layers", 1, 3),
                     dropout=trial.suggest_float("dropout", 0.0, 0.4),
                     lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                     weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                     batch_size=trial.suggest_categorical("batch_size", [64, 128]))


def tune_family(initial: pd.DataFrame, key: str, feats: list[str], tcfg: dict) -> dict:
    """Tune one family × loss on the initial window; returns params (+ inner-validation score)."""
    is_nn = not key.startswith("gbm")
    best, val, _ = tune(initial, lambda tr: _search(tr, key, feats, tcfg["max_epochs"]), n_trials=tcfg["n_trials"],
                        seed=tcfg["seed"], fit_kwargs=(lambda va: {"valid": va}) if is_nn else None)
    return {"params": best, "inner_valid_qlike": val}


# ── Statistics ──────────────────────────────────────────────────────────────────────────
def stationary_bootstrap_ratio_ci(la: np.ndarray, lb: np.ndarray, *, block: int, reps: int, seed: int) -> tuple[float, float]:
    """95% CI of 100·(mean(lb)/mean(la) − 1) by the stationary bootstrap (Politis & Romano 1994)."""
    from arch.bootstrap import StationaryBootstrap

    bs = StationaryBootstrap(max(block, 1), la, lb, seed=seed)
    vals = [100.0 * (b.mean() / a.mean() - 1.0) for (a, b), _ in bs.bootstrap(reps)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def compare(fa: pd.DataFrame, fb: pd.DataFrame, *, reps: int = 5000, seed: int = 0) -> dict:
    """DM (QLIKE & MSE) + effect size + MDE for forecasts of model A (benchmark) vs B (challenger)."""
    j = fa[["forecast", "actual"]].join(fb[["forecast"]], rsuffix="_b", how="inner")
    T = len(j)
    lag = newey_west_lag(T)
    out = {"T": T, "lag": lag}
    for name, lf in (("QLIKE", qlike), ("MSE", mse_log)):
        la, lb = lf(j["forecast"], j["actual"]), lf(j["forecast_b"], j["actual"])
        r = dm_test(la, lb, lag=lag)
        out[name] = {"mean_A": float(la.mean()), "mean_B": float(lb.mean()),
                     "pct_change_B_vs_A": 100.0 * (lb.mean() / la.mean() - 1.0), **r.as_dict()}
        if name == "QLIKE":
            out[name]["pct_change_ci95"] = stationary_bootstrap_ratio_ci(la, lb, block=lag, reps=reps, seed=seed)
            z = stats.norm.ppf(1 - 0.05 / 3 / 2) + stats.norm.ppf(0.8)
            out[name]["mde_mean_diff_holm_worst"] = float(z * r.se)
    return out


def mincer_zarnowitz_r2(f: pd.DataFrame) -> float:
    x, y = f["forecast"].to_numpy(), f["actual"].to_numpy()
    c = np.corrcoef(x, y)[0, 1]
    return float(c * c)


def summarize_forecasts(f: pd.DataFrame) -> dict:
    ql, ms = qlike(f["forecast"], f["actual"]), mse_log(f["forecast"], f["actual"])
    return {"T": len(f), "QLIKE": float(ql.mean()), "MSE_log": float(ms.mean()),
            "MAE_log": float(np.abs(f["forecast"] - f["actual"]).mean()), "MZ_R2": mincer_zarnowitz_r2(f)}


@dataclass
class AssetRun:
    asset: str
    forecasts: dict[str, pd.DataFrame]                # model → seed-averaged forecasts
    per_seed: dict[str, dict[int, pd.DataFrame]]
    tuning: dict[str, dict]
    timings: dict[str, dict]
    sample: dict


def run_asset(asset: str, df: pd.DataFrame, grid: dict, *, calendar: str, common_start_year: int | None,
              only: list[str] | None = None) -> AssetRun:
    """Run every registered model for one asset (optionally a subset by name, for tests)."""
    d = add_derived(df)
    s0 = sample_start(d, common_start_year)
    d = d[d.index >= s0]
    o0 = oos_start(d, s0, grid["initial_window_years"])
    initial = d[d.index < o0]
    models = {k: v for k, v in grid["models"].items() if asset in v.get("assets", ["btc", "spx", "nifty"])}
    if only:
        models = {k: v for k, v in models.items() if k in only}
    tuned: dict[str, dict] = {}
    forecasts, per_seed, timings = {}, {}, {}
    for name, spec in models.items():
        if spec["family"] == "ensemble":
            continue
        feats = resolve_features(spec.get("features", "HAR"), asset, grid, d.columns)
        key = _tune_key(spec)
        if key and key not in tuned:
            all_feats = resolve_features("ALL", asset, grid, d.columns)
            seed_everything(grid["tuning"]["seed"])
            tuned[key] = tune_family(initial, key, all_feats, grid["tuning"])
        params = tuned[key]["params"] if key else {}
        seeds = grid["seeds_neural"] if spec["family"] == "neural" else [0]
        runs = {}
        for sd in seeds:
            seed_everything(sd)
            res = run_oos(d, factory(spec, feats, params, sd), oos_start=o0,
                          refit_every=_refit_every(grid, spec["family"], calendar))
            runs[sd] = res.forecasts
            timings.setdefault(name, {"fit_seconds": 0.0, "n_fits": 0})
            timings[name]["fit_seconds"] += res.fit_seconds
            timings[name]["n_fits"] += res.n_fits
        per_seed[name] = runs
        f = runs[seeds[0]].copy()
        f["forecast"] = np.mean([r["forecast"].to_numpy() for r in runs.values()], axis=0)
        forecasts[name] = f
    for name, spec in models.items():                 # ensembles: inverse inner-validation loss weights
        if spec["family"] != "ensemble" or not all(m in forecasts for m in spec["members"]):
            continue
        w = np.array([1.0 / tuned[_tune_key(models[m])]["inner_valid_qlike"] for m in spec["members"]])
        w = w / w.sum()
        f = forecasts[spec["members"][0]].copy()
        f["forecast"] = sum(wi * forecasts[m]["forecast"].to_numpy() for wi, m in zip(w, spec["members"]))
        forecasts[name] = f
        timings[name] = {"weights": dict(zip(spec["members"], w.tolist()))}
    sample = {"sample_start": str(s0.date()), "oos_start": str(o0.date()), "rows": len(d), "oos_rows": int((d.index >= o0).sum())}
    return AssetRun(asset, forecasts, per_seed, tuned, timings, sample)


def evaluate_hypotheses(runs: dict[str, AssetRun], grid: dict, *, reps: int = 5000) -> dict:
    """All registered DM comparisons with Holm adjustment as specified per hypothesis."""
    out = {}
    for hid, h in grid["hypotheses"].items():
        rows = []
        for a, r in runs.items():
            for kind, pairs in (("test", h["pairs"]), ("report", h.get("also_report", []))):
                for A, B in pairs:
                    if A in r.forecasts and B in r.forecasts:
                        c = compare(r.forecasts[A], r.forecasts[B], reps=reps)
                        rows.append({"asset": a, "A": A, "B": B, "kind": kind, **c})
        tests = [x for x in rows if x["kind"] == "test"]
        adj = holm({f"{x['asset']}|{x['A']}|{x['B']}": x["QLIKE"]["p_value"] for x in tests}) if tests else {}
        for x in tests:
            x["QLIKE"]["p_holm"] = adj[f"{x['asset']}|{x['A']}|{x['B']}"]
        out[hid] = rows
    return out


def mcs_per_asset(r: AssetRun, *, reps: int = 5000) -> pd.DataFrame:
    idx = None
    for f in r.forecasts.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    L = pd.DataFrame({m: qlike(f.loc[idx, "forecast"], f.loc[idx, "actual"]) for m, f in r.forecasts.items()}, index=idx)
    block = int(math.ceil(len(L) ** (1 / 3)))
    return model_confidence_set(L, alpha=0.10, reps=reps, block_size=block, seed=0)
