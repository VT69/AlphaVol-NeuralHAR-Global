"""Experiment runner on SYNTHETIC data with a shrunken grid (code verification only)."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from alphavol.experiments import (add_derived, evaluate_hypotheses, load_grid, mcs_per_asset, resolve_features,
                                  run_asset)
from tests.test_models import synthetic


def synthetic_asset(n=1400, seed=0):
    df = synthetic(n=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    has = rng.random(len(df)) < 0.7
    df["sent_has_news"] = has.astype("int8")
    df["sent_mean_score"] = np.where(has, df["exo"].fillna(0) * 0.3 + rng.normal(0, 0.2, len(df)), np.nan)
    df["sent_n_articles"] = np.where(has, rng.integers(1, 50, len(df)), 0)
    df["sent_gdelt_volume"] = rng.integers(100, 1000, len(df)).astype(float)
    df["term_slope"], df["credit_spread"], df["macro_missing"] = rng.normal(0, 1, len(df)), 2.0, 0
    df["vix"], df["implied_vol_missing"] = 15 + rng.gamma(2, 2, len(df)), 0
    return df.drop(columns=["exo", "exo_missing", "rq"])


def small_grid():
    g = copy.deepcopy(load_grid())
    g["seeds_neural"] = [0, 1]
    g["tuning"] = {"n_trials": 2, "seed": 0, "inner_valid_frac": 0.2, "max_epochs": 3}
    g["refit_every"]["neural"] = {"exchange": 400, "continuous": 400}
    g["refit_every"]["gbm"] = {"exchange": 400, "continuous": 400}
    g["initial_window_years"] = 3
    return g


def test_feature_resolution():
    g = load_grid()
    d = add_derived(synthetic_asset())
    allf = resolve_features("ALL", "spx", g, d.columns)
    assert {"sent_mean_score", "sent_has_news", "log1p_sent_n_articles", "iv_logvar", "term_slope"} <= set(allf)
    nos = resolve_features("ALL-SENT", "spx", g, d.columns)
    assert not any(c.startswith(("sent_", "log1p_sent")) for c in nos)
    assert resolve_features("HAR+SENT", "spx", g, d.columns) == ["RV_d", "RV_w", "RV_m", "sent_mean_score", "sent_has_news"]


@pytest.mark.slow
def test_run_asset_and_hypotheses_end_to_end():
    g = small_grid()
    only = ["HAR-BC", "HAR-OLS", "LIN-SENT", "LIN-ALL", "LIN-ALLnoS", "GBM-ALL", "GBM-ALLnoS",
            "NN-GRN-ALL", "NN-GRN-ALLnoS", "NN-TF-ALL", "NN-ENS-ALL", "RW", "GARCH"]
    r = run_asset("spx", synthetic_asset(), g, calendar="exchange", common_start_year=None, only=only)
    assert set(r.forecasts) == set(only)
    assert len(r.per_seed["NN-GRN-ALL"]) == 2 and "weights" in r.timings["NN-ENS-ALL"]
    idx = {k: v.index for k, v in r.forecasts.items()}
    assert all(i.equals(idx["HAR-BC"]) for i in idx.values())          # identical OOS rows for every model
    hyp = evaluate_hypotheses({"spx": r}, g, reps=200)
    h1 = hyp["H1"][0]
    assert h1["A"] == "HAR-BC" and h1["B"] == "LIN-SENT" and "p_holm" in h1["QLIKE"]
    assert h1["QLIKE"]["pct_change_ci95"][0] <= h1["QLIKE"]["pct_change_B_vs_A"] <= h1["QLIKE"]["pct_change_ci95"][1]
    m = mcs_per_asset(r, reps=200)
    assert m["mcs_pvalue"].notna().all()
