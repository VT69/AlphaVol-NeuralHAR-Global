"""Metrics vs hand cases and independent references."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from alphavol.metrics import (dm_test, holm, model_confidence_set, mse_log, newey_west_lag, newey_west_lrv,
                              qlike)


def test_qlike_hand_case_and_minimum():
    assert math.isclose(qlike([0.0], [math.log(2)])[0], 2 - math.log(2) - 1)
    assert qlike([1.3], [1.3])[0] == 0.0
    # asymmetric: under-prediction penalised more than over-prediction by the same log distance
    assert qlike([0.0], [1.0])[0] > qlike([1.0], [0.0])[0]


def test_qlike_extreme_log_values_are_stable():
    # legacy torch loss clamped predictions at -10; real SPX/NIFTY GK log variances are near -10
    v = qlike([-12.0, -9.0], [-11.0, -11.0])
    assert np.isfinite(v).all() and (v > 0).all()


def test_mse_log_hand_case():
    assert math.isclose(mse_log([0.0], [math.log(2)])[0], math.log(2) ** 2)


def test_inputs_must_align():
    with pytest.raises(ValueError):
        qlike([0.0, 1.0], [0.0])
    with pytest.raises(ValueError):
        mse_log([np.nan], [0.0])


def test_newey_west_matches_statsmodels_bartlett():
    rng = np.random.default_rng(0)
    e = rng.normal(size=400)
    d = np.convolve(e, [1, 0.5, 0.25], mode="same") + 0.1
    lag = 5
    ours_se = math.sqrt(newey_west_lrv(d, lag) / len(d))
    ref = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lag, "kernel": "bartlett",
                                                                    "use_correction": False})
    assert math.isclose(ours_se, float(ref.bse[0]), rel_tol=1e-10)


def test_dm_matches_reference_t_stat_times_hln():
    rng = np.random.default_rng(1)
    y = rng.normal(size=300)
    la = (y - (y + rng.normal(0, 1.0, 300))) ** 2
    lb = (y - (y + rng.normal(0, 0.8, 300))) ** 2
    res = dm_test(la, lb, lag=4)
    ref = sm.OLS(la - lb, np.ones(300)).fit(cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": False})
    hln = math.sqrt((300 - 1) / 300)
    assert math.isclose(res.stat, float(ref.tvalues[0]) * hln, rel_tol=1e-10)
    assert res.better == "B" and 0 < res.p_value < 1
    same = dm_test(la, la, lag=4)
    assert same.stat == 0.0 and same.p_value == 1.0


def test_newey_west_lag_rule():
    assert newey_west_lag(100) == 4 and newey_west_lag(1000) == 6


def test_holm_hand_case():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert math.isclose(adj["a"], 0.03) and math.isclose(adj["c"], 0.06) and math.isclose(adj["b"], 0.06)


def test_mcs_wrapper_reports_all_pvalues():
    rng = np.random.default_rng(2)
    base = np.abs(rng.normal(0.5, 0.3, 300))
    L = pd.DataFrame({"worst": base + 0.3, "best": base + np.abs(rng.normal(0.0, 0.05, 300))})
    out = model_confidence_set(L, alpha=0.1, reps=300, block_size=5, seed=0)
    assert out.loc["best", "in_mcs"] and not out.loc["worst", "in_mcs"]
    assert out["mcs_pvalue"].notna().all()
