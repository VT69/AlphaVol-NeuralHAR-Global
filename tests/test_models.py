"""Models and the OOS harness on SYNTHETIC data only (code verification, not evidence)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphavol.evaluation import run_oos
from alphavol.metrics import qlike
from alphavol.models.base import Preprocessor, smearing
from alphavol.models.garch import GARCH
from alphavol.models.gbm import GBM
from alphavol.models.linear import HARQ, HistoricalMean, LinearLog, LinearQLIKE, RandomWalk
from alphavol.models.neural import NeuralHAR
from alphavol.tuning import tune

HAR = ["RV_d", "RV_w", "RV_m"]


def synthetic(n=900, seed=0):
    """Log-variance AR process with a weak exogenous signal; HAR regressors built like the pipeline."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    lv = np.zeros(n)
    for t in range(1, n):
        lv[t] = -9 + 0.8 * (lv[t - 1] + 9) + 0.15 * x[t - 1] + rng.normal(0, 0.5)
    v = pd.Series(np.exp(lv), index=pd.bdate_range("2018-01-01", periods=n, tz="UTC"))
    df = pd.DataFrame({"RV_d": np.log(v), "RV_w": np.log(v.rolling(5, min_periods=4).mean()),
                       "RV_m": np.log(v.rolling(22, min_periods=18).mean()), "exo": x,
                       "ret": rng.normal(0, 1, n) * np.sqrt(v.to_numpy()), "rq": v.to_numpy() ** 2})
    df["exo_missing"] = 0
    df.loc[df.index[::17], "exo"] = np.nan
    df.loc[df["exo"].isna(), "exo_missing"] = 1
    df["target"] = df["RV_d"].shift(-1)
    df["target_date"] = df.index.to_series().shift(-1)
    return df.dropna(subset=HAR + ["target"])


def test_preprocessor_uses_train_stats_only():
    df = synthetic()
    p = Preprocessor(["exo"]).fit(df.iloc[:100])
    a = p.transform(df.iloc[200:210])
    df2 = df.copy()
    df2.iloc[200:, df2.columns.get_loc("exo")] += 100          # perturb the "future"
    p2 = Preprocessor(["exo"]).fit(df2.iloc[:100])
    assert np.allclose(p.mean_, p2.mean_) and not np.allclose(a, p2.transform(df2.iloc[200:210]))


def test_smearing_equals_duan():
    e = np.array([-0.5, 0.0, 0.5, 1.0])
    assert np.isclose(smearing(e), np.log(np.mean(np.exp(e))))


def test_oos_harness_masks_future_targets():
    df = synthetic()
    seen = []

    class Spy:
        name = "spy"

        def fit(self, train):
            seen.append(train.index.max())
            return self

        def predict(self, context, rows):
            assert context.loc[rows, "target"].isna().all()         # targets of forecast rows masked
            assert context.index.max() == rows.max()
            return context.loc[rows, "RV_d"].to_numpy()

    res = run_oos(df, Spy, oos_start=df.index[400], refit_every=50)
    starts = res.forecasts.groupby("fit_id").apply(lambda g: g.index.min())
    assert all(s_train < s_block for s_train, s_block in zip(seen, starts))   # train strictly before block


def test_smearing_and_qlike_fit_reduce_qlike_vs_plain_ols():
    df = synthetic()
    kw = dict(oos_start=df.index[400], refit_every=1)
    q = {}
    for name, f in {"ols": lambda: LinearLog(HAR, bias="none"), "ols_bc": lambda: LinearLog(HAR),
                    "qlike": lambda: LinearQLIKE(HAR)}.items():
        r = run_oos(df.iloc[:520], f, **kw).forecasts
        q[name] = qlike(r["forecast"], r["actual"]).mean()
    assert q["ols_bc"] < q["ols"] and q["qlike"] < q["ols"]


def test_exogenous_signal_is_found_by_linear_model():
    df = synthetic(n=1200)
    feats = HAR + ["exo", "exo_missing"]
    m = LinearLog(feats).fit(df.iloc[:1000])
    assert m.beta_[4] > 0.08                                         # true coefficient 0.15 on exo


def test_naive_garch_harq_gbm_run_and_are_finite():
    df = synthetic(n=700)
    for f in (lambda: RandomWalk(), lambda: HistoricalMean(), lambda: HARQ(), lambda: GARCH(),
              lambda: GBM(HAR + ["exo", "exo_missing"], loss="qlike"), lambda: GBM(HAR + ["exo"], loss="mse")):
        r = run_oos(df, f, oos_start=df.index[500], refit_every=100).forecasts
        assert np.isfinite(r["forecast"]).all() and len(r) == len(df) - 500


def test_neural_is_deterministic_per_seed_and_seeds_differ():
    df = synthetic(n=600)
    feats = HAR + ["exo", "exo_missing"]
    mk = lambda s: NeuralHAR(feats, hidden=16, layers=1, epochs=5, seed=s)
    a = mk(0).fit(df.iloc[:500]).predict(df, df.index[500:])
    b = mk(0).fit(df.iloc[:500]).predict(df, df.index[500:])
    c = mk(1).fit(df.iloc[:500]).predict(df, df.index[500:])
    assert np.array_equal(a, b) and not np.array_equal(a, c)


@pytest.mark.parametrize("arch", ["grn", "transformer"])
@pytest.mark.parametrize("loss", ["qlike", "mse"])
def test_neural_trains_and_starts_at_har_prior(arch, loss):
    df = synthetic(n=600)
    feats = HAR + ["exo", "exo_missing"]
    m0 = NeuralHAR(feats, arch=arch, loss=loss, hidden=16, layers=1, epochs=0).fit(df.iloc[:500])
    ols = LinearLog(HAR, bias="smearing").fit(df.iloc[:500])   # QLIKE: smeared init; MSE: post-fit smearing
    p0, pols = m0.predict(df, df.index[500:]), ols.predict(df, df.index[500:])
    assert np.allclose(p0, pols, atol=1e-4)                           # zero-init head => HAR prior
    m = NeuralHAR(feats, arch=arch, loss=loss, hidden=16, layers=1, epochs=10).fit(df.iloc[:500], valid=df.iloc[450:500])
    assert len(m.history_) == 10 and np.isfinite(m.predict(df, df.index[500:])).all()


def test_tuning_is_seeded_and_records_epochs():
    df = synthetic(n=500)
    feats = HAR + ["exo", "exo_missing"]

    def make(trial):
        return NeuralHAR(feats, hidden=trial.suggest_categorical("hidden", [8, 16]), layers=1,
                         lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True), epochs=6)

    b1, v1, _ = tune(df, make, n_trials=3, seed=0, fit_kwargs=lambda va: {"valid": va})
    b2, v2, _ = tune(df, make, n_trials=3, seed=0, fit_kwargs=lambda va: {"valid": va})
    assert b1 == b2 and v1 == v2 and 1 <= b1["epochs"] <= 6
