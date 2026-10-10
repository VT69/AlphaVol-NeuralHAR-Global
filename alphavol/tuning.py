"""Seeded, equal-budget hyperparameter tuning on the initial training window only (D34)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import optuna
import pandas as pd

from alphavol.metrics import qlike
from alphavol.models.base import TARGET

optuna.logging.set_verbosity(optuna.logging.WARNING)


def inner_split(initial: pd.DataFrame, valid_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split of the initial training window (last ``valid_frac`` = inner validation)."""
    n = len(initial)
    cut = int(round(n * (1 - valid_frac)))
    return initial.iloc[:cut], initial.iloc[cut:]


def tune(initial: pd.DataFrame, make: Callable[[optuna.Trial], object], *, n_trials: int, seed: int,
         fit_kwargs: Callable[[pd.DataFrame], dict] | None = None) -> tuple[dict, float, optuna.Study]:
    """Minimize inner-validation QLIKE. ``make(trial)`` returns an unfitted model; ``fit_kwargs``
    may pass the validation frame to models that use it (e.g. for epoch selection)."""
    tr, va = inner_split(initial)
    va = va[va[TARGET].notna()]

    def objective(trial):
        model = make(trial)
        kw = fit_kwargs(va) if fit_kwargs else {}
        model.fit(tr, **kw)
        pred = model.predict(va, va.index)
        if not np.isfinite(pred).all():
            return float("inf")
        score = float(qlike(pred, va[TARGET].to_numpy()).mean())
        hist = getattr(model, "history_", None)
        if hist:                                         # neural: record best epoch on inner validation
            trial.set_user_attr("best_epoch", int(np.argmin(hist)) + 1)
            score = float(np.min(hist))
        return score

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best = dict(study.best_params)
    if "best_epoch" in study.best_trial.user_attrs:
        best["epochs"] = study.best_trial.user_attrs["best_epoch"]
    return best, float(study.best_value), study
