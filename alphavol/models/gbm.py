"""Gradient-boosting baseline on the same features (sklearn HistGradientBoostingRegressor)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from alphavol.models.base import TARGET, Preprocessor, smearing, train_xy


@dataclass
class GBM:
    """loss='qlike': gamma deviance on variance levels (log link). The gamma deviance per row is
    2·(y/μ − log(y/μ) − 1) = 2·QLIKE, so this optimizes QLIKE directly. loss='mse': squared error
    on log variance plus smearing. Missing values are passed through (native NaN handling) and
    missing indicators are kept as features."""

    features: list[str]
    loss: str = "qlike"
    params: dict = field(default_factory=dict)
    seed: int = 0
    name: str = "GBM"

    def fit(self, train: pd.DataFrame) -> "GBM":
        tr = train_xy(train, self.features)
        X = tr[self.features].to_numpy(dtype=float)
        y = tr[TARGET].to_numpy(dtype=float)
        kw = {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 40,
              "l2_regularization": 1.0, **self.params, "random_state": self.seed, "early_stopping": False}
        if self.loss == "qlike":
            self.m_ = HistGradientBoostingRegressor(loss="gamma", **kw).fit(X, np.exp(y))
            self.smear_ = 0.0
        elif self.loss == "mse":
            self.m_ = HistGradientBoostingRegressor(loss="squared_error", **kw).fit(X, y)
            self.smear_ = smearing(y - self.m_.predict(X))
        else:
            raise ValueError(self.loss)
        return self

    def predict(self, context, rows) -> np.ndarray:
        X = context.loc[rows, self.features].to_numpy(dtype=float)
        if self.loss == "qlike":
            return np.log(self.m_.predict(X))
        return self.m_.predict(X) + self.smear_
