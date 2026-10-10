"""Shared model interface and train-only preprocessing (DECISIONS.md D34)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from alphavol.logging_utils import get_logger

log = get_logger(__name__)
TARGET = "target"


class Forecaster(Protocol):
    name: str

    def fit(self, train: pd.DataFrame) -> "Forecaster": ...

    def predict(self, context: pd.DataFrame, rows: pd.Index) -> np.ndarray: ...


@dataclass
class Preprocessor:
    """Training-window imputation (mean) and optional standardization. Never sees test rows."""

    features: list[str]
    standardize: bool = True
    mean_: pd.Series = field(init=False)
    std_: pd.Series = field(init=False)

    def fit(self, train: pd.DataFrame) -> "Preprocessor":
        missing = [c for c in self.features if c not in train.columns]
        if missing:
            raise KeyError(f"features not in data: {missing}")
        X = train[self.features].astype(float)
        mean = X.mean()
        allnan = mean.isna()
        if allnan.any():
            log.warning("features entirely missing in the training window, imputed as 0: %s", list(mean.index[allnan]))
        self.mean_ = mean.fillna(0.0)
        std = X.fillna(self.mean_).std(ddof=0)
        self.std_ = std.where(std > 0, 1.0).fillna(1.0)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].astype(float).fillna(self.mean_)
        if self.standardize:
            X = (X - self.mean_) / self.std_
        return X.to_numpy(dtype=float)


def smearing(log_residuals: np.ndarray) -> float:
    """Duan (1983) smearing correction on the log scale: log mean(exp(e)). Converts a forecast of
    E[log v] into a forecast of log E[v] under homoskedastic residuals."""
    e = np.asarray(log_residuals, dtype=float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        raise ValueError("no residuals for smearing")
    m = e.max()
    return float(m + np.log(np.mean(np.exp(e - m))))


def train_xy(train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Training rows with an observed target (features may be NaN; imputed later)."""
    if TARGET not in train.columns:
        raise KeyError("training frame lacks the target column")
    return train[train[TARGET].notna()]
