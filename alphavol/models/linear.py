"""Naive and linear HAR-family forecasters (all forecasts in log variance)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from alphavol.models.base import TARGET, Preprocessor, smearing, train_xy

HAR = ["RV_d", "RV_w", "RV_m"]


@dataclass
class RandomWalk:
    """Forecast = today's log variance (RV_d). No estimation."""

    name: str = "RW"

    def fit(self, train: pd.DataFrame) -> "RandomWalk":
        return self

    def predict(self, context: pd.DataFrame, rows: pd.Index) -> np.ndarray:
        return context.loc[rows, "RV_d"].to_numpy(dtype=float)


@dataclass
class HistoricalMean:
    """Constant QLIKE-optimal forecast: log of the mean training-window variance."""

    name: str = "HistMean"
    c_: float = field(init=False, default=np.nan)

    def fit(self, train: pd.DataFrame) -> "HistoricalMean":
        y = train_xy(train, [])[TARGET].to_numpy(dtype=float)
        self.c_ = smearing(y)
        return self

    def predict(self, context, rows) -> np.ndarray:
        return np.full(len(rows), self.c_)


@dataclass
class LinearLog:
    """OLS on log variance (HAR / HAR-X), optional causal smearing bias correction (D18, R1)."""

    features: list[str]
    bias: str = "smearing"                 # "smearing" | "none"
    name: str = "HAR-OLS"
    beta_: np.ndarray = field(init=False, default=None)
    smear_: float = field(init=False, default=0.0)

    def fit(self, train: pd.DataFrame) -> "LinearLog":
        tr = train_xy(train, self.features)
        self.prep_ = Preprocessor(self.features, standardize=False).fit(tr)
        X = np.c_[np.ones(len(tr)), self.prep_.transform(tr)]
        y = tr[TARGET].to_numpy(dtype=float)
        self.beta_, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.smear_ = smearing(y - X @ self.beta_) if self.bias == "smearing" else 0.0
        return self

    def _X(self, df) -> np.ndarray:
        return np.c_[np.ones(len(df)), self.prep_.transform(df)]

    def predict(self, context, rows) -> np.ndarray:
        return self._X(context.loc[rows]) @ self.beta_ + self.smear_


@dataclass
class LinearQLIKE(LinearLog):
    """Linear index f = Xβ fitted by minimizing mean QLIKE directly (L-BFGS), initialized at OLS +
    smearing. Same information set as LinearLog; differs only in the loss (RQ3)."""

    name: str = "HAR-QLIKE"

    def fit(self, train: pd.DataFrame) -> "LinearQLIKE":
        super().fit(train)
        tr = train_xy(train, self.features)
        X, y = self._X(tr), tr[TARGET].to_numpy(dtype=float)
        b0 = self.beta_.copy()
        b0[0] += self.smear_

        def obj(b):
            u = y - X @ b
            eu = np.exp(np.clip(u, -50, 50))
            return float(np.mean(eu - u - 1.0)), (X.T @ (1.0 - eu)) / len(y)

        res = minimize(obj, b0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
        if not res.success and not np.isfinite(res.fun):
            raise RuntimeError(f"QLIKE fit failed: {res.message}")
        self.beta_, self.smear_ = res.x, 0.0
        return self


@dataclass
class HARQ:
    """HARQ in variance levels (Bollerslev, Patton & Quaedvlieg 2016), BTC only:
    v_{t+1} = b0 + (b1 + b1Q·sqrt(RQ_t))·v_t + b2·v^w_t + b3·v^m_t. Forecasts outside the training range of v
    are replaced by the training mean ("insanity filter"); returned in logs."""

    name: str = "HARQ"

    def _design(self, df):
        vd, vw, vm = (np.exp(df[c].to_numpy(dtype=float)) for c in HAR)
        rq = df["rq"].to_numpy(dtype=float)
        return np.c_[np.ones(len(df)), vd, np.sqrt(rq) * vd, vw, vm]

    def fit(self, train: pd.DataFrame) -> "HARQ":
        tr = train_xy(train, HAR).dropna(subset=HAR + ["rq"])
        X, y = self._design(tr), np.exp(tr[TARGET].to_numpy(dtype=float))
        self.beta_, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.lo_, self.hi_, self.mean_ = float(y.min()), float(y.max()), float(y.mean())
        return self

    def predict(self, context, rows) -> np.ndarray:
        f = self._design(context.loc[rows]) @ self.beta_
        f = np.where((f < self.lo_) | (f > self.hi_) | ~np.isfinite(f), self.mean_, f)
        return np.log(f)
