"""GARCH(1,1) benchmark with a causal linear calibration to the log-variance target."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from arch import arch_model

from alphavol.models.base import TARGET, smearing, train_xy

SCALE = 100.0  # returns in percent for numerical stability


@dataclass
class GARCH:
    """Student-t GARCH(1,1) with constant mean on daily close-to-close returns.

    The one-step conditional variance h_{t+1} (from returns through t) is mapped to the target by
    log v_{t+1} ≈ a + b·log h_{t+1}, fitted by OLS on training rows, plus smearing. This puts a
    daily-return model on the same scale as RV/GK targets without using future information.
    """

    name: str = "GARCH"
    params_: pd.Series = field(init=False, default=None)

    def _cond_var_next(self, ret: pd.Series, params: pd.Series | None) -> tuple[pd.Series, pd.Series]:
        r = ret.dropna() * SCALE
        am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)
        res = am.fit(disp="off", show_warning=False) if params is None else am.fix(params)
        f = res.forecast(horizon=1, start=0, reindex=False).variance.iloc[:, 0] / SCALE ** 2
        return pd.Series(f.to_numpy(), index=r.index), res.params

    def fit(self, train: pd.DataFrame) -> "GARCH":
        h, self.params_ = self._cond_var_next(train["ret"], None)
        tr = train_xy(train, []).join(h.rename("h"), how="inner").dropna(subset=["h"])
        X = np.c_[np.ones(len(tr)), np.log(tr["h"].to_numpy())]
        y = tr[TARGET].to_numpy(dtype=float)
        self.ab_, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.smear_ = smearing(y - X @ self.ab_)
        return self

    def predict(self, context, rows) -> np.ndarray:
        h, _ = self._cond_var_next(context["ret"], self.params_)
        h = h.reindex(rows)
        if h.isna().any():
            raise ValueError("GARCH: missing return history for some forecast rows")
        return self.ab_[0] + self.ab_[1] * np.log(h.to_numpy()) + self.smear_
