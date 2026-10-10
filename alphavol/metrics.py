"""Forecast losses and comparison tests — the single implementation used from Step 2 on (D18).

Conventions: forecasts and actuals are **log variances**. QLIKE is evaluated on the variance
scale: L = σ²/h − log(σ²/h) − 1 with σ² = exp(actual), h = exp(forecast) (Patton 2011).
The ±50 clip on logs is an overflow guard only; the code logs a warning if it ever binds.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from alphavol.logging_utils import get_logger

log = get_logger(__name__)
_GUARD = 50.0


def _as_arrays(pred, actual) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pred, dtype=float).ravel()
    a = np.asarray(actual, dtype=float).ravel()
    if p.shape != a.shape:
        raise ValueError(f"shape mismatch: forecasts {p.shape} vs actuals {a.shape}")
    if not (np.isfinite(p).all() and np.isfinite(a).all()):
        raise ValueError("non-finite forecasts or actuals: align and drop missing values first")
    return p, a


def qlike(pred_log, actual_log) -> np.ndarray:
    """Per-period QLIKE on the variance scale for log-variance inputs."""
    p, a = _as_arrays(pred_log, actual_log)
    if (np.abs(p) > _GUARD).any() or (np.abs(a) > _GUARD).any():
        log.warning("QLIKE overflow guard (±%s on logs) binds on %d values", _GUARD,
                    int((np.abs(p) > _GUARD).sum() + (np.abs(a) > _GUARD).sum()))
    x = np.exp(np.clip(a, -_GUARD, _GUARD) - np.clip(p, -_GUARD, _GUARD))   # σ²/h computed stably
    return x - np.log(x) - 1.0


def mse_log(pred_log, actual_log) -> np.ndarray:
    """Per-period squared error of log variance."""
    p, a = _as_arrays(pred_log, actual_log)
    return (a - p) ** 2


LOSSES = {"QLIKE": qlike, "MSE": mse_log}


def newey_west_lag(T: int) -> int:
    """Newey & West (1994) automatic lag: floor(4·(T/100)^(2/9))."""
    if T < 2:
        raise ValueError("need at least 2 observations")
    return int(math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))


def newey_west_lrv(d: np.ndarray, lag: int) -> float:
    """Bartlett-kernel long-run variance of d (1/T normalisation; always ≥ 0)."""
    d = np.asarray(d, dtype=float)
    T = len(d)
    u = d - d.mean()
    s = u @ u / T
    for j in range(1, lag + 1):
        s += 2.0 * (1.0 - j / (lag + 1.0)) * (u[j:] @ u[:-j]) / T
    return float(s)


@dataclass(frozen=True)
class DMResult:
    stat: float
    p_value: float
    mean_diff: float          # mean(L_A − L_B); > 0 means B has lower loss
    se: float
    T: int
    lag: int
    hln_factor: float
    better: str               # "B" if mean_diff > 0 else "A"

    def as_dict(self) -> dict:
        return asdict(self)


def dm_test(loss_a, loss_b, *, lag: int, h: int = 1) -> DMResult:
    """Diebold–Mariano test of equal predictive accuracy for two per-period loss series.

    Newey–West (Bartlett) HAC variance with the given ``lag`` (fixed in the pre-registration),
    the Harvey–Leybourne–Newbold (1997) small-sample factor for horizon ``h``, and a Student-t
    reference distribution with T−1 degrees of freedom. Two-sided p-value, not rounded.
    Positive statistic ⇒ model B is more accurate.
    """
    la, lb = (np.asarray(x, dtype=float).ravel() for x in (loss_a, loss_b))
    if la.shape != lb.shape:
        raise ValueError("loss series must have the same length")
    if not (np.isfinite(la).all() and np.isfinite(lb).all()):
        raise ValueError("non-finite losses")
    d = la - lb
    T = len(d)
    if T < 10:
        raise ValueError(f"too few observations for a DM test: {T}")
    lrv = newey_west_lrv(d, lag)
    dbar = float(d.mean())
    if lrv <= 0:
        if np.allclose(d, d[0]):
            return DMResult(0.0, 1.0, dbar, 0.0, T, lag, 1.0, "A")
        raise ValueError("non-positive long-run variance")
    se = math.sqrt(lrv / T)
    hln = math.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dbar / se * hln
    p = float(2.0 * stats.t.sf(abs(stat), df=T - 1))
    return DMResult(stat, p, dbar, se, T, lag, hln, "B" if dbar > 0 else "A")


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm (1979) step-down adjusted p-values (monotone, capped at 1)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def model_confidence_set(losses: pd.DataFrame, *, alpha: float, reps: int, block_size: int,
                         seed: int, method: str = "R") -> pd.DataFrame:
    """Hansen, Lunde & Nason (2011) MCS via ``arch.bootstrap.MCS`` (circular block bootstrap).

    Returns one row per model: mean loss, MCS p-value (reported for every model) and inclusion.
    """
    from arch.bootstrap import MCS

    if losses.isna().any().any():
        raise ValueError("losses contain NaN")
    mcs = MCS(losses, size=alpha, reps=reps, block_size=block_size, method=method,
              bootstrap="circular", seed=seed)
    mcs.compute()
    pv = mcs.pvalues["Pvalue"]
    out = pd.DataFrame({"mean_loss": losses.mean(), "mcs_pvalue": pv.reindex(losses.columns)})
    out["in_mcs"] = out.index.isin(mcs.included)
    return out.sort_values("mean_loss")
