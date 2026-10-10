"""Expanding-window out-of-sample evaluation with refit blocks and target masking (D34).

At each refit origin p (row position), the model is fitted on rows [start, p) and then forecasts rows
p … p+k−1 using a **context** frame in which the target columns of rows ≥ p are set to NaN.
Features at row t are observable at t by construction (DECISIONS.md D20), so forecasts at t use
only information available at t.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from alphavol.logging_utils import get_logger
from alphavol.models.base import TARGET

log = get_logger(__name__)
TARGET_COLS = (TARGET, "target_date")


@dataclass
class OOSResult:
    forecasts: pd.DataFrame          # index = origin dates; columns forecast, actual, fit_id
    fit_seconds: float
    n_fits: int


def run_oos(df: pd.DataFrame, factory: Callable[[], object], *, oos_start: pd.Timestamp, refit_every: int,
            sample_start: pd.Timestamp | None = None) -> OOSResult:
    """Expanding-window OOS forecasts for rows with index ≥ ``oos_start``."""
    data = df if sample_start is None else df[df.index >= sample_start]
    if not data.index.is_monotonic_increasing:
        raise ValueError("data must be sorted by date")
    first = int(data.index.searchsorted(oos_start))
    if first < 50:
        raise ValueError(f"too few training rows before oos_start ({first})")
    out, fit_s, n_fits = [], 0.0, 0
    for p in range(first, len(data), refit_every):
        q = min(p + refit_every, len(data))
        train = data.iloc[:p]
        context = data.iloc[:q].copy()
        context.iloc[p:, [context.columns.get_loc(c) for c in TARGET_COLS if c in context.columns]] = np.nan
        t0 = time.perf_counter()
        model = factory().fit(train)
        fit_s += time.perf_counter() - t0
        n_fits += 1
        rows = data.index[p:q]
        pred = np.asarray(model.predict(context, rows), dtype=float)
        if pred.shape != (len(rows),) or not np.isfinite(pred).all():
            raise ValueError(f"model {getattr(model, 'name', model)} returned invalid forecasts at block {p}")
        out.append(pd.DataFrame({"forecast": pred, "actual": data[TARGET].iloc[p:q].to_numpy(),
                                 "fit_id": n_fits - 1}, index=rows))
    return OOSResult(pd.concat(out), fit_s, n_fits)
