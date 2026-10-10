"""HAR regressors and the next-trading-day target (DECISIONS.md D11-D13).

Fixes vs. legacy:
* RV_d is log RV of day t (legacy used t-1, which made HAR a 2-step-ahead model).
* Weekly/monthly components are log of the *mean variance level* over windows ending at t.
* The target is the next trading day's log variance, assigned by date on the full calendar
  **before** any row is dropped. Legacy shifted after dropping rows, producing targets 35-39
  days ahead.
* An invalid day costs at most two rows (itself, and the previous day's target) instead of a
  23-row cascade.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from alphavol.logging_utils import get_logger
from alphavol.targets import log_variance

log = get_logger(__name__)
Calendar = Literal["continuous", "exchange"]


class AlignmentError(AssertionError):
    """A target is not the next trading day's log variance."""


def regularise_calendar(variance: pd.Series, calendar: Calendar) -> pd.Series:
    """Continuous calendars (BTC) are reindexed to every UTC day, so missing days appear as NaN.
    Exchange calendars keep the trading days present in the data; gaps longer than 5 calendar
    days are reported (a long weekend is at most 4)."""
    variance = variance.sort_index()
    if variance.index.has_duplicates:
        raise ValueError("duplicate dates in variance series")
    if calendar == "continuous":
        full = pd.date_range(variance.index.min(), variance.index.max(), freq="D", tz="UTC")
        missing = full.difference(variance.index)
        if len(missing):
            log.warning("continuous calendar: %d missing day(s) inserted as NaN: %s",
                        len(missing), [str(d.date()) for d in missing[:10]])
        return variance.reindex(full)
    gaps = variance.index.to_series().diff().dt.days
    long_gaps = gaps[gaps > 5]
    if len(long_gaps):
        log.warning("exchange calendar: %d gap(s) > 5 days: %s", len(long_gaps),
                    {str(d.date()): int(g) for d, g in long_gaps.items()})
    return variance


def har_regressors(variance: pd.Series, *, w: int, m: int, min_w: int, min_m: int) -> pd.DataFrame:
    """RV_d = log v_t; RV_w = log mean(v_{t-w+1..t}); RV_m = log mean(v_{t-m+1..t}).

    ``variance`` must be in levels on the asset calendar (one row per trading day, NaN for invalid
    days). Window means use the available days, requiring at least ``min_w`` / ``min_m`` of them.
    """
    if not (1 < w < m and 0 < min_w <= w and 0 < min_m <= m):
        raise ValueError(f"invalid windows w={w}, m={m}, min_w={min_w}, min_m={min_m}")
    v = variance.astype(float).where(variance > 0)
    out = pd.DataFrame(index=v.index)
    out["RV_d"] = log_variance(v)
    out["RV_w"] = log_variance(v.rolling(w, min_periods=min_w).mean())
    out["RV_m"] = log_variance(v.rolling(m, min_periods=min_m).mean())
    return out


def next_day_target(log_v: pd.Series, calendar: Calendar) -> pd.DataFrame:
    """Target_t = log variance on the next row of the calendar, with its date.

    For a continuous calendar, the next row must be exactly one day later; this always holds
    after :func:`regularise_calendar`, and is asserted.
    """
    dates = log_v.index.to_series()
    out = pd.DataFrame({"target": log_v.shift(-1), "target_date": dates.shift(-1)}, index=log_v.index)
    if calendar == "continuous":
        step = (out["target_date"] - dates).dt.days
        if not ((step == 1) | step.isna()).all():
            raise AlignmentError("continuous calendar has non-daily steps; call regularise_calendar first")
    return out


def assert_targets_aligned(df: pd.DataFrame, log_v_full: pd.Series) -> None:
    """Raise AlignmentError unless, for every row t, target_date is the next calendar date after t
    in ``log_v_full`` and target == log_v_full[target_date] exactly."""
    cal = log_v_full.index
    pos = cal.get_indexer(df.index)
    if (pos < 0).any():
        raise AlignmentError("dataset rows not on the asset calendar")
    if (pos + 1 >= len(cal)).any():
        raise AlignmentError("a row has no next calendar date but kept a target")
    expected_date = cal[pos + 1]
    if not (pd.DatetimeIndex(df["target_date"]) == expected_date).all():
        bad = df.index[pd.DatetimeIndex(df["target_date"]) != expected_date]
        raise AlignmentError(f"target_date is not the next trading day for {len(bad)} rows, e.g. {bad[:3].tolist()}")
    expected = log_v_full.to_numpy()[pos + 1]
    got = df["target"].to_numpy()
    if not np.array_equal(got, expected):
        n = int((~np.isclose(got, expected, equal_nan=False)).sum())
        raise AlignmentError(f"target differs from next trading day's log variance on {n} rows")
