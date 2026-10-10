"""Dataset construction: target, HAR regressors and exogenous feature groups (D11-D23).

``build_dataset`` is a pure function of in-memory raw inputs, so the same code builds the
development data and (in Step 4) the sealed holdout. Loading lives in ``load_raw_inputs``.

Feature groups (explicit; nothing is silently zero-filled):
  har            RV_d, RV_w, RV_m                                (always)
  macro          term_slope (FRED T10Y2Y), credit_spread (FRED BAA10Y)     lag 1 business day
  implied_vol    SPX: vix (FRED VIXCLS, lag 1 bday); NIFTY: india_vix (lag 0)
  fear_greed     BTC: crypto Fear & Greed / 100                           lag 1 day
  microstructure BTC: ofi_abs, vpin, amihud, roll                         lag 0 (day-t bars)
  sentiment      columns of the Step-3 daily sentiment file               window ends at close_t
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from alphavol.config import AssetSpec, PipelineConfig
from alphavol.har import assert_targets_aligned, har_regressors, next_day_target, regularise_calendar
from alphavol.io import MissingDataError, read_frame
from alphavol.logging_utils import get_logger
from alphavol.microstructure import daily_microstructure
from alphavol.targets import daily_realized_variance, garman_klass, log_variance

log = get_logger(__name__)

GROUP_FEATURES: dict[str, dict[str, list[str]]] = {
    "btc": {"macro": ["term_slope", "credit_spread"], "fear_greed": ["fear_greed"],
            "microstructure": ["ofi_abs", "vpin", "amihud", "roll"], "sentiment": []},
    "spx": {"macro": ["term_slope", "credit_spread"], "implied_vol": ["vix"], "sentiment": []},
    "nifty": {"macro": ["term_slope", "credit_spread"], "implied_vol": ["india_vix"], "sentiment": []},
}
FRED_CODES = {"term_slope": "T10Y2Y", "credit_spread": "BAA10Y", "vix": "VIXCLS"}


@dataclass
class RawInputs:
    """Raw inputs for one asset. ``prices``: 5-min bars (BTC) or daily OHLC (SPX/NIFTY)."""

    prices: pd.DataFrame
    fred: pd.DataFrame | None = None            # columns = FRED codes, index = observation date (UTC)
    fear_greed: pd.Series | None = None         # value in [0, 100], index = date (UTC)
    india_vix: pd.Series | None = None          # close, index = trading date (UTC)
    sentiment: pd.DataFrame | None = None       # Step-3 daily file, index = asset trading date t
    provenance: dict = field(default_factory=dict)


@dataclass
class BuildResult:
    data: pd.DataFrame
    log_variance_full: pd.Series
    features: dict[str, list[str]]
    pending: list[str]
    notes: list[str]


# ── Timing alignment ──────────────────────────────────────────────────────────

def align_lagged(series: pd.Series, row_dates: pd.DatetimeIndex, *, lag: int, unit: str,
                 max_staleness: int) -> pd.Series:
    """Value of ``series`` known at row date t: the last observation dated ≤ t − lag.

    ``unit`` is "bday" (business days) or "day" (calendar days). Observations older than
    ``max_staleness`` units at the cutoff are dropped (NaN), never carried forward indefinitely.
    """
    if unit not in ("bday", "day"):
        raise ValueError(f"unit must be 'bday' or 'day', got {unit}")
    s = series.dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    rows = pd.DatetimeIndex(row_dates)
    offset = pd.offsets.BDay(lag) if unit == "bday" else pd.Timedelta(days=lag)
    cutoff = (rows - offset) if lag else rows
    left = pd.DataFrame({"cutoff": cutoff.tz_convert("UTC").normalize()}, index=rows).sort_values("cutoff")
    right = pd.DataFrame({"obs_date": s.index, "value": s.to_numpy()}).sort_values("obs_date")
    merged = pd.merge_asof(left.reset_index(names="row"), right, left_on="cutoff", right_on="obs_date",
                           direction="backward")
    found = merged["obs_date"].notna().to_numpy()
    stale = np.full(len(merged), np.inf)
    if found.any():
        obs = merged.loc[found, "obs_date"].dt.tz_localize(None).to_numpy("datetime64[D]")
        cut = merged.loc[found, "cutoff"].dt.tz_localize(None).to_numpy("datetime64[D]")
        stale[found] = np.busday_count(obs, cut) if unit == "bday" else (cut - obs).astype(int)
    values = merged["value"].where(pd.Series(stale <= max_staleness, index=merged.index))
    out = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(merged["row"])).reindex(rows)
    return out.rename(series.name)


# ── Variance per asset ─────────────────────────────────────────────────────────

def daily_variance(asset: str, spec: AssetSpec, prices: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Return (variance levels on the asset calendar, per-day diagnostics)."""
    if spec.variance == "realized_5min":
        rv = daily_realized_variance(prices, min_bars=int(spec.min_bars_per_day or 1),
                                     flag_below=int(spec.flag_bars_below or 0))
        variance = regularise_calendar(rv["variance"], spec.calendar)
        diag = rv[["n_bars", "incomplete"]].reindex(variance.index)
        return variance, diag
    if spec.variance == "garman_klass":
        gk = garman_klass(prices)
        return regularise_calendar(gk, spec.calendar), pd.DataFrame(index=gk.index)
    raise ValueError(f"unknown variance type {spec.variance} for {asset}")


# ── Builder ─────────────────────────────────────────────────────────────────────

def _missing_indicator(frame: pd.DataFrame, cols: list[str], name: str) -> pd.Series:
    return frame[cols].isna().any(axis=1).astype("int8").rename(f"{name}_missing")


def build_dataset(asset: str, raw: RawInputs, cfg: PipelineConfig, *, groups: list[str],
                  pending: set[str] | None = None) -> BuildResult:
    """Build the modelling dataset for one asset from raw inputs.

    ``groups`` lists exogenous groups to include (the HAR group is always included).
    ``pending`` lists groups or "group.feature" items that may be missing from ``raw``; they are
    omitted and recorded. Any other missing input raises :class:`MissingDataError`.
    """
    pending = set(pending or ())
    spec = cfg.assets[asset]
    variance, diag = daily_variance(asset, spec, raw.prices)
    lv = log_variance(variance)
    w, m = spec.har_windows["w"], spec.har_windows["m"]
    X = har_regressors(variance, w=w, m=m, min_w=spec.har_min_counts["w"], min_m=spec.har_min_counts["m"])
    T = next_day_target(lv, spec.calendar)
    df = pd.concat([X, T, variance.rename("variance"), diag], axis=1)
    rows = df.index
    feats: dict[str, list[str]] = {"har": ["RV_d", "RV_w", "RV_m"]}
    omitted: list[str] = []
    notes: list[str] = []
    stale = cfg.max_staleness_business_days

    def _need(group: str, item: str, available: bool) -> bool:
        if available:
            return True
        if group in pending or f"{group}.{item}" in pending:
            omitted.append(f"{group}.{item}")
            return False
        raise MissingDataError(f"{asset}: required input for {group}.{item} is missing "
                               f"(add '{group}.{item}' to pending only if it is being fetched)")

    allowed = GROUP_FEATURES[asset]
    for group in groups:
        if group not in allowed:
            raise ValueError(f"group '{group}' not defined for {asset}; allowed: {sorted(allowed)}")
        cols: list[str] = []
        if group in ("macro", "implied_vol"):
            for name in allowed[group]:
                if name == "india_vix":
                    if _need(group, name, raw.india_vix is not None):
                        df[name] = align_lagged(raw.india_vix.rename(name), rows, lag=0, unit="bday",
                                                max_staleness=stale)
                        cols.append(name)
                    continue
                code = FRED_CODES[name]
                have = raw.fred is not None and code in raw.fred.columns and raw.fred[code].notna().any()
                if _need(group, name, have):
                    df[name] = align_lagged(raw.fred[code].rename(name), rows, lag=cfg.fred_lag_business_days,
                                            unit="bday", max_staleness=stale)
                    cols.append(name)
        elif group == "fear_greed":
            if _need(group, "fear_greed", raw.fear_greed is not None):
                df["fear_greed"] = align_lagged(raw.fear_greed.rename("fear_greed") / 100.0, rows,
                                                lag=cfg.fear_greed_lag_days, unit="day", max_staleness=stale)
                cols.append("fear_greed")
        elif group == "microstructure":
            ms = daily_microstructure(raw.prices).reindex(rows)       # lag 0: computed from day-t bars
            for c in allowed[group]:
                df[c] = ms[c]
                cols.append(c)
        elif group == "sentiment":
            if _need(group, "file", raw.sentiment is not None):
                s = raw.sentiment.reindex(rows)
                for c in s.columns:
                    df[f"sent_{c}" if not c.startswith("sent_") else c] = s[c]
                cols = [c if c.startswith("sent_") else f"sent_{c}" for c in s.columns]
        if cols:
            df[f"{group}_missing"] = _missing_indicator(df, cols, group)
            feats[group] = cols

    before = len(df)
    df = df.dropna(subset=["RV_d", "RV_w", "RV_m", "target"])
    notes.append(f"rows dropped for missing HAR/target: {before - len(df)}")
    assert_targets_aligned(df, lv)
    return BuildResult(data=df, log_variance_full=lv, features=feats, pending=sorted(omitted), notes=notes)


# ── Raw loading (development data only; the sealed holdout is built by alphavol.holdout) ─────────

def _utc_index(df: pd.DataFrame | pd.Series, col: str | None = None) -> pd.DataFrame | pd.Series:
    if col is not None:
        df = df.set_index(col)
    idx = pd.to_datetime(df.index, utc=True)
    df = df.copy()
    df.index = idx
    return df.sort_index()


def load_raw_inputs(asset: str, cfg: PipelineConfig, *, up_to: pd.Timestamp | None = None) -> RawInputs:
    """Load committed development raw data (plus ``extra_raw_dir`` additions if present),
    truncated at ``up_to`` (default ``cfg.dev_end``) inclusive, so no data after the boundary
    can enter the development set."""
    up_to = (up_to or cfg.dev_end).normalize()
    end_excl = up_to + pd.Timedelta(days=1)
    raw, extra = cfg.raw_dir, cfg.extra_raw_dir
    prov: dict = {}
    if asset == "btc":
        cols = ["close", "volume", "quote_asset_volume", "taker_buy_base_vol"]
        prices = _utc_index(read_frame(raw / "ohlcv" / "btc_5min.parquet", columns=cols))
        extra_btc = extra / "ohlcv" / "btc_5min.parquet"
        if extra_btc.exists():
            more = _utc_index(read_frame(extra_btc, columns=cols))
            prices = pd.concat([prices, more[~more.index.isin(prices.index)]]).sort_index()
        prices = prices[prices.index < end_excl]
        prov["prices"] = "btc_5min"
    else:
        prices = _utc_index(read_frame(raw / "ohlcv" / f"{asset}_1d.parquet"))[["open", "high", "low", "close"]]
        prices = prices[prices.index < end_excl]
        prov["prices"] = f"{asset}_1d"
    fred_path = extra / "macro" / "fred.parquet"
    fred = _utc_index(read_frame(fred_path if fred_path.exists() else raw / "macro" / "fred_series.parquet"))
    prov["fred"] = str(fred_path if fred_path.exists() else "macro/fred_series.parquet")
    fg = None
    if asset == "btc":
        fg_path = extra / "macro" / "crypto_fear_greed.parquet"
        g = read_frame(fg_path if fg_path.exists() else raw / "macro" / "crypto_fear_greed.parquet")
        g = _utc_index(g, "date") if "date" in g.columns else _utc_index(g)
        fg = pd.to_numeric(g["value"], errors="raise")
        prov["fear_greed"] = str(fg_path if fg_path.exists() else "macro/crypto_fear_greed.parquet")
    ivix = None
    ivix_path = extra / "ohlcv" / "india_vix_1d.parquet"
    if asset == "nifty" and ivix_path.exists():
        ivix = _utc_index(read_frame(ivix_path))["close"]
    sent = None
    sent_path = extra / "sentiment" / f"sentiment_daily_{asset}.parquet"
    if sent_path.exists():
        sent = _utc_index(read_frame(sent_path))
    for name, obj in (("fred", fred), ("fear_greed", fg), ("india_vix", ivix), ("sentiment", sent)):
        if obj is not None:
            obj.drop(obj.index[obj.index >= end_excl], inplace=True)
    return RawInputs(prices=prices, fred=fred, fear_greed=fg, india_vix=ivix, sentiment=sent, provenance=prov)
