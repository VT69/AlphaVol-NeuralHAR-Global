"""Phase-1 audit: trace shapes at every hop of the pipeline and verify lag alignment.

Read-only with respect to project sources: it imports nothing from src/ that writes,
and only reads parquet/npy artifacts. Output: analysis/artifacts/phase1_trace.json
Usage: python scripts/audit_trace_pipeline.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_collection" / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "analysis" / "artifacts" / "phase1_trace.json"
ASSETS = ["btc", "spx", "nifty"]
log = logging.getLogger("trace")


def shape_of(path: Path) -> dict:
    df = pd.read_parquet(path)
    idx = df.index
    info = {"path": str(path.relative_to(ROOT)), "shape": list(df.shape),
            "columns": list(map(str, df.columns)), "index_dtype": str(idx.dtype)}
    try:
        info["index_min"], info["index_max"] = str(idx.min()), str(idx.max())
    except Exception:  # noqa: BLE001 - non-orderable index
        pass
    return info


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out: dict = {"raw": {}, "feature_matrix": {}, "alignment": {}, "npy": {}}
    for p in sorted(RAW.rglob("*.parquet")):
        if "5min" in p.name:  # big: read metadata only
            import pyarrow.parquet as pq
            md = pq.ParquetFile(p).metadata
            out["raw"][p.name] = {"path": str(p.relative_to(ROOT)), "shape": [md.num_rows, md.num_columns],
                                  "columns": pq.ParquetFile(p).schema_arrow.names}
        else:
            out["raw"][p.name] = shape_of(p)
    for a in ASSETS:
        fm = pd.read_parquet(PROC / f"{a}_feature_matrix.parquet").sort_index()
        out["feature_matrix"][a] = {
            "shape": list(fm.shape), "columns": list(fm.columns),
            "start": str(fm.index.min()), "end": str(fm.index.max()),
            "nan_counts": {c: int(v) for c, v in fm.isna().sum().items() if v},
            "dtypes": {c: str(t) for c, t in fm.dtypes.items()},
        }
        lr = fm["log_RV"]
        # Does RV_d equal log_RV(t-1) (skips day t) or log_RV(t)?
        al = {
            "max_abs(RV_d - log_RV.shift(1))": float(np.nanmax(np.abs(fm["RV_d"] - lr.shift(1)))),
            "max_abs(RV_d - log_RV)": float(np.nanmax(np.abs(fm["RV_d"] - lr))),
            "max_abs(Target_RV - log_RV.shift(-1))": float(np.nanmax(np.abs(fm["Target_RV"] - lr.shift(-1)))),
            "corr(Target_RV, log_RV)": float(fm["Target_RV"].corr(lr)),
            "corr(Target_RV, RV_d)": float(fm["Target_RV"].corr(fm["RV_d"])),
            "calendar_gaps_days_value_counts": {str(k): int(v) for k, v in
                                                fm.index.to_series().diff().dt.days.value_counts().head(8).items()},
        }
        if "regime" in fm:
            r = fm["regime"].astype(str)
            al["regime_counts"] = r.value_counts().to_dict()
            al["regime_first_date"] = {k: str(fm.index[r == k].min()) for k in r.unique()}
        exo = [c for c in ["vpin_lag1", "obi_sq_lag1", "illiq_lag1", "FinBERT_score_lag1", "sent_surprise_lag1",
                           "credit_spread_lag1", "term_slope_lag1", "crypto_fg_lag1"] if c in fm]
        al["exo_first_valid"] = {c: str(fm[c].first_valid_index()) for c in exo}
        al["exo_frac_zero"] = {c: float((fm[c] == 0).mean()) for c in exo}
        out["alignment"][a] = al
        npy = {}
        for tag in ["har", "harss"]:
            p = PROC / f"forecasts_{tag}_{a}.npy"
            npy[tag] = list(np.load(p).shape) if p.exists() else None
        act = np.load(PROC / f"actuals_{a}.npy")
        npy["actuals"] = list(act.shape)
        tgt = fm.dropna(subset=["RV_d", "RV_w", "RV_m", "Target_RV"])["Target_RV"].to_numpy()
        npy["actuals_equal_Target_RV_tail"] = bool(np.allclose(act, tgt[-len(act):]))
        npy["neural_forecasts_present"] = (PROC / f"forecasts_neural_har_{a}.npy").exists()
        out["npy"][a] = npy
    news = pd.read_parquet(RAW / "sentiment" / "extended_news_headlines.parquet")
    news["date"] = pd.to_datetime(news["date"], utc=True)
    out["news"] = {
        "n_headlines": int(len(news)),
        "by_asset_year": {f"{a}|{y}": int(c) for (y, a), c in
                          news.groupby([news["date"].dt.year, "asset"]).size().items()},
        "by_asset_source": {f"{a}|{s}": int(c) for (a, s), c in news.groupby(["asset", "source"]).size().items()},
    }
    for a in ASSETS:
        s = pd.read_parquet(PROC / f"sentiment_daily_{a}.parquet").iloc[:, 0]
        out["news"][f"sentiment_daily_{a}"] = {"rows": int(len(s)), "start": str(s.index.min()),
                                               "end": str(s.index.max()), "nonzero_days": int((s != 0).sum())}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
