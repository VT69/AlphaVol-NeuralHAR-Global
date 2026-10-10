"""Audit: why is sentiment coverage so sparse?

Quantifies, from the committed raw news files, the date span / density / cap-hits per
source and asset, query-relevance of the returned headlines, and how much non-zero
sentiment falls into each Neural-HAR split (train/val/test, 80/10/10 chronological).

Inputs : data_collection/data/raw/sentiment/{extended_news_headlines,newsapi_headlines}.parquet
         data/processed/{asset}_feature_matrix.parquet, sentiment_daily_{asset}.parquet
Output : analysis/artifacts/sentiment_coverage.json
Usage  : python scripts/audit_sentiment_coverage.py
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SENT = ROOT / "data_collection" / "data" / "raw" / "sentiment"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "analysis" / "artifacts" / "sentiment_coverage.json"
log = logging.getLogger("sentiment_coverage")

# Regexes used to judge whether a headline is actually about the asset it is assigned to.
RELEVANCE = {
    "BTC": r"bitcoin|\bbtc\b|crypto",
    "SPX": r"s&p|\bspx\b|\bspy\b|wall street|stock market|stocks|dow|nasdaq|fed\b",
    "NIFTY": r"nifty|sensex|\bnse\b|\bbse\b|india|dalal|rbi\b",
}


def span_stats(d: pd.Series) -> dict:
    d = d.sort_values()
    days = d.dt.floor("D")
    span = (d.max() - d.min()).total_seconds() / 86400
    return {"n": int(len(d)), "first": str(d.min()), "last": str(d.max()), "span_days": round(span, 2),
            "distinct_days": int(days.nunique()),
            "articles_per_day_in_span": round(len(d) / max(span, 1e-9), 2),
            "median_gap_hours": round(float(d.diff().dt.total_seconds().median() / 3600), 3)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out: dict = {}
    ext = pd.read_parquet(SENT / "extended_news_headlines.parquet")
    ext["date"] = pd.to_datetime(ext["date"], utc=True)
    out["extended_sources_present"] = sorted(ext["source"].unique().tolist())
    out["extended"] = {}
    for (asset, src), g in ext.groupby(["asset", "source"]):
        s = span_stats(g["date"])
        text = (g["title"].fillna("") + " " + g["description"].fillna("")).str.lower()
        s["frac_relevant_to_asset"] = round(float(text.str.contains(RELEVANCE[asset], regex=True).mean()), 4)
        if asset == "NIFTY":
            s["frac_mentioning_hdfc"] = round(float(text.str.contains("hdfc").mean()), 4)
        s["frac_with_av_score"] = round(float(g["av_sentiment_score"].notna().mean()), 4)
        # density in the most recent 60 days vs older: a cap-truncated "latest N" pull is dense at the end
        last = g["date"].max()
        s["n_in_last_60d"] = int((g["date"] > last - pd.Timedelta(days=60)).sum())
        out["extended"][f"{asset}|{src}"] = s
    # cross-asset duplicates that would have been removed by drop_duplicates(url,title)
    out["extended_dup_titles_across_assets"] = int(ext.duplicated(subset=["title"], keep=False).sum())

    na = pd.read_parquet(SENT / "newsapi_headlines.parquet")
    na["published_at"] = pd.to_datetime(na["published_at"], utc=True)
    out["newsapi"] = {a: span_stats(g["published_at"]) for a, g in na.groupby("asset")}
    for a, g in na.groupby("asset"):
        text = (g["title"].fillna("") + " " + g["description"].fillna("")).str.lower()
        out["newsapi"][a]["frac_relevant_to_asset"] = round(float(text.str.contains(RELEVANCE[a]).mean()), 4)
        out["newsapi"][a]["sample_titles"] = g["title"].head(5).tolist()
    out["newsapi_used_by_finbert_pipeline"] = False  # sentiment_engine reads extended_news_headlines only

    # Non-zero FinBERT days per Neural-HAR split
    out["finbert_by_split"] = {}
    for a in ["btc", "spx", "nifty"]:
        fm = pd.read_parquet(PROC / f"{a}_feature_matrix.parquet").sort_index()
        fm = fm.dropna(subset=["RV_d", "RV_w", "RV_m", "Target_RV"])
        n = len(fm); ntr, nva = int(n * 0.8), int(n * 0.1)
        nz = (fm["FinBERT_score_lag1"].fillna(0) != 0).to_numpy()
        out["finbert_by_split"][a] = {
            "train_rows": ntr, "train_nonzero": int(nz[:ntr].sum()),
            "val_rows": nva, "val_nonzero": int(nz[ntr:ntr + nva].sum()),
            "test_rows": n - ntr - nva, "test_nonzero": int(nz[ntr + nva:].sum()),
            "train_period": [str(fm.index[0].date()), str(fm.index[ntr - 1].date())],
            "test_period": [str(fm.index[ntr + nva].date()), str(fm.index[-1].date())],
        }
    OUT.write_text(json.dumps(out, indent=2))
    log.info(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
