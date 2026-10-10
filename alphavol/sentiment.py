"""FinBERT scoring and aggregation of article sentiment to trading days (DECISIONS.md D20, D27).

* Label order comes from ``model.config.id2label`` and is asserted; nothing is hard-coded.
* The resolved model revision (Hugging Face commit hash) is returned for the manifest.
* Article → trading day t if published in (close_{t−1}, close_t] on the asset's clock.
* Days without news: ``n_articles = 0``, ``has_news = 0``, score features NaN (never 0).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from alphavol.config import AssetSpec
from alphavol.logging_utils import get_logger

log = get_logger(__name__)
EXPECTED_LABELS = {"positive", "negative", "neutral"}


class Scorer(Protocol):
    def __call__(self, texts: list[str]) -> pd.DataFrame: ...   # columns p_pos, p_neg, p_neu


def probabilities_from_logits(logits: np.ndarray, id2label: dict[int, str]) -> pd.DataFrame:
    """Softmax over logits, columns ordered by the model's own label map (asserted)."""
    labels = {int(k): str(v).lower() for k, v in id2label.items()}
    if set(labels.values()) != EXPECTED_LABELS:
        raise ValueError(f"unexpected FinBERT labels {labels}; expected {EXPECTED_LABELS}")
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    inv = {v: k for k, v in labels.items()}
    return pd.DataFrame({"p_pos": p[:, inv["positive"]], "p_neg": p[:, inv["negative"]],
                         "p_neu": p[:, inv["neutral"]]})


@dataclass
class FinbertScorer:
    """Wraps a Hugging Face sequence-classification model. Loaded lazily (local runs only)."""

    model_name: str = "ProsusAI/finbert"
    revision: str | None = None
    batch_size: int = 64
    max_length: int = 128

    def load(self) -> dict:
        import torch
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.use_deterministic_algorithms(True)
        self.tok = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, revision=self.revision)
        self.model.eval()
        resolved = getattr(self.model.config, "_commit_hash", None)
        manifest = {"model": self.model_name, "requested_revision": self.revision, "resolved_revision": resolved,
                    "id2label": {int(k): v for k, v in self.model.config.id2label.items()},
                    "transformers": transformers.__version__, "torch": torch.__version__,
                    "max_length": self.max_length}
        probabilities_from_logits(np.zeros((1, 3)), self.model.config.id2label)   # asserts labels
        log.info("FinBERT loaded: %s @ %s", self.model_name, resolved)
        return manifest

    def __call__(self, texts: list[str]) -> pd.DataFrame:
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tok(texts[i:i + self.batch_size], padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt")
                out.append(self.model(**enc).logits.numpy())
        logits = np.concatenate(out) if out else np.zeros((0, 3))
        return probabilities_from_logits(logits, self.model.config.id2label)


_WS = re.compile(r"\s+")


def normalise_title(t: str) -> str:
    return _WS.sub(" ", str(t)).strip().lower()


def dedup_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Drop cross-source duplicates within an asset: same URL or same normalised title."""
    d = df.copy()
    d["_nt"] = d["title"].map(normalise_title)
    d = d[d["_nt"] != ""]
    d = d.sort_values("published_at").drop_duplicates(subset=["asset", "url"]).drop_duplicates(subset=["asset", "_nt"])
    return d.drop(columns="_nt")


def article_text(df: pd.DataFrame) -> list[str]:
    snip = df["snippet"].fillna("").astype(str).str.replace(r"<[^>]+>", "", regex=True)
    return [f"{t}. {s}".strip(". ") if s else str(t) for t, s in zip(df["title"].astype(str), snip)]


def close_times(trading_dates: pd.DatetimeIndex, spec: AssetSpec) -> pd.DatetimeIndex:
    """UTC timestamp of the close of each trading date (end of UTC day for continuous calendars)."""
    if spec.calendar == "continuous":
        return pd.DatetimeIndex(trading_dates) + pd.Timedelta(days=1)
    local = pd.DatetimeIndex(trading_dates).tz_localize(None).normalize() + pd.Timedelta(spec.close_time + ":00")
    return local.tz_localize(spec.close_tz).tz_convert("UTC")


def assign_to_trading_day(published_at: pd.Series, trading_dates: pd.DatetimeIndex, spec: AssetSpec) -> pd.Series:
    """Map each publication time to the trading date t with published ∈ (close_{t−1}, close_t]."""
    closes = close_times(trading_dates, spec)
    if not closes.is_monotonic_increasing:
        raise ValueError("trading dates must be sorted")
    pos = np.searchsorted(closes.asi8, pd.DatetimeIndex(published_at).asi8, side="left")
    out = pd.Series(pd.NaT, index=published_at.index, dtype="datetime64[ns, UTC]")
    ok = pos < len(trading_dates)
    out[ok] = pd.DatetimeIndex(trading_dates)[pos[ok]]
    return out


def aggregate_daily(scored: pd.DataFrame, trading_dates: pd.DatetimeIndex, spec: AssetSpec) -> pd.DataFrame:
    """Daily sentiment features on the asset's trading calendar (no zero-fill)."""
    s = scored.copy()
    s["t"] = assign_to_trading_day(s["published_at"], trading_dates, spec)
    s = s.dropna(subset=["t"])
    s["score"] = s["p_pos"] - s["p_neg"]
    s["neg"] = (s["p_neg"] > s["p_pos"]).astype(float)
    g = s.groupby("t")
    agg = pd.DataFrame({"n_articles": g.size(), "mean_score": g["score"].mean(), "mean_pos": g["p_pos"].mean(),
                        "mean_neg": g["p_neg"].mean(), "share_negative": g["neg"].mean(),
                        "score_std": g["score"].std(ddof=0)})
    out = agg.reindex(pd.DatetimeIndex(trading_dates))
    out["n_articles"] = out["n_articles"].fillna(0).astype(int)
    out["has_news"] = (out["n_articles"] > 0).astype("int8")
    out.index.name = "date"
    return out


def score_articles(df: pd.DataFrame, scorer: Callable[[list[str]], pd.DataFrame]) -> pd.DataFrame:
    probs = scorer(article_text(df))
    if len(probs) != len(df):
        raise ValueError("scorer returned a different number of rows")
    return pd.concat([df.reset_index(drop=True), probs.reset_index(drop=True)], axis=1)
