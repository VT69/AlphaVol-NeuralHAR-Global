"""FinBERT label mapping, close-window assignment and aggregation (offline: fake scorer)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphavol.config import load_config
from alphavol.sentiment import (aggregate_daily, assign_to_trading_day, dedup_articles, probabilities_from_logits,
                                score_articles)

CFG = load_config()


def test_label_order_comes_from_config_not_position():
    logits = np.array([[5.0, 0.0, 0.0]])
    a = probabilities_from_logits(logits, {0: "positive", 1: "negative", 2: "neutral"})
    b = probabilities_from_logits(logits, {0: "negative", 1: "neutral", 2: "positive"})   # shuffled config
    assert a["p_pos"].iloc[0] > 0.9 and b["p_neg"].iloc[0] > 0.9
    with pytest.raises(ValueError):
        probabilities_from_logits(logits, {0: "pos", 1: "neg", 2: "neu"})


def test_spx_close_window_assignment_dst_and_weekend():
    spec = CFG.assets["spx"]
    days = pd.DatetimeIndex(["2024-03-08", "2024-03-11", "2024-03-12"], tz="UTC")   # Fri, Mon, Tue (US DST starts 03-10)
    pub = pd.Series(pd.to_datetime(["2024-03-08 20:59", "2024-03-08 21:01",       # Fri close 16:00 EST = 21:00 UTC
                                    "2024-03-10 12:00",                           # Sunday -> Monday
                                    "2024-03-11 19:59", "2024-03-11 20:01"],      # Mon close 16:00 EDT = 20:00 UTC
                                   utc=True))
    t = assign_to_trading_day(pub, days, spec)
    assert [str(x.date()) for x in t] == ["2024-03-08", "2024-03-11", "2024-03-11", "2024-03-11", "2024-03-12"]


def test_btc_window_is_utc_day():
    spec = CFG.assets["btc"]
    days = pd.date_range("2024-01-01", periods=3, tz="UTC")
    pub = pd.Series(pd.to_datetime(["2024-01-01 00:00:01", "2024-01-01 23:59:59", "2024-01-02 00:00:00"], utc=True))
    t = assign_to_trading_day(pub, days, spec)
    # an article stamped exactly at the close (00:00 of the next day) belongs to the closing day
    assert [str(x.date()) for x in t] == ["2024-01-01", "2024-01-01", "2024-01-01"]


def test_aggregate_no_zero_fill_and_dedup():
    spec = CFG.assets["btc"]
    days = pd.date_range("2024-01-01", periods=3, tz="UTC")
    arts = pd.DataFrame({"asset": "btc", "published_at": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00",
                                                                         "2024-01-03 09:00"], utc=True),
                         "title": ["Bitcoin up", "bitcoin  UP", "Crash"], "snippet": "", "url": ["a", "b", "c"]})
    arts = dedup_articles(arts)
    assert len(arts) == 2                                           # normalised-title duplicate removed
    fake = lambda texts: pd.DataFrame({"p_pos": [0.8 if "up" in x.lower() else 0.1 for x in texts],
                                       "p_neg": [0.1 if "up" in x.lower() else 0.8 for x in texts],
                                       "p_neu": [0.1] * len(texts)})
    agg = aggregate_daily(score_articles(arts, fake), days, spec)
    assert agg.loc["2024-01-01", "n_articles"] == 1 and abs(agg.loc["2024-01-01", "mean_score"] - 0.7) < 1e-12
    assert agg.loc["2024-01-02", "has_news"] == 0 and np.isnan(agg.loc["2024-01-02", "mean_score"])
    assert agg.loc["2024-01-03", "share_negative"] == 1.0
