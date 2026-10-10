"""End-to-end plumbing of scripts/local/run_local.py with FAKE vendors and a FAKE FinBERT.

No network, no real holdout. Verifies that the four stages run in order, write the expected
files, apply the selection rule, keep dev/holdout separated at dev_end, and that the holdout
stage prints only hashes.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from alphavol.config import REPO_ROOT, load_config

spec = importlib.util.spec_from_file_location("run_local", REPO_ROOT / "scripts" / "local" / "run_local.py")
RL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RL)


class FakeSource:
    def __init__(self, name, per_day):
        self.name, self.per_day = name, per_day

    def count(self, asset, s, e):
        return int(self.per_day * (e - s).days)

    def articles(self, asset, s, e, flog, sample=False):
        flog.requests += 1
        n = max(1, int(self.per_day * max((e - s).total_seconds() / 86400, 0.1)))
        return [{"source": self.name, "asset": asset, "published_at": pd.Timestamp(s) + timedelta(hours=1 + i % 20),
                 "title": f"{asset} markets {'rally' if i % 2 else 'slump'} {s:%Y%m%d%H} {i}", "snippet": "",
                 "url": f"https://x/{self.name}/{asset}/{s:%Y%m%d%H%M}/{i}", "domain": "x", "language": "en",
                 "country": ""} for i in range(n)]

    def daily_volume(self, asset, s, e):
        idx = pd.date_range(pd.Timestamp(s).normalize(), pd.Timestamp(e) - timedelta(seconds=1), freq="D")
        return pd.Series(100.0, index=idx, name="gdelt_volume")


class FakeScorer:
    def __init__(self, revision=None):
        self.revision = revision

    def load(self):
        return {"model": "fake", "resolved_revision": "abc123", "id2label": {0: "positive", 1: "negative", 2: "neutral"}}

    def __call__(self, texts):
        pos = np.array([0.7 if "rally" in t else 0.2 for t in texts])
        return pd.DataFrame({"p_pos": pos, "p_neg": 0.9 - pos, "p_neu": 0.1})


def _fake_market(monkeypatch):
    def fred(code, start, end, cache_tag=None, client=None):
        idx = pd.bdate_range(start, end, tz="UTC")
        return pd.Series({"T10Y2Y": 0.5, "BAA10Y": 2.0, "VIXCLS": 18.0}[code], index=idx, name=code)

    def fg(cache_tag=None, client=None):
        return pd.Series(50.0, index=pd.date_range("2018-01-01", "2026-12-31", tz="UTC"), name="fear_greed")

    def yahoo(ticker, start, end_inclusive):
        idx = pd.bdate_range(start, end_inclusive, tz="UTC")
        c = 100 * np.exp(np.cumsum(np.full(len(idx), 0.001)))
        return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1.0}, index=idx)

    def klines(start, end, **kw):
        idx = pd.date_range(start, end - timedelta(minutes=5), freq="5min", tz="UTC")
        c = 60000 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.001, len(idx))))
        return pd.DataFrame({"close": c, "volume": 10.0, "quote_asset_volume": 10.0 * c, "taker_buy_base_vol": 5.0},
                            index=idx)

    monkeypatch.setattr(RL, "fred_series", fred)
    monkeypatch.setattr(RL, "fear_greed", fg)
    monkeypatch.setattr(RL, "yahoo_daily", yahoo)
    monkeypatch.setattr(RL, "binance_klines", klines)


@pytest.mark.slow
def test_four_stages_end_to_end(tmp_path, monkeypatch, capsys):
    base = load_config()
    cfg = dataclasses.replace(base, extra_raw_dir=tmp_path / "raw_v2", sealed_dir=tmp_path / "sealed")
    for name, val in (("CFG", cfg), ("RAW2", cfg.extra_raw_dir), ("ART", tmp_path / "art"),
                      ("CACHE_ARTICLES", tmp_path / "cache")):
        monkeypatch.setattr(RL, name, val)
    gd, gu, ny = FakeSource("gdelt", 5), FakeSource("guardian", 3), FakeSource("nyt", 0)
    monkeypatch.setattr(RL, "GdeltDoc", lambda: gd)
    monkeypatch.setattr(RL, "Guardian", lambda: gu)
    monkeypatch.setattr(RL, "NYT", lambda: ny)
    monkeypatch.setattr(RL, "AlphaVantage", lambda: FakeSource("av", 1))
    monkeypatch.setattr(RL, "FinbertScorer", FakeScorer)
    _fake_market(monkeypatch)
    args = type("A", (), {"nyt_quota": 500, "guardian_quota": 500, "revision": None, "end": "2026-08-10",
                          "pending": []})()

    RL.stage_probe(args)
    probe = json.loads((tmp_path / "art" / "sentiment_probe.json").read_text())
    assert probe["status"]["nyt"] == "ok" and probe["selection"]["choice"]["sources"] in (["gdelt"], ["gdelt", "guardian"])
    # shorten the development news fetch for the test: pretend the chosen period starts in 2026
    probe["selection"]["choice"]["common_start_year"] = 2026
    (tmp_path / "art" / "sentiment_probe.json").write_text(json.dumps(probe))

    RL.stage_fetch_dev(args)
    assert (cfg.extra_raw_dir / "macro" / "fred.parquet").exists()
    RL.stage_score_dev(args)
    for a in ("btc", "spx", "nifty"):
        s = pd.read_parquet(cfg.extra_raw_dir / "sentiment" / f"sentiment_daily_{a}.parquet")
        assert s.index.max() <= base.dev_end and {"n_articles", "has_news", "mean_score", "vader_mean"} <= set(s.columns)
    scored = list((cfg.extra_raw_dir / "sentiment" / "articles_scored").rglob("*.parquet"))
    assert scored and "title" not in pd.read_parquet(scored[0]).columns          # D29: no text committed

    capsys.readouterr()
    RL.stage_seal_holdout(args)
    out = capsys.readouterr().out
    assert out.startswith("SEALED.") and "SHA256SUMS sha256 = " in out and len(out.splitlines()) == 3
    for a in ("btc", "spx", "nifty"):
        h = pd.read_parquet(cfg.sealed_dir / f"{a}_holdout.parquet")
        assert (h["target_date"] > base.dev_end).all() and "sent_mean_score" in h.columns
    assert (cfg.sealed_dir / "SHA256SUMS").exists() and (cfg.sealed_dir / "SEAL_INFO.json").exists()
