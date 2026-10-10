"""Seal/verify cycle on SYNTHETIC post-boundary data (the real holdout is never read by tests)."""
from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pandas as pd

from alphavol import holdout as H
from alphavol.config import load_config


def _synthetic_sealed(root, D):
    rng = np.random.default_rng(0)
    raw = root / "raw"
    (raw / "ohlcv").mkdir(parents=True)
    (raw / "macro").mkdir(parents=True)
    n = 288 * 10
    idx = pd.date_range(D + pd.Timedelta(days=1), periods=n, freq="5min", tz="UTC")
    close = 60000 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    vol = rng.uniform(5, 15, n)
    pd.DataFrame({"close": close, "volume": vol, "quote_asset_volume": vol * close,
                  "taker_buy_base_vol": vol * 0.5}, index=idx).to_parquet(raw / "ohlcv" / "btc_5min.parquet")
    days = pd.bdate_range(D + pd.Timedelta(days=1), periods=8, tz="UTC")
    for a, lvl in (("spx", 6000.0), ("nifty", 25000.0)):
        c = lvl * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
        pd.DataFrame({"open": c * 0.999, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1.0},
                     index=days).to_parquet(raw / "ohlcv" / f"{a}_1d.parquet")
    fdays = pd.bdate_range(D + pd.Timedelta(days=1), periods=8, tz="UTC")
    pd.DataFrame({"T10Y2Y": 0.5, "BAA10Y": 2.0, "VIXCLS": 15.0}, index=fdays).to_parquet(raw / "macro" / "fred.parquet")
    pd.DataFrame({"value": 50.0}, index=pd.date_range(D + pd.Timedelta(days=1), periods=10, tz="UTC")).to_parquet(
        raw / "macro" / "crypto_fear_greed.parquet")


def test_seal_build_verify_and_tamper(tmp_path, caplog):
    base = load_config()
    cfg = dataclasses.replace(base, sealed_dir=tmp_path / "sealed")
    _synthetic_sealed(cfg.sealed_dir, base.dev_end)
    pending = {"macro.credit_spread", "sentiment", "implied_vol.india_vix"}
    with caplog.at_level(logging.DEBUG, logger="alphavol"):
        hashes = H.build_holdout_features(cfg, pending=pending)
    assert not [r for r in caplog.records if r.name.startswith("alphavol")]      # nothing printed
    assert set(hashes) == {"btc_holdout.parquet", "spx_holdout.parquet", "nifty_holdout.parquet"}
    for a in ("btc", "spx", "nifty"):
        df = pd.read_parquet(cfg.sealed_dir / f"{a}_holdout.parquet")
        assert len(df) and (df["target_date"] > base.dev_end).all()
        assert df.index.min() <= base.dev_end                                # row t=D (target D+1) included
    manifest_sha = H.write_sha256sums(cfg)
    assert H.verify_sha256sums(cfg, manifest_sha) == {"manifest_sha256": manifest_sha, "manifest_matches": True,
                                                      "mismatched_files": []}
    p = cfg.sealed_dir / "btc_holdout.parquet"
    p.write_bytes(p.read_bytes() + b"x")
    res = H.verify_sha256sums(cfg, manifest_sha)
    assert res["mismatched_files"] == ["btc_holdout.parquet"]
