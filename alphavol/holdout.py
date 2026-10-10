"""Sealing (Step 4) and verification (Step 9) of the fresh holdout (DECISIONS.md D14, D21, D31).

The holdout consists of rows whose **target date is after dev_end**. They are built by the same
``build_dataset`` code from development raw data (≤ dev_end) concatenated with sealed raw data
(> dev_end). This module is the only code allowed to read sealed raw files before unsealing. It
never returns holdout data to callers and runs with pipeline logging silenced, so no statistic,
date or row count is printed: only file paths and SHA-256 hashes.
"""
from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path

import pandas as pd

from alphavol import io as aio
from alphavol.config import ASSETS, PipelineConfig
from alphavol.features import GROUP_FEATURES, RawInputs, build_dataset, load_raw_inputs

SUMS = "SHA256SUMS"
INFO = "SEAL_INFO.json"


def sealed_raw_dir(cfg: PipelineConfig) -> Path:
    return cfg.sealed_dir / "raw"


@contextlib.contextmanager
def _silenced():
    lg = logging.getLogger("alphavol")
    old = lg.level
    lg.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        lg.setLevel(old)


def _read_sealed(path: Path) -> pd.DataFrame | None:
    """Privileged read used only while sealing or verifying. Returns None if the file is absent."""
    return pd.read_parquet(path) if path.exists() else None


def _concat(dev, sealed):
    if sealed is None:
        return dev
    if dev is None:
        return sealed
    out = pd.concat([dev, sealed])
    return out[~out.index.duplicated(keep="last")].sort_index()


def _utc(df):
    if df is None:
        return None
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def _combined_inputs(asset: str, cfg: PipelineConfig) -> RawInputs:
    dev = load_raw_inputs(asset, cfg)
    r = sealed_raw_dir(cfg)
    if asset == "btc":
        sp = _utc(_read_sealed(r / "ohlcv" / "btc_5min.parquet"))
        sp = sp[["close", "volume", "quote_asset_volume", "taker_buy_base_vol"]] if sp is not None else None
    else:
        sp = _utc(_read_sealed(r / "ohlcv" / f"{asset}_1d.parquet"))
        sp = sp[["open", "high", "low", "close"]] if sp is not None else None
    if sp is None:
        raise aio.MissingDataError(f"sealed prices for {asset} not found under {r}")
    fred = _utc(_read_sealed(r / "macro" / "fred.parquet"))
    fg = _read_sealed(r / "macro" / "crypto_fear_greed.parquet")
    iv = _read_sealed(r / "ohlcv" / "india_vix_1d.parquet")
    sent = _read_sealed(r / "sentiment" / f"sentiment_daily_{asset}.parquet")
    return RawInputs(
        prices=_concat(dev.prices, sp),
        fred=_concat(dev.fred, fred),
        fear_greed=_concat(dev.fear_greed, _utc(fg)["value"] if fg is not None else None) if asset == "btc" else None,
        india_vix=_concat(dev.india_vix, _utc(iv)["close"] if iv is not None else None) if asset == "nifty" else None,
        sentiment=_concat(dev.sentiment, _utc(sent)),
    )


def build_holdout_features(cfg: PipelineConfig, *, pending: set[str] | None = None) -> dict[str, str]:
    """Write ``data/sealed/<asset>_holdout.parquet`` for all assets; return {relative path: sha256}."""
    out = {}
    with _silenced():
        for asset in ASSETS:
            raw = _combined_inputs(asset, cfg)
            res = build_dataset(asset, raw, cfg, groups=list(GROUP_FEATURES[asset]), pending=pending)
            hold = res.data[res.data["target_date"] > cfg.dev_end]
            path = cfg.sealed_dir / f"{asset}_holdout.parquet"
            out[str(path.relative_to(cfg.sealed_dir))] = aio.write_frame(hold, path)
    return out


def write_sha256sums(cfg: PipelineConfig) -> str:
    """Hash every file under data/sealed (except the manifest itself); return the manifest's hash."""
    root = cfg.sealed_dir
    lines = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name not in (SUMS,):
            lines.append(f"{aio.sha256_file(p)}  {p.relative_to(root).as_posix()}")
    (root / SUMS).write_text("\n".join(lines) + "\n")
    return aio.sha256_file(root / SUMS)


def verify_sha256sums(cfg: PipelineConfig, expected_manifest_sha: str) -> dict:
    """Step 9: check the manifest hash recorded in DECISIONS.md and every file hash it lists."""
    root = cfg.sealed_dir
    got = aio.sha256_file(root / SUMS)
    bad = []
    for line in (root / SUMS).read_text().splitlines():
        digest, rel = line.split("  ", 1)
        if aio.sha256_file(root / rel) != digest:
            bad.append(rel)
    return {"manifest_sha256": got, "manifest_matches": got == expected_manifest_sha, "mismatched_files": bad}


def write_seal_info(cfg: PipelineConfig, info: dict) -> None:
    (cfg.sealed_dir / INFO).write_text(json.dumps(info, indent=2, sort_keys=True))
