"""Config validation, sealed-holdout guard, seeding."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphavol import io as aio
from alphavol.config import load_config
from alphavol.seeding import seed_everything


def test_config_loads_and_is_consistent():
    cfg = load_config()
    assert set(cfg.assets) == {"btc", "spx", "nifty"}
    assert cfg.dev_end == pd.Timestamp("2026-07-28", tz="UTC")
    assert cfg.assets["btc"].har_windows == {"d": 1, "w": 7, "m": 30}
    assert cfg.assets["spx"].variance == "garman_klass"


def test_sealed_guard_blocks_reads(tmp_path):
    cfg = load_config()
    sealed_file = cfg.sealed_dir / "anything.parquet"
    with pytest.raises(aio.SealedDataError):
        aio.read_frame(sealed_file)
    # even with allow_sealed the unseal record must exist
    if not cfg.unseal_record.exists():
        with pytest.raises(aio.SealedDataError):
            aio.read_frame(sealed_file, allow_sealed=True)


def test_missing_input_is_loud(tmp_path):
    with pytest.raises(aio.MissingDataError):
        aio.read_frame(tmp_path / "nope.parquet")


def test_write_read_roundtrip_and_hash(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.date_range("2020-01-01", periods=2, tz="UTC"))
    h1 = aio.write_frame(df, tmp_path / "x.parquet")
    h2 = aio.sha256_file(tmp_path / "x.parquet")
    assert h1 == h2 and len(h1) == 64
    pd.testing.assert_frame_equal(aio.read_frame(tmp_path / "x.parquet"), df, check_freq=False)


def test_seeding_is_deterministic():
    import torch
    seed_everything(7)
    a = (np.random.rand(3), torch.rand(3))
    seed_everything(7)
    b = (np.random.rand(3), torch.rand(3))
    assert np.array_equal(a[0], b[0]) and torch.equal(a[1], b[1])
    with pytest.raises(ValueError):
        seed_everything(-1)
