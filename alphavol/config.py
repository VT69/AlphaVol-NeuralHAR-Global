"""Typed access to ``configs/pipeline.yaml``. All paths resolve relative to the repository root."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "pipeline.yaml"
ASSETS: tuple[str, ...] = ("btc", "spx", "nifty")


@dataclass(frozen=True)
class AssetSpec:
    """Per-asset construction rules (see DECISIONS.md D11-D14, D20)."""

    name: str
    calendar: Literal["continuous", "exchange"]
    close_tz: str
    close_time: str
    variance: Literal["realized_5min", "garman_klass"]
    har_windows: dict[str, int]
    har_min_counts: dict[str, int]
    min_bars_per_day: int | None = None
    flag_bars_below: int | None = None


@dataclass(frozen=True)
class PipelineConfig:
    raw_dir: Path
    extra_raw_dir: Path
    processed_dir: Path
    sealed_dir: Path
    unseal_record: Path
    dev_end: pd.Timestamp
    assets: dict[str, AssetSpec]
    max_staleness_business_days: int
    fred_lag_business_days: int
    fear_greed_lag_days: int
    extras: dict = field(default_factory=dict)


def _resolve(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        raise ValueError(f"Config paths must be repository-relative, got absolute path: {p}")
    return REPO_ROOT / path


@lru_cache(maxsize=4)
def load_config(path: Path = CONFIG_PATH) -> PipelineConfig:
    """Load and validate the pipeline configuration."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    assets = {}
    for name, spec in raw["assets"].items():
        if name not in ASSETS:
            raise ValueError(f"Unknown asset in config: {name}")
        windows = spec["har_windows"]
        mins = spec["har_min_counts"]
        if not (windows["d"] == 1 and 1 < windows["w"] < windows["m"]):
            raise ValueError(f"{name}: HAR windows must satisfy d=1 < w < m, got {windows}")
        if not (0 < mins["w"] <= windows["w"] and 0 < mins["m"] <= windows["m"]):
            raise ValueError(f"{name}: min counts {mins} inconsistent with windows {windows}")
        assets[name] = AssetSpec(name=name, **spec)
    exo = raw["exogenous"]
    return PipelineConfig(
        raw_dir=_resolve(raw["paths"]["raw_dir"]),
        extra_raw_dir=_resolve(raw["paths"]["extra_raw_dir"]),
        processed_dir=_resolve(raw["paths"]["processed_dir"]),
        sealed_dir=_resolve(raw["paths"]["sealed_dir"]),
        unseal_record=_resolve(raw["paths"]["unseal_record"]),
        dev_end=pd.Timestamp(raw["dev_end"], tz="UTC"),
        assets=assets,
        max_staleness_business_days=int(exo["max_staleness_business_days"]),
        fred_lag_business_days=int(exo["fred_lag_business_days"]),
        fear_greed_lag_days=int(exo["fear_greed_lag_days"]),
    )
