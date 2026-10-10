"""Guarded I/O. Reading anything under the sealed holdout directory raises unless unsealed (D21)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from alphavol.config import load_config
from alphavol.logging_utils import get_logger

log = get_logger(__name__)


class SealedDataError(PermissionError):
    """Raised on any attempt to read the sealed holdout before Step 9 (DECISIONS.md D21)."""


class MissingDataError(FileNotFoundError):
    """Raised when a required input is absent. Never silently replaced by a default."""


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def assert_not_sealed(path: Path, *, allow_sealed: bool = False) -> None:
    cfg = load_config()
    if _is_under(Path(path), cfg.sealed_dir):
        if not allow_sealed:
            raise SealedDataError(f"Refusing to read sealed holdout file {path} (pass allow_sealed=True only in Step 9)")
        if not cfg.unseal_record.exists():
            raise SealedDataError(f"{cfg.unseal_record} does not exist: the holdout has not been formally unsealed")


def read_frame(path: Path | str, *, allow_sealed: bool = False, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a parquet file after the sealed-holdout check; raise MissingDataError if absent."""
    path = Path(path)
    assert_not_sealed(path, allow_sealed=allow_sealed)
    if not path.exists():
        raise MissingDataError(f"Required input not found: {path}")
    return pd.read_parquet(path, columns=columns)


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes (does not parse contents; safe for sealed files)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_frame(df: pd.DataFrame, path: Path | str) -> str:
    """Write parquet deterministically and return its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=True)
    digest = sha256_file(path)
    log.info("wrote %s (sha256 %s)", path, digest[:16])
    return digest


def write_json(obj: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
