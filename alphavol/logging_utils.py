"""Logging setup shared by all alphavol entry points (no print statements in library code)."""
from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a module logger. Configuration is left to entry points via :func:`configure`."""
    return logging.getLogger(name)


def configure(level: int = logging.INFO) -> None:
    """Configure root logging once for a command-line entry point (idempotent)."""
    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
    root.setLevel(level)
