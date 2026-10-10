"""Global seeding for reproducible runs (DECISIONS.md R1; audit finding T3)."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed python, numpy and (if installed) torch; optionally force deterministic torch kernels.

    Optuna samplers must additionally be constructed with ``TPESampler(seed=seed)`` by the caller,
    because Optuna does not read global RNG state.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # torch is optional for data-only entry points
        return
    torch.manual_seed(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
