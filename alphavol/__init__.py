"""alphavol — rebuilt, tested pipeline for the HAR / sentiment volatility study.

Replaces the legacy code paths (run_paper.py, src/, eval/, backtest/), which are frozen for
provenance (analysis/DECISIONS.md D10).
"""
__all__ = ["config", "io", "seeding", "logging_utils", "targets", "har", "features", "metrics"]
__version__ = "0.1.0"
