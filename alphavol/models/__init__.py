"""Forecasting models. Every model maps a training frame to a fitted object whose ``predict``
returns **log-variance** forecasts for requested rows, given a context frame in which the targets
of those rows (and later rows) are masked (DECISIONS.md D34)."""
