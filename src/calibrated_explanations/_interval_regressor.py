"""Backward compatibility wrapper for :mod:`calibrated_explanations.core.interval_regressor`."""

from __future__ import annotations

# Intentionally import from the core module directly to avoid re-exporting
# miscellaneous legacy symbols and ensure a single implementation source.
from .core.interval_regressor import IntervalRegressor

__all__ = ["IntervalRegressor"]
