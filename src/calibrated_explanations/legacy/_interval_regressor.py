"""Deprecated shim for :mod:`calibrated_explanations._interval_regressor`."""

from __future__ import annotations

from warnings import warn

from ..core.interval_regressor import IntervalRegressor

warn(
    "'calibrated_explanations._interval_regressor' is deprecated; import from "
    "'calibrated_explanations.core.interval_regressor' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["IntervalRegressor"]
