"""Interval summary selection for probabilistic predictions."""

from __future__ import annotations

from enum import Enum
from typing import Any


class IntervalSummary(Enum):
    """Describe how probabilistic intervals are summarized into point estimates.

    - REGULARIZED_MEAN: regularized Venn-Abers mean (default; legacy behavior).
    - MEAN: arithmetic mean of the interval bounds.
    - LOWER: lower interval bound.
    - UPPER: upper interval bound.
    """

    REGULARIZED_MEAN = "regularized_mean"
    MEAN = "mean"
    LOWER = "lower"
    UPPER = "upper"


def coerce_interval_summary(value: Any) -> IntervalSummary:
    """Return a validated IntervalSummary, defaulting to REGULARIZED_MEAN."""
    try:
        return IntervalSummary(value)
    except Exception:  # adr002_allow
        return IntervalSummary.REGULARIZED_MEAN


__all__ = ["IntervalSummary", "coerce_interval_summary"]
