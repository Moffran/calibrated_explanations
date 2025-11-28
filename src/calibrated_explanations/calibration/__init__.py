"""Calibration data and interval learner management.

This package encapsulates all calibration-related functionality including:
- Calibration dataset state management (x_cal, y_cal)
- Calibration summary caching (categorical counts, sorted numeric values)
- Venn-Abers calibration
- Interval learner management and initialization
- Interval regressor for conformal prediction

Part of ADR-001: Core Decomposition Boundaries (Stage 1a).

This is the new top-level calibration package. Backward compatibility shim
is maintained under core.calibration to support existing imports during
the migration period.
"""

from .interval_learner import (
    assign_threshold,
    initialize_interval_learner,
    initialize_interval_learner_for_fast_explainer,
    update_interval_learner,
)
from .interval_regressor import IntervalRegressor
from .state import CalibrationState
from .summaries import get_calibration_summaries, invalidate_calibration_summaries
from .venn_abers import VennAbers

__all__ = [
    "CalibrationState",
    "IntervalRegressor",
    "VennAbers",
    "assign_threshold",
    "get_calibration_summaries",
    "initialize_interval_learner",
    "initialize_interval_learner_for_fast_explainer",
    "invalidate_calibration_summaries",
    "update_interval_learner",
]
