"""Calibration data and interval learner management.

This package encapsulates all calibration-related functionality including:
- Calibration dataset state management (x_cal, y_cal)
- Calibration summary caching (categorical counts, sorted numeric values)
- Venn-Abers calibration
- Interval learner management and initialization
- Interval regressor for conformal prediction

Part of Phase 6: Refactor Calibration Functionality (ADR-001).
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
