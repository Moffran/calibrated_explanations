"""Backward compatibility shim for calibrated_explanations.calibration package.

DEPRECATED: This module is a compatibility shim. The calibration package has been
moved to the top-level namespace (calibrated_explanations.calibration) as part of
ADR-001: Core Decomposition Boundaries (Stage 1a).

All imports from calibrated_explanations.core.calibration will be redirected to
calibrated_explanations.calibration. This shim will be removed in v1.1.0.

Migration guide:
- Old: from calibrated_explanations.core.calibration import IntervalRegressor
- New: from calibrated_explanations.calibration import IntervalRegressor
"""

import warnings

# Emit deprecation warning on first import
warnings.warn(
    "Importing from 'calibrated_explanations.core.calibration' is deprecated. "
    "Use 'calibrated_explanations.calibration' instead. "
    "This compatibility shim will be removed in v1.1.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from the new top-level package
from ...calibration import (  # noqa: F401
    CalibrationState,
    IntervalRegressor,
    VennAbers,
    assign_threshold,
    get_calibration_summaries,
    initialize_interval_learner,
    initialize_interval_learner_for_fast_explainer,
    invalidate_calibration_summaries,
    update_interval_learner,
)

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
