"""Backward compatibility shim for calibration.interval_learner.

DEPRECATED: This module is a compatibility shim. The calibration package
has been moved to the top-level namespace as part of ADR-001 (Stage 1a).

All imports from calibrated_explanations.core.calibration.interval_learner will be
redirected to calibrated_explanations.calibration.interval_learner. This shim will
be removed in v1.1.0.

Migration guide:
- Old: from calibrated_explanations.core.calibration import assign_threshold
- New: from calibrated_explanations.calibration import assign_threshold
"""

import warnings

# Emit deprecation warning on first import
warnings.warn(
    "Importing from 'calibrated_explanations.core.calibration.interval_learner' is deprecated. "
    "Use 'calibrated_explanations.calibration.interval_learner' instead. "
    "This compatibility shim will be removed in v1.1.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from the new top-level package
from ...calibration.interval_learner import (  # noqa: F401, E402
    assign_threshold,
    initialize_interval_learner,
    initialize_interval_learner_for_fast_explainer,
    update_interval_learner,
)

__all__ = [
    "assign_threshold",
    "initialize_interval_learner",
    "initialize_interval_learner_for_fast_explainer",
    "update_interval_learner",
]
