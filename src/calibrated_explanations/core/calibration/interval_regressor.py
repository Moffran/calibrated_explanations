# pylint: disable=invalid-name, line-too-long, too-many-instance-attributes
"""Backward compatibility shim for calibration.interval_regressor.

DEPRECATED: This module is a compatibility shim. The calibration package
has been moved to the top-level namespace as part of ADR-001 (Stage 1a).

All imports from calibrated_explanations.core.calibration.interval_regressor will be
redirected to calibrated_explanations.calibration.interval_regressor. This shim will
be removed in v1.1.0.

Migration guide:
- Old: from calibrated_explanations.core.calibration import IntervalRegressor
- New: from calibrated_explanations.calibration import IntervalRegressor
"""

import warnings

# Emit deprecation warning on first import
warnings.warn(
    "Importing from 'calibrated_explanations.core.calibration.interval_regressor' is deprecated. "
    "Use 'calibrated_explanations.calibration.interval_regressor' instead. "
    "This compatibility shim will be removed in v1.1.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from the new top-level package
from ...calibration.interval_regressor import IntervalRegressor  # noqa: F401, E402

__all__ = ["IntervalRegressor"]
