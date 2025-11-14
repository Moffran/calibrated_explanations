# pylint: disable=invalid-name, line-too-long, too-many-instance-attributes
"""Interval regression helpers built on conformal calibration.

DEPRECATED: This module has been moved to calibrated_explanations.core.calibration.interval_regressor.

This module is maintained for backward compatibility. All imports should be updated to use:
    from calibrated_explanations.core.calibration.interval_regressor import IntervalRegressor

Part of Phase 6: Refactor Calibration Functionality (ADR-001).
"""

import warnings

# Re-export from new location for backward compatibility
# Note: We use filterwarnings to allow pytest to properly configure warning handling
with warnings.catch_warnings():
    warnings.filterwarnings("default", category=DeprecationWarning)
    from .calibration.interval_regressor import IntervalRegressor

__all__ = ["IntervalRegressor"]



