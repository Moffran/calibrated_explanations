"""Backward compatibility shim for calibration.state.

DEPRECATED: This module is a compatibility shim. The calibration package
has been moved to the top-level namespace as part of ADR-001 (Stage 1a).

All imports from calibrated_explanations.core.calibration.state will be
redirected to calibrated_explanations.calibration.state. This shim will
be removed in v1.1.0.

Migration guide:
- Old: from calibrated_explanations.core.calibration import CalibrationState
- New: from calibrated_explanations.calibration import CalibrationState
"""

import warnings

# Emit deprecation warning on first import
warnings.warn(
    "Importing from 'calibrated_explanations.core.calibration.state' is deprecated. "
    "Use 'calibrated_explanations.calibration.state' instead. "
    "This compatibility shim will be removed in v1.1.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from the new top-level package
from ...calibration.state import CalibrationState  # noqa: F401, E402

__all__ = ["CalibrationState"]
