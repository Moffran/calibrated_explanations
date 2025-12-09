"""Backward compatibility shim for calibration.summaries.

DEPRECATED: This module is a compatibility shim. The calibration package
has been moved to the top-level namespace as part of ADR-001 (Stage 1a).

All imports from calibrated_explanations.core.calibration.summaries will be
redirected to calibrated_explanations.calibration.summaries. This shim will
be removed in v1.1.0.

Migration guide:
- Old: from calibrated_explanations.core.calibration import get_calibration_summaries
- New: from calibrated_explanations.calibration import get_calibration_summaries
"""

import warnings

# Emit deprecation warning on first import
warnings.warn(
    "Importing from 'calibrated_explanations.core.calibration.summaries' is deprecated. "
    "Use 'calibrated_explanations.calibration.summaries' instead. "
    "This compatibility shim will be removed in v1.1.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from the new top-level package
from ...calibration.summaries import (  # noqa: F401, E402
    get_calibration_summaries,
    invalidate_calibration_summaries,
)

__all__ = ["get_calibration_summaries", "invalidate_calibration_summaries"]
