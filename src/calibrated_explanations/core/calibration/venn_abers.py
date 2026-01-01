# ruff: noqa: N999
# pylint: disable=unknown-option-value
"""Backward compatibility shim for calibration.venn_abers.

DEPRECATED: This module is a compatibility shim. The calibration package
has been moved to the top-level namespace as part of ADR-001 (Stage 1a).

All imports from calibrated_explanations.core.calibration.venn_abers will be
redirected to calibrated_explanations.calibration.venn_abers. This shim will
be removed in v1.1.0.

Migration guide:
- Old: from calibrated_explanations.core.calibration import VennAbers
- New: from calibrated_explanations.calibration import VennAbers
"""

import warnings

# Emit deprecation warning on first import
warnings.warn(
    "Importing from 'calibrated_explanations.core.calibration.venn_abers' is deprecated. "
    "Use 'calibrated_explanations.calibration.venn_abers' instead. "
    "This compatibility shim will be removed in v1.1.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from the new top-level package
from ...calibration.venn_abers import VennAbers  # noqa: F401, E402

__all__ = ["VennAbers"]
