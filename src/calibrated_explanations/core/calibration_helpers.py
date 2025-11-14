"""Calibration helper delegators (DEPRECATED - use calibration.interval_learner instead).

DEPRECATED in v0.10.0: This module is deprecated and will be removed in v1.0.0.
Use ``calibrated_explanations.core.calibration.interval_learner`` instead.

This module maintains backward compatibility by re-exporting functions that have been
moved to the calibration subpackage. All new code should import from the calibration
subpackage directly.

Part of Phase 6: Refactor Calibration Functionality (ADR-001).
"""

from __future__ import annotations

import warnings

__all__ = [
    "assign_threshold",
    "initialize_interval_learner",
    "initialize_interval_learner_for_fast_explainer",
    "update_interval_learner",
]


def __getattr__(name: str):
    """Lazy-load functions from calibration.interval_learner with deprecation warning."""
    if name in __all__:
        warnings.warn(
            f"Importing {name} from calibrated_explanations.core.calibration_helpers is deprecated "
            "and will be removed in v1.0.0. "
            f"Import from calibrated_explanations.core.calibration.interval_learner instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .calibration import interval_learner as _il  # pylint: disable=import-outside-toplevel

        return getattr(_il, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

