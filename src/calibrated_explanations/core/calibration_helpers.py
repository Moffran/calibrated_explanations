"""Calibration helper delegators and utilities.

This module provides:
- Backward-compatible delegators (DEPRECATED - use calibration.interval_learner instead)
- New calibration state and preprocessing utilities

Part of Phase 6: Refactor Calibration Functionality (ADR-001).
"""

from __future__ import annotations

import warnings
import numpy as np

__all__ = [
    "assign_threshold",
    "initialize_interval_learner",
    "initialize_interval_learner_for_fast_explainer",
    "update_interval_learner",
    "identify_constant_features",
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


def identify_constant_features(x_cal: np.ndarray) -> list:
    """Identify constant (non-varying) columns in calibration data.

    Parameters
    ----------
    x_cal : np.ndarray
        Calibration input data of shape (n_samples, n_features).

    Returns
    -------
    list
        Indices of features that have constant values across all calibration samples.
    """
    constant_columns = [
        f for f in range(x_cal.shape[1]) if np.all(x_cal[:, f] == x_cal[0, f])
    ]
    return constant_columns


