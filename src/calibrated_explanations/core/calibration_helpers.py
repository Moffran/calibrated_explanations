"""Calibration helper utilities.

Part of Phase 6: Refactor Calibration Functionality (ADR-001).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "identify_constant_features",
]


def __getattr__(name: str):
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
    constant_columns = [f for f in range(x_cal.shape[1]) if np.all(x_cal[:, f] == x_cal[0, f])]
    return constant_columns
