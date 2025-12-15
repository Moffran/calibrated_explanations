"""Custom exception hierarchy for calibrated_explanations (Phase 1B).

This module is a backward-compatibility shim.
Exceptions have been moved to `calibrated_explanations.utils.exceptions`
to avoid circular dependencies.
"""

from __future__ import annotations

from ..utils.exceptions import (
    CalibratedError,
    ConfigurationError,
    ConvergenceError,
    DataShapeError,
    ModelNotSupportedError,
    NotFittedError,
    SerializationError,
    ValidationError,
    explain_exception,
)

__all__ = [
    "CalibratedError",
    "ValidationError",
    "DataShapeError",
    "ConfigurationError",
    "ModelNotSupportedError",
    "NotFittedError",
    "ConvergenceError",
    "SerializationError",
    "explain_exception",
]
