"""Custom exception hierarchy for calibrated_explanations (Phase 1B).

These exceptions standardize error signaling across the core library without
changing the existing successful code paths.

Do not export these in the package-level ``calibrated_explanations.core.__all__`` yet
(to avoid API surface churn). Import them from this module directly.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CalibratedError",
    "ValidationError",
    "DataShapeError",
    "ConfigurationError",
    "ModelNotSupportedError",
    "NotFittedError",
    "ConvergenceError",
    "SerializationError",
]


class CalibratedError(Exception):
    """Base class for library-specific errors."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details: dict[str, Any] | None = details

    def __repr__(self) -> str:  # pragma: no cover - repr stability check in tests
        cls = self.__class__.__name__
        return f"{cls}({super().__str__()!r})"


class ValidationError(CalibratedError):
    """Inputs or configuration failed validation."""


class DataShapeError(ValidationError):
    """Provided data has incompatible shape or dtype (e.g., X/y mismatch)."""


class ConfigurationError(CalibratedError):
    """Invalid or conflicting configuration/parameter combination."""


class ModelNotSupportedError(CalibratedError):
    """Unsupported model type or missing required methods for task (e.g., predict_proba)."""


class NotFittedError(CalibratedError):
    """Operation requires a fitted estimator/explainer."""


class ConvergenceError(CalibratedError):
    """Optimization or calibration failed to converge within limits."""


class SerializationError(CalibratedError):
    """Failed to serialize/deserialize explanation artifacts."""
