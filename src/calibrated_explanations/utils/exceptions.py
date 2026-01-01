"""Custom exception hierarchy for calibrated_explanations.

These exceptions standardize error signaling across the library.
Moved from core to utils to avoid circular dependencies (ADR-001).

ADR-002 compliance: All exceptions inherit from CalibratedError and support
structured error payloads via the ``details`` kwarg.
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
    "PlotPluginError",
    "explain_exception",
]


class CalibratedError(Exception):
    """Base class for library-specific errors."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        """Attach structured error details alongside the user-facing message."""
        super().__init__(message)
        self.details: dict[str, Any] | None = details

    def __repr__(self) -> str:  # pragma: no cover - repr stability check in tests
        """Return the exception representation with the message payload."""
        cls = self.__class__.__name__
        return f"{cls}({super().__str__()!r})"


class ValidationError(CalibratedError):
    """Inputs or configuration failed validation."""


class DataShapeError(ValidationError):
    """Provided data has incompatible shape or dtype (e.g., x/y mismatch)."""


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


def explain_exception(e: Exception) -> str:
    """Return a human-readable multi-line description of an exception.

    Formats library-specific ``CalibratedError`` instances with structured
    details for diagnostics and logging. For other exceptions, returns the
    standard string representation.

    Parameters
    ----------
    e : Exception
        The exception to format.

    Returns
    -------
    str
        Multi-line human-readable message.
        For CalibratedError, includes class name, message, and details dict if present.

    Examples
    --------
    >>> from calibrated_explanations.utils.exceptions import ValidationError, explain_exception
    >>> e = ValidationError("x must not be empty", details={"param": "x", "requirement": "non-empty"})
    >>> print(explain_exception(e))
    ValidationError: x must not be empty
      Details: {'param': 'x', 'requirement': 'non-empty'}
    """
    if isinstance(e, CalibratedError):
        lines = [f"{e.__class__.__name__}: {str(e)}"]
        if e.details is not None:
            lines.append(f"  Details: {e.details}")
        return "\n".join(lines)
    return str(e)


class PlotPluginError(CalibratedError):
    """Raised when a plot plugin encounters an unrecoverable error or is misconfigured."""
