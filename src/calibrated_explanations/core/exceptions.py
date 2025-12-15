"""Compatibility shim exposing the exception hierarchy under ``core``.

Historically the central exception types lived directly under
``calibrated_explanations.core``.  They were migrated to
``calibrated_explanations.utils.exceptions`` (ADR-002) but documentation,
type-check configuration and downstream code still import from
``calibrated_explanations.core.exceptions``.  Sphinx therefore attempts to
``autodoc`` this module during the docs build.  Without a real module the
import fails, bringing the entire docs workflow down.

To keep the public API stable we provide a light re-export module that simply
forwards the canonical implementations from ``utils.exceptions``.  This keeps
documentation, mypy configuration, and any downstream imports working without
duplicating logic.
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
    "ConfigurationError",
    "ConvergenceError",
    "DataShapeError",
    "ModelNotSupportedError",
    "NotFittedError",
    "SerializationError",
    "ValidationError",
    "explain_exception",
]
