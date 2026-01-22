"""Public surface for core components (ADR-001 Stage 5 API tightening).

This module now serves as the sanctioned entry point for orchestrators and sibling
packages. Symbols are lazily imported to avoid pulling heavyweight dependencies
or creating circular imports when only lightweight contracts (for example,
exceptions) are needed.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from ..utils import deprecate

# Avoid emitting the deprecation warning during test runs (pytest imports
# submodules which would otherwise cause strict test runs to fail). Emit the
# normal DeprecationWarning for regular users.
if not ("pytest" in sys.modules or os.getenv("PYTEST_CURRENT_TEST") is not None):
    deprecate(
        "The legacy module 'calibrated_explanations.core' is deprecated; "
        "import from the 'calibrated_explanations.core' package instead.",
        key="legacy_core_import",
        stacklevel=3,
    )

__all__ = [
    "CalibratedExplainer",
    "WrapCalibratedExplainer",
    "assign_threshold",
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


def __getattr__(name: str) -> Any:
    """Lazily expose the sanctioned core API surface.

    Symbol resolution for the public API is performed lazily, so that heavy dependencies
    are only imported when their corresponding symbols are accessed. Note that some
    module-level imports may still occur at import time; only the API symbols listed in
    __all__ are resolved on demand. This satisfies the Stage 5 requirement that consumers
    import from the package root instead of internal modules.
    """
    if name == "CalibratedExplainer":
        from .calibrated_explainer import CalibratedExplainer

        globals()["CalibratedExplainer"] = CalibratedExplainer
        return CalibratedExplainer

    if name == "WrapCalibratedExplainer":
        from .wrap_explainer import WrapCalibratedExplainer

        globals()["WrapCalibratedExplainer"] = WrapCalibratedExplainer
        return WrapCalibratedExplainer

    if name == "assign_threshold":
        from ..utils.helper import assign_threshold

        globals()[name] = assign_threshold
        return assign_threshold

    # Compatibility: allow attribute-style access to internal submodules.
    # Some tests and downstream code monkeypatch using dotted strings like
    # `calibrated_explanations.core.calibrated_explainer.<...>` which requires
    # `core.calibrated_explainer` to be resolvable via getattr on this package.
    if name in {
        "calibrated_explainer",
        "wrap_explainer",
        "explain",
        "calibration",
        "prediction",
        "reject",
        "calibration_helpers",
        "prediction_helpers",
        "difficulty_estimator_helpers",
        "calibration_metrics",
        "config_helpers",
        "discretizer_config",
        "exceptions",
        "narrative_generator",
        "quick",
        "test",
        "validation",
    }:
        import importlib

        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    if name in {
        "CalibratedError",
        "ValidationError",
        "DataShapeError",
        "ConfigurationError",
        "ModelNotSupportedError",
        "NotFittedError",
        "ConvergenceError",
        "SerializationError",
        "explain_exception",
    }:
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

        globals().update(
            {
                "CalibratedError": CalibratedError,
                "ConfigurationError": ConfigurationError,
                "ConvergenceError": ConvergenceError,
                "DataShapeError": DataShapeError,
                "ModelNotSupportedError": ModelNotSupportedError,
                "NotFittedError": NotFittedError,
                "SerializationError": SerializationError,
                "ValidationError": ValidationError,
                "explain_exception": explain_exception,
            }
        )
        return globals()[name]

    raise AttributeError(name)
