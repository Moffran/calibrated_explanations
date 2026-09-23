"""
Calibrated Explanations (calibrated_explanations).

is a Python package for explaining black-box models.

It is based on the paper "Calibrated Explanations: with Uncertainty Information and Counterfactuals"
by Helena Löfström et al.
"""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import logging as _logging
from typing import Any


def _resolve_package_version() -> str:
    """Return the installed package version or the checked-in fallback."""
    for distribution_name in ("calibrated_explanations", "calibrated-explanations"):
        try:
            return importlib_metadata.version(distribution_name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return "1.0.2.dev0"


__version__ = _resolve_package_version()

# Expose viz namespace lazily via __getattr__ (avoid importing heavy backends eagerly)
# Note: avoid eager imports of explanation, viz and discretizer modules here.
# Those modules import heavy dependencies (numpy, pandas, plotting backends)
# and should be loaded lazily via __getattr__ below. Importing them at
# package import time increases startup cost significantly.

# Provide a default no-op handler to avoid "No handler" warnings for library users.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

# Note: core submodules are intentionally not imported here to avoid importing
# large backends and to make deprecation transitions explicit. We still expose
# the public symbols lazily so `from calibrated_explanations import CalibratedExplainer`
# works without triggering an eager import of `calibrated_explanations.core`.
__all__ = [
    "CalibratedExplainer",
    "ExplainerBuilder",
    "ExplainerConfig",
    "GuardedOptions",
    "NormalizationStrategy",
    "RejectPolicySpec",
    "WrapCalibratedExplainer",
    "configure_logging",
    "transform_to_numeric",
]


def __getattr__(name: str) -> Any:
    """Lazy import for sanctioned public symbols.

    This avoids importing `calibrated_explanations.core` at package import time
    while preserving the public API surface for users and tests.

    Sanctioned symbols (no deprecation warning):
    - CalibratedExplainer, WrapCalibratedExplainer, transform_to_numeric

    Removed in v0.11.0:
    - Top-level compatibility exports for explanation classes/discretizers/calibrators
    - Top-level ``viz`` and ``plotting`` compatibility aliases
    """
    # ===================================================================
    # SANCTIONED SYMBOLS (no deprecation warning)
    # ===================================================================

    if name == "CalibratedExplainer":
        from .core.calibrated_explainer import CalibratedExplainer

        globals()[name] = CalibratedExplainer
        return CalibratedExplainer

    if name == "WrapCalibratedExplainer":
        from .core.wrap_explainer import WrapCalibratedExplainer

        globals()[name] = WrapCalibratedExplainer
        return WrapCalibratedExplainer

    if name == "transform_to_numeric":
        module = importlib.import_module(f"{__name__}.utils")
        value = getattr(module, name)
        globals()[name] = value
        return value

    if name == "GuardedOptions":
        from .explanations.guarded_options import GuardedOptions

        globals()[name] = GuardedOptions
        return GuardedOptions

    if name == "RejectPolicySpec":
        from .explanations.reject import RejectPolicySpec

        globals()[name] = RejectPolicySpec
        return RejectPolicySpec

    if name in ("ExplainerBuilder", "ExplainerConfig"):
        from .api.config import ExplainerBuilder, ExplainerConfig  # noqa: F401

        globals()["ExplainerBuilder"] = ExplainerBuilder
        globals()["ExplainerConfig"] = ExplainerConfig
        return globals()[name]

    if name == "NormalizationStrategy":
        from .calibration.normalization_strategy import NormalizationStrategy

        globals()[name] = NormalizationStrategy
        return NormalizationStrategy

    if name == "configure_logging":
        from .logging import configure_logging

        globals()[name] = configure_logging
        return configure_logging

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__getattr__.__annotations__["return"] = Any
