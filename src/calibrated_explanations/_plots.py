"""Backward compatibility wrapper for :mod:`calibrated_explanations.plotting`.

This shim mirrors the public API of ``calibrated_explanations.plotting`` while
emitting a deprecation warning to guide callers to the canonical module.
"""

from __future__ import annotations

import importlib
import warnings

# Emit a deprecation warning when the shim is imported
warnings.warn(
    "calibrated_explanations._plots is deprecated; import from "
    "calibrated_explanations.plotting instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all public names from the canonical plotting module
_canonical = importlib.import_module("calibrated_explanations.plotting")
__all__ = tuple(name for name in dir(_canonical) if not name.startswith("__"))

globals().update({name: getattr(_canonical, name) for name in __all__})
