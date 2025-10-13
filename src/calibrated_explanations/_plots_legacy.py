"""Backward compatibility wrapper for legacy plotting helpers.

This shim mirrors the public API of ``calibrated_explanations.legacy.plotting``
while emitting a deprecation warning. It also reattaches itself to the package
namespace for callers importing ``calibrated_explanations._plots_legacy`` via
the parent package.
"""

from __future__ import annotations

import importlib
import sys
import warnings

# Emit a deprecation warning when the shim is imported
warnings.warn(
    "calibrated_explanations._plots_legacy is deprecated; import from "
    "calibrated_explanations.legacy.plotting instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all public names from the legacy plotting module
_legacy = importlib.import_module("calibrated_explanations.legacy.plotting")
__all__ = tuple(name for name in dir(_legacy) if not name.startswith("__"))

globals().update({name: getattr(_legacy, name) for name in __all__})

# Attach the shim back to the parent package namespace for future imports
parent = sys.modules.get("calibrated_explanations")
if parent is not None:
    setattr(parent, "_plots_legacy", sys.modules[__name__])

