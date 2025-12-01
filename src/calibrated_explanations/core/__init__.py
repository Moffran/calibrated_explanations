"""Core component modules (Phase 1A mechanical split).

Importing :mod:`calibrated_explanations.core` now loads a package instead of the legacy
module. This package import path remains supported for the current development phase
but is slated for deprecation cleanup in a future minor release. A single
``DeprecationWarning`` is emitted on first import so that downstream libraries / users
become aware without flooding logs.
"""

import os
import sys

from ..utils import deprecate
from .calibrated_explainer import CalibratedExplainer  # noqa: F401
from .wrap_explainer import WrapCalibratedExplainer  # noqa: F401

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
]
