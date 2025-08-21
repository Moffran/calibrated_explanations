"""Core component modules (Phase 1A mechanical split).

Importing :mod:`calibrated_explanations.core` now loads a package instead of the legacy
module. This package import path remains supported for the current development phase
but is slated for deprecation cleanup in a future minor release. A single
``DeprecationWarning`` is emitted on first import so that downstream libraries / users
become aware without flooding logs.
"""

from warnings import warn as _warn

from .calibrated_explainer import CalibratedExplainer  # noqa: F401
from .online_explainer import OnlineCalibratedExplainer  # noqa: F401
from .wrap_explainer import WrapCalibratedExplainer  # noqa: F401

_warn(
    "Importing 'calibrated_explanations.core' now loads the core package; the legacy module "
    "form is deprecated and will be removed in a future minor release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CalibratedExplainer",
    "WrapCalibratedExplainer",
    "OnlineCalibratedExplainer",
]
