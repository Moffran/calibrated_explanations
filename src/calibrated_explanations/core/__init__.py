"""Core component modules (Phase 1A mechanical split).

Importing ``calibrated_explanations.core`` emits a deprecation warning because the legacy
module form has been replaced by a package. The import path remains the same; users
can continue using it until at least v0.8.0.
"""

from warnings import warn as _warn

from .calibrated_explainer import CalibratedExplainer  # noqa: F401
from .online_explainer import OnlineCalibratedExplainer  # noqa: F401
from .wrap_explainer import WrapCalibratedExplainer  # noqa: F401

_warn(
    "Importing 'calibrated_explanations.core' now loads the core package; the legacy module"
    " form is deprecated (no action needed other than updating any messaging).",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CalibratedExplainer",
    "WrapCalibratedExplainer",
    "OnlineCalibratedExplainer",
]
