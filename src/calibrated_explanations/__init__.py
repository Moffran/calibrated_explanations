"""
Calibrated Explanations (calibrated_explanations).

is a Python package for explaining black-box models.

It is based on the paper "Calibrated Explanations: with Uncertainty Information and Counterfactuals"
by Helena Löfström et al.
"""

import copyreg
import importlib
import logging as _logging
from types import MappingProxyType

# Ensure MappingProxyType objects can be pickled project-wide by reducing them
# to plain dicts. This avoids "cannot pickle 'mappingproxy' object" errors
# when serializing objects that embed immutable mapping proxies (telemetry,
# plugin metadata, etc.). Registering here ensures the reducer is active
# as soon as the package is imported.
try:  # pragma: no cover - defensive

    def _reduce_mappingproxy(mp: MappingProxyType):
        return dict, (dict(mp),)

    copyreg.pickle(MappingProxyType, _reduce_mappingproxy)
except (TypeError, AttributeError) as exc:
    # ADR-002: avoid catching broad Exception; handle the specific
    # expected failures when registering reducers (type errors or
    # attribute errors in constrained packaging environments).
    _logging.getLogger(__name__).debug("MappingProxyType reducer registration skipped: %s", exc)

# Expose viz namespace lazily via __getattr__ (avoid importing heavy backends eagerly)
# Note: avoid eager imports of explanation, viz and discretizer modules here.
# Those modules import heavy dependencies (numpy, pandas, plotting backends)
# and should be loaded lazily via __getattr__ below. Importing them at
# package import time increases startup cost significantly.

# Provide a default no-op handler to avoid "No handler" warnings for library users.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

__version__ = "v0.11.1"

# Note: core submodules are intentionally not imported here to avoid importing
# large backends and to make deprecation transitions explicit. We still expose
# the public symbols lazily so `from calibrated_explanations import CalibratedExplainer`
# works without triggering an eager import of `calibrated_explanations.core`.
__all__ = [
    "CalibratedExplainer",
    "WrapCalibratedExplainer",
    "transform_to_numeric",
    "RejectPolicySpec",
]


def __getattr__(name: str):
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

    if name == "RejectPolicySpec":
        from .explanations.reject import RejectPolicySpec

        globals()[name] = RejectPolicySpec
        return RejectPolicySpec

    raise AttributeError(name)
