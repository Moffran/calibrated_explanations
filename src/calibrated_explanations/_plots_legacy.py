"""Deprecated legacy plotting shim.

The legacy implementations now live under calibrated_explanations.legacy.plotting.
"""

from warnings import warn

from .legacy import plotting as _legacy_plotting

warn(
    "'calibrated_explanations._plots_legacy' is deprecated; import from "
    "'calibrated_explanations.legacy.plotting' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [name for name in dir(_legacy_plotting) if not name.startswith("__")]

globals().update({name: getattr(_legacy_plotting, name) for name in __all__})

