"""Deprecated plotting shim kept for backwards compatibility.

Import from calibrated_explanations.plotting instead.
"""

from warnings import warn

from . import plotting as _plotting

warn(
    "'calibrated_explanations._plots' is deprecated; import from "
    "'calibrated_explanations.plotting' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [name for name in dir(_plotting) if not name.startswith("__")]

globals().update({name: getattr(_plotting, name) for name in __all__})

