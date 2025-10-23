"""Deprecated shim for :mod:`calibrated_explanations._plots`."""

from __future__ import annotations

from warnings import warn

from ..viz import plots as _plots

warn(
    "'calibrated_explanations._plots' is deprecated; import from "
    "'calibrated_explanations.viz.plots' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [name for name in dir(_plots) if not name.startswith("__")]

globals().update({name: getattr(_plots, name) for name in __all__})
