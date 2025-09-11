"""Visualization namespace (experimental).

Contains a minimal PlotSpec abstraction and a matplotlib adapter.
Public API stability is not guaranteed yet; use for experimentation.
"""

from . import matplotlib_adapter
from .builders import build_regression_bars_spec
from .plotspec import BarHPanelSpec, BarItem, IntervalHeaderSpec, PlotSpec
from .serializers import PLOTSPEC_VERSION, plotspec_from_dict, plotspec_to_dict, validate_plotspec

__all__ = [
    "PlotSpec",
    "IntervalHeaderSpec",
    "BarHPanelSpec",
    "BarItem",
    "build_regression_bars_spec",
    "matplotlib_adapter",
    "plotspec_to_dict",
    "plotspec_from_dict",
    "validate_plotspec",
    "PLOTSPEC_VERSION",
]
