"""Visualization namespace (experimental).

Contains a minimal PlotSpec abstraction and a matplotlib adapter.
Public API stability is not guaranteed yet; use for experimentation.
"""

from . import matplotlib_adapter
from .builders import build_regression_bars_spec
from .plotspec import BarHPanelSpec, BarItem, IntervalHeaderSpec, PlotSpec

__all__ = [
    "PlotSpec",
    "IntervalHeaderSpec",
    "BarHPanelSpec",
    "BarItem",
    "build_regression_bars_spec",
    "matplotlib_adapter",
]
