"""Visualization namespace (experimental).

Contains a minimal PlotSpec abstraction and a matplotlib adapter.
Public API stability is not guaranteed yet; use for experimentation.

Note: this subpackage requires matplotlib. When matplotlib is not
available we raise ModuleNotFoundError at import time so callers using
pytest.importorskip("calibrated_explanations.viz") will correctly skip
tests that depend on the visualization extra.
"""

try:  # guard to make the "viz" package behave like an optional extra
    import matplotlib  # type: ignore  # only to detect availability
except Exception as _exc:  # pragma: no cover - optional dependency path
    raise ModuleNotFoundError(
        "Visualization requires matplotlib. Install the 'viz' extra: pip install calibrated_explanations[viz]"
    )

from . import matplotlib_adapter, plots
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
    "plots",
    "PLOTSPEC_VERSION",
]
