"""Visualization namespace (experimental).

Contains a minimal PlotSpec abstraction and a matplotlib adapter.
Public API stability is not guaranteed yet; use for experimentation.

Note: this subpackage requires matplotlib. When matplotlib is not
available we raise ModuleNotFoundError at import time so callers using
pytest.importorskip("calibrated_explanations.viz") will correctly skip
tests that depend on the visualization extra.
"""

try:  # guard to make the "viz" package behave like an optional extra
    # Prefer non-binding availability check to avoid importing the heavy package
    from importlib import util as _importlib_util

    if _importlib_util.find_spec("matplotlib") is None:
        raise ModuleNotFoundError(
            "Visualization requires matplotlib."
            " Install the 'viz' extra: pip install calibrated_explanations[viz]"
        )
except Exception as _exc:  # pragma: no cover - optional dependency path
    # Chain the exception to distinguish import-time failures from our guard
    raise ModuleNotFoundError(
        "Visualization requires matplotlib."
        " Install the 'viz' extra: pip install calibrated_explanations[viz]"
    ) from _exc

from . import matplotlib_adapter, plots
from .builders import build_regression_bars_spec, is_valid_probability_values
from .narrative_plugin import NarrativePlotPlugin
from .plotspec import BarHPanelSpec, BarItem, IntervalHeaderSpec, PlotSpec
from .serializers import PLOTSPEC_VERSION, plotspec_from_dict, plotspec_to_dict, validate_plotspec

__all__ = [
    "PlotSpec",
    "IntervalHeaderSpec",
    "BarHPanelSpec",
    "BarItem",
    "build_regression_bars_spec",
    "is_valid_probability_values",
    "matplotlib_adapter",
    "plotspec_to_dict",
    "plotspec_from_dict",
    "validate_plotspec",
    "plots",
    "PLOTSPEC_VERSION",
    "NarrativePlotPlugin",
]
