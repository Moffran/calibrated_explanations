"""Visualization namespace (ADR-001 Stage 5 façade).

Plotting primitives and adapters are re-exported from the package root to keep
the public surface narrow and lintable. Backend-specific modules remain
internal. The optional matplotlib dependency is checked at import time so
``pytest.importorskip("calibrated_explanations.viz")`` continues to behave as
expected.
"""

from __future__ import annotations

from importlib import import_module, util as _importlib_util
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from . import matplotlib_adapter, plots
    from .builders import (
        REGRESSION_BAR_COLOR,
        REGRESSION_BASE_COLOR,
        _legacy_get_fill_color,
        build_alternative_probabilistic_spec,
        build_alternative_regression_spec,
        build_global_plotspec_dict,
        build_probabilistic_bars_spec,
        build_regression_bars_spec,
        build_triangular_plotspec_dict,
        build_factual_probabilistic_plotspec_dict,
        is_valid_probability_values,
    )
    from .narrative_plugin import NarrativePlotPlugin
    from .plotspec import BarHPanelSpec, BarItem, IntervalHeaderSpec, PlotSpec
    from .serializers import (
        PLOTSPEC_VERSION,
        plotspec_from_dict,
        plotspec_to_dict,
        validate_plotspec,
    )
    from .matplotlib_adapter import render


def _require_matplotlib() -> None:
    if _importlib_util.find_spec("matplotlib") is None:
        raise ModuleNotFoundError(
            "Visualization requires matplotlib."
            " Install the 'viz' extra: pip install calibrated_explanations[viz]"
        )


__all__ = (
    "PlotSpec",
    "IntervalHeaderSpec",
    "BarHPanelSpec",
    "BarItem",
    "build_regression_bars_spec",
    "build_alternative_probabilistic_spec",
    "build_alternative_regression_spec",
    "build_probabilistic_bars_spec",
    "build_triangular_plotspec_dict",
    "build_global_plotspec_dict",
    "build_factual_probabilistic_plotspec_dict",
    "is_valid_probability_values",
    "REGRESSION_BAR_COLOR",
    "REGRESSION_BASE_COLOR",
    "_legacy_get_fill_color",
    "render",
    "matplotlib_adapter",
    "plotspec_to_dict",
    "plotspec_from_dict",
    "validate_plotspec",
    "plots",
    "PLOTSPEC_VERSION",
    "NarrativePlotPlugin",
)

_NAME_TO_MODULE = {
    "PlotSpec": ("plotspec", "PlotSpec"),
    "IntervalHeaderSpec": ("plotspec", "IntervalHeaderSpec"),
    "BarHPanelSpec": ("plotspec", "BarHPanelSpec"),
    "BarItem": ("plotspec", "BarItem"),
    "build_regression_bars_spec": ("builders", "build_regression_bars_spec"),
    "build_alternative_probabilistic_spec": ("builders", "build_alternative_probabilistic_spec"),
    "build_alternative_regression_spec": ("builders", "build_alternative_regression_spec"),
    "build_probabilistic_bars_spec": ("builders", "build_probabilistic_bars_spec"),
    "build_triangular_plotspec_dict": ("builders", "build_triangular_plotspec_dict"),
    "build_global_plotspec_dict": ("builders", "build_global_plotspec_dict"),
    "build_factual_probabilistic_plotspec_dict": (
        "builders",
        "build_factual_probabilistic_plotspec_dict",
    ),
    "is_valid_probability_values": ("builders", "is_valid_probability_values"),
    "REGRESSION_BAR_COLOR": ("builders", "REGRESSION_BAR_COLOR"),
    "REGRESSION_BASE_COLOR": ("builders", "REGRESSION_BASE_COLOR"),
    "_legacy_get_fill_color": ("builders", "_legacy_get_fill_color"),
    "render": ("matplotlib_adapter", "render"),
    "matplotlib_adapter": ("matplotlib_adapter", None),
    "plots": ("plots", None),
    "plotspec_to_dict": ("serializers", "plotspec_to_dict"),
    "plotspec_from_dict": ("serializers", "plotspec_from_dict"),
    "validate_plotspec": ("serializers", "validate_plotspec"),
    "PLOTSPEC_VERSION": ("serializers", "PLOTSPEC_VERSION"),
    "NarrativePlotPlugin": ("narrative_plugin", "NarrativePlotPlugin"),
}

_MATPLOTLIB_REQUIRED = {"matplotlib_adapter", "plots", "render"}


def __getattr__(name: str) -> Any:
    """Lazily load plotting primitives and adapters."""

    if name not in __all__:
        raise AttributeError(name)

    if name in _MATPLOTLIB_REQUIRED:
        _require_matplotlib()

    module_name, attr_name = _NAME_TO_MODULE[name]
    module = import_module(f"{__name__}.{module_name}")
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value
