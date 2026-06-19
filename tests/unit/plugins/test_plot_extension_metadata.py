"""ADR-037 §4 extension metadata: plot plugins must declare supported kinds and modes (Task 15-E).

Validates that:
- validate_plugin_meta accepts plot plugins without plot_kinds/plot_modes (uses defaults).
- validate_plugin_meta rejects invalid plot_kinds / plot_modes values with ValidationError.
- Built-in plot plugins register cleanly with validate_plugin_meta.
"""

from __future__ import annotations

import pytest


def _minimal_plot_meta(**overrides):
    base = {
        "schema_version": 1,
        "name": "test.plot.builder",
        "version": "0.1.0",
        "provider": "test",
        "capabilities": ["plot:builder"],
        "data_modalities": ("tabular",),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Default injection
# ---------------------------------------------------------------------------


def test_plot_plugin_without_plot_kinds_gets_defaults():
    """A plot plugin that omits plot_kinds receives the full default set."""
    from calibrated_explanations.plugins.base import validate_plugin_meta

    meta = _minimal_plot_meta()
    validate_plugin_meta(meta)
    assert "plot_kinds" in meta
    assert set(meta["plot_kinds"]) == {
        "factual_probabilistic",
        "factual_regression",
        "alternative_probabilistic",
        "alternative_regression",
        "global_probabilistic",
        "global_regression",
    }


def test_plot_plugin_without_plot_modes_gets_defaults():
    """A plot plugin that omits plot_modes receives the full default set."""
    from calibrated_explanations.plugins.base import validate_plugin_meta

    meta = _minimal_plot_meta()
    validate_plugin_meta(meta)
    assert "plot_modes" in meta
    assert set(meta["plot_modes"]) == {"factual", "alternative", "fast"}


# ---------------------------------------------------------------------------
# Validation of declared values
# ---------------------------------------------------------------------------


def test_plot_plugin_with_valid_plot_kinds_accepted():
    """Declaring a valid subset of plot_kinds is accepted."""
    from calibrated_explanations.plugins.base import validate_plugin_meta

    meta = _minimal_plot_meta(plot_kinds=["factual_probabilistic", "global_regression"])
    validate_plugin_meta(meta)
    assert set(meta["plot_kinds"]) == {"factual_probabilistic", "global_regression"}


def test_plot_plugin_with_category_plot_kinds_warns():
    """Declaring legacy category plot_kinds remains transitional but warns."""
    from calibrated_explanations.plugins.base import validate_plugin_meta

    meta = _minimal_plot_meta(plot_kinds=["instance", "global"])
    with pytest.warns(DeprecationWarning, match="category vocabulary"):
        validate_plugin_meta(meta)
    assert set(meta["plot_kinds"]) == {"instance", "global"}


def test_plot_plugin_with_valid_plot_modes_accepted():
    """Declaring a valid subset of plot_modes is accepted."""
    from calibrated_explanations.plugins.base import validate_plugin_meta

    meta = _minimal_plot_meta(plot_modes=["factual"])
    validate_plugin_meta(meta)
    assert tuple(meta["plot_modes"]) == ("factual",)


def test_plot_plugin_with_invalid_plot_kinds_raises():
    """Declaring an unrecognised plot_kind value raises ValidationError."""
    from calibrated_explanations.plugins.base import validate_plugin_meta
    from calibrated_explanations.utils.exceptions import ValidationError

    meta = _minimal_plot_meta(plot_kinds=["instance", "UNSUPPORTED_KIND"])
    with pytest.raises(ValidationError, match="plot_kinds"):
        validate_plugin_meta(meta)


def test_plot_plugin_with_triangular_plot_kind_raises():
    """Triangular is an internal routing kind, not plugin metadata vocabulary."""
    from calibrated_explanations.plugins.base import validate_plugin_meta
    from calibrated_explanations.utils.exceptions import ValidationError

    meta = _minimal_plot_meta(plot_kinds=["triangular"])
    with pytest.raises(ValidationError, match="plot_kinds"):
        validate_plugin_meta(meta)


def test_plot_plugin_with_invalid_plot_modes_raises():
    """Declaring an unrecognised plot_mode value raises ValidationError."""
    from calibrated_explanations.plugins.base import validate_plugin_meta
    from calibrated_explanations.utils.exceptions import ValidationError

    meta = _minimal_plot_meta(plot_modes=["factual", "UNKNOWN_MODE"])
    with pytest.raises(ValidationError, match="plot_modes"):
        validate_plugin_meta(meta)


def test_non_plot_plugin_ignores_plot_fields():
    """A non-plot plugin (explanation:*) is not subject to plot_kinds/plot_modes validation."""
    from calibrated_explanations.plugins.base import validate_plugin_meta

    meta = {
        "schema_version": 1,
        "name": "test.explanation.plugin",
        "version": "0.1.0",
        "provider": "test",
        "capabilities": ["explanation:factual"],
        "data_modalities": ("tabular",),
    }
    validate_plugin_meta(meta)
    assert "plot_kinds" not in meta
    assert "plot_modes" not in meta


# ---------------------------------------------------------------------------
# Built-in plugin registration
# ---------------------------------------------------------------------------


def test_legacy_plot_builder_registers_cleanly():
    """LegacyPlotBuilder plugin_meta passes validate_plugin_meta."""
    from calibrated_explanations.plugins.base import validate_plugin_meta
    from calibrated_explanations.plugins.builtins import LegacyPlotBuilder

    meta = dict(LegacyPlotBuilder.plugin_meta)
    validate_plugin_meta(meta)
    assert "plot_kinds" in meta
    assert "plot_modes" in meta


def test_legacy_plot_renderer_registers_cleanly():
    """LegacyPlotRenderer plugin_meta passes validate_plugin_meta."""
    from calibrated_explanations.plugins.base import validate_plugin_meta
    from calibrated_explanations.plugins.builtins import LegacyPlotRenderer

    meta = dict(LegacyPlotRenderer.plugin_meta)
    validate_plugin_meta(meta)
    assert "plot_kinds" in meta
    assert "plot_modes" in meta


def test_plotspec_default_builder_registers_cleanly():
    """PlotSpecDefaultBuilder plugin_meta passes validate_plugin_meta."""
    from calibrated_explanations.plugins.base import validate_plugin_meta
    from calibrated_explanations.plugins.builtins import PlotSpecDefaultBuilder

    meta = dict(PlotSpecDefaultBuilder.plugin_meta)
    validate_plugin_meta(meta)
    assert "plot_kinds" in meta
    assert "plot_modes" in meta


def test_plotspec_default_renderer_registers_cleanly():
    """PlotSpecDefaultRenderer plugin_meta passes validate_plugin_meta."""
    from calibrated_explanations.plugins.base import validate_plugin_meta
    from calibrated_explanations.plugins.builtins import PlotSpecDefaultRenderer

    meta = dict(PlotSpecDefaultRenderer.plugin_meta)
    validate_plugin_meta(meta)
    assert "plot_kinds" in meta
    assert "plot_modes" in meta
