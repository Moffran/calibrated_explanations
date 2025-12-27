"""Unit tests for CalibratedExplainer fallback paths.

These tests exercise defensive branches that are primarily used when a
fully initialized :class:`~calibrated_explanations.core.calibrated_explainer.CalibratedExplainer`
instance is not available (for example, in legacy usage or during error
handling). The goal is to ensure the fallback logic remains functional and
boost coverage for branches not hit by higher-level integration tests.
"""

from __future__ import annotations

import pytest

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer


def minimal_explainer() -> CalibratedExplainer:
    """Return a minimally constructed explainer without invoking ``__init__``."""

    return CalibratedExplainer.__new__(CalibratedExplainer)


def test_plugin_manager_property_should_raise_not_fitted_error_when_missing():
    """plugin_manager property should raise NotFittedError when the manager is absent."""
    from calibrated_explanations.utils.exceptions import NotFittedError

    explainer = minimal_explainer()

    with pytest.raises(NotFittedError, match="PluginManager is not initialized"):
        _ = explainer.plugin_manager


def test_plugin_overrides_should_not_raise_when_manager_unavailable():
    """Property setters should not raise when PluginManager is unavailable."""

    explainer = minimal_explainer()

    # Setting public properties should not raise
    explainer.explanation_plugin_overrides = {"factual": "explicit"}
    explainer.interval_plugin_override = "interval"
    explainer.fast_interval_plugin_override = "fast-interval"
    explainer.plot_style_override = "plot-style"

    # Since no manager, getters return defaults
    assert explainer.explanation_plugin_overrides == {}
    assert explainer.interval_plugin_override is None
    assert explainer.fast_interval_plugin_override is None
    assert explainer.plot_style_override is None


def test_build_plot_style_chain_should_return_empty_tuple_without_manager():
    """build_plot_style_chain should return empty tuple without an initialized manager."""

    explainer = minimal_explainer()
    assert explainer.build_plot_style_chain() == ()

    # Provide a lightweight manager to ensure the delegation path is exercised
    class Manager:
        def _build_plot_chain(self):
            return ("plot",)

    explainer._plugin_manager = Manager()
    assert explainer.build_plot_style_chain() == ("plot",)
