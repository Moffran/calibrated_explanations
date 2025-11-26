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


def _minimal_explainer() -> CalibratedExplainer:
    """Return a minimally constructed explainer without invoking ``__init__``."""

    return CalibratedExplainer.__new__(CalibratedExplainer)


def test_coerce_plugin_override_fallbacks():
    """Fallback coercion should cover None, strings, callables, and manager delegation."""

    explainer = _minimal_explainer()

    # No plugin manager -> direct passthrough and callable instantiation
    assert explainer._coerce_plugin_override(None) is None
    assert explainer._coerce_plugin_override("identifier") == "identifier"

    def _callable_override():
        return {"constructed": True}

    assert explainer._coerce_plugin_override(_callable_override) == {"constructed": True}

    # With plugin manager we should delegate instead of using fallback logic
    class _PM:
        def __init__(self):
            self.seen = None

        def coerce_plugin_override(self, override):
            self.seen = override
            return "delegated"

    explainer._plugin_manager = _PM()
    assert explainer._coerce_plugin_override("value") == "delegated"
    assert explainer._plugin_manager.seen == "value"


def test_require_plugin_manager_raises_when_missing():
    """_require_plugin_manager should raise when the manager is absent."""

    explainer = _minimal_explainer()

    with pytest.raises(RuntimeError, match="PluginManager is not initialized"):
        explainer._require_plugin_manager()


def test_plugin_state_accessors_cache_without_manager():
    """Property setters should cache values when PluginManager is unavailable."""

    explainer = _minimal_explainer()

    assert explainer._explanation_plugin_overrides == {}
    explainer._explanation_plugin_overrides = {"factual": "explicit"}
    assert explainer._plugin_manager_cache_explanation_overrides == {"factual": "explicit"}

    assert explainer._interval_plugin_override is None
    explainer._interval_plugin_override = "interval"
    assert explainer._plugin_manager_cache_interval_override == "interval"

    assert explainer._fast_interval_plugin_override is None
    explainer._fast_interval_plugin_override = "fast-interval"
    assert explainer._plugin_manager_cache_fast_interval_override == "fast-interval"

    assert explainer._plot_style_override is None
    explainer._plot_style_override = "plot-style"
    assert explainer._plugin_manager_cache_plot_style_override == "plot-style"


def test_chain_builders_use_fallback_without_manager():
    """Chain builders should return empty tuples without an initialized manager."""

    explainer = _minimal_explainer()
    assert explainer._build_explanation_chain("factual") == ()
    assert explainer._build_interval_chain(fast=False) == ()
    assert explainer._build_plot_style_chain() == ()

    # Provide a lightweight manager to ensure the delegation path is exercised
    class _Manager:
        def __init__(self):
            self._default_explanation_identifiers = {"factual": "default"}

        def _build_explanation_chain(self, mode, default_id):
            return (mode, default_id)

        def _build_interval_chain(self, *, fast: bool):
            return ("interval", fast)

        def _build_plot_chain(self):
            return ("plot",)

    explainer._plugin_manager = _Manager()
    assert explainer._build_explanation_chain("factual") == ("factual", "default")
    assert explainer._build_interval_chain(fast=True) == ("interval", True)
    assert explainer._build_plot_style_chain() == ("plot",)

