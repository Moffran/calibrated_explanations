"""Integration test for complete plugin workflow: register → discover → initialize.

Tests the end-to-end plugin lifecycle with real plugins, not mocks.

Ref: ADR-013 (Plugin Protocol), ADR-014 (Plugin Registry), ADR-015 (Predict Bridge)
"""

# ruff: noqa: E402
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from calibrated_explanations.plugins import ExplanationContext
from calibrated_explanations.plugins import registry
from calibrated_explanations.plugins import (
    clear_explanation_plugins,
    ensure_builtin_plugins,
)
from calibrated_explanations.plugins.builtins import LegacyFactualExplanationPlugin


class SimpleExplainerPlugin(LegacyFactualExplanationPlugin):
    """A real test plugin for workflow validation."""

    plugin_meta = {
        **LegacyFactualExplanationPlugin.plugin_meta,
        "name": "tests.integration.simple_explainer",
        "version": "1.0.0-test",
        "provider": "tests.integration",
        "tasks": ("classification", "regression"),
        "trust": True,
    }


@pytest.fixture(autouse=True)
def _registry_fixture():
    """Clear registry before and after each test."""
    clear_explanation_plugins()
    ensure_builtin_plugins()
    yield
    clear_explanation_plugins()
    ensure_builtin_plugins()


class TestPluginWorkflowIntegration:
    """Integration tests for complete plugin lifecycle."""

    def test_plugin_registration_and_list(self):
        """Verify plugin registration appears in list."""
        plugin = SimpleExplainerPlugin()
        registry.register(plugin)

        # Verify it's in the list
        all_plugins = registry.list_plugins()
        assert plugin in all_plugins

        # Verify plugin has required methods
        assert callable(plugin.supports)
        assert callable(plugin.supports_mode)
        assert callable(plugin.initialize)
        assert callable(plugin.explain_batch)

    def test_plugin_supports_mode_contract(self):
        """Verify that plugin supports_mode returns boolean."""
        plugin = SimpleExplainerPlugin()

        # Should support these combinations
        assert plugin.supports_mode("factual", task="classification")
        assert plugin.supports_mode("factual", task="regression")

        # Verify method is callable and returns bool
        result = plugin.supports_mode("unknown_mode", task="classification")
        assert isinstance(result, bool)

    def test_plugin_initialization_with_context(self):
        """Verify plugin can be initialized with context."""
        plugin = SimpleExplainerPlugin()

        # Create mock explainer for the plugin
        class MockExplainer:  # pylint: disable=missing-docstring
            pass

        # Create explanation context with required explainer handle
        ctx = ExplanationContext(
            task="classification",
            mode="factual",
            feature_names=("f0", "f1", "f2"),
            categorical_features=(),
            categorical_labels={},
            discretizer=None,
            helper_handles={"explainer": MockExplainer()},
            predict_bridge=object(),
            interval_settings={},
            plot_settings={},
        )

        # Initialize plugin - this should not raise
        plugin.initialize(ctx)

        # Verify plugin state after initialization
        assert plugin is not None
        assert hasattr(plugin, "plugin_meta")
        assert "classification" in plugin.plugin_meta["tasks"]

    def test_plugin_context_immutability_after_init(self):
        """Verify context remains immutable after plugin initialization."""
        plugin = SimpleExplainerPlugin()

        class MockExplainer:  # pylint: disable=missing-docstring
            pass

        ctx = ExplanationContext(
            task="classification",
            mode="factual",
            feature_names=("f0", "f1"),
            categorical_features=(),
            categorical_labels={},
            discretizer=None,
            helper_handles={"explainer": MockExplainer()},
            predict_bridge=object(),
            interval_settings={},
            plot_settings={},
        )

        # Initialize plugin
        plugin.initialize(ctx)

        # Verify context immutability after initialization
        original_task = ctx.task
        with pytest.raises(Exception):
            ctx.task = "regression"  # type: ignore[misc]
        assert ctx.task == original_task

    def test_unregister_removes_plugin(self):
        """Verify that unregistering a plugin removes it from discovery."""
        plugin = SimpleExplainerPlugin()
        registry.register(plugin)

        # Verify registered
        assert plugin in registry.list_plugins()

        # Unregister
        registry.unregister(plugin)

        # Verify unregistered
        assert plugin not in registry.list_plugins()

    def test_plugin_metadata_accessible(self):
        """Verify that plugin metadata contains required fields."""
        plugin = SimpleExplainerPlugin()
        registry.register(plugin)

        # Access plugin metadata
        meta = plugin.plugin_meta

        # Required fields per ADR-013/ADR-014
        assert "name" in meta
        assert "trust" in meta
        assert callable(plugin.supports_mode)
        assert callable(plugin.initialize)
        assert callable(plugin.explain_batch)
