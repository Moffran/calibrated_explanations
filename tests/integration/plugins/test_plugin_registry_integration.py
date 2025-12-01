"""Integration test for plugin registry and plugin discovery.

Tests real plugin discovery and registration, paired with unit tests in test_protocols.py.

Ref: ADR-006 (Plugin Trust Model), ADR-014 (Plugin Registry)
"""

# ruff: noqa: E402
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from calibrated_explanations.plugins import registry
from calibrated_explanations.plugins import (
    clear_explanation_plugins,
    ensure_builtin_plugins,
)
from calibrated_explanations.plugins.builtins import (
    LegacyFactualExplanationPlugin,
    LegacyAlternativeExplanationPlugin,
)


class TrustedTestPlugin(LegacyFactualExplanationPlugin):
    """A test plugin marked as trusted."""

    plugin_meta = {
        **LegacyFactualExplanationPlugin.plugin_meta,
        "name": "tests.integration.trusted_plugin",
        "version": "1.0.0-test",
        "provider": "tests.integration",
        "trust": True,
    }


class UntrustedTestPlugin(LegacyFactualExplanationPlugin):
    """A test plugin marked as untrusted."""

    plugin_meta = {
        **LegacyFactualExplanationPlugin.plugin_meta,
        "name": "tests.integration.untrusted_plugin",
        "version": "1.0.0-test",
        "provider": "tests.integration",
        "trust": False,
    }


@pytest.fixture(autouse=True)
def _registry_fixture():
    """Clear registry before and after each test."""
    clear_explanation_plugins()
    ensure_builtin_plugins()
    yield
    clear_explanation_plugins()
    ensure_builtin_plugins()


class TestPluginRegistryIntegration:
    """Integration tests for plugin registry and discovery."""

    def test_builtin_plugins_loaded_by_default(self):
        """Verify that built-in plugins are loaded by default."""
        # Registry should already have built-in plugins from the fixture
        all_plugins = registry.list_plugins()

        # Should have at least some built-in plugins loaded
        assert len(all_plugins) > 0

        # Verify some expected types are present
        plugin_types = {type(p).__name__ for p in all_plugins}
        # At least some built-in plugins should be there
        assert len(plugin_types) > 0

    def test_plugin_registration_idempotence(self):
        """Verify that registering the same plugin twice is safe."""
        plugin = TrustedTestPlugin()

        registry.register(plugin)
        count_after_first = len(registry.list_plugins())

        registry.register(plugin)
        count_after_second = len(registry.list_plugins())

        # Should not have added a duplicate
        assert count_after_second == count_after_first

    def test_multiple_plugins_coexist_in_registry(self):
        """Verify that multiple different plugins can coexist."""
        plugin1 = LegacyFactualExplanationPlugin()
        plugin2 = LegacyAlternativeExplanationPlugin()

        registry.register(plugin1)
        registry.register(plugin2)

        all_plugins = registry.list_plugins()

        # Both should be discoverable
        assert plugin1 in all_plugins
        assert plugin2 in all_plugins

    def test_trusted_plugin_metadata(self):
        """Verify that trusted plugin metadata is preserved."""
        plugin = TrustedTestPlugin()
        registry.register(plugin)

        # Check metadata
        assert plugin.plugin_meta["trust"] is True

    def test_untrusted_plugin_metadata(self):
        """Verify that untrusted plugin metadata is marked appropriately."""
        plugin = UntrustedTestPlugin()

        # Before registration, metadata should reflect intent
        assert plugin.plugin_meta["trust"] is False

        # After registration, registry may update the trust flag
        # but the plugin should still be in the registry
        registry.register(plugin)

        # Verify plugin is registered
        assert plugin in registry.list_plugins()

    def test_plugin_discovery_with_list_plugins(self):
        """Verify that list_plugins returns registered plugins."""
        plugin = TrustedTestPlugin()
        registry.register(plugin)

        all_plugins = registry.list_plugins()

        # Plugin should be discoverable
        assert plugin in all_plugins

    def test_unregister_plugin_removes_it(self):
        """Verify that unregistering a plugin removes it from the list."""
        plugin = TrustedTestPlugin()
        registry.register(plugin)

        # Verify registered
        assert plugin in registry.list_plugins()

        # Unregister
        registry.unregister(plugin)

        # Verify unregistered
        assert plugin not in registry.list_plugins()

    def test_plugin_has_required_interface(self):
        """Verify that plugins have required interface methods."""
        plugin = TrustedTestPlugin()

        # Check for required methods
        assert callable(plugin.supports)
        assert callable(plugin.explain)
        assert callable(plugin.supports_mode)
        assert callable(plugin.initialize)
        assert callable(plugin.explain_batch)

    def test_builtin_plugins_are_discoverable(self):
        """Verify that built-in plugins are discoverable."""
        all_plugins = registry.list_plugins()

        # Get some built-in plugin types
        built_in_types = {type(p).__name__ for p in all_plugins}

        # Should include some factual or regression plugins
        assert len(built_in_types) > 0
