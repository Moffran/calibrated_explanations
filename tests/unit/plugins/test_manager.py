"""Unit tests for PluginManager.

Tests for the PluginManager class which centralizes plugin state,
override configuration, and instance caching.
"""

import pytest
import copy
import types
from unittest.mock import Mock

from calibrated_explanations.utils.exceptions import ConfigurationError
from calibrated_explanations.plugins.manager import PluginManager


class TestPluginManagerInitialization:
    """Tests for PluginManager initialization."""


class TestPluginManagerInitializeFromKwargs:
    """Tests for initialize_from_kwargs method."""


class TestCoercePluginOverride:
    """Tests for coerce_plugin_override method."""

    def test_coerce_callable_calls_and_returns_result(self):
        """should_coerce_callable_by_invoking_and_returning_result."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        def factory():
            return {"plugin": True}

        result = manager.coerce_plugin_override(factory)
        assert result == {"plugin": True}

    def test_coerce_callable_raising_raises_configuration_error(self):
        """should_raise_configuration_error_when_callable_raises."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        def bad_factory():
            raise RuntimeError("boom")

        with pytest.raises(ConfigurationError):
            manager.coerce_plugin_override(bad_factory)


class TestBridgeMonitorManagement:
    """Tests for bridge monitor caching."""

    def test_clear_bridge_monitors_empties_cache(self):
        """should_empty_bridge_monitor_cache_on_clear."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        _ = manager.get_bridge_monitor("my_plugin")
        assert len(manager.bridge_monitors) == 1

        manager.clear_bridge_monitors()
        assert len(manager.bridge_monitors) == 0


class TestExplanationPluginInstanceManagement:
    """Tests for explanation plugin instance caching."""

    def test_set_and_get_explanation_plugin_instance(self):
        """should_cache_and_retrieve_explanation_plugin_instance."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        instance = Mock()
        manager.set_explanation_plugin_instance("my_id", instance)

        result = manager.get_explanation_plugin_instance("my_id")
        assert result is instance

    def test_clear_explanation_plugin_instances(self):
        """should_clear_all_cached_explanation_plugin_instances."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        manager.set_explanation_plugin_instance("id1", Mock())
        manager.set_explanation_plugin_instance("id2", Mock())
        assert len(manager.explanation_plugin_instances) == 2

        manager.clear_explanation_plugin_instances()
        assert len(manager.explanation_plugin_instances) == 0


class TestExplanationPluginIdentifierManagement:
    """Tests for explanation plugin identifier caching."""

    def test_set_and_get_explanation_plugin_identifier(self):
        """should_cache_and_retrieve_explanation_plugin_identifier."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        manager.set_explanation_plugin_identifier("factual", "core.explanation.factual")

        result = manager.get_explanation_plugin_identifier("factual")
        assert result == "core.explanation.factual"

    def test_clear_explanation_plugin_identifiers(self):
        """should_clear_all_cached_explanation_plugin_identifiers."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        manager.set_explanation_plugin_identifier("factual", "id1")
        manager.set_explanation_plugin_identifier("alternative", "id2")
        assert len(manager.explanation_plugin_identifiers) == 2

        manager.clear_explanation_plugin_identifiers()
        assert len(manager.explanation_plugin_identifiers) == 0


class TestIntervalPluginState:
    """Tests for interval plugin state management."""

    def test_resolve_interval_plugin_returns_object_override(self):
        """should_return_non_string_override_directly_for_interval_resolution."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)
        override = Mock()
        override.plugin_meta = {"name": "custom.interval"}
        manager.interval_plugin_override = override

        plugin, identifier = manager.resolve_interval_plugin(fast=False)

        assert plugin is override
        assert identifier == "custom.interval"


class TestPlotResolution:
    """Tests for plot style and plugin resolution."""

    def test_resolve_plot_style_chain_includes_explicit_and_mode_fallbacks(self):
        """should_merge_explicit_style_manager_chain_and_mode_hints."""
        mock_explainer = Mock()
        mock_explainer.last_explanation_mode = "factual"
        manager = PluginManager(mock_explainer)
        manager.plot_plugin_fallbacks = {"factual": ("mode.one",)}
        manager.build_plot_chain = Mock(return_value=("env.style", "legacy"))

        chain = manager.resolve_plot_style_chain(explicit_style="explicit.style")

        assert chain[0] == "explicit.style"
        assert "env.style" in chain
        assert "mode.one" in chain
        assert "plot_spec.default" in chain
        assert chain[-1] == "legacy"

    def test_resolve_plot_plugin_raises_when_no_candidates(self, monkeypatch):
        """should_raise_configuration_error_when_plot_plugins_missing."""
        import calibrated_explanations.plugins.manager as manager_module

        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)
        manager.build_plot_chain = Mock(return_value=("missing.style", "legacy"))

        monkeypatch.setattr(manager_module, "ensure_builtin_plugins", lambda: None)
        monkeypatch.setattr(manager_module, "find_plot_plugin_trusted", lambda _ident: None)
        monkeypatch.setattr(manager_module, "find_plot_plugin", lambda _ident: None)

        with pytest.raises(ConfigurationError, match="Unable to resolve plot plugin"):
            manager.resolve_plot_plugin(explicit_style="missing.style")


class TestPluginManagerDeepCopy:
    """Tests for PluginManager deepcopy behavior."""

    def test_deepcopy_handles_mappingproxy(self):
        """should_handle_mappingproxy_when_deepcopied."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        # Simulate the state that causes issues (mappingproxy in a dict)
        unpicklable_dict = types.MappingProxyType({"a": 1})
        manager.explanation_contexts = {"test_context": unpicklable_dict}

        # This should not raise TypeError
        copied_manager = copy.deepcopy(manager)

        # Verify the copy has the data (shallow copied or reference)
        assert copied_manager.explanation_contexts["test_context"] == unpicklable_dict
        assert isinstance(
            copied_manager.explanation_contexts["test_context"], types.MappingProxyType
        )

        # Verify it's a different manager instance
        assert copied_manager is not manager
        assert manager.interval_context_metadata["fast"] == {}

    def test_deepcopy_preserves_mappingproxy_attribute(self):
        """should_preserve_mappingproxy_attribute_when_deepcopied."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        proxy = types.MappingProxyType({"a": 1})
        manager.test_proxy = proxy

        copied_manager = copy.deepcopy(manager)

        assert isinstance(copied_manager.test_proxy, types.MappingProxyType)
        assert copied_manager.test_proxy is not proxy
        assert dict(copied_manager.test_proxy) == dict(proxy)

    def test_deepcopy_mappingproxy_failure_falls_back_to_reference(self, monkeypatch):
        """should_fallback_to_reference_when_mappingproxy_recreation_fails."""
        import calibrated_explanations.plugins.manager as manager_module

        class FakeProxy:
            fail_on_init = False

            def __init__(self, data):
                if self.__class__.fail_on_init:
                    raise TypeError("boom")
                self.data = dict(data)

            def __iter__(self):
                return iter(self.data.items())

            def __len__(self):
                return len(self.data)

            def __getitem__(self, key):
                return self.data[key]

        monkeypatch.setattr(manager_module, "MappingProxyType", FakeProxy)

        mock_explainer = Mock()
        manager = manager_module.PluginManager(mock_explainer)

        fake_proxy = FakeProxy({"a": 1})
        manager.test_proxy = fake_proxy
        FakeProxy.fail_on_init = True

        copied_manager = copy.deepcopy(manager)

        assert copied_manager.test_proxy is fake_proxy

    def test_deepcopy_dict_preserve_failure_falls_back_to_shallow_copy(self):
        """should_fallback_to_shallow_dict_copy_when_dict_preserve_fails."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        class BadDeepcopy:
            def __deepcopy__(self, memo):
                raise TypeError("boom")

        bad_value = BadDeepcopy()
        manager.test_dict = {"bad": bad_value}

        copied_manager = copy.deepcopy(manager)

        assert copied_manager.test_dict is not manager.test_dict
        assert copied_manager.test_dict["bad"] is bad_value

    def test_deepcopy_typeerror_falls_back_for_list_and_object(self):
        """should_fallback_on_typeerror_for_list_and_object_values."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        class BadDeepcopy:
            def __deepcopy__(self, memo):
                raise TypeError("boom")

        bad_value = BadDeepcopy()
        manager.test_list = [bad_value]
        manager.test_obj = bad_value

        copied_manager = copy.deepcopy(manager)

        assert copied_manager.test_list is not manager.test_list
        assert copied_manager.test_list[0] is bad_value
        assert copied_manager.test_obj is bad_value
