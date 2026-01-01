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
from calibrated_explanations.plugins.predict_monitor import PredictBridgeMonitor


class TestPluginManagerInitialization:
    """Tests for PluginManager initialization."""

    def test_init_creates_empty_state(self):
        """should_create_empty_state_when_initialized."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        assert manager.explainer is mock_explainer
        assert manager.explanation_plugin_overrides == {}
        assert manager._interval_plugin_override is None
        assert manager._fast_interval_plugin_override is None
        assert manager.plot_style_override is None

    def test_init_creates_empty_caches(self):
        """should_create_empty_caches_when_initialized."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        assert manager._bridge_monitors == {}
        assert manager._explanation_plugin_instances == {}
        assert manager.explanation_plugin_identifiers == {}

    def test_init_creates_empty_fallback_chains(self):
        """should_create_empty_fallback_chains_when_initialized."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        assert manager.explanation_plugin_fallbacks == {}
        assert manager.plot_plugin_fallbacks == {}
        assert manager.interval_plugin_hints == {}
        assert manager.interval_plugin_fallbacks == {}


class TestPluginManagerInitializeFromKwargs:
    """Tests for initialize_from_kwargs method."""

    def test_initialize_explanation_overrides(self):
        """should_initialize_explanation_plugin_overrides_from_kwargs."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        kwargs = {
            "factual_plugin": "my_factual",
            "alternative_plugin": "my_alternative",
            "fast_plugin": "my_fast",
        }
        manager.initialize_from_kwargs(kwargs)

        assert manager.explanation_plugin_overrides["factual"] == "my_factual"
        assert manager.explanation_plugin_overrides["alternative"] == "my_alternative"
        assert manager.explanation_plugin_overrides["fast"] == "my_fast"

    def test_initialize_interval_overrides(self):
        """should_initialize_interval_plugin_overrides_from_kwargs."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        kwargs = {"interval_plugin": "my_interval", "fast_interval_plugin": "my_fast_interval"}
        manager.initialize_from_kwargs(kwargs)

        assert manager._interval_plugin_override == "my_interval"
        assert manager._fast_interval_plugin_override == "my_fast_interval"

    def test_initialize_plot_override(self):
        """should_initialize_plot_style_override_from_kwargs."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        kwargs = {"plot_style": "my_style"}
        manager.initialize_from_kwargs(kwargs)

        assert manager.plot_style_override == "my_style"

    def test_initialize_missing_overrides_remain_none(self):
        """should_keep_overrides_none_when_not_in_kwargs."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        kwargs = {}
        manager.initialize_from_kwargs(kwargs)

        assert manager._interval_plugin_override is None
        assert manager._fast_interval_plugin_override is None
        assert manager.plot_style_override is None


class TestCoercePluginOverride:
    """Tests for coerce_plugin_override method."""

    def test_coerce_none_returns_none(self):
        """should_coerce_none_to_none."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        result = manager.coerce_plugin_override(None)
        assert result is None

    def test_coerce_string_returns_string(self):
        """should_coerce_string_to_string."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        result = manager.coerce_plugin_override("my_identifier")
        assert result == "my_identifier"

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

    def test_coerce_instance_with_plugin_meta_returns_as_is(self):
        """should_return_instance_with_plugin_meta_unchanged."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        mock_plugin = Mock()
        mock_plugin.plugin_meta = {"name": "test"}

        result = manager.coerce_plugin_override(mock_plugin)
        assert result is mock_plugin

    def test_coerce_dict_returns_as_is(self):
        """should_return_dict_unchanged."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        obj = {"key": "value"}
        result = manager.coerce_plugin_override(obj)
        assert result is obj


class TestBridgeMonitorManagement:
    """Tests for bridge monitor caching."""

    def test_get_bridge_monitor_creates_new(self):
        """should_create_new_bridge_monitor_when_not_cached."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        monitor = manager.get_bridge_monitor("my_plugin")
        assert isinstance(monitor, PredictBridgeMonitor)

    def test_get_bridge_monitor_returns_cached(self):
        """should_return_cached_bridge_monitor_on_subsequent_calls."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        monitor1 = manager.get_bridge_monitor("my_plugin")
        monitor2 = manager.get_bridge_monitor("my_plugin")
        assert monitor1 is monitor2

    def test_clear_bridge_monitors_empties_cache(self):
        """should_empty_bridge_monitor_cache_on_clear."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        _ = manager.get_bridge_monitor("my_plugin")
        assert len(manager._bridge_monitors) == 1

        manager.clear_bridge_monitors()
        assert len(manager._bridge_monitors) == 0


class TestExplanationPluginInstanceManagement:
    """Tests for explanation plugin instance caching."""

    def test_get_explanation_plugin_instance_not_cached(self):
        """should_return_none_when_plugin_instance_not_cached."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        result = manager.get_explanation_plugin_instance("unknown")
        assert result is None

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
        assert len(manager._explanation_plugin_instances) == 2

        manager.clear_explanation_plugin_instances()
        assert len(manager._explanation_plugin_instances) == 0


class TestExplanationPluginIdentifierManagement:
    """Tests for explanation plugin identifier caching."""

    def test_get_explanation_plugin_identifier_not_cached(self):
        """should_return_none_when_identifier_not_cached."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        result = manager.get_explanation_plugin_identifier("factual")
        assert result is None

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

    def test_init_creates_interval_state(self):
        """should_initialize_interval_plugin_state."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        assert "default" in manager.interval_plugin_identifiers
        assert "fast" in manager.interval_plugin_identifiers
        assert manager.interval_plugin_identifiers["default"] is None
        assert manager.interval_plugin_identifiers["fast"] is None

    def test_init_creates_interval_context_metadata(self):
        """should_initialize_interval_context_metadata."""
        mock_explainer = Mock()
        manager = PluginManager(mock_explainer)

        assert "default" in manager.interval_context_metadata
        assert "fast" in manager.interval_context_metadata
        assert manager.interval_context_metadata["default"] == {}
        assert manager.interval_context_metadata["fast"] == {}


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
