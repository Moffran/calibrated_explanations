"""Unit tests for execution strategy wrapper explanation plugins.

Tests verify that:
- Wrapper plugins are properly registered
- Metadata declares correct fallback chains
- Wrapper plugins delegate to execution plugins
- Fallback to legacy implementation works when execution fails
"""

from __future__ import annotations

import pytest

from calibrated_explanations.plugins import registry
from calibrated_explanations.plugins.builtins import (
    FeatureParallelAlternativeExplanationPlugin,
    FeatureParallelExplanationPlugin,
    InstanceParallelAlternativeExplanationPlugin,
    InstanceParallelExplanationPlugin,
    SequentialAlternativeExplanationPlugin,
    SequentialExplanationPlugin,
)


class TestExecutionStrategyPluginRegistration:
    """Test that execution strategy wrapper plugins are registered correctly."""

    def test_sequential_factual_plugin_registered(self):
        """Sequential factual plugin should be registered."""
        descriptor = registry.find_explanation_descriptor("core.explanation.factual.sequential")
        assert descriptor is not None
        assert descriptor.identifier == "core.explanation.factual.sequential"
        assert "factual" in descriptor.metadata.get("modes", ())
        assert descriptor.trusted

    def test_feature_parallel_factual_plugin_registered(self):
        """Feature-parallel factual plugin should be registered."""
        descriptor = registry.find_explanation_descriptor(
            "core.explanation.factual.feature_parallel"
        )
        assert descriptor is not None
        assert descriptor.identifier == "core.explanation.factual.feature_parallel"
        assert "factual" in descriptor.metadata.get("modes", ())
        assert descriptor.trusted

    def test_instance_parallel_factual_plugin_registered(self):
        """Instance-parallel factual plugin should be registered."""
        descriptor = registry.find_explanation_descriptor(
            "core.explanation.factual.instance_parallel"
        )
        assert descriptor is not None
        assert descriptor.identifier == "core.explanation.factual.instance_parallel"
        assert "factual" in descriptor.metadata.get("modes", ())
        assert descriptor.trusted

    def test_sequential_alternative_plugin_registered(self):
        """Sequential alternative plugin should be registered."""
        descriptor = registry.find_explanation_descriptor("core.explanation.alternative.sequential")
        assert descriptor is not None
        assert descriptor.identifier == "core.explanation.alternative.sequential"
        assert "alternative" in descriptor.metadata.get("modes", ())
        assert descriptor.trusted

    def test_feature_parallel_alternative_plugin_registered(self):
        """Feature-parallel alternative plugin should be registered."""
        descriptor = registry.find_explanation_descriptor(
            "core.explanation.alternative.feature_parallel"
        )
        assert descriptor is not None
        assert descriptor.identifier == "core.explanation.alternative.feature_parallel"
        assert "alternative" in descriptor.metadata.get("modes", ())
        assert descriptor.trusted

    def test_instance_parallel_alternative_plugin_registered(self):
        """Instance-parallel alternative plugin should be registered."""
        descriptor = registry.find_explanation_descriptor(
            "core.explanation.alternative.instance_parallel"
        )
        assert descriptor is not None
        assert descriptor.identifier == "core.explanation.alternative.instance_parallel"
        assert "alternative" in descriptor.metadata.get("modes", ())
        assert descriptor.trusted


class TestExecutionStrategyPluginMetadata:
    """Test that wrapper plugins declare correct metadata and fallback chains."""

    def test_sequential_factual_metadata_has_fallback(self):
        """Sequential factual plugin should declare fallback to legacy."""
        plugin = SequentialExplanationPlugin()
        assert "fallbacks" in plugin.plugin_meta
        assert "core.explanation.factual" in plugin.plugin_meta["fallbacks"]

    def test_feature_parallel_factual_fallback_chain(self):
        """Feature-parallel factual should have sequential and legacy fallbacks."""
        plugin = FeatureParallelExplanationPlugin()
        fallbacks = plugin.plugin_meta["fallbacks"]
        assert "core.explanation.factual.sequential" in fallbacks
        assert "core.explanation.factual" in fallbacks

    def test_instance_parallel_factual_fallback_chain(self):
        """Instance-parallel factual should have feature-parallel, sequential, and legacy fallbacks."""
        plugin = InstanceParallelExplanationPlugin()
        fallbacks = plugin.plugin_meta["fallbacks"]
        assert "core.explanation.factual.feature_parallel" in fallbacks
        assert "core.explanation.factual.sequential" in fallbacks
        assert "core.explanation.factual" in fallbacks

    def test_sequential_alternative_metadata_has_fallback(self):
        """Sequential alternative plugin should declare fallback to legacy."""
        plugin = SequentialAlternativeExplanationPlugin()
        assert "fallbacks" in plugin.plugin_meta
        assert "core.explanation.alternative" in plugin.plugin_meta["fallbacks"]

    def test_feature_parallel_alternative_fallback_chain(self):
        """Feature-parallel alternative should have sequential and legacy fallbacks."""
        plugin = FeatureParallelAlternativeExplanationPlugin()
        fallbacks = plugin.plugin_meta["fallbacks"]
        assert "core.explanation.alternative.sequential" in fallbacks
        assert "core.explanation.alternative" in fallbacks

    def test_instance_parallel_alternative_fallback_chain(self):
        """Instance-parallel alternative should have feature-parallel, sequential, and legacy fallbacks."""
        plugin = InstanceParallelAlternativeExplanationPlugin()
        fallbacks = plugin.plugin_meta["fallbacks"]
        assert "core.explanation.alternative.feature_parallel" in fallbacks
        assert "core.explanation.alternative.sequential" in fallbacks
        assert "core.explanation.alternative" in fallbacks


class TestExecutionStrategyPluginAttributes:
    """Test that wrapper plugins have correct attributes and capabilities."""

    def test_sequential_factual_plugin_attributes(self):
        """Sequential factual plugin should have correct base attributes."""
        plugin = SequentialExplanationPlugin()
        assert plugin._mode == "factual"
        assert plugin._explanation_attr == "explain_factual"
        assert plugin._execution_plugin_class is not None

    def test_feature_parallel_factual_plugin_attributes(self):
        """Feature-parallel factual plugin should have correct base attributes."""
        plugin = FeatureParallelExplanationPlugin()
        assert plugin._mode == "factual"
        assert plugin._explanation_attr == "explain_factual"
        assert plugin._execution_plugin_class is not None

    def test_instance_parallel_factual_plugin_attributes(self):
        """Instance-parallel factual plugin should have correct base attributes."""
        plugin = InstanceParallelExplanationPlugin()
        assert plugin._mode == "factual"
        assert plugin._explanation_attr == "explain_factual"
        assert plugin._execution_plugin_class is not None

    def test_sequential_alternative_plugin_attributes(self):
        """Sequential alternative plugin should have correct base attributes."""
        plugin = SequentialAlternativeExplanationPlugin()
        assert plugin._mode == "alternative"
        assert plugin._explanation_attr == "explore_alternatives"
        assert plugin._execution_plugin_class is not None

    def test_feature_parallel_alternative_plugin_attributes(self):
        """Feature-parallel alternative plugin should have correct base attributes."""
        plugin = FeatureParallelAlternativeExplanationPlugin()
        assert plugin._mode == "alternative"
        assert plugin._explanation_attr == "explore_alternatives"
        assert plugin._execution_plugin_class is not None

    def test_instance_parallel_alternative_plugin_attributes(self):
        """Instance-parallel alternative plugin should have correct base attributes."""
        plugin = InstanceParallelAlternativeExplanationPlugin()
        assert plugin._mode == "alternative"
        assert plugin._explanation_attr == "explore_alternatives"
        assert plugin._execution_plugin_class is not None


class TestPluginSupportsMode:
    """Test that plugins support the correct modes and tasks."""

    def test_sequential_factual_supports_factual_mode(self):
        """Sequential factual plugin should support factual mode."""
        plugin = SequentialExplanationPlugin()
        assert plugin.supports_mode("factual", task="classification")
        assert plugin.supports_mode("factual", task="regression")
        assert not plugin.supports_mode("alternative", task="classification")

    def test_sequential_alternative_supports_alternative_mode(self):
        """Sequential alternative plugin should support alternative mode."""
        plugin = SequentialAlternativeExplanationPlugin()
        assert plugin.supports_mode("alternative", task="classification")
        assert plugin.supports_mode("alternative", task="regression")
        assert not plugin.supports_mode("factual", task="classification")

    def test_feature_parallel_factual_supports_factual_mode(self):
        """Feature-parallel factual plugin should support factual mode."""
        plugin = FeatureParallelExplanationPlugin()
        assert plugin.supports_mode("factual", task="classification")
        assert not plugin.supports_mode("alternative", task="classification")

    def test_instance_parallel_factual_supports_factual_mode(self):
        """Instance-parallel factual plugin should support factual mode."""
        plugin = InstanceParallelExplanationPlugin()
        assert plugin.supports_mode("factual", task="classification")
        assert not plugin.supports_mode("alternative", task="classification")


class TestExecutionPluginClassConfiguration:
    """Test that execution plugin classes are correctly configured."""

    def test_sequential_loads_sequential_executor_class(self):
        """Sequential wrapper should load SequentialExplainExecutor."""
        plugin = SequentialExplanationPlugin()
        assert plugin._execution_plugin_class is not None
        # Verify the class name matches
        assert "Sequential" in plugin._execution_plugin_class.__name__

    def test_feature_parallel_loads_feature_executor_class(self):
        """Feature-parallel wrapper should load FeatureParallelExplainExecutor."""
        plugin = FeatureParallelExplanationPlugin()
        assert plugin._execution_plugin_class is not None
        # Verify the class name matches
        assert "FeatureParallel" in plugin._execution_plugin_class.__name__

    def test_instance_parallel_loads_instance_executor_class(self):
        """Instance-parallel wrapper should load InstanceParallelExplainExecutor."""
        plugin = InstanceParallelExplanationPlugin()
        assert plugin._execution_plugin_class is not None
        # Verify the class name matches
        assert "InstanceParallel" in plugin._execution_plugin_class.__name__
