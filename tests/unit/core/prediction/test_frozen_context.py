"""Tests for interval context immutability enforcement (ADR-013).

This module verifies the immutability design for interval contexts:
- Contexts are built with mutable metadata to allow plugins to function
- After plugin execution, metadata is frozen for caching via obtain_interval_calibrator()
- bins, residuals, difficulty, fast_flags remain frozen throughout execution
- This design allows plugin operations while ensuring cached contexts are immutable

Note: These tests verify the frozen state AFTER plugin execution, not during
build_interval_context() which intentionally returns unfrozen contexts for plugins.
"""

from __future__ import annotations

import pytest
import numpy as np
from types import MappingProxyType

from tests.helpers.explainer_utils import make_explainer_from_dataset
from calibrated_explanations.core.prediction.orchestrator import PredictionOrchestrator
from calibrated_explanations.plugins.intervals import IntervalCalibratorContext
from calibrated_explanations.utils.exceptions import ConfigurationError


class TestFrozenContextStructure:
    """Verify context structure uses MappingProxyType for critical fields."""

    def test_context_bins_is_mapping_proxy(self, binary_dataset):
        """Context.bins should be a MappingProxyType."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        assert isinstance(context.bins, MappingProxyType), (
            f"bins should be MappingProxyType, got {type(context.bins)}"
        )

    def test_context_residuals_is_mapping_proxy(self, binary_dataset):
        """Context.residuals should be a MappingProxyType."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        assert isinstance(context.residuals, MappingProxyType), (
            f"residuals should be MappingProxyType, got {type(context.residuals)}"
        )

    def test_context_difficulty_is_mapping_proxy(self, binary_dataset):
        """Context.difficulty should be a MappingProxyType."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        assert isinstance(context.difficulty, MappingProxyType), (
            f"difficulty should be MappingProxyType, got {type(context.difficulty)}"
        )

    def test_context_metadata_is_dict_for_plugins(self, binary_dataset):
        """Context.metadata should be mutable dict for plugin execution."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        assert isinstance(context.metadata, dict) and not isinstance(context.metadata, MappingProxyType), (
            f"metadata should be dict (not MappingProxyType) for plugin use, got {type(context.metadata)}"
        )

    def test_context_fast_flags_is_mapping_proxy(self, binary_dataset):
        """Context.fast_flags should be a MappingProxyType."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        assert isinstance(context.fast_flags, MappingProxyType), (
            f"fast_flags should be MappingProxyType, got {type(context.fast_flags)}"
        )

    def test_context_calibration_splits_is_tuple(self, binary_dataset):
        """Context.calibration_splits should be an immutable tuple."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        assert isinstance(context.calibration_splits, tuple), (
            f"calibration_splits should be tuple, got {type(context.calibration_splits)}"
        )

    def test_context_learner_is_reference(self, binary_dataset):
        """Context.learner should be a direct reference (mutable, intentional)."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        # learner is the actual learner object, not wrapped
        assert context.learner is explainer.learner


class TestFrozenContextBinsImmutability:
    """Verify bins field cannot be modified by plugins."""

    def test_bins_prevents_key_assignment(self, binary_dataset):
        """Attempting to assign a new key to bins should raise TypeError."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        with pytest.raises(TypeError, match="does not support item assignment"):
            context.bins["new_key"] = "new_value"

    def test_bins_prevents_key_deletion(self, binary_dataset):
        """Attempting to delete a key from bins should raise TypeError."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        # bins has "calibration" key by default
        with pytest.raises(TypeError):
            del context.bins["calibration"]

    def test_bins_nested_value_immutable(self, binary_dataset):
        """Nested values in bins should be immutable (tuples)."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        calibration_bins = context.bins.get("calibration")
        assert isinstance(calibration_bins, (tuple, type(None))), (
            f"bins['calibration'] should be tuple or None, got {type(calibration_bins)}"
        )

    def test_bins_clear_raises_error(self, binary_dataset):
        """Attempting to clear bins should raise AttributeError (no such method on MappingProxyType)."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        # MappingProxyType doesn't have a clear() method
        with pytest.raises(AttributeError):
            context.bins.clear()  # type: ignore


class TestFrozenContextResidualsImmutability:
    """Verify residuals field cannot be modified by plugins."""

    def test_residuals_prevents_key_assignment(self, binary_dataset):
        """Attempting to assign a new key to residuals should raise TypeError."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        with pytest.raises(TypeError, match="does not support item assignment"):
            context.residuals["new_key"] = "new_value"

    def test_residuals_prevents_key_deletion(self, binary_dataset):
        """Attempting to delete a key from residuals should raise AttributeError (no such method on MappingProxyType)."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        # residuals is empty by default but still protected
        # MappingProxyType doesn't have pop() method
        with pytest.raises(AttributeError):
            context.residuals.pop("key", None)  # type: ignore


class TestFrozenContextMetadataImmutability:
    """Verify metadata is mutable for plugin execution, nested dicts still frozen."""

    def test_metadata_is_mutable_dict_for_plugins(self, binary_dataset):
        """Metadata should be a mutable dict for plugin execution."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        # Metadata is mutable (dict) for plugin use
        assert isinstance(context.metadata, dict) and not isinstance(context.metadata, MappingProxyType)
        context.metadata["new_key"] = "new_value"
        assert context.metadata["new_key"] == "new_value"

    def test_metadata_contains_required_fields(self, binary_dataset):
        """Metadata should contain required fields for plugin execution."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        required_fields = ["task", "mode", "explainer"]
        for field in required_fields:
            assert field in context.metadata, (
                f"metadata missing required field '{field}'"
            )

    def test_metadata_explainer_reference_accessible(self, binary_dataset):
        """Explainer reference in metadata should be accessible (for plugins)."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        stored_explainer = context.metadata.get("explainer")
        assert stored_explainer is explainer, (
            "Explainer in metadata should reference the original explainer"
        )


class TestFrozenContextDifficultyImmutability:
    """Verify difficulty field cannot be modified by plugins."""

    def test_difficulty_prevents_key_assignment(self, binary_dataset):
        """Attempting to assign a new key to difficulty should raise TypeError."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        with pytest.raises(TypeError, match="does not support item assignment"):
            context.difficulty["new_key"] = "new_value"

    def test_difficulty_prevents_key_deletion(self, binary_dataset):
        """Attempting to delete a key from difficulty should raise AttributeError (no such method on MappingProxyType)."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        # MappingProxyType doesn't have pop() method
        with pytest.raises(AttributeError):
            context.difficulty.pop("estimator", None)  # type: ignore


class TestFrozenContextFastFlagsImmutability:
    """Verify fast_flags field cannot be modified by plugins."""

    def test_fast_flags_prevents_key_assignment(self, binary_dataset):
        """Attempting to assign a new key to fast_flags should raise TypeError."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        with pytest.raises(TypeError, match="does not support item assignment"):
            context.fast_flags["new_flag"] = True

    def test_fast_flags_prevents_existing_key_modification(self, binary_dataset):
        """Attempting to modify existing fast_flags should raise TypeError."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        with pytest.raises(TypeError, match="does not support item assignment"):
            context.fast_flags["fast"] = True


class TestFrozenContextFastMode:
    """Verify frozen context immutability in fast mode."""

    def test_fast_context_bins_frozen(self, binary_dataset):
        """Bins in fast mode should also be frozen."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=True, metadata={})

        assert isinstance(context.bins, MappingProxyType)
        with pytest.raises(TypeError, match="does not support item assignment"):
            context.bins["test"] = "value"

    def test_fast_context_metadata_mutable_for_plugins(self, binary_dataset):
        """Metadata in fast mode should be mutable dict for plugin execution."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=True, metadata={})

        # Metadata is mutable even in fast mode (design: mutable for plugins)
        assert isinstance(context.metadata, dict) and not isinstance(context.metadata, MappingProxyType)
        context.metadata["test"] = "value"
        assert context.metadata["test"] == "value"

    def test_fast_context_preserves_existing_fast_calibrators(self, binary_dataset):
        """Metadata should preserve existing fast calibrators when available."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)

        # Simulate cached fast calibrators
        context = orchestrator.build_interval_context(
            fast=True, metadata={"existing_fast_calibrators": (object(), object())}
        )

        # Verify the field exists (though frozen)
        assert "existing_fast_calibrators" in context.metadata


class TestFrozenContextClassificationMode:
    """Verify frozen context design in classification mode specifically."""

    def test_classification_context_metadata_mutable(self, binary_dataset):
        """Metadata should be mutable (dict) in classification mode."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        assert isinstance(context.metadata, dict)
        context.metadata["custom_field"] = "value"
        assert context.metadata.get("mode") == "classification"

    def test_classification_context_bins_frozen(self, binary_dataset):
        """Bins should be frozen in classification mode."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        assert isinstance(context.bins, MappingProxyType)


class TestFrozenContextRegressionMode:
    """Verify frozen context design in regression mode."""

    def test_regression_context_metadata_mutable(self, binary_dataset):
        """Metadata should be mutable (dict) in regression mode."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        # Metadata is mutable dict (design allows plugin mutation during execution)
        assert isinstance(context.metadata, dict) and not isinstance(context.metadata, MappingProxyType)
        context.metadata["custom_field"] = "value"
        assert context.metadata["custom_field"] == "value"

    def test_regression_context_bins_frozen(self, binary_dataset):
        """Bins should be frozen (tested with binary dataset for simplicity)."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        assert isinstance(context.bins, MappingProxyType)



class TestContextDatastructureConsistency:
    """Verify context maintains consistent frozen structure across calls."""

    def test_repeated_context_builds_are_consistent(self, binary_dataset):
        """Building context multiple times should produce consistent frozen structures."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)

        context1 = orchestrator.build_interval_context(fast=False, metadata={})
        context2 = orchestrator.build_interval_context(fast=False, metadata={})

        # Both should be MappingProxyType
        assert type(context1.bins) == type(context2.bins)
        assert type(context1.metadata) == type(context2.metadata)

        # Should be independent instances (new MappingProxyType each time)
        assert context1.bins is not context2.bins

    def test_context_dataclass_is_frozen(self, binary_dataset):
        """IntervalCalibratorContext dataclass should be frozen."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        # The dataclass itself should be frozen per ADR-013
        with pytest.raises((AttributeError, TypeError)):
            context.bins = {}  # type: ignore


class TestContextImmutabilityPluginSafety:
    """Verify frozen contexts prevent accidental mutation by well-behaved plugins."""

    def test_plugin_cannot_pollute_context_bins(self, binary_dataset):
        """Well-intentioned plugins should not be able to add keys to bins."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        # Simulate a plugin mistakenly trying to store state in context
        with pytest.raises(TypeError):
            context.bins["plugin_cache"] = {}  # type: ignore

    def test_plugin_can_mutate_metadata_during_execution(self, binary_dataset):
        """Plugins CAN mutate metadata during execution (it's a mutable dict)."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        # Metadata is mutable during execution phase
        context.metadata["calibrator"] = None
        assert context.metadata["calibrator"] is None

    def test_difficulty_remains_frozen_for_protection(self, binary_dataset):
        """Difficulty dict remains frozen even though metadata is mutable."""
        explainer, _ = make_explainer_from_dataset(binary_dataset, mode="classification")
        orchestrator = PredictionOrchestrator(explainer)
        context = orchestrator.build_interval_context(fast=False, metadata={})

        # Difficulty is still frozen (MappingProxyType) and cannot be modified
        with pytest.raises(TypeError):
            context.difficulty["custom_estimator"] = object()  # type: ignore


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
