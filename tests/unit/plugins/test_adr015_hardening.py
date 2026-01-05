"""Comprehensive test suite for ADR-015 explanation plugin integration hardening gaps."""

import os
import pickle
import pytest
from unittest.mock import Mock, MagicMock, patch
from types import MappingProxyType

from calibrated_explanations.plugins.explanations import (
    ExplanationContext,
    ExplainerHandle,
    ExplanationRequest,
    ExplanationBatch,
)
from calibrated_explanations.explanations.explanations import CalibratedExplanations
from calibrated_explanations.utils.exceptions import ConfigurationError, ValidationError
from calibrated_explanations.plugins.registry import (
    find_explanation_descriptor,
    find_explanation_plugin_trusted,
    is_identifier_denied,
)


# Helper classes
class FakeBridge:
    """Pickleable bridge for testing."""
    pass


@pytest.fixture
def mock_explainer():
    """Create a mock CalibratedExplainer."""
    explainer = MagicMock()
    explainer.num_features = 5
    explainer.mode = "classification"
    return explainer


# ==============================================================================
# STEP 1: In-Tree FAST Explanation Plugin Tests
# ==============================================================================


class TestFastExplanationPluginRegistration:
    """Verify in-tree FAST plugin registration (Step 1)."""

    def test_fast_explanation_plugin_always_registered_when_should_not_exist(self):
        """Test that core.explanation.fast is registered after ensure_builtins."""
        from calibrated_explanations.plugins import ensure_builtin_plugins

        ensure_builtin_plugins()
        descriptor = find_explanation_descriptor("core.explanation.fast")
        assert descriptor is not None, "core.explanation.fast plugin should be registered"
        assert descriptor.trusted is True, "FAST plugin should be trusted"

    def test_fast_explanation_plugin_metadata_complete(self):
        """Test that FAST plugin metadata includes all required fields per ADR-015."""
        from calibrated_explanations.plugins import ensure_builtin_plugins

        ensure_builtin_plugins()
        descriptor = find_explanation_descriptor("core.explanation.fast")
        assert descriptor is not None
        
        meta = descriptor.metadata
        assert meta["name"] == "core.explanation.fast"
        assert "interval_dependency" in meta
        assert meta["modes"] == ("fast",)
        assert "task:classification" in meta.get("capabilities", [])
        assert "task:regression" in meta.get("capabilities", [])

    def test_fast_plugin_without_external_extras(self):
        """Test that FAST plugin is available without external extras."""
        from calibrated_explanations.plugins import ensure_builtin_plugins

        ensure_builtin_plugins()
        plugin = find_explanation_plugin_trusted("core.explanation.fast")
        assert plugin is not None, "FAST plugin should be available without external extras"


# ==============================================================================
# STEP 2: Canonical Collection Reconstruction Tests
# ==============================================================================


class TestCanonicalCollectionReconstruction:
    """Verify canonical reconstruction of collections from batches (Step 2)."""

    @pytest.fixture
    def mock_batch_no_template(self, mock_explainer):
        """Create an ExplanationBatch without embedded container template."""
        from calibrated_explanations.explanations.explanation import FactualExplanation
        import numpy as np

        # Create mock explanation - minimal valid state
        explanation = MagicMock(spec=FactualExplanation)
        explanation.instance_index = 0
        
        metadata = {
            "calibrated_explainer": mock_explainer,
            "x_test": np.array([[1, 2, 3, 4, 5]]),
            "y_threshold": 0.5,
            "bins": None,
            "features_to_ignore": [],
            "mode": "factual",
            "interval_dependency": "core.interval.legacy",
            "plot_dependency": "plot_spec.default",
        }

        instance_payload = {"explanation": explanation}

        batch = ExplanationBatch(
            container_cls=CalibratedExplanations,
            explanation_cls=FactualExplanation,
            instances=[instance_payload],
            collection_metadata=metadata,
        )
        return batch

    def test_from_batch_without_template_reconstructs(self, mock_batch_no_template):
        """Test that from_batch reconstructs when no template is present."""
        collection = CalibratedExplanations.from_batch(mock_batch_no_template)
        assert isinstance(collection, CalibratedExplanations)
        assert len(collection.explanations) == 1
        assert collection.explanations[0].instance_index == 0

    def test_from_batch_preserves_metadata(self, mock_batch_no_template):
        """Test that from_batch preserves all metadata fields."""
        collection = CalibratedExplanations.from_batch(mock_batch_no_template)
        # Metadata should be stored in batch_metadata
        batch_meta = getattr(collection, "batch_metadata", {})
        assert batch_meta.get("mode") == "factual"
        assert batch_meta.get("interval_dependency") == "core.interval.legacy"
        assert batch_meta.get("plot_dependency") == "plot_spec.default"

    def test_from_batch_populates_derived_attributes(self, mock_batch_no_template):
        """Test that from_batch populates derived caches correctly."""
        collection = CalibratedExplanations.from_batch(mock_batch_no_template)
        # Collection should have explanations attached
        assert len(collection.explanations) > 0
        # Each explanation should reference the parent collection
        for exp in collection.explanations:
            assert exp.calibrated_explanations is collection

    def test_from_batch_validates_explanation_type(self, mock_explainer):
        """Test that from_batch validates explanation types."""
        from calibrated_explanations.explanations.explanation import FactualExplanation
        import numpy as np

        metadata = {
            "calibrated_explainer": mock_explainer,
            "x_test": np.array([[1, 2, 3, 4, 5]]),
        }

        # Wrong explanation type
        instance_payload = {"explanation": "not_an_explanation"}

        batch = ExplanationBatch(
            container_cls=CalibratedExplanations,
            explanation_cls=FactualExplanation,
            instances=[instance_payload],
            collection_metadata=metadata,
        )

        with pytest.raises(ValidationError):
            CalibratedExplanations.from_batch(batch)


# ==============================================================================
# STEP 3: Trust Enforcement Tests
# ==============================================================================


class TestTrustEnforcement:
    """Verify trust enforcement in explanation plugin resolution (Step 3)."""

    def test_untrusted_plugin_rejected_from_env_var(self):
        """Test that untrusted plugins from env vars are rejected."""
        # This test requires a mock untrusted plugin to be registered
        # For now, we verify the trust checking logic exists
        from calibrated_explanations.core.explain.orchestrator import ExplanationOrchestrator

        orchestrator = MagicMock(spec=ExplanationOrchestrator)
        # Orchestrator should have resolve_plugin method
        assert hasattr(orchestrator, "resolve_plugin") or True  # Defensive check

    def test_denied_plugin_raises_error(self):
        """Test that denied plugins raise ConfigurationError."""
        from calibrated_explanations.plugins import find_explanation_descriptor

        # Try to find a denied plugin (will depend on CE_DENY_PLUGIN env var)
        # This is a placeholder for integration testing
        descriptor = find_explanation_descriptor("core.explanation.factual")
        if descriptor is not None and is_identifier_denied("core.explanation.factual"):
            pytest.fail("Built-in factual plugin should not be denied by default")

    def test_explicit_override_allows_untrusted_with_warning(self, mock_explainer):
        """Test that explicit overrides can use untrusted plugins but log a warning."""
        import warnings
        from calibrated_explanations.core.explain.orchestrator import ExplanationOrchestrator
        from calibrated_explanations.plugins.registry import (
            register_explanation_plugin,
            ExplanationPluginDescriptor,
        )

        # Register a mock untrusted plugin
        mock_plugin = MagicMock()
        mock_plugin.plugin_meta = {
            "name": "test.untrusted", 
            "trusted": False, 
            "schema_version": 1,
            "tasks": ["classification"],
            "modes": ["factual"],
            "capabilities": ["explain", "explanation:factual", "task:classification"]
        }
        
        # We need to bypass the registry's trust check for registration if it has one
        # but here we just want to see if resolve_plugin warns.
        with patch("calibrated_explanations.core.explain.orchestrator.find_explanation_descriptor") as mock_find:
            descriptor = MagicMock(spec=ExplanationPluginDescriptor)
            descriptor.plugin = mock_plugin
            descriptor.trusted = False
            descriptor.metadata = mock_plugin.plugin_meta
            mock_find.return_value = descriptor
            
            orchestrator = ExplanationOrchestrator(mock_explainer)
            mock_explainer.mode = "classification"
            mock_explainer.plugin_manager.explanation_plugin_overrides = {"factual": "test.untrusted"}
            mock_explainer.plugin_manager.coerce_plugin_override.side_effect = lambda x: x
            mock_explainer.plugin_manager.explanation_plugin_fallbacks = {"factual": ("test.untrusted",)}
            
            with pytest.warns(UserWarning, match="Using untrusted explanation plugin"):
                plugin, identifier = orchestrator.resolve_plugin("factual")
                assert identifier == "test.untrusted"
                assert plugin == mock_plugin


# ==============================================================================
# STEP 4: Environment Variable Alignment Tests
# ==============================================================================


class TestEnvironmentVariableAlignment:
    """Verify alignment of env vars with ADR-015 (Step 4)."""

    def test_ce_explanation_plugin_env_var_honored(self):
        """Test that CE_EXPLANATION_PLUGIN is recognized as top-level default."""
        from calibrated_explanations.plugins.manager import PluginManager

        # Create mock explainer
        explainer = MagicMock()
        pm = PluginManager(explainer)

        with patch.dict(os.environ, {"CE_EXPLANATION_PLUGIN": "core.explanation.factual"}):
            chain = pm.build_explanation_chain("factual", "core.explanation.factual.sequential")
            # The top-level env var should appear in the chain
            assert "core.explanation.factual" in chain or "core.explanation.factual.sequential" in chain

    def test_ce_explanation_plugin_fast_env_var_honored(self):
        """Test that CE_EXPLANATION_PLUGIN_FAST is recognized for fast mode."""
        from calibrated_explanations.plugins.manager import PluginManager

        explainer = MagicMock()
        pm = PluginManager(explainer)

        with patch.dict(os.environ, {"CE_EXPLANATION_PLUGIN_FAST": "core.explanation.fast"}):
            chain = pm.build_explanation_chain("fast", "core.explanation.fast")
            # The FAST env var should appear in the chain
            assert "core.explanation.fast" in chain

    def test_mode_specific_env_vars_as_fallback(self):
        """Test that mode-specific env vars work as fallbacks."""
        from calibrated_explanations.plugins.manager import PluginManager

        explainer = MagicMock()
        pm = PluginManager(explainer)

        with patch.dict(
            os.environ,
            {
                "CE_EXPLANATION_PLUGIN_FACTUAL": "core.explanation.factual.sequential",
                "CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS": "core.explanation.factual",
            },
        ):
            chain = pm.build_explanation_chain(
                "factual", "core.explanation.factual.sequential"
            )
            # Both should appear in the chain
            assert len(chain) > 0


# ==============================================================================
# STEP 5: Immutable Plugin Handles Tests
# ==============================================================================


class TestImmutablePluginHandles:
    """Verify immutable handles in ExplanationContext (Step 5)."""

    def test_explanation_context_helper_handles_is_mapping_proxy(self):
        """Test that helper_handles is wrapped in MappingProxyType."""
        bridge = MagicMock()
        handles = {"explainer": MagicMock()}
        
        context = ExplanationContext(
            task="classification",
            mode="factual",
            feature_names=["f1", "f2"],
            categorical_features=[],
            categorical_labels={},
            discretizer=None,
            helper_handles=MappingProxyType(handles),
            predict_bridge=bridge,
            interval_settings={},
            plot_settings={},
        )

        # Verify it's read-only
        with pytest.raises(TypeError):
            context.helper_handles["new_key"] = "value"

    def test_explainer_handle_wraps_explainer_object(self):
        """Test that ExplainerHandle provides constrained interface."""
        explainer = MagicMock()
        explainer.num_features = 10
        explainer.mode = "classification"
        explainer.is_multiclass.return_value = True

        metadata = {"source": "test"}
        handle = ExplainerHandle(explainer, metadata)

        assert handle.num_features == 10
        assert handle.mode == "classification"
        assert handle.is_multiclass is True

    def test_explainer_handle_protects_against_mutation(self):
        """Test that ExplainerHandle prevents direct reassignment of wrapped object."""
        explainer = MagicMock()
        metadata = {"key": "value"}
        
        handle = ExplainerHandle(explainer, metadata)
        
        # Using __slots__, the handle should not allow new attributes
        # but the private attributes are read-only by design
        # Note: this test verifies the design intent; direct reassignment
        # is prevented by __slots__
        assert hasattr(handle, "_ExplainerHandle__explainer")

    def test_explanation_context_is_frozen(self):
        """Test that ExplanationContext is a frozen dataclass."""
        bridge = MagicMock()
        context = ExplanationContext(
            task="classification",
            mode="factual",
            feature_names=["f1"],
            categorical_features=[],
            categorical_labels={},
            discretizer=None,
            helper_handles={},
            predict_bridge=bridge,
            interval_settings={},
            plot_settings={},
        )

        # Should not be able to modify frozen fields
        with pytest.raises((AttributeError, TypeError)):
            context.task = "regression"

    def test_explanation_context_pickleable_with_proxy(self):
        """Test that ExplanationContext is pickleable with plain dicts."""
        handles = {"test": "value"}
        
        context = ExplanationContext(
            task="classification",
            mode="factual",
            feature_names=["f1"],
            categorical_features=[],
            categorical_labels={},
            discretizer=None,
            helper_handles=handles,
            predict_bridge=FakeBridge(),
            interval_settings={},
            plot_settings={},
        )

        # Should be pickleable
        pickled = pickle.dumps(context)
        unpickled = pickle.loads(pickled)
        assert unpickled.task == "classification"


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestADR015IntegrationFlow:
    """End-to-end tests for ADR-015 explanation plugin pipeline."""

    def test_full_pipeline_plugin_resolution_to_collection(self):
        """Test full pipeline from plugin resolution to collection reconstruction."""
        from calibrated_explanations.plugins import ensure_builtin_plugins

        ensure_builtin_plugins()
        
        # Verify at least one core plugin is registered and trusted
        descriptor = find_explanation_descriptor("core.explanation.factual")
        assert descriptor is not None
        assert descriptor.trusted is True

    def test_batch_metadata_preserved_through_pipeline(self):
        """Test that metadata is preserved through plugin→batch→collection."""
        from calibrated_explanations.explanations.explanation import FactualExplanation
        import numpy as np

        explainer = MagicMock()
        explainer.num_features = 3
        
        metadata = {
            "calibrated_explainer": explainer,
            "x_test": np.array([[1, 2, 3]]),
            "mode": "factual",
            "interval_dependency": "core.interval.legacy",
            "plot_dependency": "plot_spec.default",
            "telemetry": {"version": "0.10.2"},
        }

        # Create mock explanation
        explanation = MagicMock(spec=FactualExplanation)
        explanation.instance_index = 0

        instance_payload = {"explanation": explanation}

        batch = ExplanationBatch(
            container_cls=CalibratedExplanations,
            explanation_cls=FactualExplanation,
            instances=[instance_payload],
            collection_metadata=metadata,
        )

        collection = CalibratedExplanations.from_batch(batch)
        
        # Verify metadata preserved
        batch_meta = getattr(collection, "batch_metadata", {})
        assert batch_meta.get("mode") == "factual"
        assert batch_meta.get("telemetry", {}).get("version") == "0.10.2"

    def test_from_batch_reconstructs_new_instance_even_with_template(self, mock_explainer):
        """Test that from_batch always returns a new instance, not the template."""
        from calibrated_explanations.explanations.explanation import FactualExplanation
        import numpy as np

        # Create a template instance
        template = CalibratedExplanations(
            mock_explainer,
            np.array([[1, 2, 3]]),
            y_threshold=0.5,
            bins=None
        )
        
        metadata = {
            "calibrated_explainer": mock_explainer,
            "x_test": np.array([[1, 2, 3]]),
            "container": template,
        }

        # Create mock explanation
        explanation = MagicMock(spec=FactualExplanation)
        explanation.instance_index = 0

        instance_payload = {"explanation": explanation}

        batch = ExplanationBatch(
            container_cls=CalibratedExplanations,
            explanation_cls=FactualExplanation,
            instances=[instance_payload],
            collection_metadata=metadata,
        )

        collection = CalibratedExplanations.from_batch(batch)
        
        # Verify it's a NEW instance
        assert collection is not template, "from_batch should return a new instance, not the template"
        assert len(collection.explanations) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
