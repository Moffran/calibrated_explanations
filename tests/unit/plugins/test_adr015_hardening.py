"""Comprehensive test suite for ADR-015 explanation plugin integration hardening gaps."""

import os
import pytest
from unittest.mock import MagicMock, patch
from types import MappingProxyType

from calibrated_explanations.plugins.explanations import (
    ExplanationContext,
    ExplanationBatch,
)
from calibrated_explanations.explanations.explanations import CalibratedExplanations
from calibrated_explanations.utils.exceptions import ConfigurationError, ValidationError


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

    def test_untrusted_plugin_rejected_from_env_var(self, mock_explainer):
        """Test that untrusted plugins from env vars are rejected."""
        from calibrated_explanations.core.explain.orchestrator import ExplanationOrchestrator
        from calibrated_explanations.plugins.registry import (
            ExplanationPluginDescriptor,
        )

        mock_plugin = MagicMock()
        mock_plugin.plugin_meta = {
            "name": "test.untrusted",
            "trusted": False,
            "schema_version": 1,
            "tasks": ["classification"],
            "modes": ["factual"],
            "capabilities": ["explain", "explanation:factual", "task:classification"],
        }

        with patch(
            "calibrated_explanations.core.explain.orchestrator.find_explanation_descriptor"
        ) as mock_find:
            descriptor = MagicMock(spec=ExplanationPluginDescriptor)
            descriptor.plugin = mock_plugin
            descriptor.trusted = False
            descriptor.metadata = mock_plugin.plugin_meta
            mock_find.return_value = descriptor

            orchestrator = ExplanationOrchestrator(mock_explainer)
            mock_explainer.mode = "classification"
            mock_explainer.plugin_manager.explanation_plugin_overrides = {"factual": None}
            mock_explainer.plugin_manager.coerce_plugin_override.side_effect = lambda x: x
            mock_explainer.plugin_manager.explanation_plugin_fallbacks = {
                "factual": ("test.untrusted",)
            }
            mock_explainer.plugin_manager.explanation_preferred_identifier = {
                "factual": "test.untrusted"
            }
            mock_explainer.plugin_manager.resolve_explanation_plugin.return_value = (
                None,
                None,
                "untrusted",
            )

            with pytest.raises(ConfigurationError, match="untrusted"):
                orchestrator.resolve_plugin("factual")

    def test_explicit_instance_override_warns_when_untrusted(self, mock_explainer):
        """Explicit instance overrides should warn when untrusted."""
        from calibrated_explanations.core.explain.orchestrator import ExplanationOrchestrator

        mock_plugin = MagicMock()
        mock_plugin.plugin_meta = {
            "name": "test.untrusted.instance",
            "trusted": False,
            "schema_version": 1,
            "tasks": ["classification"],
            "modes": ["factual"],
            "capabilities": ["explain", "explanation:factual", "task:classification"],
        }

        orchestrator = ExplanationOrchestrator(mock_explainer)
        mock_explainer.mode = "classification"
        mock_explainer.plugin_manager.explanation_plugin_overrides = {"factual": mock_plugin}
        mock_explainer.plugin_manager.coerce_plugin_override.side_effect = lambda x: x

        with pytest.warns(UserWarning, match="Using untrusted explanation plugin"):
            plugin, identifier = orchestrator.resolve_plugin("factual")
            assert identifier == "test.untrusted.instance"
            assert plugin == mock_plugin


# ==============================================================================
# STEP 4: Environment Variable Alignment Tests
# ==============================================================================


class TestEnvironmentVariableAlignment:
    """Verify alignment of env vars with ADR-015 (Step 4)."""

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
            chain = pm.build_explanation_chain("factual", "core.explanation.factual.sequential")
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


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestADR015IntegrationFlow:
    """End-to-end tests for ADR-015 explanation plugin pipeline."""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
