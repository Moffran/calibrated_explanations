import pytest
from unittest.mock import MagicMock
from calibrated_explanations.core.prediction.orchestrator import PredictionOrchestrator
from calibrated_explanations.plugins.explanations import (
    validate_explanation_batch,
    ExplanationBatch,
)
from calibrated_explanations.explanations.explanations import CalibratedExplanations
from calibrated_explanations.explanations.explanation import CalibratedExplanation


class DummyContainer(CalibratedExplanations):
    pass


class DummyExplanation(CalibratedExplanation):
    pass


class TestPredictionOrchestratorInvariants:
    def setup_method(self):
        self.explainer = MagicMock()
        # Mock attributes accessed in __init__
        self.explainer.plugin_manager.interval_plugin_identifiers = {}
        self.explainer.plugin_manager.interval_plugin_fallbacks = {}
        self.explainer.plugin_manager.interval_plugin_hints = {}
        self.explainer.plugin_manager.interval_context_metadata = {}
        self.explainer.telemetry_interval_sources = {}
        self.explainer.interval_preferred_identifier = {}

        # Mock attributes accessed in _predict
        self.explainer.perf_cache = MagicMock()
        self.explainer.perf_cache.enabled = False
        self.explainer.mode = "regression"

        self.orchestrator = PredictionOrchestrator(self.explainer)


class TestExplanationBatchInvariants:
    def test_validate_batch_invalid_low_gt_high(self):
        batch = ExplanationBatch(
            container_cls=DummyContainer,
            explanation_cls=DummyExplanation,
            instances=[{"prediction": {"predict": 0.5, "low": 0.7, "high": 0.6}}],
            collection_metadata={"task": "regression", "mode": "test"},
        )
        with pytest.warns(UserWarning, match="low > high"):
            validate_explanation_batch(batch, expected_task="regression", expected_mode="test")
