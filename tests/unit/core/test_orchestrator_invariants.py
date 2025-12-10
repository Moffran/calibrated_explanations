import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from calibrated_explanations.core.prediction.orchestrator import PredictionOrchestrator
from calibrated_explanations.core.exceptions import ValidationError
from calibrated_explanations.plugins.explanations import validate_explanation_batch, ExplanationBatch
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
        self.explainer._interval_plugin_identifiers = {}
        self.explainer._interval_plugin_fallbacks = {}
        self.explainer._interval_plugin_hints = {}
        self.explainer._interval_context_metadata = {}
        self.explainer._telemetry_interval_sources = {}
        self.explainer._interval_preferred_identifier = {}
        
        # Mock attributes accessed in _predict
        self.explainer._perf_cache = MagicMock()
        self.explainer._perf_cache.enabled = False
        self.explainer.mode = "regression"
        
        self.orchestrator = PredictionOrchestrator(self.explainer)

    def test_predict_valid_invariant(self):
        # predict, low, high, extra
        valid_result = (np.array([0.5]), np.array([0.4]), np.array([0.6]), None)
        
        with patch.object(self.orchestrator, '_predict_impl', return_value=valid_result):
            result = self.orchestrator._predict(np.array([[1]]))
            assert result == valid_result

    def test_predict_invalid_low_gt_high(self):
        # low > high
        invalid_result = (np.array([0.5]), np.array([0.7]), np.array([0.6]), None)
        
        with patch.object(self.orchestrator, '_predict_impl', return_value=invalid_result):
            with pytest.warns(RuntimeWarning, match="low > high"):
                self.orchestrator._predict(np.array([[1]]))

    def test_predict_invalid_predict_lt_low(self):
        # predict < low
        invalid_result = (np.array([0.3]), np.array([0.4]), np.array([0.6]), None)
        
        with patch.object(self.orchestrator, '_predict_impl', return_value=invalid_result):
            with pytest.warns(RuntimeWarning, match="predict not in"):
                self.orchestrator._predict(np.array([[1]]))

    def test_predict_invalid_predict_gt_high(self):
        # predict > high
        invalid_result = (np.array([0.7]), np.array([0.4]), np.array([0.6]), None)
        
        with patch.object(self.orchestrator, '_predict_impl', return_value=invalid_result):
            with pytest.warns(RuntimeWarning, match="predict not in"):
                self.orchestrator._predict(np.array([[1]]))

class TestExplanationBatchInvariants:
    def test_validate_batch_valid(self):
        batch = ExplanationBatch(
            container_cls=DummyContainer,
            explanation_cls=DummyExplanation,
            instances=[
                {"prediction": {"predict": 0.5, "low": 0.4, "high": 0.6}}
            ],
            collection_metadata={"task": "regression", "mode": "test"}
        )
        # Should not raise
        validate_explanation_batch(batch, expected_task="regression", expected_mode="test")

    def test_validate_batch_invalid_low_gt_high(self):
        batch = ExplanationBatch(
            container_cls=DummyContainer,
            explanation_cls=DummyExplanation,
            instances=[
                {"prediction": {"predict": 0.5, "low": 0.7, "high": 0.6}}
            ],
            collection_metadata={"task": "regression", "mode": "test"}
        )
        with pytest.warns(RuntimeWarning, match="low > high"):
            validate_explanation_batch(batch, expected_task="regression", expected_mode="test")

    def test_validate_batch_invalid_predict_out_of_bounds(self):
        batch = ExplanationBatch(
            container_cls=DummyContainer,
            explanation_cls=DummyExplanation,
            instances=[
                {"prediction": {"predict": 0.3, "low": 0.4, "high": 0.6}}
            ],
            collection_metadata={"task": "regression", "mode": "test"}
        )
        with pytest.warns(RuntimeWarning, match="predict not in"):
            validate_explanation_batch(batch, expected_task="regression", expected_mode="test")
