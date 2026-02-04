"""
Unit tests for RejectOrchestrator resilience and fallbacks.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock
from calibrated_explanations.core.reject.orchestrator import RejectOrchestrator

class FakeIntervalLearner:
    def predict_proba(self, x, bins=None):
        n = len(x)
        return np.ones((n, 2)) * 0.5 

class FakeExplainer:
    def __init__(self):
        self.interval_learner = FakeIntervalLearner()
        self.reject_learner = MagicMock()
        self.mode = "classification"
        self.is_multiclass = lambda: False
        self.bins = None
        self.seed = 42

@pytest.fixture
def orchestrator():
    explainer = FakeExplainer()
    orch = RejectOrchestrator(explainer)
    return orch

def test_fallback_from_predict_p_to_predict_set(orchestrator):
    # Setup
    x = np.zeros((5, 2))
    orchestrator.explainer.reject_learner.predict_p.side_effect = Exception("Simulated Failure")
    
    # predict_set should be called as fallback
    orchestrator.explainer.reject_learner.predict_set.return_value = np.zeros((5, 2), dtype=bool)
    
    with pytest.warns(UserWarning, match="Reject prediction fallback engaged: predict_p failed"):
        breakdown = orchestrator.predict_reject_breakdown(x, confidence=0.9)

    assert orchestrator.explainer.reject_learner.predict_p.called
    assert orchestrator.explainer.reject_learner.predict_set.called
    assert breakdown["prediction_set"].shape == (5, 2)

def test_fallback_from_bulk_predict_set_to_per_instance(orchestrator):
    # Setup
    x = np.zeros((3, 2))
    # Remove predict_p so it goes straight to predict_set
    del orchestrator.explainer.reject_learner.predict_p
    
    # Bulk predict_set raises Exception
    orchestrator.explainer.reject_learner.predict_set.side_effect = [
        Exception("Bulk Failure"), # First call (bulk)
        np.array([[False, True]]), # Instance 0
        np.array([[True, False]]), # Instance 1
        np.array([[False, False]]) # Instance 2
    ]
    
    with pytest.warns(UserWarning, match="Reject prediction fallback engaged: bulk predict_set failed"):
        breakdown = orchestrator.predict_reject_breakdown(x, confidence=0.9)
    
    # Should be called 4 times: 1 bulk (fail) + 3 individual
    assert orchestrator.explainer.reject_learner.predict_set.call_count == 4
    prediction_set = breakdown["prediction_set"]
    assert prediction_set.shape == (3, 2)
    assert np.all(prediction_set[0] == [False, True])

def test_fallback_from_bad_shape_to_per_instance(orchestrator):
    # Setup
    x = np.zeros((3, 2))
    del orchestrator.explainer.reject_learner.predict_p
    
    # Bulk predict_set returns wrong shape
    # Expected (3, 2), returning (1, 2) or something weird
    # Note: alphas_test len is 3. 
    wrong_shape_result = np.zeros((1, 2), dtype=bool)
    
    # Side effects:
    # 1. Bulk call -> returns wrong shape (no exception)
    # 2. Instance 0
    # 3. Instance 1
    # 4. Instance 2
    orchestrator.explainer.reject_learner.predict_set.side_effect = [
        wrong_shape_result,
        np.array([[True, True]]),
        np.array([[True, True]]),
        np.array([[True, True]])
    ]
    
    with pytest.warns(UserWarning, match="Reject prediction fallback engaged: predict_set returned unexpected shape"):
        breakdown = orchestrator.predict_reject_breakdown(x, confidence=0.9)
        
    assert orchestrator.explainer.reject_learner.predict_set.call_count == 4
    prediction_set = breakdown["prediction_set"]
    assert prediction_set.shape == (3, 2)

