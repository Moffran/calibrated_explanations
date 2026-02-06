import pytest
import numpy as np
import warnings
from unittest.mock import MagicMock
from calibrated_explanations.explanations._conjunctions import ConjunctionState
from calibrated_explanations.explanations.explanation import FactualExplanation

def test_conjunction_state_normalization():
    # Test normalization of various feature types
    initial_rules = {
        "rule": ["A", "B"],
        "feature": [0, [1]],
        "weight": [0.1, None],
        "sampled_values": [None, None],
    }
    state = ConjunctionState(initial_rules)
    
    # Check normalization
    assert state.state["feature"] == [0, [1]]
    assert state.state["weight"] == [0.1, pytest.approx(np.nan, nan_ok=True)]
    assert state.state["is_conjunctive"] == [False, False]

def test_robust_ranking_nan():
    # We can test rank_features by creating a minimal FactualExplanation
    f = FactualExplanation.__new__(FactualExplanation)
    f.mode = "classification"
    
    weights = np.array([0.1, np.nan, 0.3])
    ranked = FactualExplanation.rank_features(f, feature_weights=weights, num_to_show=3)
    assert list(ranked) == [1, 0, 2]

def test_add_conjunctions_diagnostic_warning():
    f = FactualExplanation.__new__(FactualExplanation)
    f.has_rules = True
    f.rules = {
        "rule": ["feat1", "feat2"],
        "feature": [0, 1],
        "weight": [0.1, 0.2],
        "weight_low": [0.0, 0.1],
        "weight_high": [0.2, 0.3],
        "sampled_values": [1, 2],
        "value": ["1", "2"],
        "classes": 1,
    }
    f.has_conjunctive_rules = False
    f.conjunctive_rules = []
    f.y_threshold = None
    f.x_test = np.array([1, 2])
    f.prediction = {"predict": 0.5}
    f.bin = None
    
    # Mock methods
    f.rank_features = MagicMock(return_value=[0, 1])
    f.predict_conjunctive = MagicMock(side_effect=Exception("Predict failed"))
    
    with pytest.warns(UserWarning, match="add_conjunctions: created=0"):
        FactualExplanation.add_conjunctions(f, n_top_features=2, max_rule_size=2)

def test_fallback_to_legacy(monkeypatch):
    f = FactualExplanation.__new__(FactualExplanation)
    f.has_rules = True
    f.rules = {
        "rule": ["feat1", "feat2"],
        "feature": [0, 1],
        "weight": [0.1, 0.2],
        "weight_low": [0.0, 0.1],
        "weight_high": [0.2, 0.3],
        "sampled_values": [1, 2],
        "value": ["1", "2"],
        "classes": 1,
    }
    f.has_conjunctive_rules = False
    f.conjunctive_rules = []
    f.y_threshold = None
    f.x_test = np.array([1, 2])
    f.prediction = {"predict": 0.5}
    f.bin = None
    
    # Mock methods
    f.rank_features = MagicMock(return_value=[0, 1])
    f.predict_conjunctive = MagicMock(side_effect=Exception("Predict failed"))
    
    # Mock legacy function
    mock_legacy = MagicMock()
    monkeypatch.setattr("calibrated_explanations.explanations.legacy_conjunctions.add_conjunctions_factual_legacy", mock_legacy)
    
    # Call with fallback
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        FactualExplanation.add_conjunctions(f, n_top_features=2, max_rule_size=2, _fallback_to_legacy_on_zero=True)
    
    mock_legacy.assert_called_once()
