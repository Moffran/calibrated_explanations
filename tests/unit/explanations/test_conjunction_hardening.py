import pytest
import numpy as np
import warnings
from unittest.mock import MagicMock
from calibrated_explanations.explanations._conjunctions import ConjunctionState
from calibrated_explanations.explanations.explanation import FactualExplanation, AlternativeExplanation

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


# --- Phase 4A: Conjunction output validation tests ---


def test_raise_on_predict_error_surfaces_exceptions():
    """Calling with raise_on_predict_error=True should propagate prediction errors."""
    f = FactualExplanation.__new__(FactualExplanation)
    f.has_rules = True
    f.rules = {
        "rule": ["feat1", "feat2"],
        "feature": [0, 1],
        "weight": [0.1, 0.2],
        "weight_low": [0.0, 0.1],
        "weight_high": [0.2, 0.3],
        "sampled_values": [np.array([1.0]), np.array([2.0])],
        "value": ["1", "2"],
        "classes": 1,
    }
    f.has_conjunctive_rules = False
    f.conjunctive_rules = []
    f.y_threshold = None
    f.x_test = np.array([1, 2])
    f.prediction = {"predict": 0.5}
    f.bin = None

    f.rank_features = MagicMock(return_value=[0, 1])
    f.predict_conjunctive = MagicMock(side_effect=ValueError("broken predictor"))

    with pytest.raises(ValueError, match="broken predictor"):
        FactualExplanation.add_conjunctions(
            f, n_top_features=2, max_rule_size=2, raise_on_predict_error=True
        )


def test_conjunction_stats_accuracy():
    """Verify attempts == created + sum(skipped) when conjunction_stats is available."""
    f = FactualExplanation.__new__(FactualExplanation)
    f.has_rules = True
    f.rules = {
        "rule": ["feat1", "feat2", "feat3"],
        "feature": [0, 1, 2],
        "weight": [0.1, 0.2, 0.3],
        "weight_low": [0.0, 0.1, 0.2],
        "weight_high": [0.2, 0.3, 0.4],
        "sampled_values": [np.array([1.0]), np.array([2.0]), np.array([3.0])],
        "value": ["1", "2", "3"],
        "predict": [0.6, 0.7, 0.8],
        "predict_low": [0.5, 0.6, 0.7],
        "predict_high": [0.7, 0.8, 0.9],
        "base_predict": [0.5],
        "base_predict_low": [0.4],
        "base_predict_high": [0.6],
        "classes": 1,
    }
    f.has_conjunctive_rules = False
    f.conjunctive_rules = []
    f.y_threshold = None
    f.x_test = np.array([1, 2, 3])
    f.prediction = {"predict": 0.5}
    f.bin = None

    # Alternate: first call succeeds, second raises
    call_count = [0]

    def mock_predict(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] % 2 == 0:
            raise ValueError("intermittent failure")
        return 0.6, 0.5, 0.7

    f.rank_features = MagicMock(return_value=[0, 1, 2])
    f.predict_conjunctive = MagicMock(side_effect=mock_predict)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        FactualExplanation.add_conjunctions(f, n_top_features=3, max_rule_size=2)

    stats = f.conjunction_stats
    total_skipped = sum(stats["skipped"].values())
    assert stats["attempts"] == stats["created"] + total_skipped


def test_conjunction_diagnostic_includes_predict_errors():
    """When all predictions fail, the warning should include predict_errors."""
    f = FactualExplanation.__new__(FactualExplanation)
    f.has_rules = True
    f.rules = {
        "rule": ["feat1", "feat2"],
        "feature": [0, 1],
        "weight": [0.1, 0.2],
        "weight_low": [0.0, 0.1],
        "weight_high": [0.2, 0.3],
        "sampled_values": [np.array([1.0]), np.array([2.0])],
        "value": ["1", "2"],
        "classes": 1,
    }
    f.has_conjunctive_rules = False
    f.conjunctive_rules = []
    f.y_threshold = None
    f.x_test = np.array([1, 2])
    f.prediction = {"predict": 0.5}
    f.bin = None

    f.rank_features = MagicMock(return_value=[0, 1])
    f.predict_conjunctive = MagicMock(side_effect=RuntimeError("shape mismatch"))

    with pytest.warns(UserWarning, match="predict_errors"):
        FactualExplanation.add_conjunctions(f, n_top_features=2, max_rule_size=2)


# --- Phase 4C: Edge case tests ---


def test_max_rule_size_1_returns_self():
    """max_rule_size=1 should return self without creating conjunctions."""
    f = FactualExplanation.__new__(FactualExplanation)
    f.has_rules = True
    f.rules = {
        "rule": ["feat1", "feat2"],
        "feature": [0, 1],
        "weight": [0.1, 0.2],
        "weight_low": [0.0, 0.1],
        "weight_high": [0.2, 0.3],
        "sampled_values": [np.array([1.0]), np.array([2.0])],
        "value": ["1", "2"],
        "classes": 1,
    }
    f.has_conjunctive_rules = False
    f.conjunctive_rules = []
    f.y_threshold = None
    f.x_test = np.array([1, 2])
    f.prediction = {"predict": 0.5}
    f.bin = None
    f.rank_features = MagicMock(return_value=[0, 1])
    f.predict_conjunctive = MagicMock()

    result = FactualExplanation.add_conjunctions(f, n_top_features=2, max_rule_size=1)

    # predict_conjunctive should never be called with max_rule_size=1
    f.predict_conjunctive.assert_not_called()
    assert result is f


def test_single_feature_explanation_no_crash():
    """An explanation with only one feature should not crash."""
    f = FactualExplanation.__new__(FactualExplanation)
    f.has_rules = True
    f.rules = {
        "rule": ["feat1"],
        "feature": [0],
        "weight": [0.1],
        "weight_low": [0.0],
        "weight_high": [0.2],
        "sampled_values": [np.array([1.0])],
        "value": ["1"],
        "classes": 1,
    }
    f.has_conjunctive_rules = False
    f.conjunctive_rules = []
    f.y_threshold = None
    f.x_test = np.array([1])
    f.prediction = {"predict": 0.5}
    f.bin = None
    f.rank_features = MagicMock(return_value=[0])
    f.predict_conjunctive = MagicMock()

    # Should not raise
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = FactualExplanation.add_conjunctions(f, n_top_features=1, max_rule_size=2)
    assert result is f


def test_alternative_raise_on_predict_error():
    """AlternativeExplanation should also support raise_on_predict_error."""
    f = AlternativeExplanation.__new__(AlternativeExplanation)
    f.has_rules = True
    f.rules = {
        "rule": ["feat1", "feat2"],
        "feature": [0, 1],
        "weight": [0.1, 0.2],
        "weight_low": [0.0, 0.1],
        "weight_high": [0.2, 0.3],
        "sampled_values": [np.array([1.0]), np.array([2.0])],
        "value": ["1", "2"],
        "feature_value": ["fv1", "fv2"],
        "predict": [0.6, 0.7],
        "predict_low": [0.5, 0.6],
        "predict_high": [0.7, 0.8],
        "is_conjunctive": [False, False],
        "classes": 1,
    }
    f.has_conjunctive_rules = False
    f.conjunctive_rules = []
    f.y_threshold = None
    f.x_test = np.array([1, 2])
    f.prediction = {"predict": 0.5}
    f.bin = None

    f.rank_features = MagicMock(return_value=[0, 1])
    f.predict_conjunctive = MagicMock(side_effect=ValueError("alt broken"))

    with pytest.raises(ValueError, match="alt broken"):
        AlternativeExplanation.add_conjunctions(
            f, n_top_features=2, max_rule_size=2, raise_on_predict_error=True
        )
