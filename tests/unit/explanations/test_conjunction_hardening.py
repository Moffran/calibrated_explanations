import pytest
import numpy as np
import warnings
from unittest.mock import MagicMock
from sklearn.ensemble import RandomForestClassifier
from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.explanations._conjunctions import ConjunctionState
from calibrated_explanations.explanations.explanation import FactualExplanation


def _make_binary_explainer(binary_dataset):
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _,
        _,
        _,
        categorical_features,
        feature_names,
    ) = binary_dataset

    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(x_prop_train, y_prop_train)

    explainer = WrapCalibratedExplainer(model)
    explainer.calibrate(
        x_cal, y_cal, feature_names=feature_names, categorical_features=categorical_features
    )
    return explainer, x_test

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


def test_conjunctions_actually_created(binary_dataset):
    explainer, x_test = _make_binary_explainer(binary_dataset)
    explanation = explainer.explain_factual(x_test[:1])[0]

    explanation.add_conjunctions(n_top_features=5, max_rule_size=2)

    stats = explanation.conjunction_stats
    assert stats["created"] > 0


def test_conjunction_stats_accuracy(binary_dataset):
    explainer, x_test = _make_binary_explainer(binary_dataset)
    explanation = explainer.explain_factual(x_test[:1])[0]

    explanation.add_conjunctions(n_top_features=5, max_rule_size=2)

    stats = explanation.conjunction_stats
    assert stats["attempts"] == stats["created"] + sum(stats["skipped"].values())


def test_raise_on_predict_error_surfaces_exceptions():
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

    f.rank_features = MagicMock(return_value=[0, 1])
    f.predict_conjunctive = MagicMock(side_effect=ValueError("Predict failed"))

    with pytest.raises(ValueError, match="Predict failed"):
        FactualExplanation.add_conjunctions(
            f, n_top_features=2, max_rule_size=2, raise_on_predict_error=True
        )


def test_calibration_invariant_on_conjunctions(binary_dataset):
    explainer, x_test = _make_binary_explainer(binary_dataset)
    explanation = explainer.explain_factual(x_test[:1])[0]

    explanation.add_conjunctions(n_top_features=5, max_rule_size=2)

    stats = explanation.conjunction_stats
    assert stats["created"] > 0

    rules = explanation.conjunctive_rules
    for idx, is_conjunctive in enumerate(rules["is_conjunctive"]):
        if not is_conjunctive:
            continue
        predict = rules["predict"][idx]
        low = rules["predict_low"][idx]
        high = rules["predict_high"][idx]
        assert low <= predict <= high


def test_single_feature_explanation_no_crash():
    f = FactualExplanation.__new__(FactualExplanation)
    f.has_rules = True
    f.rules = {
        "rule": ["feat1"],
        "feature": [0],
        "weight": [0.1],
        "weight_low": [0.0],
        "weight_high": [0.2],
        "sampled_values": [1],
        "value": ["1"],
        "classes": 1,
    }
    f.has_conjunctive_rules = False
    f.conjunctive_rules = []
    f.y_threshold = None
    f.x_test = np.array([1])
    f.prediction = {"predict": 0.5}
    f.bin = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = FactualExplanation.add_conjunctions(f, n_top_features=1, max_rule_size=2)

    assert result is f


def test_all_nan_weights_fallback():
    f = FactualExplanation.__new__(FactualExplanation)
    f.mode = "classification"
    weights = np.array([np.nan, np.nan, np.nan])
    ranked = FactualExplanation.rank_features(f, feature_weights=weights, num_to_show=3)
    assert set(ranked) == {0, 1, 2}


def test_max_rule_size_1_returns_self():
    f = FactualExplanation.__new__(FactualExplanation)
    assert FactualExplanation.add_conjunctions(f, max_rule_size=1) is f


def test_conjunction_deduplication_correctness():
    f = FactualExplanation.__new__(FactualExplanation)
    f.has_rules = True
    f.rules = {
        "rule": ["feat1", "feat2", "feat3"],
        "feature": [0, 1, 2],
        "weight": [0.1, 0.2, 0.3],
        "weight_low": [0.0, 0.1, 0.2],
        "weight_high": [0.2, 0.3, 0.4],
        "sampled_values": [1, 2, 3],
        "value": ["1", "2", "3"],
        "classes": 1,
    }
    f.has_conjunctive_rules = True
    f.conjunctive_rules = {
        "rule": ["feat1", "feat2", "feat3", "feat1 & \nfeat2"],
        "feature": [0, 1, 2, [0, 1]],
        "weight": [0.1, 0.2, 0.3, 0.05],
        "weight_low": [0.0, 0.1, 0.2, 0.01],
        "weight_high": [0.2, 0.3, 0.4, 0.1],
        "sampled_values": [1, 2, 3, [1, 2]],
        "value": ["1", "2", "3", "1\n2"],
        "classes": 1,
        "is_conjunctive": [False, False, False, True],
    }
    f.y_threshold = None
    f.x_test = np.array([1, 2, 3])
    f.prediction = {"predict": 0.5}
    f.bin = None

    f.rank_features = MagicMock(return_value=[0, 1, 2])
    f.predict_conjunctive = MagicMock(return_value=(0.6, 0.5, 0.7))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        FactualExplanation.add_conjunctions(f, n_top_features=3, max_rule_size=2)

    stats = f.conjunction_stats
    assert stats["skipped"]["duplicate_combo"] > 0
    assert stats["created"] > 0
