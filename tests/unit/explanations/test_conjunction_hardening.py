import pytest
import numpy as np
import warnings
from unittest.mock import MagicMock
from sklearn.ensemble import RandomForestClassifier
from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.explanations._conjunctions import ConjunctionState
from calibrated_explanations.explanations.explanation import (
    FactualExplanation,
    AlternativeExplanation,
)


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
    monkeypatch.setattr(
        "calibrated_explanations.explanations.legacy_conjunctions.add_conjunctions_factual_legacy",
        mock_legacy,
    )

    # Call with fallback
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        FactualExplanation.add_conjunctions(
            f, n_top_features=2, max_rule_size=2, _fallback_to_legacy_on_zero=True
        )

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
        FactualExplanation.add_conjunctions(f, n_top_features=2, max_rule_size=2, verbose=True)


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
