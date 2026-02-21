"""Unit tests verifying strict alignment of guarded explanations with standard CE conventions."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.explanations.guarded_explanation import (
    GuardedAlternativeExplanation,
    GuardedFactualExplanation,
)


def test_should_align_factual_weight_with_ce_impact_semantics():
    """Guarded factual should use the same CE factual weight definition as standard factual."""
    # Arrange
    x_train = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    y_train = np.array([0, 2, 1, 1])
    
    model = RandomForestRegressor(n_estimators=5, random_state=42)
    model.fit(x_train, y_train)
    
    explainer = CalibratedExplainer(model, x_train, y_train, mode="regression")
    
    # Act
    x_test = np.array([[0.5, 0.5]])
    result = explainer.explain_guarded_factual(x_test, significance=0.01)
    
    expl = result.explanations[0]
    rules = expl.get_rules()
    
    # Assert
    for i in range(len(rules["rule"])):
        w = rules["weight"][i]
        f = rules["feature"][i]
        p = rules["predict"][i]
        feature_background = expl.feature_predict["predict"][f]

        # Standard factual CE invariant:
        # weight = prediction - per-feature background prediction.
        assert np.isclose(w, p - feature_background, atol=1e-5)


def test_should_match_standard_factual_baseline_payload_shape():
    """Guarded factual baseline payload should match standard factual schema."""
    x_train = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    y_train = np.array([0, 1, 1, 0])

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(x_train, y_train)

    explainer = CalibratedExplainer(model, x_train, y_train, mode="classification")
    x_test = np.array([[0, 0]])

    factual = explainer.explain_factual(x_test).explanations[0].get_rules()
    guarded = explainer.explain_guarded_factual(x_test, significance=1.0).explanations[0].get_rules()

    assert len(factual["base_predict"]) == 1
    assert len(guarded["base_predict"]) == 1
    assert len(factual["base_predict_low"]) == 1
    assert len(guarded["base_predict_low"]) == 1
    assert len(factual["base_predict_high"]) == 1
    assert len(guarded["base_predict_high"]) == 1


def test_should_format_categorical_conditions_with_single_equals():
    """Categorical rules should use 'Feature = Value' not 'Feature == Value'."""
    # Arrange
    x_train = np.array([[0, 0], [1, 1]])
    y_train = np.array([0, 1])
    
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(x_train, y_train)
    
    # Force categorical feature
    categorical_features = [0, 1]
    explainer = CalibratedExplainer(model, x_train, y_train, categorical_features=categorical_features, mode="classification")
    
    # Act
    x_test = np.array([[0, 0]])
    result = explainer.explain_guarded_factual(x_test, significance=0.01)
    
    expl = result.explanations[0]
    rules = expl.get_rules()
    
    # Assert
    for i in range(len(rules["rule"])):
        rule_str = rules["rule"][i]
        # In classification binary/multi, categorical rule strings should use '='
        assert " == " not in rule_str, f"Found double equals in rule string: {rule_str}"
        assert " = " in rule_str, f"Missing single equals in rule string: {rule_str}"


def test_should_populate_internal_feature_weights_for_conjunction_compatibility():
    """Guarded explanations should populate internal caches used by plugins like conjunctions."""
    # Arrange
    x_train = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    y_train = np.array([0, 1, 1, 0])
    
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(x_train, y_train)
    
    explainer = CalibratedExplainer(model, x_train, y_train, mode="classification")
    
    # Act
    x_test = np.array([[0.5, 0.5]])
    result = explainer.explain_guarded_factual(x_test, significance=0.01)
    expl = result.explanations[0]
    
    # These attributes are dicts in singular explanations holding (predict, low, high)
    assert hasattr(expl, "feature_weights")
    assert hasattr(expl, "feature_predict")
    
    # Singular expl.feature_weights is a dict with 3 keys: predict, low, high
    # Each value is an array of size n_features
    assert len(expl.feature_weights["predict"]) == x_train.shape[1]
    assert len(expl.feature_predict["predict"]) == x_train.shape[1]
    
    # Standard CE convention: plural container should also have them
    assert hasattr(result, "feature_weights")
    assert len(result.feature_weights["predict"]) == 1 # 1 instance
    assert len(result.feature_weights["predict"][0]) == x_train.shape[1]

def test_should_include_prob_in_prediction_data_for_classification():
    """Classification predictions should include 'prob' key in prediction data."""
    # Arrange
    x_train = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    y_train = np.array([0, 1, 1, 0])
    
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(x_train, y_train)
    
    explainer = CalibratedExplainer(model, x_train, y_train, mode="classification")
    
    # Act
    x_test = np.array([[0, 0]])
    result = explainer.explain_guarded_factual(x_test, significance=1.0)
    expl = result.explanations[0]
    
    # Standard CE convention: Prediction metadata for classification MUST have 'prob'
    assert "prob" in expl.prediction
    assert 0 <= expl.prediction["prob"] <= 1
