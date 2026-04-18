"""Internal consistency tests for guarded CE-compatible compatibility shims."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer


def test_should_keep_factual_weights_internally_consistent_with_feature_backgrounds():
    """Guarded factual weights should stay consistent with guarded background caches."""
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

        # Guarded internal consistency invariant:
        # weight = emitted rule prediction - guarded background prediction.
        assert np.isclose(w, p - feature_background, atol=1e-5)


def test_should_preserve_ce_compatible_baseline_payload_shape():
    """Guarded factual baseline payload should preserve CE-compatible schema shape."""
    x_train = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    y_train = np.array([0, 1, 1, 0])

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(x_train, y_train)

    explainer = CalibratedExplainer(model, x_train, y_train, mode="classification")
    x_test = np.array([[0, 0]])

    factual = explainer.explain_factual(x_test).explanations[0].get_rules()
    guarded = (
        explainer.explain_guarded_factual(x_test, significance=1.0).explanations[0].get_rules()
    )

    assert len(factual["base_predict"]) == 1
    assert len(guarded["base_predict"]) == 1
    assert len(factual["base_predict_low"]) == 1
    assert len(guarded["base_predict_low"]) == 1
    assert len(factual["base_predict_high"]) == 1
    assert len(guarded["base_predict_high"]) == 1


def test_should_format_categorical_conditions_with_single_equals_for_helper_compatibility():
    """Categorical rules should keep CE-compatible condition formatting."""
    # Arrange
    x_train = np.array([[0, 0], [1, 1]])
    y_train = np.array([0, 1])

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(x_train, y_train)

    # Force categorical feature
    categorical_features = [0, 1]
    explainer = CalibratedExplainer(
        model, x_train, y_train, categorical_features=categorical_features, mode="classification"
    )

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


def test_should_populate_internal_feature_caches_for_conjunction_compatibility():
    """Guarded explanations should populate the caches used by conjunction helpers."""
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

    # CE-compatible containers should also expose the same helper caches.
    assert hasattr(result, "feature_weights")
    assert len(result.feature_weights["predict"]) == 1  # 1 instance
    assert len(result.feature_weights["predict"][0]) == x_train.shape[1]


def test_should_include_prob_key_in_classification_prediction_metadata():
    """Classification predictions should retain the `prob` key expected by CE helpers."""
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

    # CE-compatible helper surfaces expect a `prob` key for classification.
    assert "prob" in expl.prediction
    assert 0 <= expl.prediction["prob"] <= 1
