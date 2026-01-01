# pylint: disable=protected-access, missing-function-docstring, missing-module-docstring
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from calibrated_explanations import WrapCalibratedExplainer


@pytest.fixture
def binary_dataset():
    x_vals = np.random.default_rng().random((100, 10))
    y = (x_vals[:, 0] + x_vals[:, 1] > 1).astype(int)
    feature_names = [f"f{i}" for i in range(10)]
    categorical_features = []

    # Split
    x_prop_train, y_prop_train = x_vals[:50], y[:50]
    x_cal, y_cal = x_vals[50:80], y[50:80]
    x_test, y_test = x_vals[80:], y[80:]

    return (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        y_test,
        None,
        None,
        categorical_features,
        feature_names,
    )


def test_large_conjunctions(binary_dataset):
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        y_test,
        _,
        _,
        categorical_features,
        feature_names,
    ) = binary_dataset
    # Use a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(x_prop_train, y_prop_train)

    explainer = WrapCalibratedExplainer(model)
    explainer.calibrate(
        x_cal, y_cal, feature_names=feature_names, categorical_features=categorical_features
    )

    # Get an explanation
    explanation = explainer.explain_factual(x_test[0].reshape(1, -1))

    # Try to add conjunctions with size 4
    # This should now succeed as _use_batched=True is the default
    explanation.add_conjunctions(n_top_features=5, max_rule_size=4)

    # This should also succeed explicitly
    explanation.add_conjunctions(n_top_features=5, max_rule_size=4, _use_batched=True)

    # Check if we have any rules of size 3 or 4
    # Note: With random data and small n_top_features, we might not find any *good* rules of size 4,
    # but the code should run without error.

    # Let's check the conjunctive_rules payload
    # explanation is a CalibratedExplanations object (collection), so we need to access the first item
    rules = explanation[0].conjunctive_rules

    # Just verify that rules is not None
    assert rules is not None


def test_large_conjunctions_alternative(binary_dataset):
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        y_test,
        _,
        _,
        categorical_features,
        feature_names,
    ) = binary_dataset
    # Use a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(x_prop_train, y_prop_train)

    explainer = WrapCalibratedExplainer(model)
    explainer.calibrate(
        x_cal, y_cal, feature_names=feature_names, categorical_features=categorical_features
    )

    # Get an explanation
    explanation = explainer.explain_counterfactual(x_test[0].reshape(1, -1))

    # Try to add conjunctions with size 4
    # This should now succeed as _use_batched=True is the default
    explanation.add_conjunctions(n_top_features=5, max_rule_size=4)

    # This should also succeed explicitly
    explanation.add_conjunctions(n_top_features=5, max_rule_size=4, _use_batched=True)

    # Check if we have any rules of size 3 or 4
    # explanation is a CalibratedExplanations object (collection), so we need to access the first item
    rules = explanation[0].conjunctive_rules

    # Just verify that rules is not None
    assert rules is not None
