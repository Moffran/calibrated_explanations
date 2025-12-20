# pylint: disable=protected-access, missing-function-docstring, missing-module-docstring
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from calibrated_explanations import WrapCalibratedExplainer

@pytest.fixture
def binary_dataset():
    X = np.random.rand(100, 10)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    feature_names = [f"f{i}" for i in range(10)]
    categorical_features = []
    
    # Split
    X_prop_train, y_prop_train = X[:50], y[:50]
    X_cal, y_cal = X[50:80], y[50:80]
    X_test, y_test = X[80:], y[80:]
    
    return X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, None, None, categorical_features, feature_names

def test_large_conjunctions(binary_dataset):
    X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, _, _, categorical_features, feature_names = binary_dataset
    # Use a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_prop_train, y_prop_train)
    
    explainer = WrapCalibratedExplainer(model)
    explainer.calibrate(X_cal, y_cal, feature_names=feature_names, categorical_features=categorical_features)
    
    # Get an explanation
    explanation = explainer.explain_factual(X_test[0].reshape(1, -1))
    
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
    X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, _, _, categorical_features, feature_names = binary_dataset
    # Use a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_prop_train, y_prop_train)
    
    explainer = WrapCalibratedExplainer(model)
    explainer.calibrate(X_cal, y_cal, feature_names=feature_names, categorical_features=categorical_features)
    
    # Get an explanation
    explanation = explainer.explain_counterfactual(X_test[0].reshape(1, -1))
    
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
