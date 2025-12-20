import pytest
import numpy as np
import numbers
from sklearn.ensemble import RandomForestClassifier
from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.explanations.legacy_conjunctions import (
    add_conjunctions_factual_legacy,
    add_conjunctions_alternative_legacy,
)

def compare_payloads(p1, p2):
    assert p1.keys() == p2.keys()
    for k in p1:
        v1 = p1[k]
        v2 = p2[k]
        if isinstance(v1, list):
            assert len(v1) == len(v2)
            for i in range(len(v1)):
                if isinstance(v1[i], (np.ndarray, list)):
                    if isinstance(v1[i], list):
                        assert len(v1[i]) == len(v2[i])
                        for j in range(len(v1[i])):
                            if isinstance(v1[i][j], np.ndarray):
                                np.testing.assert_array_equal(v1[i][j], v2[i][j])
                            elif isinstance(v1[i][j], (float, np.floating, numbers.Real)):
                                assert float(v1[i][j]) == pytest.approx(float(v2[i][j]))
                            else:
                                assert v1[i][j] == v2[i][j]
                    else:
                        np.testing.assert_array_equal(v1[i], v2[i])
                elif isinstance(v1[i], (float, np.floating, numbers.Real)):
                    assert float(v1[i]) == pytest.approx(float(v2[i]))
                else:
                    assert v1[i] == v2[i]
        elif isinstance(v1, np.ndarray):
            np.testing.assert_array_equal(v1, v2)
        elif isinstance(v1, (float, np.floating, numbers.Real)):
            assert float(v1) == pytest.approx(float(v2))
        else:
            assert v1 == v2

def test_factual_conjunction_parity(binary_dataset):
    X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, _, _, categorical_features, feature_names = binary_dataset
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_prop_train, y_prop_train)
    
    explainer = WrapCalibratedExplainer(model)
    explainer.calibrate(X_cal, y_cal, feature_names=feature_names, categorical_features=categorical_features)
    
    # Get an explanation for the first test instance
    explanation = explainer.explain_factual(X_test[0].reshape(1, -1))
    explanation_legacy = explainer.explain_factual(X_test[0].reshape(1, -1))
    
    # Run new (currently same as legacy)
    explanation.add_conjunctions(n_top_features=5, max_rule_size=2)
    
    # Run legacy
    for exp in explanation_legacy:
        add_conjunctions_factual_legacy(exp, n_top_features=5, max_rule_size=2)
    # Compare
    for e1, e2 in zip(explanation, explanation_legacy):
        compare_payloads(e1.conjunctive_rules, e2.conjunctive_rules)

def test_alternative_conjunction_parity(binary_dataset):
    X_prop_train, y_prop_train, X_cal, y_cal, X_test, y_test, _, _, categorical_features, feature_names = binary_dataset
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_prop_train, y_prop_train)
    
    explainer = WrapCalibratedExplainer(model)
    explainer.calibrate(X_cal, y_cal, feature_names=feature_names, categorical_features=categorical_features)
    
    # Get an explanation for the first test instance
    explanation = explainer.explore_alternatives(X_test[0].reshape(1, -1))
    explanation_legacy = explainer.explore_alternatives(X_test[0].reshape(1, -1))
    
    # Run new
    explanation.add_conjunctions(n_top_features=5, max_rule_size=2)
    
    # Run legacy
    for exp in explanation_legacy:
        add_conjunctions_alternative_legacy(exp, n_top_features=5, max_rule_size=2)
    # Compare
    for e1, e2 in zip(explanation, explanation_legacy):
        compare_payloads(e1.conjunctive_rules, e2.conjunctive_rules)
