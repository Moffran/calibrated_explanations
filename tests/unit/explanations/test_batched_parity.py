import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from calibrated_explanations import WrapCalibratedExplainer


def compare_payloads(p1, p2):
    assert p1.keys() == p2.keys()
    for k in p1:
        v1 = p1[k]
        v2 = p2[k]
        if isinstance(v1, list):
            assert len(v1) == len(v2)
            for j in range(len(v1)):
                if isinstance(v1[j], (np.ndarray, list)):
                    if isinstance(v1[j], list):
                        assert len(v1[j]) == len(v2[j])
                        for idx in range(len(v1[j])):
                            if isinstance(v1[j][idx], np.ndarray):
                                np.testing.assert_allclose(v1[j][idx], v2[j][idx])
                            elif isinstance(v1[j][idx], float):
                                assert v1[j][idx] == pytest.approx(v2[j][idx])
                            else:
                                assert v1[j][idx] == v2[j][idx]
                    else:
                        np.testing.assert_allclose(v1[j], v2[j])
                elif isinstance(v1[j], float):
                    assert v1[j] == pytest.approx(v2[j])
                else:
                    assert v1[j] == v2[j]
        elif isinstance(v1, np.ndarray):
            np.testing.assert_allclose(v1, v2)
        elif isinstance(v1, float):
            assert v1 == pytest.approx(v2)
        else:
            assert v1 == v2


def test_batched_factual_parity(binary_dataset):
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

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(x_prop_train, y_prop_train)

    explainer = WrapCalibratedExplainer(model)
    explainer.calibrate(
        x_cal, y_cal, feature_names=feature_names, categorical_features=categorical_features
    )

    # Get an explanation for the first test instance
    explanation_iterative = explainer.explain_factual(x_test[0].reshape(1, -1))
    explanation_batched = explainer.explain_factual(x_test[0].reshape(1, -1))

    # Run iterative
    explanation_iterative.add_conjunctions(n_top_features=5, max_rule_size=2, _use_batched=False)

    # Run batched
    explanation_batched.add_conjunctions(n_top_features=5, max_rule_size=2, _use_batched=True)

    # Compare
    for e1, e2 in zip(explanation_iterative, explanation_batched):
        compare_payloads(e1.conjunctive_rules, e2.conjunctive_rules)


def test_batched_alternative_parity(binary_dataset):
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

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(x_prop_train, y_prop_train)

    explainer = WrapCalibratedExplainer(model)
    explainer.calibrate(
        x_cal, y_cal, feature_names=feature_names, categorical_features=categorical_features
    )

    # Get an explanation for the first test instance
    explanation_iterative = explainer.explore_alternatives(x_test[0].reshape(1, -1))
    explanation_batched = explainer.explore_alternatives(x_test[0].reshape(1, -1))

    # Run iterative
    explanation_iterative.add_conjunctions(n_top_features=5, max_rule_size=2, _use_batched=False)

    # Run batched
    explanation_batched.add_conjunctions(n_top_features=5, max_rule_size=2, _use_batched=True)

    # Compare
    for e1, e2 in zip(explanation_iterative, explanation_batched):
        compare_payloads(e1.conjunctive_rules, e2.conjunctive_rules)
