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


def build_explainer(binary_dataset):
    """Build and return (explainer, x_test) from dataset."""
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
    return explainer, x_test


def test_factual_conjunction_parity(binary_dataset):
    explainer, x_test = build_explainer(binary_dataset)

    # Get an explanation for the first test instance
    explanation = explainer.explain_factual(x_test[0].reshape(1, -1))
    explanation_legacy = explainer.explain_factual(x_test[0].reshape(1, -1))

    # Run new (currently same as legacy)
    explanation.add_conjunctions(n_top_features=5, max_rule_size=2)

    # Run legacy
    for exp in explanation_legacy:
        add_conjunctions_factual_legacy(exp, n_top_features=5, max_rule_size=2)
    # Compare
    for e1, e2 in zip(explanation, explanation_legacy):
        assert len(e1.conjunctive_rules["rule"]) > 0
        compare_payloads(e1.conjunctive_rules, e2.conjunctive_rules)


def test_alternative_conjunction_parity(binary_dataset):
    explainer, x_test = build_explainer(binary_dataset)

    # Get an explanation for the first test instance
    explanation = explainer.explore_alternatives(x_test[0].reshape(1, -1))
    explanation_legacy = explainer.explore_alternatives(x_test[0].reshape(1, -1))

    # Run new
    explanation.add_conjunctions(n_top_features=5, max_rule_size=2)

    # Run legacy
    for exp in explanation_legacy:
        add_conjunctions_alternative_legacy(exp, n_top_features=5, max_rule_size=2)
    # Compare
    for e1, e2 in zip(explanation, explanation_legacy):
        assert len(e1.conjunctive_rules["rule"]) > 0
        compare_payloads(e1.conjunctive_rules, e2.conjunctive_rules)


def test_alternative_conjunction_parity_max_rule_size_3(binary_dataset):
    explainer, x_test = build_explainer(binary_dataset)

    explanation = explainer.explore_alternatives(x_test[0].reshape(1, -1))
    explanation_legacy = explainer.explore_alternatives(x_test[0].reshape(1, -1))

    explanation.add_conjunctions(n_top_features=5, max_rule_size=3)

    for exp in explanation_legacy:
        add_conjunctions_alternative_legacy(exp, n_top_features=5, max_rule_size=3)

    for e1, e2 in zip(explanation, explanation_legacy):
        assert len(e1.conjunctive_rules["rule"]) > 0
        compare_payloads(e1.conjunctive_rules, e2.conjunctive_rules)


# --- Phase 4D: Strengthened parity tests ---


def test_factual_conjunctions_actually_created(binary_dataset):
    """Verify that add_conjunctions actually produces conjunctive rules."""
    explainer, x_test = build_explainer(binary_dataset)

    explanation = explainer.explain_factual(x_test[0].reshape(1, -1))
    explanation.add_conjunctions(n_top_features=5, max_rule_size=2)

    for exp in explanation:
        assert exp.has_conjunctive_rules, "has_conjunctive_rules should be True"
        assert exp.conjunctive_rules is not None, "conjunctive_rules should not be None"
        # Public API: once conjunctions are added, get_rules() should expose them.
        assert exp.get_rules() is exp.conjunctive_rules
        rules = exp.conjunctive_rules
        assert len(rules["rule"]) > 0, (
            f"Expected conjunctive rules to be created but got 0. "
            f"Stats: {getattr(exp, 'conjunction_stats', 'N/A')}"
        )


def test_alternative_conjunctions_actually_created(binary_dataset):
    """Verify that add_conjunctions actually produces conjunctive rules for alternatives."""
    explainer, x_test = build_explainer(binary_dataset)

    explanation = explainer.explore_alternatives(x_test[0].reshape(1, -1))
    explanation.add_conjunctions(n_top_features=5, max_rule_size=2)

    for exp in explanation:
        assert exp.has_conjunctive_rules, "has_conjunctive_rules should be True"
        assert exp.conjunctive_rules is not None, "conjunctive_rules should not be None"
        # Public API: once conjunctions are added, get_rules() should expose them.
        assert exp.get_rules() is exp.conjunctive_rules
        rules = exp.conjunctive_rules
        assert len(rules["rule"]) > 0, (
            f"Expected conjunctive rules to be created but got 0. "
            f"Stats: {getattr(exp, 'conjunction_stats', 'N/A')}"
        )


def test_factual_conjunction_parity_max_rule_size_3(binary_dataset):
    """Test parity with max_rule_size=3."""
    explainer, x_test = build_explainer(binary_dataset)

    explanation = explainer.explain_factual(x_test[0].reshape(1, -1))
    explanation_legacy = explainer.explain_factual(x_test[0].reshape(1, -1))

    explanation.add_conjunctions(n_top_features=5, max_rule_size=3)

    for exp in explanation_legacy:
        add_conjunctions_factual_legacy(exp, n_top_features=5, max_rule_size=3)

    for e1, e2 in zip(explanation, explanation_legacy):
        compare_payloads(e1.conjunctive_rules, e2.conjunctive_rules)


@pytest.mark.parametrize("instance_idx", [0, 1])
def test_factual_parity_multiple_instances(binary_dataset, instance_idx):
    """Test factual parity across multiple test instances."""
    explainer, x_test = build_explainer(binary_dataset)

    if instance_idx >= len(x_test):
        pytest.skip("Not enough test instances")

    explanation = explainer.explain_factual(x_test[instance_idx].reshape(1, -1))
    explanation_legacy = explainer.explain_factual(x_test[instance_idx].reshape(1, -1))

    explanation.add_conjunctions(n_top_features=5, max_rule_size=2)

    for exp in explanation_legacy:
        add_conjunctions_factual_legacy(exp, n_top_features=5, max_rule_size=2)

    for e1, e2 in zip(explanation, explanation_legacy):
        compare_payloads(e1.conjunctive_rules, e2.conjunctive_rules)


@pytest.mark.parametrize("instance_idx", [0, 1])
def test_alternative_parity_multiple_instances(binary_dataset, instance_idx):
    """Test alternative parity across multiple test instances."""
    explainer, x_test = build_explainer(binary_dataset)

    if instance_idx >= len(x_test):
        pytest.skip("Not enough test instances")

    explanation = explainer.explore_alternatives(x_test[instance_idx].reshape(1, -1))
    explanation_legacy = explainer.explore_alternatives(x_test[instance_idx].reshape(1, -1))

    explanation.add_conjunctions(n_top_features=5, max_rule_size=2)

    for exp in explanation_legacy:
        add_conjunctions_alternative_legacy(exp, n_top_features=5, max_rule_size=2)

    for e1, e2 in zip(explanation, explanation_legacy):
        compare_payloads(e1.conjunctive_rules, e2.conjunctive_rules)


def test_calibration_invariant_on_conjunctions(binary_dataset):
    """For every conjunctive rule, assert predict_low <= predict <= predict_high."""
    explainer, x_test = build_explainer(binary_dataset)

    explanation = explainer.explain_factual(x_test[0].reshape(1, -1))
    explanation.add_conjunctions(n_top_features=5, max_rule_size=2)

    for exp in explanation:
        rules = exp.conjunctive_rules
        if rules is None or len(rules["rule"]) == 0:
            continue
        for i in range(len(rules["rule"])):
            p = rules["predict"][i]
            lo = rules["predict_low"][i]
            hi = rules["predict_high"][i]
            # Skip non-finite values (some interval calibrators may produce -inf/inf)
            if np.isfinite(lo) and np.isfinite(hi) and np.isfinite(p):
                assert lo <= p + 1e-10, f"Rule {i}: predict_low ({lo}) > predict ({p})"
                assert p <= hi + 1e-10, f"Rule {i}: predict ({p}) > predict_high ({hi})"
