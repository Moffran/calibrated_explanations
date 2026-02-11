import types

import pytest

from calibrated_explanations.explanations.explanation import (
    FactualExplanation,
    AlternativeExplanation,
)


def make_factual_instance():
    inst = object.__new__(FactualExplanation)
    # minimal explainer providing feature names
    inst.get_explainer = lambda: types.SimpleNamespace(feature_names=["age", "income"])
    # synthetic rules payload (two rules)
    rules = {
        "rule": ["age > 50", "income < 30k"],
        "feature": [0, 1],
        "feature_value": [55, 25000],
        "value": ["55", "25000"],
        "weight": [0.42, -0.23],
        "weight_low": [0.30, -0.40],
        "weight_high": [0.55, -0.05],
        "predict": [0.8, 0.6],
        "predict_low": [0.7, 0.5],
        "predict_high": [0.9, 0.7],
        "base_predict": [0.4, 0.8],
        "base_predict_low": [0.3, 0.7],
        "base_predict_high": [0.5, 0.9],
    }
    inst.get_rules = lambda: rules
    return inst


def make_alternative_instance():
    inst = object.__new__(AlternativeExplanation)
    inst.get_explainer = lambda: types.SimpleNamespace(feature_names=["age", "income"])
    rules = {
        "rule": ["age > 50", "income < 30k"],
        "feature": [0, 1],
        "sampled_values": [55, 25000],
        "value": ["60", "20000"],
        "weight": [0.10, -0.15],
        "weight_low": [0.05, -0.25],
        "weight_high": [0.15, -0.05],
        "predict": [0.85, 0.35],
        "predict_low": [0.75, 0.25],
        "predict_high": [0.95, 0.45],
        "base_predict": [0.5],
        "base_predict_low": [0.4],
        "base_predict_high": [0.6],
    }
    inst.get_rules = lambda: rules
    return inst


def test_factual_get_rule_by_index_and_list():
    inst = make_factual_instance()
    r0 = inst.get_rule_by_index(0)
    assert r0["feature"] == "age"
    assert "condition" in r0 and r0["condition"] == "age > 50"
    assert isinstance(r0["uncertainty_interval"], (tuple, dict))

    all_rules = inst.list_rules()
    assert len(all_rules) == 2


def test_factual_get_rules_by_feature_and_errors():
    inst = make_factual_instance()
    matches = inst.get_rules_by_feature("income")
    assert len(matches) == 1
    assert matches[0]["feature"] == "income"

    with pytest.raises(KeyError):
        inst.get_rules_by_feature("nonexistent")


def test_alternative_get_rule_by_index_and_feature():
    inst = make_alternative_instance()
    r1 = inst.get_rule_by_index(1)
    assert r1["feature"] == "income"
    assert "alternative_prediction" in r1

    matches = inst.get_rules_by_feature("age")
    assert len(matches) == 1

    with pytest.raises(IndexError):
        inst.get_rule_by_index(10)
