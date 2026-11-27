"""Regression tests for rule-table serialization in JSON exports (issue #262)."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from calibrated_explanations import CalibratedExplainer
from calibrated_explanations.explanations import CalibratedExplanations


def _as_float(value: Any) -> float:
    """Convert serialized numeric or stringified infinity values to float."""
    if isinstance(value, str) and value in ("-inf", "inf", "+inf"):
        return float(value)
    return float(value)


def _assert_close(actual: Any, expected: Any, label: str) -> None:
    """Assert numeric equality while handling infinite bounds cleanly."""
    act = _as_float(actual)
    exp = _as_float(expected)
    if math.isinf(exp):
        assert math.isinf(act) and (act > 0) == (exp > 0), f"{label}: expected {exp}, got {act}"
    else:
        assert math.isclose(act, exp, rel_tol=1e-9, abs_tol=1e-9), (
            f"{label}: expected {exp}, got {act}"
        )


def _assert_payload_matches_live_rules(
    payload_exp: Mapping[str, Any],
    live_exp: Any,
) -> None:
    """Assert serialized rule_weight and rule_prediction match the live rule table."""
    live_rules = live_exp.get_rules()
    serialized_rules = payload_exp["rules"]
    assert len(serialized_rules) == len(live_rules["rule"])
    assert len(serialized_rules) > 0

    for idx, rule_entry in enumerate(serialized_rules):
        assert rule_entry["rule"] == live_rules["rule"][idx]
        rw = rule_entry["rule_weight"]
        rp = rule_entry["rule_prediction"]
        _assert_close(rw["predict"], live_rules["weight"][idx], f"rule[{idx}].weight")
        _assert_close(rw["low"], live_rules["weight_low"][idx], f"rule[{idx}].weight_low")
        _assert_close(rw["high"], live_rules["weight_high"][idx], f"rule[{idx}].weight_high")
        _assert_close(rp["predict"], live_rules["predict"][idx], f"rule[{idx}].predict")
        _assert_close(rp["low"], live_rules["predict_low"][idx], f"rule[{idx}].predict_low")
        _assert_close(rp["high"], live_rules["predict_high"][idx], f"rule[{idx}].predict_high")


def _assert_domain_matches_live_rules(domain_exp: Any, live_exp: Any) -> None:
    """Assert deserialized domain FeatureRule values match the live rule table."""
    live_rules = live_exp.get_rules()
    assert len(domain_exp.rules) == len(live_rules["rule"])
    for idx, rule_obj in enumerate(domain_exp.rules):
        assert rule_obj.rule == live_rules["rule"][idx]
        _assert_close(
            rule_obj.rule_weight["predict"], live_rules["weight"][idx], f"domain[{idx}].weight"
        )
        _assert_close(
            rule_obj.rule_weight["low"], live_rules["weight_low"][idx], f"domain[{idx}].weight_low"
        )
        _assert_close(
            rule_obj.rule_weight["high"],
            live_rules["weight_high"][idx],
            f"domain[{idx}].weight_high",
        )
        _assert_close(
            rule_obj.rule_prediction["predict"],
            live_rules["predict"][idx],
            f"domain[{idx}].predict",
        )
        _assert_close(
            rule_obj.rule_prediction["low"],
            live_rules["predict_low"][idx],
            f"domain[{idx}].predict_low",
        )
        _assert_close(
            rule_obj.rule_prediction["high"],
            live_rules["predict_high"][idx],
            f"domain[{idx}].predict_high",
        )


def _extract_stream_explanations(collection: Any) -> list[dict[str, Any]]:
    """Collect serialized explanation items from a JSONL stream."""
    chunks = list(collection.to_json_stream(format="jsonl"))
    assert len(chunks) >= 3
    return [json.loads(chunk) for chunk in chunks[1:-1]]


@pytest.fixture(scope="module")
def binary_explainer_and_test():
    """Create a deterministic binary classification CalibratedExplainer."""
    x, y = make_classification(
        n_samples=90,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )
    model = RandomForestClassifier(n_estimators=12, random_state=42)
    model.fit(x[:45], y[:45])
    explainer = CalibratedExplainer(model, x[45:85], y[45:85], mode="classification")
    return explainer, x[85:88]


@pytest.fixture(scope="module")
def multiclass_explainer_and_test():
    """Create a deterministic 3-class CalibratedExplainer."""
    x, y = make_classification(
        n_samples=120,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        random_state=7,
    )
    model = RandomForestClassifier(n_estimators=12, random_state=7)
    model.fit(x[:60], y[:60])
    explainer = CalibratedExplainer(model, x[60:115], y[60:115], mode="classification")
    return explainer, x[115:117]


@pytest.fixture(scope="module")
def regression_explainer_and_test():
    """Create a deterministic regression CalibratedExplainer."""
    x, y = make_regression(
        n_samples=90,
        n_features=5,
        n_informative=4,
        noise=0.5,
        random_state=11,
    )
    model = RandomForestRegressor(n_estimators=12, random_state=11)
    model.fit(x[:45], y[:45])
    explainer = CalibratedExplainer(model, x[45:85], y[45:85], mode="regression")
    return explainer, x[85:87]


def test_should_serialize_factual_rule_values_when_features_are_ignored(
    binary_explainer_and_test,
):
    """Factual JSON exports match get_rules() when leading features are ignored."""
    explainer, x_test = binary_explainer_and_test
    explanations = explainer.explain_factual(x_test, features_to_ignore=[0, 1])

    payload = explanations.to_json()
    roundtripped = json.loads(json.dumps(payload))
    exported = CalibratedExplanations.from_json(roundtripped)
    streamed = _extract_stream_explanations(explanations)

    assert len(payload["explanations"]) == len(explanations)
    assert len(streamed) == len(explanations)
    for idx, live_exp in enumerate(explanations):
        _assert_payload_matches_live_rules(roundtripped["explanations"][idx], live_exp)
        _assert_payload_matches_live_rules(streamed[idx], live_exp)
        _assert_domain_matches_live_rules(exported.explanations[idx], live_exp)


def test_should_serialize_fast_rule_values_from_rule_table(binary_explainer_and_test):
    """FAST JSON exports match get_rules() across to_json, stream, and round trip."""
    explainer, x_test = binary_explainer_and_test
    explanations = explainer.explain_fast(x_test)

    payload = explanations.to_json()
    roundtripped = json.loads(json.dumps(payload))
    exported = CalibratedExplanations.from_json(roundtripped)
    streamed = _extract_stream_explanations(explanations)

    assert len(roundtripped["explanations"]) == len(explanations)
    assert len(streamed) == len(explanations)
    for idx, live_exp in enumerate(explanations):
        _assert_payload_matches_live_rules(roundtripped["explanations"][idx], live_exp)
        _assert_payload_matches_live_rules(streamed[idx], live_exp)
        _assert_domain_matches_live_rules(exported.explanations[idx], live_exp)


def test_should_serialize_alternative_rule_values_from_rule_table(binary_explainer_and_test):
    """Alternative JSON exports match get_rules() for both weights and predictions."""
    explainer, x_test = binary_explainer_and_test
    explanations = explainer.explore_alternatives(x_test)

    payload = explanations.to_json()
    roundtripped = json.loads(json.dumps(payload))
    exported = CalibratedExplanations.from_json(roundtripped)
    streamed = _extract_stream_explanations(explanations)

    assert len(roundtripped["explanations"]) == len(explanations)
    assert len(streamed) == len(explanations)
    for idx, live_exp in enumerate(explanations):
        _assert_payload_matches_live_rules(roundtripped["explanations"][idx], live_exp)
        _assert_payload_matches_live_rules(streamed[idx], live_exp)
        _assert_domain_matches_live_rules(exported.explanations[idx], live_exp)


def test_should_serialize_conjunctive_factual_and_alternative_rule_values(
    binary_explainer_and_test,
):
    """Conjunctive factual and alternative exports match conjunctive_rules."""
    explainer, x_test = binary_explainer_and_test
    factual_conj = explainer.explain_factual(x_test[:1]).add_conjunctions(n_top_features=3)
    alt_conj = explainer.explore_alternatives(x_test[:1]).add_conjunctions(n_top_features=3)

    for collection in (factual_conj, alt_conj):
        live_exp = collection[0]
        live_rules = live_exp.get_rules()
        assert any(live_rules["is_conjunctive"])

        payload = collection.to_json()
        roundtripped = json.loads(json.dumps(payload))
        exported = CalibratedExplanations.from_json(roundtripped)
        streamed = _extract_stream_explanations(collection)

        _assert_payload_matches_live_rules(roundtripped["explanations"][0], live_exp)
        _assert_payload_matches_live_rules(streamed[0], live_exp)
        _assert_domain_matches_live_rules(exported.explanations[0], live_exp)


def test_should_serialize_multiclass_rule_values_from_rule_table(
    multiclass_explainer_and_test,
):
    """Multiclass to_json and to_json_stream match per-class live rule tables."""
    explainer, x_test = multiclass_explainer_and_test
    mc_factual = explainer.explain_factual(
        x_test, multi_labels_enabled=True, features_to_ignore=[0]
    )
    mc_alt = explainer.explore_alternatives(x_test, multi_labels_enabled=True)

    for collection in (mc_factual, mc_alt):
        payload = collection.to_json()
        roundtripped = json.loads(json.dumps(payload))
        streamed = _extract_stream_explanations(collection)

        flat_live = [exp for class_dict in collection.explanations for exp in class_dict.values()]
        assert len(roundtripped["explanations"]) == len(flat_live)
        assert len(streamed) == len(flat_live)
        for idx, live_exp in enumerate(flat_live):
            _assert_payload_matches_live_rules(roundtripped["explanations"][idx], live_exp)
            _assert_payload_matches_live_rules(streamed[idx], live_exp)


@pytest.mark.parametrize(
    "percentiles",
    [(-np.inf, 90), (10, np.inf)],
    ids=["lower_unbounded", "upper_unbounded"],
)
def test_should_serialize_one_sided_regression_rules_and_preserve_infinities(
    regression_explainer_and_test,
    percentiles,
):
    """One-sided regression exports pass invariants and preserve -inf/inf after round trip."""
    explainer, x_test = regression_explainer_and_test
    factual = explainer.explain_factual(
        x_test[:1], low_high_percentiles=percentiles, features_to_ignore=[0]
    )
    alternative = explainer.explore_alternatives(x_test[:1], low_high_percentiles=percentiles)

    for collection in (factual, alternative):
        payload = collection.to_json()
        roundtripped = json.loads(json.dumps(payload))
        exported = CalibratedExplanations.from_json(roundtripped)
        streamed = _extract_stream_explanations(collection)

        assert len(roundtripped["explanations"]) == 1
        assert len(streamed) == 1
        _assert_payload_matches_live_rules(roundtripped["explanations"][0], collection[0])
        _assert_payload_matches_live_rules(streamed[0], collection[0])
        _assert_domain_matches_live_rules(exported.explanations[0], collection[0])
