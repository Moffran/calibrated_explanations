"""Tests for ``metadata.narrative_context`` and the deterministic NarrativeSpec (ADR-041)."""

from __future__ import annotations

import copy
import dataclasses
import json

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from calibrated_explanations import RejectPolicySpec
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
from calibrated_explanations.explanations import (
    CalibratedExplanations,
    NarrativeSpec,
    build_narrative_spec,
)
from calibrated_explanations.serialization import from_json, to_json
from calibrated_explanations.utils.exceptions import ValidationError

_SEED = 0
_LEGACY_KEYS = {
    "task",
    "index",
    "explanation_type",
    "prediction",
    "rules",
    "provenance",
    "metadata",
    "schema_version",
}


@pytest.fixture(scope="module")
def binary():
    X, y = make_classification(n_samples=300, n_features=4, random_state=_SEED)
    explainer = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=10, random_state=_SEED))
    explainer.fit(X[:100], y[:100])
    explainer.calibrate(X[100:250], y[100:250])
    return explainer, X[250:]


@pytest.fixture(scope="module")
def multiclass():
    X, y = make_classification(
        n_samples=300,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        random_state=_SEED,
    )
    explainer = WrapCalibratedExplainer(RandomForestClassifier(n_estimators=10, random_state=_SEED))
    explainer.fit(X[:100], y[:100])
    explainer.calibrate(X[100:250], y[100:250])
    return explainer, X[250:]


@pytest.fixture(scope="module")
def regression():
    X, y = make_regression(n_samples=300, n_features=4, random_state=_SEED)
    explainer = WrapCalibratedExplainer(RandomForestRegressor(n_estimators=10, random_state=_SEED))
    explainer.fit(X[:100], y[:100])
    explainer.calibrate(X[100:250], y[100:250])
    return explainer, X[250:]


def _payloads(explanations):
    return explanations.to_json()["explanations"]


def _alternative_payload(context=None):
    """Classification alternative from the ADR-042 Appendix A example (base 0.6)."""
    base = 0.6
    predict, low, high = [0.2, 0.75, 0.8], [0.05, 0.74, 0.7], [0.45, 0.76, 0.9]
    rules = [
        {
            "feature": i,
            "rule": f"r{i}",
            "rule_weight": {"predict": p - base, "low": lo - base, "high": hi - base},
            "rule_prediction": {"predict": p, "low": lo, "high": hi},
        }
        for i, (p, lo, hi) in enumerate(zip(predict, low, high))
    ]
    metadata = {"narrative_context": context} if context is not None else None
    return {
        "task": "classification",
        "index": 0,
        "explanation_type": "alternative",
        "prediction": {"predict": base, "low": 0.5, "high": 0.9},
        "rules": rules,
        "metadata": metadata,
    }


def _factual_payload(weights, metadata=None, task="regression"):
    return {
        "task": task,
        "index": 0,
        "explanation_type": "factual",
        "prediction": {"predict": 1.0, "low": 0.0, "high": 2.0},
        "rules": [
            {"feature": i, "rule": f"r{i}", "rule_weight": dict(zip(("predict", "low", "high"), w))}
            for i, w in enumerate(weights)
        ],
        "metadata": metadata,
    }


def _rule_names(spec):
    return [fact.rule for fact in spec.rules]


# ---------------------------------------------------------------------------
# metadata.narrative_context (#228)
# ---------------------------------------------------------------------------


def test_should_add_binary_narrative_context_without_changing_existing_keys(binary):
    explainer, X = binary

    payload = _payloads(explainer.explain_factual(X[:2]))[0]

    assert set(payload) == _LEGACY_KEYS
    assert payload["metadata"]["calibration_metadata"] == {"method": "classification"}
    assert payload["metadata"]["model_metadata"] == {"type": "RandomForestClassifier"}
    assert payload["metadata"]["narrative_context"] == {
        "problem_type": "binary_classification",
        "class_index": 1,
        "class_label": "1",
    }


def test_should_name_explained_class_in_multiclass_narrative_context(multiclass):
    explainer, X = multiclass

    payloads = _payloads(explainer.explain_factual(X[:3]))

    predicted = explainer.predict(X[:3])
    for payload, label in zip(payloads, predicted):
        context = payload["metadata"]["narrative_context"]
        assert context["problem_type"] == "multiclass_classification"
        assert context["class_index"] == int(label)


def test_should_carry_percentiles_for_regression_and_threshold_for_probabilistic(regression):
    explainer, X = regression

    plain = _payloads(explainer.explain_factual(X[:1]))[0]["metadata"]["narrative_context"]
    scalar = _payloads(explainer.explain_factual(X[:1], threshold=0.0))[0]["metadata"][
        "narrative_context"
    ]
    interval = _payloads(explainer.explore_alternatives(X[:1], threshold=(-10.0, 10.0)))[0][
        "metadata"
    ]["narrative_context"]

    assert plain == {"problem_type": "regression", "low_high_percentiles": [5.0, 95.0]}
    assert scalar == {"problem_type": "probabilistic_regression", "threshold": 0.0}
    assert interval == {"problem_type": "probabilistic_regression", "threshold": [-10.0, 10.0]}


def test_should_record_reject_state_only_when_a_reject_policy_is_active(binary):
    explainer, X = binary

    without_policy = _payloads(explainer.explain_factual(X[:10]))
    with_policy = explainer.explain_factual(X[:10], reject_policy=RejectPolicySpec.flag())
    payloads = _payloads(with_policy)

    assert all("reject" not in p["metadata"]["narrative_context"] for p in without_policy)
    for explanation, payload in zip(with_policy, payloads):
        reject = payload["metadata"]["narrative_context"]["reject"]
        assert reject["rejected"] is bool(explanation.reject_context.rejected)
        assert reject["reject_type"] == explanation.reject_context.reject_type
        assert reject["prediction_set_size"] == explanation.reject_context.prediction_set_size


def test_should_round_trip_payloads_with_and_without_narrative_context(binary):
    explainer, X = binary
    payload = json.loads(json.dumps(_payloads(explainer.explain_factual(X[:1]))[0]))
    legacy = copy.deepcopy(payload)
    del legacy["metadata"]["narrative_context"]

    restored = to_json(from_json(payload))
    restored_legacy = to_json(from_json(legacy))
    collection = CalibratedExplanations.from_json({"explanations": [payload, legacy]})

    assert restored["metadata"]["narrative_context"] == payload["metadata"]["narrative_context"]
    assert "narrative_context" not in restored_legacy["metadata"]
    assert restored_legacy["rules"] == legacy["rules"]
    assert collection.explanations[0].metadata["narrative_context"]["class_index"] == 1
    assert "narrative_context" not in collection.explanations[1].metadata


# ---------------------------------------------------------------------------
# NarrativeSpec facts (#229, #230)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("explain_factual", {}),
        ("explore_alternatives", {}),
        ("explain_factual", {"threshold": 0.0}),
    ],
)
def test_should_restate_calibrated_prediction_without_rounding(regression, method, kwargs):
    explainer, X = regression
    if kwargs:
        expected, (low, high) = explainer.predict_proba(X[:3], uq_interval=True, **kwargs)
        expected = expected[:, 1]
    else:
        expected, (low, high) = explainer.predict(X[:3], uq_interval=True)

    payloads = _payloads(getattr(explainer, method)(X[:3], **kwargs))
    specs = [build_narrative_spec(p) for p in payloads]

    for i, (payload, spec) in enumerate(zip(payloads, specs)):
        assert spec.prediction.predict == payload["prediction"]["predict"]
        assert spec.prediction.low == payload["prediction"]["low"]
        assert spec.prediction.high == payload["prediction"]["high"]
        assert spec.prediction.predict == pytest.approx(expected[i], rel=1e-12)
        assert (spec.prediction.low, spec.prediction.high) == pytest.approx(
            (low[i], high[i]), rel=1e-12
        )
        by_source = {fact.source: fact for fact in spec.rules}
        for j, rule in enumerate(payload["rules"]):
            fact = by_source[f"/rules/{j}"]
            assert fact.rule == rule["rule"]
            assert (fact.weight.predict, fact.weight.low, fact.weight.high) == (
                rule["rule_weight"]["predict"],
                rule["rule_weight"]["low"],
                rule["rule_weight"]["high"],
            )


def test_should_separate_counterfactual_alternatives_from_factual_rules(binary):
    explainer, X = binary

    factual = build_narrative_spec(_payloads(explainer.explain_factual(X[:1]))[0])
    alternative = build_narrative_spec(_payloads(explainer.explore_alternatives(X[:1]))[0])

    assert factual.explanation_type == "factual"
    assert not any(fact.counterfactual for fact in factual.rules)
    assert alternative.explanation_type == "alternative"
    assert all(fact.counterfactual and fact.prediction is not None for fact in alternative.rules)


def test_should_cover_fast_explanations_as_factual_rules(binary):
    explainer, X = binary

    spec = build_narrative_spec(_payloads(explainer.explain_fast(X[:1]))[0])

    assert spec.explanation_type == "fast"
    assert dict(spec.ranking) == {"metric": "feature_weight"}
    assert not any(fact.counterfactual for fact in spec.rules)


def test_should_build_identical_canonical_json_for_identical_semantic_input(multiclass):
    explainer, X = multiclass
    payload = _payloads(explainer.explain_factual(X[:1]))[0]
    reparsed = json.loads(json.dumps(payload))
    reordered = dict(reversed(list(reparsed.items())))

    first = build_narrative_spec(payload)
    second = build_narrative_spec(reordered)

    assert first == second
    assert first.to_json() == second.to_json()
    assert first.spec_id == second.spec_id
    assert json.loads(first.to_json())["spec_id"] == first.spec_id


def test_should_treat_integer_and_float_values_as_the_same_semantic_input():
    as_int = _factual_payload([(1, 0, 2)])
    as_float = _factual_payload([(1.0, 0.0, 2.0)])

    assert build_narrative_spec(as_int).to_json() == build_narrative_spec(as_float).to_json()


def test_should_not_mutate_the_source_payload(regression):
    explainer, X = regression
    payload = _payloads(explainer.explore_alternatives(X[:1]))[0]
    before = copy.deepcopy(payload)

    build_narrative_spec(payload)

    assert payload == before


def test_should_accept_a_domain_explanation(binary):
    explainer, X = binary
    payload = _payloads(explainer.explain_factual(X[:1]))[0]

    assert build_narrative_spec(from_json(payload)) == build_narrative_spec(payload)


def test_should_be_immutable(binary):
    explainer, X = binary
    spec = build_narrative_spec(_payloads(explainer.explain_factual(X[:1]))[0])

    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.rules = ()  # type: ignore[misc]
    with pytest.raises(TypeError):
        spec.context["problem_type"] = "regression"  # type: ignore[index]
    assert isinstance(spec, NarrativeSpec)


def test_should_qualify_rejected_instances_and_missing_context():
    rejected = _alternative_payload(
        {
            "problem_type": "binary_classification",
            "class_index": 1,
            "reject": {"rejected": True, "reject_type": "ambiguity", "prediction_set_size": 2},
        }
    )

    assert build_narrative_spec(rejected).qualifiers == ("rejected",)
    assert build_narrative_spec(_alternative_payload()).qualifiers == ("context_unavailable",)


def test_should_flag_factual_weight_intervals_that_contain_zero():
    spec = build_narrative_spec(_factual_payload([(0.5, 0.2, 0.8), (0.1, -0.1, 0.3)]))

    by_rule = {fact.rule: fact.qualifiers for fact in spec.rules}
    assert by_rule == {"r0": (), "r1": ("weight_interval_contains_zero",)}


# ---------------------------------------------------------------------------
# ADR-021 interval semantics
# ---------------------------------------------------------------------------


def test_should_mark_placeholder_endpoints_of_infinite_percentiles_unbounded(regression):
    explainer, X = regression

    spec = build_narrative_spec(
        _payloads(explainer.explain_factual(X[:1], low_high_percentiles=(-np.inf, 90)))[0]
    )

    assert spec.prediction.low_bounded is False
    assert spec.prediction.high_bounded is True
    assert "unbounded_interval" in spec.qualifiers
    # A factual weight is predict - perturbed, so the open side is mirrored.
    assert all(not fact.weight.high_bounded and fact.weight.low_bounded for fact in spec.rules)


def test_should_represent_infinite_endpoints_as_unbounded_in_canonical_json():
    payload = _factual_payload(
        [(0.5, 0.2, 0.8)],
        metadata={
            "narrative_context": {
                "problem_type": "regression",
                "low_high_percentiles": [-np.inf, 90.0],
            }
        },
    )
    payload["prediction"] = {"predict": 1.0, "low": -np.inf, "high": 2.0}

    spec = build_narrative_spec(payload)

    assert spec.prediction.low == -np.inf
    assert json.loads(spec.to_json())["prediction"]["low"] == "-inf"


def test_should_skip_containment_on_sides_excluded_by_the_percentile_scope():
    # ADR-021 §4: with percentiles (-inf, 40) the CPS median may exceed `high`.
    payload = _factual_payload(
        [(0.5, 0.2, 0.8)],
        metadata={
            "narrative_context": {
                "problem_type": "regression",
                "low_high_percentiles": [-np.inf, 40.0],
            }
        },
    )
    payload["prediction"] = {"predict": 3.0, "low": -10.0, "high": 2.0}

    spec = build_narrative_spec(payload)

    assert spec.prediction.predict == 3.0


@pytest.mark.parametrize(
    "prediction",
    [
        {"predict": 1.0, "low": 2.0, "high": 0.0},
        {"predict": 5.0, "low": 0.0, "high": 2.0},
        {"predict": float("nan"), "low": 0.0, "high": 2.0},
        {"predict": "1.0", "low": 0.0, "high": 2.0},
        {"predict": 1.0, "low": 0.0},
        None,
    ],
)
def test_should_reject_invalid_prediction_intervals(prediction):
    payload = _factual_payload([(0.5, 0.2, 0.8)])
    payload["prediction"] = prediction

    with pytest.raises(ValidationError):
        build_narrative_spec(payload)


def test_should_reject_invalid_rule_weight_intervals():
    with pytest.raises(ValidationError, match="ADR-021"):
        build_narrative_spec(_factual_payload([(0.9, 0.2, 0.8)]))


def test_should_reject_non_integer_rule_features():
    payload = _factual_payload([(0.5, 0.2, 0.8)])
    payload["rules"][0]["feature"] = "age"

    with pytest.raises(ValidationError, match="feature"):
        build_narrative_spec(payload)


@pytest.mark.parametrize(
    ("overrides", "context"),
    [
        ({"explanation_type": "global"}, None),
        ({"task": "ranking"}, None),
        ({}, {"problem_type": "binary_classification"}),
        ({}, {"problem_type": "regression", "threshold": 0.5}),
        ({}, {"problem_type": "probabilistic_regression"}),
        ({}, {"problem_type": "regression", "low_high_percentiles": [95.0, 5.0]}),
        ({}, {"problem_type": "regression", "reject": {"rejected": "yes"}}),
        ({}, ["regression"]),
    ],
)
def test_should_reject_missing_or_contradictory_semantics(overrides, context):
    payload = _factual_payload([(0.5, 0.2, 0.8)])
    payload.update(overrides)
    if context is not None:
        payload["metadata"] = {"narrative_context": context}

    with pytest.raises(ValidationError):
        build_narrative_spec(payload)


# ---------------------------------------------------------------------------
# ADR-042 canonical rule order
# ---------------------------------------------------------------------------


def test_should_order_factual_rules_by_magnitude_then_narrower_width_then_index():
    spec = build_narrative_spec(
        _factual_payload(
            [
                (0.3, 0.0, 0.6),  # r0: |w| 0.3, wide
                (-0.5, -0.6, -0.4),  # r1: strongest
                (0.3, 0.25, 0.35),  # r2: |w| 0.3, narrower than r0
                (-0.3, -0.35, -0.25),  # r3: ties r2 exactly, higher index
            ]
        )
    )

    assert _rule_names(spec) == ["r1", "r2", "r3", "r0"]
    assert dict(spec.ranking) == {"metric": "feature_weight"}


@pytest.mark.parametrize(
    ("kwargs", "expected", "ranking"),
    [
        ({}, ["r1", "r2", "r0"], {"metric": "ensured", "weight": 0.5}),
        ({"rnk_metric": "feature_weight"}, ["r0", "r2", "r1"], {"metric": "feature_weight"}),
        ({"rnk_weight": -0.5}, ["r0", "r1", "r2"], {"metric": "ensured", "weight": -0.5}),
        ({"rnk_metric": "uncertainty"}, ["r1", "r2", "r0"], {"metric": "ensured", "weight": 0.0}),
    ],
)
def test_should_order_alternatives_by_the_adr042_ranking(kwargs, expected, ranking):
    spec = build_narrative_spec(_alternative_payload(), **kwargs)

    assert _rule_names(spec) == expected
    assert dict(spec.ranking) == ranking


def test_should_reject_alternative_metrics_on_factual_explanations():
    with pytest.raises(ValidationError, match="ADR-042"):
        build_narrative_spec(_factual_payload([(0.5, 0.2, 0.8)]), rnk_metric="ensured")


def test_should_require_context_for_ensured_ranking_of_regression_alternatives():
    payload = _alternative_payload()
    payload["task"] = "regression"

    with pytest.raises(ValidationError, match="narrative_context"):
        build_narrative_spec(payload)
    assert _rule_names(build_narrative_spec(payload, rnk_metric="feature_weight")) == [
        "r0",
        "r2",
        "r1",
    ]
