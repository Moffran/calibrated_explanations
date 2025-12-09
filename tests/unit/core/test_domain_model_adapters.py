from __future__ import annotations

from typing import Any, Dict

import pytest

import numpy as np

from calibrated_explanations.explanations import domain_to_legacy, legacy_to_domain
from calibrated_explanations.explanations import models


def _make_feature_rule(
    feature: Any,
    rule: str,
    *,
    rule_weight: Dict[str, Any] | None = None,
    rule_prediction: Dict[str, Any] | None = None,
    **extras: Any,
) -> models.FeatureRule:
    """Small helper to build feature rules with sensible defaults for tests."""

    defaults: Dict[str, Any] = {
        "instance_prediction": None,
        "feature_value": None,
        "is_conjunctive": False,
        "value_str": None,
        "bin_index": None,
    }
    defaults.update(extras)
    return models.FeatureRule(
        feature=feature,
        rule=rule,
        rule_weight=rule_weight or {},
        rule_prediction=rule_prediction or {},
        **defaults,
    )


def _make_legacy_payload_zero_rules(task: str = "classification") -> Dict[str, Any]:
    return {
        "task": task,
        "prediction": {"predict": [0.7], "low": [0.6], "high": [0.8]},
        "rules": {"rule": [], "feature": []},
        "feature_weights": {"predict": [], "low": [], "high": []},
        "feature_predict": {"predict": [], "low": [], "high": []},
    }


def _make_legacy_payload_two_rules(task: str = "classification") -> Dict[str, Any]:
    return {
        "task": task,
        "prediction": {"predict": [0.7], "low": [0.6], "high": [0.8]},
        "rules": {"rule": ["x0 <= 3.1", "x2 > 5.0"], "feature": [0, 2]},
        "feature_weights": {
            "predict": [0.11, 0.22],
            "low": [0.05, 0.15],
            "high": [0.18, 0.30],
        },
        "feature_predict": {
            "predict": [0.51, 0.66],
            "low": [0.40, 0.55],
            "high": [0.60, 0.77],
        },
    }


@pytest.mark.parametrize(
    "payload",
    [
        _make_legacy_payload_zero_rules(),
        _make_legacy_payload_two_rules(),
    ],
)
def test_round_trip_legacy_domain_legacy_preserves_shape(payload: Dict[str, Any]) -> None:
    idx = 0
    domain = legacy_to_domain(idx, payload)
    back = domain_to_legacy(domain)

    # Top-level keys present
    assert {"task", "prediction", "rules", "feature_weights", "feature_predict"} <= set(back)

    # Task and prediction preserved (prediction may be shallow-copied)
    assert back["task"] == payload.get("task")
    # When original prediction is per-instance arrays, ensure keys preserved
    assert set(back["prediction"]) == set(payload["prediction"])  # type: ignore[index]

    # Rules block shape preserved
    assert isinstance(back["rules"], dict)
    assert list(back["rules"].keys()) == ["rule", "feature"]
    assert len(back["rules"]["rule"]) == len(back["rules"]["feature"])  # type: ignore[index]

    # Feature arrays stay aligned by index across weights and predicts
    n = len(back["rules"]["rule"])  # type: ignore[index]
    for k, arr in back["feature_weights"].items():  # type: ignore[assignment]
        assert len(arr) == n
    for k, arr in back["feature_predict"].items():  # type: ignore[assignment]
        assert len(arr) == n


def test_domain_to_legacy_emits_parallel_arrays_with_multiple_rules() -> None:
    domain = models.Explanation(
        task="classification",
        index=2,
        explanation_type="factual",
        prediction={"predict": [0.8], "alt": [0.2]},
        rules=[
            _make_feature_rule(
                0,
                "x0 <= 0.3",
                rule_weight={"predict": 0.11, "alt": 0.04},
                rule_prediction={"predict": 0.56, "alt": 0.33},
                feature_value=0.25,
                value_str="<= 0.3",
            ),
            _make_feature_rule(
                2,
                "x2 > 1.0",
                rule_weight={"predict": 0.23, "alt": 0.09, "lift": 1.2},
                rule_prediction={"predict": 0.61, "alt": 0.18},
                feature_value=1.7,
                value_str="> 1.0",
            ),
        ],
    )

    out = domain_to_legacy(domain)

    assert out["task"] == "classification"
    assert out["prediction"] == {"predict": [0.8], "alt": [0.2]}
    assert out["rules"] == {"rule": ["x0 <= 0.3", "x2 > 1.0"], "feature": [0, 2]}
    assert out["feature_weights"] == {
        "predict": [0.11, 0.23],
        "alt": [0.04, 0.09],
        "lift": [1.2],
    }
    assert out["feature_predict"] == {
        "predict": [0.56, 0.61],
        "alt": [0.33, 0.18],
    }

    # ensure arrays are decoupled from the domain model structures
    out["feature_weights"]["predict"][0] = 42
    assert domain.rules[0].rule_weight["predict"] == 0.11


def test_legacy_to_domain_handles_conjunctive_rules_and_short_vectors() -> None:
    payload: Dict[str, Any] = {
        "task": "classification",
        "prediction": {"predict": [0.71]},
        "rules": {
            "rule": ["x0 <= 1", "x1 > 3", "x5 in [1, 2]"],
            "feature": [0, np.array([1, 2])],
            "feature_value": ["0.2"],
            "value": ["<= 1", "> 3"],
            "is_conjunctive": [False, True],
        },
        "feature_weights": {
            "predict": [0.11, 0.24],
            "support": [],
        },
        "feature_predict": {
            "predict": [0.55, 0.66, 0.72],
            "support": [0.40, 0.52],
        },
    }

    domain = legacy_to_domain(5, payload)

    assert domain.task == "classification"
    assert domain.index == 5
    assert len(domain.rules) == 3

    first, second, third = domain.rules
    assert first.feature == 0
    assert second.feature == [1, 2]
    assert third.feature == 2
    assert second.is_conjunctive is True
    assert third.is_conjunctive is True  # propagated from the final value
    assert third.rule_weight["predict"] == 0.24
    assert third.rule_weight["support"] is None
    assert third.rule_prediction["support"] == 0.52
    assert third.feature_value == "0.2"
    assert third.value_str == "> 3"
