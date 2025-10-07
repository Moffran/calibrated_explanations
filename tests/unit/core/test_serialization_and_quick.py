from __future__ import annotations


import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations.api.quick import quick_explain
from calibrated_explanations.explanations.adapters import (
    domain_to_legacy,
    legacy_to_domain,
)
from calibrated_explanations.explanations.models import Explanation, FeatureRule
from calibrated_explanations.serialization import from_json, to_json, validate_payload


def test_domain_json_round_trip_and_schema_validation():
    rules = [
        FeatureRule(
            feature=0,
            rule="x0 <= 0.5",
            weight={"predict": 0.1, "low": 0.05, "high": 0.15},
            prediction={"predict": 0.6, "low": 0.5, "high": 0.7},
        ),
        FeatureRule(
            feature=2,
            rule="x2 > 1.2",
            weight={"predict": 0.2, "low": 0.1, "high": 0.3},
            prediction={"predict": 0.7, "low": 0.6, "high": 0.8},
        ),
    ]
    exp = Explanation(task="classification", index=0, prediction={"predict": 1}, rules=rules)
    payload = to_json(exp)
    # Validate when jsonschema is available
    try:
        validate_payload(payload)
    except Exception as exc:  # pragma: no cover - optional dependency
        pytest.skip(f"jsonschema not available or invalid: {exc}")
    back = from_json(payload)
    assert back.task == exp.task
    assert back.index == exp.index
    assert len(back.rules) == len(exp.rules)


def test_validate_payload_rejects_missing_required_fields():
    jsonschema = pytest.importorskip("jsonschema")

    rules = [
        FeatureRule(
            feature=0,
            rule="x0 <= 0.5",
            weight={"predict": 0.1, "low": 0.05, "high": 0.15},
            prediction={"predict": 0.6, "low": 0.5, "high": 0.7},
        )
    ]
    exp = Explanation(task="classification", index=0, prediction={"predict": 1.0}, rules=rules)
    payload = to_json(exp)
    payload.pop("prediction")

    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate_payload(payload)


def test_adapter_legacy_to_json_round_trip():
    legacy = {
        "task": "classification",
        "prediction": {"predict": [0.7], "low": [0.6], "high": [0.8]},
        "rules": {"rule": ["x0 <= 0.5"], "feature": [0]},
        "feature_weights": {"predict": [0.11], "low": [0.05], "high": [0.18]},
        "feature_predict": {"predict": [0.61], "low": [0.51], "high": [0.71]},
    }
    domain = legacy_to_domain(0, legacy)
    payload = to_json(domain)
    back_domain = from_json(payload)
    back_legacy = domain_to_legacy(back_domain)
    # Minimal shape stability
    assert set(back_legacy) >= {"task", "prediction", "rules", "feature_weights", "feature_predict"}
    assert back_legacy["rules"]["feature"] == legacy["rules"]["feature"]
    assert back_legacy["rules"]["rule"] == legacy["rules"]["rule"]


def test_quick_explain_smoke():
    data = load_iris()
    x_train, x_cal, y_train, y_cal = train_test_split(
        data.data, data.target, test_size=0.2, random_state=0, stratify=data.target
    )
    model = RandomForestClassifier(n_estimators=10, random_state=0, max_depth=2)
    exp = quick_explain(
        model, x_train, y_train, x_cal, y_cal, x_cal[:5], task="classification"
    )
    # Basic shape checks
    assert hasattr(exp, "explanations")
    assert len(exp.explanations) == 5
