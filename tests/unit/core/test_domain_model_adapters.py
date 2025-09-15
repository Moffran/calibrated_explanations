from __future__ import annotations

from typing import Any, Dict

import pytest

from calibrated_explanations.explanations.adapters import domain_to_legacy, legacy_to_domain


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
