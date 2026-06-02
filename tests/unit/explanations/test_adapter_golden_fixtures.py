"""Golden fixture parity tests for legacy/domain adapter round-trips (ADR-008 gap 4).

These tests encode known inputs and their exact expected outputs as golden fixtures.
Any change to adapter behaviour will cause these tests to fail, providing early
warning of regressions in the legacy ↔ domain round-trip.
"""

from __future__ import annotations

from calibrated_explanations.explanations.adapters import domain_to_legacy, legacy_to_domain
from calibrated_explanations.explanations.models import Explanation, FeatureRule

# ---------------------------------------------------------------------------
# Golden fixture: classification factual explanation
# ---------------------------------------------------------------------------

_CLASSIFICATION_LEGACY_PAYLOAD = {
    "task": "classification",
    "prediction": {"predict": 0.72, "low": 0.55, "high": 0.88},
    "rules": {
        "rule": ["x0 > 0.5", "x1 <= 1.2"],
        "feature": [0, 1],
    },
    "feature_weights": {"predict": [0.15, -0.08]},
    "feature_predict": {"predict": [0.65, 0.50]},
    "provenance": {"library_version": "0.11.3", "run_id": "golden-001"},
    "metadata": {"class_index": 1, "class_label": "positive"},
}

_CLASSIFICATION_DOMAIN_GOLDEN = Explanation(
    task="classification",
    index=0,
    explanation_type="factual",
    prediction={"predict": 0.72, "low": 0.55, "high": 0.88},
    rules=[
        FeatureRule(
            feature=0,
            rule="x0 > 0.5",
            rule_weight={"predict": 0.15},
            rule_prediction={"predict": 0.65},
        ),
        FeatureRule(
            feature=1,
            rule="x1 <= 1.2",
            rule_weight={"predict": -0.08},
            rule_prediction={"predict": 0.50},
        ),
    ],
    provenance={"library_version": "0.11.3", "run_id": "golden-001"},
    metadata={"class_index": 1, "class_label": "positive"},
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_should_produce_golden_domain_model_when_legacy_to_domain():
    """legacy_to_domain must produce the exact golden Explanation for the classification fixture."""
    domain = legacy_to_domain(0, _CLASSIFICATION_LEGACY_PAYLOAD)

    assert domain.task == _CLASSIFICATION_DOMAIN_GOLDEN.task
    assert domain.index == _CLASSIFICATION_DOMAIN_GOLDEN.index
    assert domain.prediction == _CLASSIFICATION_DOMAIN_GOLDEN.prediction
    assert domain.provenance == _CLASSIFICATION_DOMAIN_GOLDEN.provenance
    assert domain.metadata == _CLASSIFICATION_DOMAIN_GOLDEN.metadata

    assert len(domain.rules) == len(_CLASSIFICATION_DOMAIN_GOLDEN.rules)
    for i, (actual, expected) in enumerate(
        zip(domain.rules, _CLASSIFICATION_DOMAIN_GOLDEN.rules, strict=True)
    ):
        assert actual.feature == expected.feature, f"Rule {i}: feature mismatch"
        assert actual.rule == expected.rule, f"Rule {i}: rule text mismatch"
        assert actual.rule_weight == expected.rule_weight, f"Rule {i}: rule_weight mismatch"
        assert (
            actual.rule_prediction == expected.rule_prediction
        ), f"Rule {i}: rule_prediction mismatch"


def test_should_produce_golden_legacy_payload_when_domain_to_legacy():
    """domain_to_legacy must produce the exact golden legacy payload for the domain fixture."""
    out = domain_to_legacy(_CLASSIFICATION_DOMAIN_GOLDEN)

    assert out["task"] == "classification"
    assert out["prediction"] == {"predict": 0.72, "low": 0.55, "high": 0.88}
    assert out["provenance"] == {"library_version": "0.11.3", "run_id": "golden-001"}
    assert out["metadata"] == {"class_index": 1, "class_label": "positive"}
    assert out["rules"]["rule"] == ["x0 > 0.5", "x1 <= 1.2"]
    assert out["rules"]["feature"] == [0, 1]
    assert out["feature_weights"] == {"predict": [0.15, -0.08]}
    assert out["feature_predict"] == {"predict": [0.65, 0.50]}


def test_should_round_trip_without_data_loss():
    """legacy→domain→legacy must preserve all fields in the golden classification fixture."""
    domain = legacy_to_domain(0, _CLASSIFICATION_LEGACY_PAYLOAD)
    reconstructed = domain_to_legacy(domain)

    assert reconstructed["task"] == _CLASSIFICATION_LEGACY_PAYLOAD["task"]
    assert reconstructed["prediction"] == _CLASSIFICATION_LEGACY_PAYLOAD["prediction"]
    assert reconstructed["rules"]["rule"] == _CLASSIFICATION_LEGACY_PAYLOAD["rules"]["rule"]
    assert reconstructed["rules"]["feature"] == _CLASSIFICATION_LEGACY_PAYLOAD["rules"]["feature"]
    assert reconstructed["feature_weights"] == _CLASSIFICATION_LEGACY_PAYLOAD["feature_weights"]
    assert reconstructed["feature_predict"] == _CLASSIFICATION_LEGACY_PAYLOAD["feature_predict"]
    assert reconstructed["provenance"] == _CLASSIFICATION_LEGACY_PAYLOAD["provenance"]
    assert reconstructed["metadata"] == _CLASSIFICATION_LEGACY_PAYLOAD["metadata"]


def test_should_round_trip_domain_to_legacy_to_domain_without_data_loss():
    """domain→legacy→domain must preserve all fields in the golden domain fixture."""
    legacy = domain_to_legacy(_CLASSIFICATION_DOMAIN_GOLDEN)
    reconstructed = legacy_to_domain(0, legacy)

    assert reconstructed.task == _CLASSIFICATION_DOMAIN_GOLDEN.task
    assert reconstructed.prediction == _CLASSIFICATION_DOMAIN_GOLDEN.prediction
    assert reconstructed.provenance == _CLASSIFICATION_DOMAIN_GOLDEN.provenance
    assert reconstructed.metadata == _CLASSIFICATION_DOMAIN_GOLDEN.metadata
    assert len(reconstructed.rules) == len(_CLASSIFICATION_DOMAIN_GOLDEN.rules)
    for i, (r_actual, r_expected) in enumerate(
        zip(reconstructed.rules, _CLASSIFICATION_DOMAIN_GOLDEN.rules, strict=True)
    ):
        assert r_actual.feature == r_expected.feature, f"Round-trip rule {i}: feature mismatch"
        assert r_actual.rule == r_expected.rule, f"Round-trip rule {i}: rule text mismatch"
        assert (
            r_actual.rule_weight == r_expected.rule_weight
        ), f"Round-trip rule {i}: rule_weight mismatch"
