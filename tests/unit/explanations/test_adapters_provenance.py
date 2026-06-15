"""Regression tests for legacy/domain adapter provenance propagation and round-trips."""

from __future__ import annotations

from calibrated_explanations.explanations.adapters import domain_to_legacy, legacy_to_domain
from calibrated_explanations.explanations.models import Explanation, FeatureRule


def test_should_preserve_provenance_and_metadata_when_legacy_to_domain():
    payload = {
        "task": "classification",
        "prediction": {"predict": 0.7},
        "rules": {"rule": ["x0 <= 1.0"], "feature": [0]},
        "feature_weights": {"predict": [0.2]},
        "feature_predict": {"predict": [0.8]},
        "provenance": {"library_version": "0.11.0", "run_id": "abc"},
        "metadata": {"class_index": 1, "class_label": "yes"},
    }

    domain = legacy_to_domain(0, payload)

    assert domain.provenance == {"library_version": "0.11.0", "run_id": "abc"}
    assert domain.metadata == {"class_index": 1, "class_label": "yes"}


def test_should_preserve_provenance_and_metadata_when_domain_to_legacy():
    exp = Explanation(
        task="classification",
        index=0,
        explanation_type="factual",
        prediction={"predict": 0.9},
        rules=[
            FeatureRule(
                feature=0,
                rule="x0 <= 1.0",
                rule_weight={"predict": 0.2},
                rule_prediction={"predict": 0.8},
            )
        ],
        provenance={"library_version": "0.11.0"},
        metadata={"class_index": 0, "class_label": "a"},
    )

    out = domain_to_legacy(exp)

    assert out["provenance"] == {"library_version": "0.11.0"}
    assert out["metadata"] == {"class_index": 0, "class_label": "a"}


def test_should_survive_round_trip_for_conjunction_feature():
    """Multi-feature conjunction must survive legacy_to_domain → domain_to_legacy without data loss (ADR-008 gap 3)."""
    exp = Explanation(
        task="classification",
        index=0,
        explanation_type="factual",
        prediction={"predict": 0.7},
        rules=[
            FeatureRule(
                feature=(0, 2),
                rule="x0 <= 1.0 AND x2 > 0.5",
                rule_weight={"predict": 0.3},
                rule_prediction={"predict": 0.6},
                is_conjunctive=True,
            )
        ],
    )

    legacy = domain_to_legacy(exp)
    assert legacy["rules"]["feature"] == [(0, 2)]

    roundtripped = legacy_to_domain(0, legacy)
    assert roundtripped.rules[0].feature == (0, 2)


def test_should_survive_round_trip_for_single_feature():
    """Single-feature rule must also round-trip without type change (ADR-008 gap 3)."""
    exp = Explanation(
        task="classification",
        index=0,
        explanation_type="factual",
        prediction={"predict": 0.8},
        rules=[
            FeatureRule(
                feature=3,
                rule="x3 > 0.0",
                rule_weight={"predict": 0.1},
                rule_prediction={"predict": 0.9},
            )
        ],
    )

    legacy = domain_to_legacy(exp)
    roundtripped = legacy_to_domain(0, legacy)

    assert roundtripped.rules[0].feature == 3
    assert isinstance(roundtripped.rules[0].feature, int)
