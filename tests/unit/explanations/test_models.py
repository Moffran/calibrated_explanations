"""Unit tests for src/calibrated_explanations/explanations/models.py."""

import pytest

from calibrated_explanations.explanations.models import (
    FeatureRule,
    Explanation,
    from_legacy_dict,
)


class TestModels:
    def test_feature_rule_instantiation(self):
        """Test basic instantiation of FeatureRule."""
        rule = FeatureRule(
            feature=0,
            rule="f0 < 5",
            rule_weight={"predict": 0.1},
            rule_prediction={"predict": 0.8},
            instance_prediction={"predict": 0.9},
            feature_value=3.5,
            is_conjunctive=False,
            value_str="3.50",
        )
        assert rule.feature == 0
        assert rule.rule == "f0 < 5"
        assert rule.is_conjunctive is False
        assert rule.instance_prediction["predict"] == 0.9

    def test_explanation_instantiation(self):
        """Test instantiation of Explanation dataclass."""
        expl = Explanation(
            task="classification",
            index=10,
            explanation_type="factual",
            prediction={"predict": 0.5},
            rules=[],
            provenance={"source": "test"},
            metadata={"version": "1.0"},
        )
        assert expl.task == "classification"
        assert expl.index == 10
        assert expl.rules == []
        assert expl.provenance["source"] == "test"

    def test_from_legacy_dict_empty(self):
        """Test adapter with minimal/empty dictionary."""
        legacy = {}
        expl = from_legacy_dict(0, legacy)
        assert expl.prediction == {}
        assert expl.rules == []
        assert expl.index == 0  # Default doesn't come from dict but arg

    def test_from_legacy_dict_full(self):
        """Test adapter with full populated dictionary."""
        legacy = {
            "prediction": {"predict": 0.5},
            "rules": {
                "rule": ["f0 < 5", "f1 > 2"],
                "feature": [0, 1],
                "value": ["3.5", "4.2"],
            },
            "feature_weights": {"predict": [0.1, 0.2], "low": [0.05, 0.15]},
            "feature_predict": {"predict": [0.8, 0.7], "high": [0.9, 0.8]},
        }
        expl = from_legacy_dict(0, legacy)

        assert len(expl.rules) == 2

        r0 = expl.rules[0]
        assert r0.feature == 0
        assert r0.rule == "f0 < 5"
        assert r0.rule_weight["predict"] == 0.1
        assert r0.rule_prediction["predict"] == 0.8

        r1 = expl.rules[1]
        assert r1.feature == 1
        assert r1.rule == "f1 > 2"
        assert r1.rule_weight["predict"] == 0.2
        assert r1.rule_prediction["predict"] == 0.7

    def test_from_legacy_dict_ragged_arrays(self):
        """Test adapter handling of ragged/mismatched array lengths (safe_pick)."""
        # Rules has 2 entries
        # Weights has 1 entry (should repeat last or handle missing)
        # Predicts has 0 entries (should be None)
        legacy = {
            "rules": {
                "rule": ["r1", "r2"],
                "feature": [0, [1, 2]],  # Test single int and list (conjunctive)
            },
            "feature_weights": {"w": [0.1]},
            "feature_predict": {"p": []},
        }

        expl = from_legacy_dict(0, legacy)
        assert len(expl.rules) == 2

        # First rule
        r0 = expl.rules[0]
        assert r0.feature == 0
        assert r0.rule_weight["w"] == 0.1  # Takes valid index 0
        assert r0.rule_prediction["p"] is None  # Empty array returns None

        # Second rule
        r1 = expl.rules[1]
        assert r1.feature == [1, 2]  # Converts to list
        # We expect safe_pick to return last element if index out of bounds
        assert r1.rule_weight["w"] == 0.1  # w has 1 element, index 1 -> takes last (0.1)
        assert r1.rule_prediction["p"] is None

    def test_from_legacy_dict_malformed_rules(self):
        """Test adapter robustness when rules is not a dict."""
        legacy = {"rules": []}  # List instead of dict
        expl = from_legacy_dict(0, legacy)
        assert expl.rules == []


@pytest.mark.parametrize(
    ("explicit_type", "expected"),
    [
        ("alternative", "alternative"),
        ("factual", "factual"),
        ("fast", "fast"),
        ("unknown", "factual"),
        ("fastish", "factual"),
    ],
)
def test_from_legacy_dict_respects_explicit_type(explicit_type, expected):
    legacy = {
        "explanation_type": explicit_type,
        "prediction": {},
        "rules": {"rule": [], "feature": []},
        "feature_weights": {},
        "feature_predict": {"predict": []},
    }
    expl = from_legacy_dict(0, legacy)
    assert expl.explanation_type == expected


def test_from_legacy_dict_legacy_feature_predict_heuristic():
    legacy = {
        "prediction": {},
        "rules": {"rule": [], "feature": []},
        "feature_weights": {},
        "feature_predict": {"predict": []},
    }
    expl = from_legacy_dict(0, legacy)
    assert expl.explanation_type == "alternative"

    legacy.pop("feature_predict", None)
    expl = from_legacy_dict(0, legacy)
    assert expl.explanation_type == "factual"
