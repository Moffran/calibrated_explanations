"""Contract tests for guarded audit payloads."""

import json

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.utils.exceptions import ValidationError
from calibrated_explanations.explanations.guarded_explanation import (
    GuardedAlternativeExplanation,
    GuardedBin,
    GuardedFactualExplanation,
)


def make_classification_explainer(*, seed: int = 0) -> tuple[CalibratedExplainer, np.ndarray]:
    data = load_iris()
    x_train, x_cal, y_train, y_cal = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=seed,
        stratify=data.target,
    )

    model = RandomForestClassifier(n_estimators=15, random_state=seed, max_depth=3)
    model.fit(x_train, y_train)

    explainer = CalibratedExplainer(model, x_cal, y_cal, mode="classification", seed=seed)
    return explainer, x_cal


class DummyExplainer:
    def __init__(self) -> None:
        self.y_cal = np.array([0, 1])
        self.mode = "classification"
        self.feature_names = ["f0", "f1"]
        self.class_labels = [0, 1]
        self.categorical_features = []
        self.categorical_labels = None

    def is_multiclass(self) -> bool:
        return False


class DummyCollection:
    def __init__(self, features_to_ignore=None) -> None:
        self.explainer = DummyExplainer()
        self.features_to_ignore = list(features_to_ignore or [])
        self.feature_filter_per_instance_ignore = None

    def get_explainer(self):
        return self.explainer


def minimal_payload():
    return {
        "binned": {"rule_values": [{0: ([0.1], 0.1, 0.1), 1: ([0.2], 0.2, 0.2)}]},
        "feature_weights": {
            "predict": np.array([[0.0, 0.0]]),
            "low": np.array([[0.0, 0.0]]),
            "high": np.array([[0.0, 0.0]]),
        },
        "feature_predict": {
            "predict": np.array([[0.2, 0.4]]),
            "low": np.array([[0.1, 0.3]]),
            "high": np.array([[0.3, 0.5]]),
        },
        "prediction": {
            "predict": np.array([0.8]),
            "low": np.array([0.7]),
            "high": np.array([0.9]),
            "classes": np.array([1.0]),
        },
    }


def test_guarded_factual_audit_returns_full_interval_table():
    explainer, x_cal = make_classification_explainer(seed=31)
    res = explainer.explain_guarded_factual(x_cal[:1], significance=0.05)
    audit = res.explanations[0].get_guarded_audit()
    assert audit["mode"] == "factual"
    assert isinstance(audit["intervals"], list)
    assert "summary" in audit
    required = {
        "feature",
        "feature_name",
        "lower",
        "upper",
        "emitted_lower",
        "emitted_upper",
        "representative",
        "p_value",
        "conforming",
        "is_factual",
        "is_merged",
        "emitted",
        "emission_reason",
        "condition",
        "predict",
        "low",
        "high",
    }
    if audit["intervals"]:
        assert required.issubset(set(audit["intervals"][0].keys()))


def test_guarded_alternative_audit_returns_full_interval_table():
    explainer, x_cal = make_classification_explainer(seed=32)
    res = explainer.explore_guarded_alternatives(x_cal[:1], significance=0.05)
    audit = res.explanations[0].get_guarded_audit()
    assert audit["mode"] == "alternative"
    assert isinstance(audit["intervals"], list)
    assert "summary" in audit


def test_guarded_audit_removed_count_equals_nonconforming_count():
    """Removed-guard counts should track candidates rejected by the shipped guard rule."""
    explainer, x_cal = make_classification_explainer(seed=33)
    res = explainer.explain_guarded_factual(x_cal[:1], significance=0.2)
    audit = res.explanations[0].get_guarded_audit()
    nonconforming = sum(1 for rec in audit["intervals"] if not rec["conforming"])
    assert audit["summary"]["intervals_removed_guard"] == nonconforming


def test_guarded_audit_emitted_count_matches_rules_length_factual():
    explainer, x_cal = make_classification_explainer(seed=34)
    res = explainer.explain_guarded_factual(x_cal[:1], significance=0.05)
    rules = res.explanations[0].get_rules()
    audit = res.explanations[0].get_guarded_audit()
    assert audit["summary"]["intervals_emitted"] == len(rules["rule"])


def test_guarded_audit_emitted_count_matches_rules_length_alternative():
    explainer, x_cal = make_classification_explainer(seed=35)
    res = explainer.explore_guarded_alternatives(x_cal[:1], significance=0.05)
    rules = res.explanations[0].get_rules()
    audit = res.explanations[0].get_guarded_audit()
    assert audit["summary"]["intervals_emitted"] == len(rules["rule"])


def test_guarded_audit_includes_p_values_for_all_tested_intervals():
    explainer, x_cal = make_classification_explainer(seed=36)
    res = explainer.explain_guarded_factual(x_cal[:1], significance=0.1)
    audit = res.explanations[0].get_guarded_audit()
    assert all("p_value" in rec for rec in audit["intervals"])
    assert all(0.0 <= float(rec["p_value"]) <= 1.0 for rec in audit["intervals"])


def test_guarded_audit_order_is_deterministic():
    explainer, x_cal = make_classification_explainer(seed=37)
    res = explainer.explain_guarded_factual(x_cal[:1], significance=0.1)
    a1 = res.explanations[0].get_guarded_audit()["intervals"]
    a2 = res.explanations[0].get_guarded_audit()["intervals"]
    assert a1 == a2
    keys = [(rec["feature"], rec["lower"], rec["upper"]) for rec in a1]
    assert keys == sorted(keys, key=lambda t: (t[0], t[1], t[2]))


def test_guarded_audit_handles_zero_emitted_rules_with_nonempty_intervals():
    payload = minimal_payload()
    expl = GuardedFactualExplanation(
        DummyCollection(),
        0,
        np.array([0.1, 0.2]),
        guarded_bins={
            0: [
                GuardedBin(
                    lower=-np.inf,
                    upper=np.inf,
                    representative=0.1,
                    predict=0.2,
                    low=0.1,
                    high=0.3,
                    conforming=True,
                    p_value=0.9,
                    is_factual=True,
                )
            ]
        },
        feature_names=["f0", "f1"],
        **payload,
    )
    # Force zero-impact gating.
    expl.feature_predict["predict"][0] = expl.prediction["predict"]
    audit = expl.get_guarded_audit()
    assert audit["summary"]["intervals_tested"] > 0
    assert audit["summary"]["intervals_emitted"] == 0


def test_collection_guarded_audit_aggregates_instance_summaries():
    explainer, x_cal = make_classification_explainer(seed=38)
    res = explainer.explain_guarded_factual(x_cal[:2], significance=0.1)
    audit = res.get_guarded_audit()
    assert audit["summary"]["n_instances"] == 2
    assert len(audit["instances"]) == 2
    assert audit["summary"]["intervals_tested"] == sum(
        inst["summary"]["intervals_tested"] for inst in audit["instances"]
    )


def test_collection_guarded_audit_raises_for_non_guarded_collection():
    explainer, x_cal = make_classification_explainer(seed=39)
    res = explainer.explain_factual(x_cal[:1])
    with pytest.raises(ValidationError, match="only available for guarded explanation collections"):
        _ = res.get_guarded_audit()


def test_guarded_audit_respects_ignored_features_marking():
    payload = minimal_payload()
    expl = GuardedFactualExplanation(
        DummyCollection(features_to_ignore=[0]),
        0,
        np.array([0.1, 0.2]),
        guarded_bins={
            0: [
                GuardedBin(
                    lower=-np.inf,
                    upper=np.inf,
                    representative=0.1,
                    predict=0.6,
                    low=0.5,
                    high=0.7,
                    conforming=True,
                    p_value=0.9,
                    is_factual=True,
                )
            ]
        },
        feature_names=["f0", "f1"],
        **payload,
    )
    audit = expl.get_guarded_audit()
    assert audit["intervals"][0]["emission_reason"] == "ignored_feature"
    assert audit["intervals"][0]["emitted"] is False


def test_guarded_audit_merge_adjacent_marks_is_merged_and_retains_p_values():
    payload = minimal_payload()
    expl = GuardedAlternativeExplanation(
        DummyCollection(),
        0,
        np.array([0.1, 0.2]),
        guarded_bins={
            0: [
                GuardedBin(
                    lower=0.0,
                    upper=1.0,
                    representative=0.5,
                    predict=0.3,
                    low=0.2,
                    high=0.4,
                    conforming=True,
                    p_value=0.42,
                    is_factual=False,
                    is_merged=True,
                )
            ]
        },
        feature_names=["f0", "f1"],
        **payload,
    )
    audit = expl.get_guarded_audit()
    rec = audit["intervals"][0]
    assert rec["is_merged"] is True
    assert rec["p_value"] == pytest.approx(0.42)


def test_guarded_audit_uses_emitted_bounds_in_condition_strings():
    payload = minimal_payload()
    expl = GuardedAlternativeExplanation(
        DummyCollection(),
        0,
        np.array([0.1, 0.2]),
        guarded_bins={
            0: [
                GuardedBin(
                    lower=0.0,
                    upper=1.0,
                    representative=0.5,
                    predict=0.3,
                    low=0.2,
                    high=0.4,
                    conforming=True,
                    p_value=0.42,
                    is_factual=False,
                    emitted_lower=0.25,
                    emitted_upper=0.75,
                )
            ]
        },
        feature_names=["f0", "f1"],
        **payload,
    )
    rec = expl.get_guarded_audit()["intervals"][0]
    assert rec["emitted_lower"] == pytest.approx(0.25)
    assert rec["emitted_upper"] == pytest.approx(0.75)
    assert rec["condition"] == "0.25 < f0 <= 0.75"


def test_guarded_audit_serialization_smoke():
    explainer, x_cal = make_classification_explainer(seed=40)
    res = explainer.explain_guarded_factual(x_cal[:1], significance=0.1)
    audit = res.get_guarded_audit()
    assert isinstance(json.dumps(audit), str)
