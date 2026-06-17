"""Unit tests for ADR-008 Gap 1 and Gap 3 — domain model authority and structured metadata."""

from __future__ import annotations

from calibrated_explanations.explanations.models import (
    CalibrationDescriptor,
    Explanation,
    ModelDescriptor,
)


# ---------------------------------------------------------------------------
# Gap 3: typed descriptor fields on Explanation dataclass
# ---------------------------------------------------------------------------


def test_explanation_defaults_calibration_metadata_to_none():
    exp = Explanation(
        task="classification",
        index=0,
        explanation_type="factual",
        prediction={"predict": 0.7},
        rules=[],
    )
    assert exp.calibration_metadata is None
    assert exp.model_metadata is None


def test_explanation_accepts_typed_calibration_metadata():
    exp = Explanation(
        task="classification",
        index=0,
        explanation_type="factual",
        prediction={"predict": 0.7},
        rules=[],
        calibration_metadata=CalibrationDescriptor(method="mondrian"),
        model_metadata=ModelDescriptor(type="DecisionTreeClassifier"),
    )
    assert exp.calibration_metadata.method == "mondrian"
    assert exp.model_metadata.type == "DecisionTreeClassifier"


def test_legacy_to_domain_roundtrip_preserves_typed_metadata_fields():
    """calibration_metadata and model_metadata survive to_json -> legacy_to_domain as typed objects."""
    from calibrated_explanations.explanations.adapters import legacy_to_domain

    payload = {
        "task": "classification",
        "prediction": {"predict": 0.8, "low": 0.6, "high": 0.95},
        "rules": {"rule": ["x0 > 0.5"], "feature": [0]},
        "feature_weights": {"predict": [0.2]},
        "feature_predict": {"predict": [0.75]},
        "metadata": {
            "calibration_metadata": {"method": "venn_abers"},
            "model_metadata": {"type": "RandomForestClassifier"},
        },
    }
    domain = legacy_to_domain(0, payload)
    assert isinstance(domain.calibration_metadata, CalibrationDescriptor)
    assert domain.calibration_metadata.method == "venn_abers"
    assert isinstance(domain.model_metadata, ModelDescriptor)
    assert domain.model_metadata.type == "RandomForestClassifier"


def test_legacy_to_domain_metadata_absent_leaves_fields_none():
    """When metadata dict has no calibration_metadata key, typed fields stay None."""
    from calibrated_explanations.explanations.adapters import legacy_to_domain

    payload = {
        "task": "classification",
        "prediction": {"predict": 0.5},
        "rules": {},
        "feature_weights": {},
        "feature_predict": {},
    }
    domain = legacy_to_domain(0, payload)
    assert domain.calibration_metadata is None
    assert domain.model_metadata is None


# ---------------------------------------------------------------------------
# Gap 1: _exp_to_domain replaces _legacy_payload + legacy_to_domain chain
# (verified through the public to_json / from_json API)
# ---------------------------------------------------------------------------


def _make_minimal_factual():
    """Build a minimal CalibratedExplanations with two FactualExplanations via WrapCalibratedExplainer."""
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier

    from calibrated_explanations import WrapCalibratedExplainer

    rng = np.random.default_rng(42)
    x = rng.random((20, 3))
    y = (x[:, 0] > 0.5).astype(int)
    model = DecisionTreeClassifier(random_state=42).fit(x[:10], y[:10])
    ce = WrapCalibratedExplainer(model)
    ce.calibrate(x[10:], y[10:])
    result = ce.explain_factual(x[:2])
    return result


def test_exp_to_domain_returns_explanation_instance():
    """`_exp_to_domain` must return an Explanation dataclass instance."""
    ce = _make_minimal_factual()
    domain = ce._exp_to_domain(ce.explanations[0])
    assert isinstance(domain, Explanation)


def test_exp_to_domain_populates_calibration_metadata():
    """`_exp_to_domain` must set calibration_metadata as a CalibrationDescriptor with a non-None method."""
    ce = _make_minimal_factual()
    domain = ce._exp_to_domain(ce.explanations[0])
    assert isinstance(domain.calibration_metadata, CalibrationDescriptor)
    assert domain.calibration_metadata.method is not None
    assert domain.calibration_metadata.method == ce.calibrated_explainer.mode


def test_exp_to_domain_populates_model_metadata():
    """`_exp_to_domain` must set model_metadata as a ModelDescriptor with the learner class name."""
    ce = _make_minimal_factual()
    domain = ce._exp_to_domain(ce.explanations[0])
    assert isinstance(domain.model_metadata, ModelDescriptor)
    assert domain.model_metadata.type == "DecisionTreeClassifier"


def test_to_json_task_field_matches_explainer_mode():
    """to_json 'task' field must match the explainer mode."""
    ce = _make_minimal_factual()
    payload = ce.to_json()
    for item in payload["explanations"]:
        assert item["task"] == ce.calibrated_explainer.mode


def test_to_json_explanation_type_is_factual():
    """to_json explanation_type must be 'factual' for explain_factual output."""
    ce = _make_minimal_factual()
    payload = ce.to_json()
    for item in payload["explanations"]:
        assert item["explanation_type"] == "factual"


def test_to_json_explanation_type_is_alternative():
    """to_json explanation_type must be 'alternative' for explore_alternatives output."""
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier

    from calibrated_explanations import WrapCalibratedExplainer

    rng = np.random.default_rng(42)
    x = rng.random((20, 3))
    y = (x[:, 0] > 0.5).astype(int)
    model = DecisionTreeClassifier(random_state=42).fit(x[:10], y[:10])
    ce = WrapCalibratedExplainer(model)
    ce.calibrate(x[10:], y[10:])
    result = ce.explore_alternatives(x[:1])
    payload = result.to_json()
    for item in payload["explanations"]:
        assert item["explanation_type"] == "alternative"


def test_to_json_metadata_contains_calibration_and_model_keys():
    """to_json output must include calibration_metadata and model_metadata under 'metadata'."""
    ce = _make_minimal_factual()
    payload = ce.to_json()
    for item in payload["explanations"]:
        assert item.get("metadata") is not None
        assert "calibration_metadata" in item["metadata"]
        assert "model_metadata" in item["metadata"]
        assert item["metadata"]["calibration_metadata"].get("method") is not None
        assert item["metadata"]["model_metadata"].get("type") == "DecisionTreeClassifier"


def test_to_json_then_from_json_roundtrip_preserves_typed_metadata():
    """to_json → from_json must produce Explanation objects with typed CalibrationDescriptor."""
    from calibrated_explanations.explanations.explanations import CalibratedExplanations

    ce = _make_minimal_factual()
    exported_json = ce.to_json()
    restored = CalibratedExplanations.from_json(exported_json)
    for domain_exp in restored.explanations:
        assert isinstance(domain_exp.calibration_metadata, CalibrationDescriptor)
        assert domain_exp.calibration_metadata.method == ce.calibrated_explainer.mode


def test_to_json_with_conjunctive_rules_uses_them():
    """When conjunctive_rules are present, to_json output must reflect them."""
    ce = _make_minimal_factual()
    exp = ce.explanations[0]
    exp.has_conjunctive_rules = True
    exp.conjunctive_rules = {
        "rule": ["x0 > 0.5 AND x1 <= 1.0"],
        "feature": [[0, 1]],
        "is_conjunctive": [True],
    }
    payload = ce.to_json()
    # The serialized explanation for index 0 must contain the conjunctive rule text
    first_item = payload["explanations"][0]
    rule_texts = [r["rule"] for r in first_item.get("rules", [])]
    assert any(
        "AND" in t for t in rule_texts
    ), "Conjunctive rule text must appear in to_json output when has_conjunctive_rules=True"


def test_to_json_does_not_call_legacy_payload(monkeypatch):
    """After Gap 1 closure, to_json must NOT call legacy_payload (the public wrapper)."""
    ce = _make_minimal_factual()
    calls = []
    original = ce.legacy_payload
    monkeypatch.setattr(ce, "legacy_payload", lambda exp: (calls.append(1), original(exp))[1])
    ce.to_json()
    assert calls == [], "legacy_payload must not be called by to_json after Gap 1 closure"


# ---------------------------------------------------------------------------
# Multiclass class_index bug fix
# ---------------------------------------------------------------------------


def test_multiclass_to_json_includes_class_index_in_metadata():
    """After the bug fix, multiclass to_json must include class_index in each explanation's metadata."""
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier

    from calibrated_explanations import WrapCalibratedExplainer

    rng = np.random.default_rng(0)
    x = rng.random((30, 3))
    y = (x[:, 0] * 3).astype(int).clip(0, 2)
    model = DecisionTreeClassifier(random_state=0).fit(x[:15], y[:15])
    ce = WrapCalibratedExplainer(model)
    ce.calibrate(x[15:], y[15:])
    result = ce.explain_factual(x[:1], multi_labels_enabled=True)
    payload = result.to_json()
    for item in payload["explanations"]:
        assert item.get("metadata") is not None
        assert "class_index" in item["metadata"]
