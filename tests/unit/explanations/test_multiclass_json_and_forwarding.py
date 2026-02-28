"""Tests for multiclass JSON streaming and orchestration forwarding behavior."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from calibrated_explanations.core.explain.orchestrator import ExplanationOrchestrator
from calibrated_explanations.explanations.explanations import MultiClassCalibratedExplanations


class SimpleExplainer:
    def __init__(self):
        self.class_labels = {0: "a", 1: "b"}
        self.x_cal = np.array([[0.0], [0.0]])
        self.y_cal = np.array([0, 1])
        self.num_features = 1
        self.interval_summary = "ISUM"


class SimpleExp:
    def __init__(self, index: int, klass: int):
        self.index = index
        self.klass = klass

    def get_class_labels(self) -> Dict[int, str]:
        return {0: "a", 1: "b"}

    def to_narrative(self, *args, **kwargs):
        return {"text": f"instance={self.index},class={self.klass}"}


def test_to_json_stream_yields_expected_count():
    expl = SimpleExplainer()
    x = np.array([[1.0], [2.0]])
    explanations = [
        {0: SimpleExp(0, 0), 1: SimpleExp(0, 1)},
        {0: SimpleExp(1, 0), 1: SimpleExp(1, 1)},
    ]
    coll = MultiClassCalibratedExplanations(
        expl, x, bins=None, num_classes=2, explanations=explanations
    )

    gen = coll.to_json_stream(format="jsonl")
    fragments = list(gen)

    # metadata + 4 explanation lines (2 instances * 2 classes) + telemetry
    assert len(fragments) == 1 + 4 + 1


def test_orchestrator_forwards_features_and_interval(monkeypatch):
    # Prepare a minimal explainer-like object for the orchestrator
    class DummyExplainer:
        def __init__(self):
            self.y_cal = np.array([0, 1, 2])
            self.class_labels = {0: "a", 1: "b", 2: "c"}
            self.interval_summary = "FROM_EXPLAINER"
            self.num_features = 1

    expl = DummyExplainer()
    orch = ExplanationOrchestrator(expl)

    calls: List[Dict[str, Any]] = []

    def fake_legacy_explain(
        explainer, x, threshold, low_high_percentiles, bins, labels=None, **kwargs
    ):
        # record kwargs the orchestrator forwarded
        calls.append(dict(kwargs))
        # return a list of placeholder objects (one per instance)
        return [SimpleExp(i, int(labels[0])) for i in range(len(x))]

    # Monkeypatch the module used by orchestrator at call-time
    import calibrated_explanations.core.explain._legacy_explain as _legacy

    monkeypatch.setattr(_legacy, "explain", fake_legacy_explain)

    x = np.array([[0.0], [1.0]])
    # Invoke with explicit features_to_ignore and interval_summary
    features_to_ignore = [0]
    orch.invoke_factual(
        x,
        threshold=None,
        low_high_percentiles=None,
        bins=None,
        features_to_ignore=features_to_ignore,
        discretizer=None,
        _use_plugin=True,
        reject_policy=None,
        multi_labels_enabled=True,
        interval_summary="EXPLICIT_SUM",
    )

    # legacy_explain should have been called once per class (3 classes)
    assert len(calls) == 3
    for call in calls:
        assert call.get("features_to_ignore") == features_to_ignore
        # interval_summary should be forwarded from kwargs
        assert call.get("interval_summary") == "EXPLICIT_SUM"


# ---------------------------------------------------------------------------
# Multiclass public method coverage
# ---------------------------------------------------------------------------


class StubExp:
    """Minimal explanation stub supporting multiclass method dispatch."""

    def __init__(self, index: int, klass: int):
        self.index = index
        self.klass = klass
        self.conjunctions_added = False
        self.was_reset = False

    def get_class_labels(self) -> Dict[int, str]:
        return {0: "a", 1: "b"}

    def to_narrative(self, *args, **kwargs):
        return {"text": f"inst={self.index},cls={self.klass}"}

    def add_conjunctions(self, n_top_features=5, max_rule_size=2, **kw):
        self.conjunctions_added = True

    def remove_conjunctions(self):
        self.conjunctions_added = False

    def reset(self):
        self.was_reset = True

    def filter_rule_sizes(self, *, rule_sizes=None, size_range=None, copy=True):
        if copy:
            new = StubExp(self.index, self.klass)
            return new
        return self

    def filter_features(self, *, exclude_features=None, include_features=None, copy=True):
        if copy:
            new = StubExp(self.index, self.klass)
            return new
        return self

    def get_rules(self):
        return [{"feature": f"f{self.index}", "weight": 0.5}]

    def __str__(self):
        return f"StubExp(idx={self.index}, cls={self.klass})"


def make_multiclass(n_instances=2, n_classes=2):
    """Build a MultiClassCalibratedExplanations with stub data."""
    expl = SimpleExplainer()
    x = np.array([[float(i)] for i in range(n_instances)])
    explanations = [{c: StubExp(i, c) for c in range(n_classes)} for i in range(n_instances)]
    return MultiClassCalibratedExplanations(
        expl, x, bins=None, num_classes=n_classes, explanations=explanations
    )


def test_multiclass_repr():
    """Verify __repr__ includes class labels and explanations."""
    coll = make_multiclass()
    text = repr(coll)
    assert "MultiClassCalibratedExplanations" in text
    assert "StubExp" in text


def test_multiclass_iter():
    """Verify __iter__ yields one item per instance."""
    coll = make_multiclass(n_instances=3)
    items = list(coll)
    assert len(items) == 3


def test_multiclass_add_remove_conjunctions():
    """Verify conjunctions dispatch to each class explanation."""
    coll = make_multiclass()
    coll.add_conjunctions(n_top_features=3, max_rule_size=2)
    for class_dict in coll.explanations:
        for exp in class_dict.values():
            assert exp.conjunctions_added is True

    coll.remove_conjunctions()
    for class_dict in coll.explanations:
        for exp in class_dict.values():
            assert exp.conjunctions_added is False


def test_multiclass_reset():
    """Verify reset dispatches to each class explanation."""
    coll = make_multiclass()
    coll.reset()
    for class_dict in coll.explanations:
        for exp in class_dict.values():
            assert exp.was_reset is True


def test_multiclass_filter_rule_sizes_copy():
    """Verify filter_rule_sizes with copy=True returns new object."""
    coll = make_multiclass()
    filtered = coll.filter_rule_sizes(rule_sizes=[1], copy=True)
    assert filtered is not coll
    assert len(filtered.explanations) == len(coll.explanations)


def test_multiclass_filter_rule_sizes_inplace():
    """Verify filter_rule_sizes with copy=False modifies in-place."""
    coll = make_multiclass()
    result = coll.filter_rule_sizes(rule_sizes=[1], copy=False)
    assert result is coll


def test_multiclass_filter_features_copy():
    """Verify filter_features with copy=True returns new object."""
    coll = make_multiclass()
    filtered = coll.filter_features(exclude_features=["f0"], copy=True)
    assert filtered is not coll


def test_multiclass_filter_features_inplace():
    """Verify filter_features with copy=False modifies in-place."""
    coll = make_multiclass()
    result = coll.filter_features(exclude_features=["f0"], copy=False)
    assert result is coll


def test_multiclass_get_rules():
    """Verify get_rules returns per-instance per-class payloads."""
    coll = make_multiclass(n_instances=2, n_classes=2)
    rules = coll.get_rules()
    assert len(rules) == 2
    for inst_rules in rules:
        assert isinstance(inst_rules, dict)


def test_multiclass_to_narrative_dict():
    """Verify to_narrative with output_format='dict' returns list."""
    coll = make_multiclass()
    result = coll.to_narrative(output_format="dict")
    assert isinstance(result, list)
    assert len(result) == 2


def test_multiclass_to_narrative_text():
    """Verify to_narrative with output_format='text' returns string."""
    coll = make_multiclass()
    result = coll.to_narrative(output_format="text")
    assert isinstance(result, str)
    assert "Instance 0" in result
    assert "inst=" in result
