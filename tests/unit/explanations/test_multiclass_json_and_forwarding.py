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
