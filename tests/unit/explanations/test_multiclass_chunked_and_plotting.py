"""Tests for multiclass chunked JSON streaming and multiclass plotting orchestration."""

from __future__ import annotations

import json
import numpy as np

from calibrated_explanations.explanations.explanations import MultiClassCalibratedExplanations


class SimpleChunkExp:
    def __init__(self, index: int, klass: int):
        self.index = index
        self.klass = klass

    def get_class_labels(self):
        return {0: "a", 1: "b"}

    def _legacy_payload(self):
        return {"predict": 0.5}


def test_to_json_stream_chunked_emits_expected_chunks():
    # 2 instances x 2 classes = 4 items, chunk_size=2 -> 2 chunks
    x = np.array([[1.0], [2.0]])
    explanations = [
        {0: SimpleChunkExp(0, 0), 1: SimpleChunkExp(0, 1)},
        {0: SimpleChunkExp(1, 0), 1: SimpleChunkExp(1, 1)},
    ]

    class DummyExplainer:
        def __init__(self, num_features):
            self.num_features = num_features

    expl = DummyExplainer(x.shape[1])
    coll = MultiClassCalibratedExplanations(
        expl, x, bins=None, num_classes=2, explanations=explanations
    )

    gen = coll.to_json_stream(format="chunked", chunk_size=2)
    fragments = list(gen)

    # metadata + 2 chunk arrays + telemetry
    assert len(fragments) == 1 + 2 + 1

    # Validate the chunk arrays contain JSON arrays with two elements each
    meta = json.loads(fragments[0])
    assert "collection" in meta

    chunk1 = json.loads(fragments[1])
    chunk2 = json.loads(fragments[2])
    assert isinstance(chunk1, list) and isinstance(chunk2, list)
    assert len(chunk1) == 2 and len(chunk2) == 2


class SimplePlotExp:
    def __init__(self):
        self.index = 0
        self.prediction = 0.8

    def get_rules(self):
        return {
            "weight": [1],
            "weight_low": [0.5],
            "weight_high": [1.5],
            "value": [0.1],
            "predict": [0.8],
            "predict_low": [0.7],
            "predict_high": [0.9],
            "classes": ["a"],
            "rule": ["r1"],
            "feature": ["f1"],
            "feature_value": [1],
            "is_conjunctive": [False],
            "base_predict": [0.8],
            "base_predict_low": [0.7],
            "base_predict_high": [0.9],
        }

    def _check_preconditions(self):
        return True

    def _rank_features(self, *args, **kwargs):
        return [0]


def test_plot_factual_calls_plot_wrapper(monkeypatch):
    # Build a small multiclass collection with two instances
    x = np.array([[0.0], [1.0]])
    explanations = [
        {0: SimplePlotExp(), 1: SimplePlotExp()},
        {0: SimplePlotExp(), 1: SimplePlotExp()},
    ]

    class DummyExplainer:
        def __init__(self, num_features):
            self.num_features = num_features

    expl = DummyExplainer(x.shape[1])
    coll = MultiClassCalibratedExplanations(
        expl, x, bins=None, num_classes=2, explanations=explanations
    )

    calls = []

    def fake_plot_probabilistic(*args, **kwargs):
        calls.append((args, kwargs))

    # Monkeypatch the module-level wrapper used by MultiClass plotting
    import calibrated_explanations.explanations.explanations as impl

    monkeypatch.setattr(impl, "_plot_probabilistic_dict", fake_plot_probabilistic)

    coll.plot_factual(index=None)

    # Should call the plot wrapper once per instance and forward idx
    assert len(calls) == len(explanations)
    # idx kwarg should match the instance index (0 then 1)
    assert calls[0][1].get("idx") == 0
    assert calls[1][1].get("idx") == 1
