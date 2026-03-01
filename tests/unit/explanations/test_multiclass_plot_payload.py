"""Tests for the plotting payload builder used by MultiClassCalibratedExplanations."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from calibrated_explanations.explanations.explanations import MultiClassCalibratedExplanations


class SimpleExp:
    def __init__(self, idx: int, cls: int):
        self.index = idx
        self.class_id = cls

    def get_rules(self) -> Dict[str, Any]:
        # minimal factual dict shape expected by the payload builder
        return {
            "base_predict": [0.6, 0.6],
            "base_predict_low": [0.55, 0.55],
            "base_predict_high": [0.65, 0.65],
            "weight": [0.1, 0.2],
            "weight_low": [0.05, 0.15],
            "weight_high": [0.15, 0.25],
            "predict": [0.6, 0.4],
            "predict_low": [0.55, 0.35],
            "predict_high": [0.65, 0.45],
            "value": ["a", "b"],
            "rule": ["r1", "r2"],
            "feature": [0, 1],
            "feature_value": [None, None],
            "is_conjunctive": [False, False],
            "classes": [self.class_id, self.class_id],
        }

    def _rank_features(self, *args, **kwargs):
        # simple rank implementation for tests: return indices up to num_to_show
        num = (
            kwargs.get("num_to_show") if "num_to_show" in kwargs else (len(args[0]) if args else 0)
        )
        try:
            num = int(num)
        except Exception:
            num = 0
        return list(range(min(num, len(args[0]) if args else 0)))


def test_plot_factual_payload_via_public_api(monkeypatch):
    pytest.importorskip("matplotlib")
    # Build a small multiclass collection with a single instance containing two class explanations
    x = np.array([[1.0]])
    explanations = [
        {0: SimpleExp(0, 0), 1: SimpleExp(0, 1)},
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

    # Should call the plot wrapper once (one instance)
    assert len(calls) == 1
    args, kw = calls[0]
    # positional args correspond to: class_explanations_list, factual_values, predicts, feature_weights_list, features_list_to_plot, filter_top, colors, column_names_list
    assert len(args) >= 8
    class_explanations_list = args[0]
    _factual_values = args[1]
    _predicts = args[2]
    feature_weights_list = args[3]
    features_list_to_plot = args[4]
    filter_top = args[5]

    assert isinstance(class_explanations_list, list)
    assert len(feature_weights_list) == 2
    assert len(features_list_to_plot) == 2
    assert filter_top[0] == 2
