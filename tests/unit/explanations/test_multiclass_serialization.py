"""Unit tests for multiclass collection serialization and indexing."""

from __future__ import annotations

import pickle
from typing import Dict

import numpy as np

from calibrated_explanations.explanations.explanations import MultiClassCalibratedExplanations


class SimpleExplainer:
    def __init__(self):
        self.class_labels = {0: "a", 1: "b"}
        self.x_cal = np.array([[0.0]])
        self.y_cal = np.array([0, 1])
        self.num_features = 1


class SimpleExp:
    def __init__(self, index: int, klass: int):
        self.index = index
        self.klass = klass

    def get_class_labels(self) -> Dict[int, str]:
        return {0: "a", 1: "b"}

    def to_narrative(self, *args, **kwargs):
        return {"text": f"instance={self.index},class={self.klass}"}


def test_multiclass_pickle_roundtrip():
    expl = SimpleExplainer()
    x = np.array([[1.0], [2.0]])
    # two instances, each with two class explanations
    explanations = [
        {0: SimpleExp(0, 0), 1: SimpleExp(0, 1)},
        {0: SimpleExp(1, 0), 1: SimpleExp(1, 1)},
    ]
    coll = MultiClassCalibratedExplanations(
        expl, x, bins=None, num_classes=2, explanations=explanations
    )

    data = pickle.dumps(coll)
    restored = pickle.loads(data)

    assert isinstance(restored, MultiClassCalibratedExplanations)
    assert len(restored.explanations) == 2
    assert set(restored.explanations[0].keys()) == {0, 1}


def test_getitem_accepts_numpy_integer():
    expl = SimpleExplainer()
    x = np.array([[1.0]])
    explanations = [{0: SimpleExp(0, 0), 1: SimpleExp(0, 1)}]
    coll = MultiClassCalibratedExplanations(
        expl, x, bins=None, num_classes=2, explanations=explanations
    )

    # np.integer should be accepted for class_idx
    ex = coll[(0, np.int64(1))]
    assert ex is not None
    assert getattr(ex, "klass", 1) == 1
