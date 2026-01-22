import numpy as np
import pytest

from calibrated_explanations.plugins.builtins import (
    derive_threshold_labels,
    LegacyPredictBridge,
    collection_to_batch,
)
from calibrated_explanations.utils.exceptions import ValidationError


def test_derive_threshold_labels_variants():
    lohi = derive_threshold_labels((1, 5))
    assert "1.00" in lohi[0]

    scalar = derive_threshold_labels(0.5)
    assert "0.50" in scalar[0]

    # non-numeric fallback
    fallback = derive_threshold_labels("abc")
    assert isinstance(fallback, tuple)


def test_legacy_predict_bridge_interval_invariant():
    class FakeExplainer:
        def predict(self, x, **kwargs):
            # interval call
            if kwargs.get("uq_interval"):
                preds = np.array([0.2, 0.3])
                low = np.array([0.5, 0.6])
                high = np.array([0.1, 0.2])
                return preds, (low, high)
            # classification calibrated path
            if kwargs.get("calibrated"):
                return np.array([0, 1])
            return np.array([0.2, 0.3])

    bridge = LegacyPredictBridge(FakeExplainer())
    with pytest.raises(ValidationError):
        bridge.predict(np.zeros((2, 1)), mode="regression", task="regression")


def test_collection_to_batch_minimal():
    class DummyCollection:
        def __init__(self):
            self.explanations = []
            self.mode = "regression"

    c = DummyCollection()
    batch = collection_to_batch(c)
    assert batch.container_cls is type(c)
