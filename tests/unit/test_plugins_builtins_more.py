import numpy as np
import pytest

from calibrated_explanations.plugins.builtins import derive_threshold_labels, LegacyPredictBridge
from calibrated_explanations.utils.exceptions import ValidationError
from calibrated_explanations.plugins.builtins import collection_to_batch


def test_derive_threshold_labels_fallback():
    # non-numeric threshold should return fallback labels
    pos, neg = derive_threshold_labels("not-a-number")
    assert "Target" in pos or "threshold" in pos


def test_legacy_predict_bridge_classification_returns_classes():
    class _ExplainerForClass:
        def predict(self, x, uq_interval=False, bins=None, calibrated=False):
            if uq_interval:
                return np.array([0.5]), (np.array([0.1]), np.array([0.9]))
            if calibrated:
                return np.array(["cls1", "cls2"])
            return np.array([0.5])

    expl = _ExplainerForClass()
    bridge = LegacyPredictBridge(expl)
    out = bridge.predict([[1]], mode="classification", task="classification")
    assert "classes" in out


def test_collection_to_batch_basic():
    class C:
        pass

    col = C()
    col.explanations = [object()]
    col.mode = "regression"
    batch = collection_to_batch(col)
    assert batch.instances and batch.collection_metadata["mode"] == "regression"


def test_derive_threshold_labels_interval_and_scalar():
    lohi = (10, 20)
    pos, neg = derive_threshold_labels(lohi)
    assert "10.00" in pos and "20.00" in pos

    pos2, neg2 = derive_threshold_labels(5)
    assert pos2.startswith("Y < 5.00")


class _ExplainerSimple:
    def predict(self, x, uq_interval=False, bins=None, calibrated=False):
        if uq_interval:
            # return preds, (low, high)
            return np.array([0.5]), (np.array([0.1]), np.array([0.9]))
        if calibrated:
            return np.array(["a", "b"])
        return np.array([0.5])


def test_legacy_predict_bridge_regression_happy_path():
    expl = _ExplainerSimple()
    bridge = LegacyPredictBridge(expl)
    out = bridge.predict([[1]], mode="regression", task="regression")
    assert "predict" in out and out["predict"].shape[0] == 1
    assert "low" in out and "high" in out


class _ExplainerBadInterval:
    def predict(self, x, uq_interval=False, bins=None, calibrated=False):
        if uq_interval:
            # low > high to trigger ValidationError
            return np.array([0.5, 0.6]), (np.array([0.9, 0.2]), np.array([0.1, 0.3]))
        return np.array([0.5, 0.6])


def test_legacy_predict_bridge_invalid_interval_raises():
    expl = _ExplainerBadInterval()
    bridge = LegacyPredictBridge(expl)
    with pytest.raises(ValidationError):
        bridge.predict([[1], [2]], mode="regression", task="regression")
