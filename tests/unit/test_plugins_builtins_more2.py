import numpy as np
import pytest

from calibrated_explanations.plugins.builtins import (
    derive_threshold_labels,
    LegacyPredictBridge,
    collection_to_batch,
)
from calibrated_explanations.utils.exceptions import ValidationError


def test_derive_threshold_labels_interval_sequence():
    labels = derive_threshold_labels((10, 20))
    assert labels[0].startswith("10.00 <= Y < 20.00")


def test_derive_threshold_labels_scalar():
    labels = derive_threshold_labels(5)
    assert labels == ("Y < 5.00", "Y ≥ 5.00")


def test_derive_threshold_labels_non_numeric():
    labels = derive_threshold_labels("n/a")
    assert labels == ("Target within threshold", "Outside threshold")


class DummyExplainer:
    def __init__(self, preds, low=None, high=None, classes=None):
        self.preds = np.asarray(preds)
        self.low = None if low is None else np.asarray(low)
        self.high = None if high is None else np.asarray(high)
        self.classes = None if classes is None else np.asarray(classes)

    def predict(self, x, uq_interval=False, calibrated=False, bins=None, threshold=None):
        # calibrated=True path used by LegacyPredictBridge for classes
        if calibrated:
            return self.classes
        if self.low is not None and self.high is not None:
            return (self.preds, (self.low, self.high))
        return self.preds


def test_legacy_predict_bridge_regression_valid():
    expl = DummyExplainer(preds=[1.0, 2.0], low=[0.9, 1.9], high=[1.1, 2.1])
    bridge = LegacyPredictBridge(expl)
    out = bridge.predict(np.array([[0]]), mode="regression", task="regression")
    assert "predict" in out
    assert np.allclose(out["predict"], [1.0, 2.0])
    assert "low" in out and "high" in out


def test_legacy_predict_bridge_regression_invalid_interval_raises():
    # Create low > high to trigger ValidationError
    expl = DummyExplainer(preds=[1.0], low=[2.0], high=[1.0])
    bridge = LegacyPredictBridge(expl)
    with pytest.raises(ValidationError):
        bridge.predict(np.array([[0]]), mode="regression", task="regression")


def test_legacy_predict_bridge_classification_includes_classes():
    expl = DummyExplainer(preds=[0.1, 0.9], low=[0.0, 0.0], high=[1.0, 1.0], classes=[0, 1])
    bridge = LegacyPredictBridge(expl)
    out = bridge.predict(np.array([[0]]), mode="classification", task="classification")
    assert "classes" in out


def test_collection_to_batch_basic():
    class DummyCollection:
        def __init__(self):
            self.explanations = [object(), object()]
            self.mode = "regression"
            self.calibrated_explainer = None
            self.x_test = None
            self.y_threshold = None
            self.bins = None
            self.features_to_ignore = None
            self.low_high_percentiles = None
            self.feature_filter_per_instance_ignore = None
            self.filter_telemetry = None

    col = DummyCollection()
    batch = collection_to_batch(col)
    # metadata should include our container
    assert batch.collection_metadata["container"] is col
    assert len(batch.instances) == 2
