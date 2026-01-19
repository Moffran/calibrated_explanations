import sys
import types
import numpy as np
import pytest

from calibrated_explanations.plugins.builtins import (
    derive_threshold_labels,
    LegacyPredictBridge,
    collection_to_batch,
    _supports_calibrated_explainer,
)
from calibrated_explanations.utils.exceptions import ValidationError


def test_derive_threshold_labels_variants_and_fallback():
    # interval
    labels = derive_threshold_labels((10, 20))
    assert isinstance(labels, tuple) and "10.00" in labels[0]

    # scalar
    scalar = derive_threshold_labels(5)
    assert scalar[0].startswith("Y < 5.00")

    # non-numeric fallback
    fallback = derive_threshold_labels("abc")
    assert isinstance(fallback, tuple)


def test_derive_threshold_labels_expected_outputs():
    pos, neg = derive_threshold_labels((10, 20))
    assert "10.00" in pos and "20.00" in pos
    pos2, neg2 = derive_threshold_labels(5)
    assert pos2.startswith("Y < 5.00")


def test_supports_calibrated_explainer_true():
    # Create a fake module and class at the expected import path
    mod_name = "calibrated_explanations.core.calibrated_explainer"
    fake_mod = types.ModuleType(mod_name)

    class CalibratedExplainer:
        pass

    fake_mod.CalibratedExplainer = CalibratedExplainer
    sys.modules[mod_name] = fake_mod

    inst = CalibratedExplainer()
    assert _supports_calibrated_explainer(inst) is True


def test_collection_to_batch_variants():
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
    assert batch.collection_metadata["container"] is col
    assert len(batch.instances) == 2


def test_legacy_predict_bridge_classification_and_regression():
    class DummyExplainer:
        def __init__(self, preds, low=None, high=None, classes=None):
            self.preds = np.asarray(preds)
            self.low = None if low is None else np.asarray(low)
            self.high = None if high is None else np.asarray(high)
            self.classes = None if classes is None else np.asarray(classes)

        def predict(self, x, uq_interval=False, calibrated=False, bins=None, threshold=None):
            if calibrated:
                return self.classes
            if self.low is not None and self.high is not None:
                return (self.preds, (self.low, self.high))
            return self.preds

    # regression happy path
    expl = DummyExplainer(preds=[1.0, 2.0], low=[0.9, 1.9], high=[1.1, 2.1])
    bridge = LegacyPredictBridge(expl)
    out = bridge.predict(np.array([[0]]), mode="regression", task="regression")
    assert "predict" in out
    assert np.allclose(out["predict"], [1.0, 2.0])
    assert "low" in out and "high" in out

    # classification includes classes
    expl2 = DummyExplainer(preds=[0.1, 0.9], low=[0.0, 0.0], high=[1.0, 1.0], classes=[0, 1])
    bridge2 = LegacyPredictBridge(expl2)
    out2 = bridge2.predict(np.array([[0]]), mode="classification", task="classification")
    assert "classes" in out2


def test_legacy_predict_bridge_invalid_intervals_raise():
    class BadExplainer:
        def predict(self, x, uq_interval=False, calibrated=False, bins=None, threshold=None):
            if uq_interval:
                # low > high to trigger ValidationError
                return np.array([0.5, 0.6]), (np.array([0.9, 0.2]), np.array([0.1, 0.3]))
            return np.array([0.5, 0.6])

    bridge = LegacyPredictBridge(BadExplainer())
    with pytest.raises(ValidationError):
        bridge.predict(np.zeros((2, 1)), mode="regression", task="regression")


class DummyExplainer:
    def __init__(self, interval_ok=True, classification=False):
        self.interval_ok = interval_ok
        self.classification = classification

    def predict(self, x, uq_interval=False, calibrated=False, bins=None, **kwargs):
        # When asking for intervals, return (preds, (low, high))
        if uq_interval:
            preds = np.asarray([0.5] * (len(x) if hasattr(x, "__len__") else 1))
            if self.interval_ok:
                low = np.asarray([0.4] * len(preds))
                high = np.asarray([0.6] * len(preds))
            else:
                # produce invalid interval low > high
                low = np.asarray([1.0] * len(preds))
                high = np.asarray([0.0] * len(preds))
            return preds, (low, high)
        if calibrated:
            # classification classes
            return np.asarray([1] * (len(x) if hasattr(x, "__len__") else 1))
        return np.asarray([0.5] * (len(x) if hasattr(x, "__len__") else 1))


def test_derive_threshold_labels():
    assert derive_threshold_labels((1, 2))[0].startswith("1.00")
    assert "Outside" in derive_threshold_labels((1, 2))[1]
    assert derive_threshold_labels(0.3)[0].startswith("Y < 0.30")
    assert derive_threshold_labels("not-a-number")[0] == "Target within threshold"


def test_legacy_predict_bridge_valid_and_invalid():
    good = DummyExplainer(interval_ok=True)
    bridge = LegacyPredictBridge(good)
    payload = bridge.predict(np.array([[1.0]]), mode="regression", task="regression")
    assert "predict" in payload
    assert "low" in payload and "high" in payload

    bad = DummyExplainer(interval_ok=False)
    bridge_bad = LegacyPredictBridge(bad)
    try:
        bridge_bad.predict(np.array([[1.0]]), mode="regression", task="regression")
        raised = False
    except ValidationError:
        raised = True
    assert raised


def test_legacy_predict_bridge_classification_classes_present():
    cl = DummyExplainer(interval_ok=True)
    bridge = LegacyPredictBridge(cl)
    payload = bridge.predict(np.array([[1.0]]), mode="classification", task="classification")
    assert "classes" in payload


def test_supports_calibrated_explainer_false_for_plain_object():
    assert _supports_calibrated_explainer(object()) is False
