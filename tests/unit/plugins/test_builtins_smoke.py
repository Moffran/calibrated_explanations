import numpy as np


def test_derive_threshold_labels():
    from calibrated_explanations.plugins.builtins import derive_threshold_labels

    # Interval-like sequence
    lo_hi = derive_threshold_labels((0.2, 0.8))
    assert "<= Y" in lo_hi[0] or "Y <" in lo_hi[0]

    # Scalar threshold
    scalar = derive_threshold_labels(0.5)
    assert "Y <" in scalar[0] or "Target within threshold" in scalar[0]


def test_legacy_predict_bridge_predict():
    from calibrated_explanations.plugins.builtins import LegacyPredictBridge

    class DummyExplainer:
        def predict(self, x, uq_interval=False, bins=None, calibrated=False):
            # Return (preds, (low, high)) when uq_interval True
            preds = np.asarray([[0.2, 0.8] for _ in x])
            low = np.asarray([[0.1, 0.7] for _ in x])
            high = np.asarray([[0.3, 0.9] for _ in x])
            if uq_interval:
                return preds, (low, high)
            return preds

        def predict_proba(self, x, uq_interval=False, calibrated=False, bins=None):
            return np.asarray([[0.2, 0.8] for _ in x])

    bridge = LegacyPredictBridge(DummyExplainer())
    x = [[1.0, 2.0]]
    payload = bridge.predict(x, mode="classification", task="classification")
    assert "predict" in payload and payload["predict"].shape[0] == 1
    assert payload.get("classes") is not None
