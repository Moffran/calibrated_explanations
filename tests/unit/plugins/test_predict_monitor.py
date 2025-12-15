import pytest
import numpy as np
from unittest.mock import create_autospec

from calibrated_explanations.plugins.predict_monitor import PredictBridgeMonitor
from calibrated_explanations.plugins.predict import PredictBridge

class DummyBridge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.predictions = {
            "predict": {"result": "predict"},
            "predict_interval": ("interval",),
            "predict_proba": (0.1, 0.9)
        }

    def predict(self, x, *, mode, task, bins=None):
        self.calls.append(("predict", (mode, task, bins)))
        return self.predictions["predict"]

    def predict_interval(self, x, *, task, bins=None):
        self.calls.append(("predict_interval", (task, bins)))
        return self.predictions["predict_interval"]

    def predict_proba(self, x, bins=None):
        self.calls.append(("predict_proba", (bins,)))
        return self.predictions["predict_proba"]

def test_predict_bridge_monitor_tracks_usage_and_passthrough():
    """Test that PredictBridgeMonitor correctly tracks bridge method calls and passes results."""
    bridge = DummyBridge()
    monitor = PredictBridgeMonitor(bridge)

    assert monitor.used is False

    predict_result = monitor.predict(np.array([[1.0]]), mode="factual", task="classification")
    interval_result = monitor.predict_interval(np.array([[1.0]]), task="classification")
    proba_result = monitor.predict_proba(np.array([[1.0]]))

    assert monitor.calls == ("predict", "predict_interval", "predict_proba")
    assert monitor.used is True
    
    # Ensure the wrapped bridge is called transparently.
    assert predict_result is bridge.predictions["predict"]
    assert interval_result is bridge.predictions["predict_interval"]
    assert proba_result is bridge.predictions["predict_proba"]
    
    assert bridge.calls[0][0] == "predict"
    assert bridge.calls[1][0] == "predict_interval"
    assert bridge.calls[2][0] == "predict_proba"

def test_predict_bridge_monitor_reset_usage():
    """Test that usage tracking can be reset."""
    bridge = create_autospec(PredictBridge, instance=True)
    monitor = PredictBridgeMonitor(bridge)

    payload = {"x": np.ones((2, 2))}
    monitor.predict(payload, mode="factual", task="classification")
    
    assert monitor.used
    assert len(monitor.calls) > 0

    monitor.reset_usage()
    assert monitor.calls == ()
    assert not monitor.used
