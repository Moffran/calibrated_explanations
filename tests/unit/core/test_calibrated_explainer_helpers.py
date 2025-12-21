import numpy as np


from calibrated_explanations.plugins.predict_monitor import PredictBridgeMonitor
from calibrated_explanations.plugins import EXPLANATION_PROTOCOL_VERSION


class DummyBridge:
    def predict(self, x, *, mode, task, bins=None):
        return {"ok": True, "x": x}

    def predict_interval(self, x, *, task, bins=None):
        return (x, x)

    def predict_proba(self, x, bins=None):
        return np.asarray(x) * 0 + 0.5


def test_predict_bridge_monitor_records_calls():
    """Test that PredictBridgeMonitor records method calls."""
    bridge = DummyBridge()
    monitor = PredictBridgeMonitor(bridge)
    assert not monitor.used
    _ = monitor.predict(np.array([1, 2, 3]), mode="m", task="t")
    assert "predict" in monitor.calls
    assert monitor.used
    _ = monitor.predict_interval(np.array([1, 2]), task="t")
    _ = monitor.predict_proba(np.array([1, 2]))
    assert tuple(monitor.calls).count("predict") == 1
    assert "predict_interval" in monitor.calls
    assert "predict_proba" in monitor.calls


def test_check_explanation_runtime_metadata_various(explainer_factory):
    """Test ExplanationOrchestrator metadata validation through delegating method."""
    orch = explainer_factory()._explanation_orchestrator

    # None metadata
    msg = orch._check_metadata(None, identifier=None, mode="factual")
    assert "metadata unavailable" in msg

    # bad schema
    bad_meta = {"schema_version": "bad"}
    msg = orch._check_metadata(bad_meta, identifier=None, mode="factual")
    assert "unsupported" in msg

    # missing tasks
    good_schema = {"schema_version": EXPLANATION_PROTOCOL_VERSION}
    meta_missing_tasks = dict(good_schema)
    msg = orch._check_metadata(meta_missing_tasks, identifier="id", mode="factual")
    assert "missing tasks declaration" in msg

    # tasks incompatible
    meta_tasks = dict(good_schema)
    meta_tasks["tasks"] = "regression"
    msg = orch._check_metadata(meta_tasks, identifier="id", mode="factual")
    assert "does not support task" in msg

    # missing modes
    meta_tasks["tasks"] = "both"
    msg = orch._check_metadata({**meta_tasks}, identifier="id", mode="factual")
    assert "missing modes declaration" in msg

    # modes not matching
    meta_ok = {
        "schema_version": EXPLANATION_PROTOCOL_VERSION,
        "tasks": "both",
        "modes": ("fast",),
    }
    msg = orch._check_metadata(meta_ok, identifier="id", mode="factual")
    assert "does not declare mode" in msg

    # missing capabilities
    meta_ok["modes"] = ("factual",)
    meta_ok["capabilities"] = []
    msg = orch._check_metadata(meta_ok, identifier="id", mode="factual")
    assert "missing required capabilities" in msg

    # valid metadata
    meta_valid = {
        "schema_version": EXPLANATION_PROTOCOL_VERSION,
        "tasks": ("both",),
        "modes": ("factual", "alternative"),
        "capabilities": [
            "explain",
            "explanation:classification",
            "explanation:factual",
            "task:classification",
        ],
    }
    assert orch._check_metadata(meta_valid, identifier="id", mode="factual") is None


def test_slice_threshold_and_bins():
    """Test threshold and bins slicing behavior through explain helpers.

    Tests should call explain module functions directly,
    not private methods on CalibratedExplainer.
    """
    from calibrated_explanations.core.explain._helpers import slice_threshold, slice_bins

    # threshold None or scalar
    assert slice_threshold(None, 0, 1, 1) is None
    assert slice_threshold(0.5, 0, 1, 1) == 0.5

    # list slicing
    th = [1, 2, 3, 4]
    assert slice_threshold(th, 1, 3, 4) == [2, 3]

    # numpy array slicing
    arr = np.array([10, 20, 30])
    assert np.array_equal(slice_threshold(arr, 0, 2, 3), np.array([10, 20]))

    # bins None
    assert slice_bins(None, 0, 1) is None
    bins = np.array([0, 1, 2])
    assert np.array_equal(slice_bins(bins, 1, 3), np.array([1, 2]))


def test_compute_weight_delta_basic():
    """Test weight delta computation through explain helpers.

    Tests should call explain module functions directly,
    not private methods on CalibratedExplainer.
    """
    from calibrated_explanations.core.explain._helpers import compute_weight_delta

    # scalar baseline vs array perturbed
    res = compute_weight_delta(1.0, np.array([0.5, 1.5]))
    assert np.allclose(res, np.array([0.5, -0.5]))

    # matching shapes
    base = np.array([2.0, 3.0])
    pert = np.array([1.0, 5.0])
    res = compute_weight_delta(base, pert)
    assert np.allclose(res, np.array([1.0, -2.0]))
