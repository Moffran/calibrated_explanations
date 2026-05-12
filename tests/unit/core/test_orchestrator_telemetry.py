import numpy as np

from calibrated_explanations.core.explain.orchestrator import ExplanationOrchestrator
from calibrated_explanations.logging import reset_module_config_manager


class DummyExplanation:
    def __init__(self, probs, deps=None):
        self.prediction = {"__full_probabilities__": probs}
        self.deps = deps or ("a", "b")

    def to_telemetry(self):
        return {"interval_dependencies": tuple(self.deps)}


def test_build_instance_telemetry_payload_includes_full_probs_when_diag_enabled(monkeypatch):
    monkeypatch.setenv("CE_TELEMETRY_DIAGNOSTIC_MODE", "true")
    reset_module_config_manager()
    arr = np.array([[0.2, 0.8], [0.6, 0.4]])
    expl = DummyExplanation(arr, deps=("interval_b",))

    payload = ExplanationOrchestrator.build_instance_telemetry_payload([expl])

    assert payload["full_probabilities_shape"] == (2, 2)
    assert "full_probabilities_summary" in payload
    assert np.allclose(payload["full_probabilities"], arr)
    assert payload["interval_dependencies"] == ("interval_b",)


def test_should_capture_ce_interval_plugin_fallbacks_in_config_manager_snapshot(
    monkeypatch,
) -> None:
    """ConfigManager.from_sources() must capture CE_INTERVAL_PLUGIN_FALLBACKS at construction time.

    This verifies that PredictionOrchestrator's ConfigManager.from_sources() call in __init__
    correctly snapshots the env so that CE_INTERVAL_PLUGIN_FALLBACKS is readable via
    the ConfigManager API (the public .env() method).
    """
    from calibrated_explanations.core.config_manager import ConfigManager

    monkeypatch.setenv("CE_INTERVAL_PLUGIN_FALLBACKS", "none")
    mgr = ConfigManager.from_sources()
    assert mgr.env("CE_INTERVAL_PLUGIN_FALLBACKS") == "none"

    # Without the env var the key must return None (default, not raise).
    monkeypatch.delenv("CE_INTERVAL_PLUGIN_FALLBACKS", raising=False)
    mgr_absent = ConfigManager.from_sources()
    assert mgr_absent.env("CE_INTERVAL_PLUGIN_FALLBACKS") is None
