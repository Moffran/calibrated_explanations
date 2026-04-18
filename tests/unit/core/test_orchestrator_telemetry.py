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
