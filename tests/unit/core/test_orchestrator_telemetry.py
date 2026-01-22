import numpy as np

from calibrated_explanations.core.explain.orchestrator import ExplanationOrchestrator


class DummyExplanation:
    def __init__(self, probs, deps=None):
        self.prediction = {"__full_probabilities__": probs}
        self.deps = deps or ("a", "b")

    def to_telemetry(self):
        return {"interval_dependencies": tuple(self.deps)}


def test_build_instance_telemetry_payload_includes_full_prob_summary_and_deps():
    # create a small full probability cube
    arr = np.array([[0.1, 0.9], [0.8, 0.2]])
    expl = DummyExplanation(arr, deps=("interval_a",))
    payload = ExplanationOrchestrator.build_instance_telemetry_payload([expl])

    # shape and summary should be present
    assert payload.get("full_probabilities_shape") == (2, 2)
    assert "full_probabilities_summary" in payload

    # interval dependencies should be preserved from to_telemetry
    assert payload.get("interval_dependencies") == ("interval_a",)


def test_build_instance_telemetry_payload_includes_full_probs_when_diag_enabled(monkeypatch):
    monkeypatch.setenv("CE_TELEMETRY_DIAGNOSTIC_MODE", "true")
    arr = np.array([[0.2, 0.8], [0.6, 0.4]])
    expl = DummyExplanation(arr, deps=("interval_b",))

    payload = ExplanationOrchestrator.build_instance_telemetry_payload([expl])

    assert payload["full_probabilities_shape"] == (2, 2)
    assert "full_probabilities_summary" in payload
    assert np.allclose(payload["full_probabilities"], arr)
    assert payload["interval_dependencies"] == ("interval_b",)
