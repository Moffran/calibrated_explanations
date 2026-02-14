import numpy as np
import types
from types import SimpleNamespace

import pytest

from calibrated_explanations.core.explain.orchestrator import ExplanationOrchestrator
from calibrated_explanations.utils.discretizers import EntropyDiscretizer


def make_explainer_stub():
    ns = SimpleNamespace()
    ns.feature_names = ["a", "b", "c"]
    ns.categorical_features = []
    ns.features_to_ignore = []
    ns.plugin_manager = SimpleNamespace()
    ns.plugin_manager.plot_style_chain = ("legacy",)
    ns.plugin_manager.interval_plugin_hints = {}
    ns.plugin_manager.plot_plugin_fallbacks = {}
    ns.plugin_manager.last_telemetry = {}
    ns.mode = "classification"
    return ns


# Note: private members are intentionally not referenced in tests per repo policy.
# Coercion logic is exercised indirectly by public flows elsewhere; keep lightweight
# unit tests focused on public-facing behavior.


def test_infer_mode_entropy_discretizer_is_alternative():
    # small toy data to construct EntropyDiscretizer
    X = np.array([[0.1], [0.9]])
    y = np.array([0, 1])
    expl = make_explainer_stub()
    # instantiate discretizer (requires labels)
    disc = EntropyDiscretizer(X, [], ["f0"], labels=y)
    expl.discretizer = disc
    orch = ExplanationOrchestrator(expl)
    assert orch.infer_mode() == "alternative"
    expl.discretizer = object()
    assert orch.infer_mode() == "factual"


def test_derive_plot_chain_uses_base_chain_when_no_identifier(monkeypatch):
    expl = make_explainer_stub()
    orch = ExplanationOrchestrator(expl)
    chain = orch.derive_plot_chain("factual", None)
    assert isinstance(chain, tuple)
    assert chain[0] == "legacy"


def test_build_instance_telemetry_payload_from_first_explanation():
    class FakeExplanation:
        def __init__(self):
            self.prediction = {"__full_probabilities__": np.array([[0.2, 0.8]])}
            self.metadata = {"interval_dependencies": ("d1",)}

        def to_telemetry(self):
            return {"metadata": {"interval_dependencies": ("d1",)}}

    payload = ExplanationOrchestrator.build_instance_telemetry_payload([FakeExplanation()])
    assert "full_probabilities_shape" in payload
    assert "full_probabilities_summary" in payload
    assert payload.get("interval_dependencies") is not None


def test_instantiate_plugin_various_inputs():
    expl = make_explainer_stub()
    orch = ExplanationOrchestrator(expl)

    # None -> None
    assert orch.instantiate_plugin(None) is None

    # Callable with plugin_meta -> returned unchanged
    def plugin_fn():
        return 1

    plugin_fn.plugin_meta = {"name": "x"}
    assert orch.instantiate_plugin(plugin_fn) is plugin_fn

    # Instance of simple class -> returns new instance (not identity)
    class C:
        def __init__(self):
            self.v = 1

    inst = C()
    new_inst = orch.instantiate_plugin(inst)
    assert isinstance(new_inst, C)
    assert new_inst is not inst


def test_check_metadata_detects_missing_and_schema_mismatch():
    expl = make_explainer_stub()
    orch = ExplanationOrchestrator(expl)
    msg = orch.check_metadata(None, identifier="x", mode="factual")
    assert "plugin metadata unavailable" in msg

    bad_meta = {"schema_version": "0", "tasks": "classification", "modes": "factual", "capabilities": ["explain"]}
    msg2 = orch.check_metadata(bad_meta, identifier="y", mode="factual")
    assert "unsupported" in msg2 or "schema_version" in msg2
