import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


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

    bad_meta = {
        "schema_version": "0",
        "tasks": "classification",
        "modes": "factual",
        "capabilities": ["explain"],
    }
    msg2 = orch.check_metadata(bad_meta, identifier="y", mode="factual")
    assert "unsupported" in msg2 or "schema_version" in msg2


def make_invoke_explainer_stub():
    """Extended stub with attributes required for invoke telemetry tests.

    Uses a MagicMock for plugin_manager so that runtime attribute access (e.g.
    `initialize_orchestrators()`) succeeds without an explicit SimpleNamespace entry.
    """
    ns = make_explainer_stub()
    pm = MagicMock()
    pm.get_bridge_monitor = MagicMock(return_value=None)
    pm.telemetry_interval_sources = {"default": "core.interval.test"}
    pm.interval_plugin_hints = {"factual": ("core.interval.test",)}
    pm.last_explanation_mode = None
    pm.plot_plugin_fallbacks = {}
    pm.last_telemetry = {}
    ns.plugin_manager = pm
    ns.preprocessor_metadata = None
    ns.interval_summary = None
    ns.latest_explanation = None
    ns.initialized = True
    return ns


def test_should_include_interval_dependencies_in_batch_telemetry_when_plugin_provides_hints():
    """ADR-026 gap 2: batch-level last_telemetry must include interval_dependencies key."""
    expl = make_invoke_explainer_stub()
    orch = ExplanationOrchestrator(expl)

    mock_plugin = MagicMock()
    mock_batch = MagicMock()
    mock_batch.collection_metadata = {}
    mock_container = MagicMock()
    mock_batch.container_cls = mock_container
    mock_result = MagicMock()
    mock_container.from_batch.return_value = mock_result
    mock_plugin.explain_batch.return_value = mock_batch

    with (
        patch.object(orch, "ensure_plugin", return_value=(mock_plugin, "core.test")),
        patch("calibrated_explanations.core.explain.orchestrator.validate_explanation_batch"),
        patch.object(ExplanationOrchestrator, "build_instance_telemetry_payload", return_value={}),
    ):
        orch.invoke(
            mode="factual",
            x=np.array([[1, 2, 3]]),
            threshold=None,
            low_high_percentiles=None,
            bins=None,
            features_to_ignore=None,
        )

    telemetry = expl.plugin_manager.last_telemetry
    assert "interval_dependencies" in telemetry
    assert telemetry["interval_dependencies"] == ("core.interval.test",)
