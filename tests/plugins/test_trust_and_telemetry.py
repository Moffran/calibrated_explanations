import types
import numpy as np
import logging
import pytest

from calibrated_explanations.plugins import register_explanation_plugin, find_explanation_descriptor
from calibrated_explanations.plugins.manager import PluginManager
from calibrated_explanations.core.explain.orchestrator import ExplanationOrchestrator
from calibrated_explanations.utils.exceptions import ConfigurationError


class DummyPlugin:
    plugin_meta = {
        "name": "test.untrusted.plugin",
        "schema_version": 1,
        "version": "0",
        "provider": "tests",
        "modes": ("factual",),
        "tasks": ("classification",),
        "capabilities": ("explain",),
        "dependencies": (),
        "trusted": False,
    }

    def supports_mode(self, mode: str, *, task: str) -> bool:
        return True

    def initialize(self, context):
        return None

    def explain_batch(self, x, request):
        raise NotImplementedError()


def test_resolve_preferred_untrusted_plugin_raises_configuration_error():
    """When the preferred identifier is untrusted, resolution should fail."""
    identifier = "test.untrusted.plugin"
    # register untrusted plugin explicitly (manual source)
    register_explanation_plugin(
        identifier, DummyPlugin(), metadata=DummyPlugin.plugin_meta, source="manual"
    )
    desc = find_explanation_descriptor(identifier)
    assert desc is not None and desc.trusted is False

    # Minimal explainer shim with plugin_manager and required attributes
    explainer = types.SimpleNamespace()
    explainer.mode = "classification"
    explainer.plugin_manager = PluginManager(explainer)
    # Force fallback chain and preferred identifier for 'factual' mode
    explainer.plugin_manager.explanation_plugin_fallbacks["factual"] = (identifier,)
    explainer.plugin_manager.explanation_preferred_identifier["factual"] = identifier

    orch = ExplanationOrchestrator(explainer)
    try:
        orch.resolve_plugin("factual")
        raised = False
    except ConfigurationError:
        raised = True
    assert raised, "Expected ConfigurationError when preferred plugin is untrusted"


def test_build_instance_telemetry_payload_includes_probability_cube_shape():
    class DummyExplanationWithTelemetry:
        def to_telemetry(self):
            return {"__full_probabilities__": np.zeros((2, 3, 4))}

    container = [DummyExplanationWithTelemetry()]
    payload = ExplanationOrchestrator.build_instance_telemetry_payload(container)
    assert isinstance(payload, dict)
    cube = payload.get("__full_probabilities__")
    assert cube is not None
    assert getattr(cube, "shape", None) == (2, 3, 4)


def test_governance_logging_for_untrusted_plugin(caplog, monkeypatch):
    """Test that governance logs are emitted for untrusted plugin warnings."""
    from calibrated_explanations.plugins.registry import _warn_untrusted_plugin

    # Mock the warnings.warn to avoid actual warnings
    warned = []
    original_warn = __import__('warnings').warn
    def mock_warn(*args, **kwargs):
        warned.append(args)
    monkeypatch.setattr('warnings.warn', mock_warn)

    # Clear any previous warnings
    from calibrated_explanations.plugins.registry import _WARNED_UNTRUSTED
    _WARNED_UNTRUSTED.clear()

    meta = {"name": "test.untrusted", "provider": "test"}
    with caplog.at_level(logging.INFO):
        _warn_untrusted_plugin(meta, source="test")

    # Check that governance log was emitted
    governance_logs = [record for record in caplog.records if 'governance' in record.name]
    assert len(governance_logs) == 1
    log = governance_logs[0]
    assert log.name == "calibrated_explanations.governance.plugins"
    assert "Plugin trust decision: skipped untrusted plugin" in log.message
    assert log.plugin_identifier == "test.untrusted"  # Should be set by context
