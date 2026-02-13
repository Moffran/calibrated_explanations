import os
from types import MappingProxyType

import numpy as np
import pytest

from calibrated_explanations.plugins import (
    find_explanation_descriptor,
    find_explanation_plugin_trusted,
    register_explanation_plugin,
    clear_explanation_plugins,
    ensure_builtin_plugins,
)
from calibrated_explanations.plugins.manager import PluginManager
from calibrated_explanations.core.explain.orchestrator import ExplanationOrchestrator
from calibrated_explanations.utils.exceptions import ConfigurationError
from calibrated_explanations.plugins.explanations import ExplainerHandle
from calibrated_explanations.plugins import ExplanationBatch
from calibrated_explanations.explanations.explanations import CalibratedExplanations


def test_core_fast_plugin_registered_and_trusted():
    """`core.explanation.fast` must be registered as an in-tree builtin and trusted."""
    ensure_builtin_plugins()
    desc = find_explanation_descriptor("core.explanation.fast")
    assert desc is not None, "core.explanation.fast not registered"
    assert desc.trusted is True, "core.explanation.fast should be trusted in-tree"
    plugin = find_explanation_plugin_trusted("core.explanation.fast")
    assert plugin is not None




def test_registry_respects_denylist_on_resolution_and_explicit_override_allows_untrusted():
    """When identifier is denied via CE_DENY_PLUGIN resolution must fail; explicit override warns but allows."""
    # Ensure a clean registry
    clear_explanation_plugins()
    ensure_builtin_plugins()

    class DummyPlugin:
        plugin_meta = {
            "name": "test.explain.plugin",
            "schema_version": 1,
            "version": "0",
            "provider": "test",
            "capabilities": ("explain", "explanation:fast", "task:classification"),
            "modes": ("fast",),
            "tasks": ("classification",),
            "dependencies": (),
            "trusted": False,
        }

        def supports_mode(self, mode, *, task: str):
            return True

        def initialize(self, context):
            return None

        def explain_batch(self, x, request):
            return ExplanationBatch(
                container_cls=CalibratedExplanations,
                explanation_cls=object,
                instances=[{"explanation": object()}],
                collection_metadata={},
            )

    identifier = "test.explain.plugin"
    # Register the plugin (untrusted by default since not builtin and no operator trust)
    register_explanation_plugin(identifier, DummyPlugin(), source="external")

    # Create dummy explainer with plugin manager and orchestrator
    class DummyExplainerObj:
        def __init__(self):
            self.mode = "classification"
            self.feature_names = []
            self.categorical_features = []
            self.categorical_labels = {}
            self.discretizer = None
            self.plugin_manager = PluginManager(self)

    expl = DummyExplainerObj()

    # Explicit override set to identifier
    expl.plugin_manager.explanation_plugin_overrides = {"fast": identifier}

    orch = ExplanationOrchestrator(expl)

    # Deny via env should cause ConfigurationError when resolving preferred identifier
    os.environ["CE_DENY_PLUGIN"] = identifier
    with pytest.raises(ConfigurationError):
        orch.resolve_plugin("fast")
    del os.environ["CE_DENY_PLUGIN"]

    # Explicit override should allow untrusted plugin with a UserWarning
    with pytest.warns(UserWarning):
        plugin, ident = orch.resolve_plugin("fast")
        assert ident == identifier




def test_env_var_precedence_for_explanation_selection():
    mgr = PluginManager(object())
    # Global env wins over mode-specific per current manager logic
    os.environ["CE_EXPLANATION_PLUGIN"] = "global.plugin"
    os.environ["CE_EXPLANATION_PLUGIN_FACTUAL"] = "mode.plugin"
    chain = mgr.build_explanation_chain("factual", "core.explanation.factual")
    # first entry should be the global env value (preferred_identifier selection favors it)
    assert "global.plugin" in chain
    assert chain[0] == "global.plugin"
    # cleanup
    del os.environ["CE_EXPLANATION_PLUGIN"]
    del os.environ["CE_EXPLANATION_PLUGIN_FACTUAL"]


def test_explainerhandle_metadata_is_immutable():
    class Dummy:
        pass

    dummy = Dummy()
    h = ExplainerHandle(dummy, {"k": "v"})
    meta = h.get_metadata()
    assert isinstance(meta, (MappingProxyType, dict))
    with pytest.raises(TypeError):
        meta["k"] = "x"
