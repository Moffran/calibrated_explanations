from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer


def build_stub_explainer() -> CalibratedExplainer:
    explainer = object.__new__(CalibratedExplainer)
    explainer.default_reject_policy = RejectPolicy.NONE

    plugin_manager = MagicMock()
    plugin_manager.explanation_orchestrator = MagicMock()
    plugin_manager.prediction_orchestrator = MagicMock()
    plugin_manager.reject_orchestrator = MagicMock()
    explainer.plugin_manager = plugin_manager
    return explainer


def test_should_raise_attribute_error_when_nonessential_methods_are_accessed() -> None:
    explainer = build_stub_explainer()

    for name in (
        "build_plot_style_chain",
        "plot_style_chain",
        "instantiate_plugin",
        "_invoke_explanation_plugin",
        "invoke_explanation_plugin",
        "ensure_interval_runtime_state",
        "gather_interval_hints",
        "initialize_reject_learner",
        "predict_reject",
    ):
        assert not hasattr(explainer, name)
        with pytest.raises(AttributeError):
            getattr(explainer, name)


def test_should_raise_attribute_error_when_plugin_state_aliases_are_accessed() -> None:
    explainer = build_stub_explainer()

    for name in (
        "interval_plugin_hints",
        "interval_plugin_fallbacks",
        "explanation_plugin_overrides",
        "interval_plugin_override",
        "fast_interval_plugin_override",
        "plot_style_override",
        "interval_preferred_identifier",
        "telemetry_interval_sources",
        "interval_plugin_identifiers",
        "interval_context_metadata",
    ):
        assert not hasattr(explainer, name)
        with pytest.raises(AttributeError):
            getattr(explainer, name)


def test_should_raise_attribute_error_when_wrapper_reject_delegators_are_accessed() -> None:
    wrapper = object.__new__(WrapCalibratedExplainer)

    for name in ("initialize_reject_learner", "predict_reject"):
        assert not hasattr(wrapper, name)
        with pytest.raises(AttributeError):
            getattr(wrapper, name)


def test_should_keep_canonical_orchestrator_paths_available() -> None:
    explainer = build_stub_explainer()

    assert explainer.reject_orchestrator is explainer.plugin_manager.reject_orchestrator
    assert explainer.plugin_manager is not None
    assert explainer.prediction_orchestrator is explainer.plugin_manager.prediction_orchestrator
    assert explainer.explanation_orchestrator is explainer.plugin_manager.explanation_orchestrator
