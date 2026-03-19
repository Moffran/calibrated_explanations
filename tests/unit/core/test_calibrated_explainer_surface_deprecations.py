from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy
from tests.helpers.deprecation import warns_or_raises


def build_stub_explainer() -> CalibratedExplainer:
    explainer = object.__new__(CalibratedExplainer)
    explainer.initialized = False
    explainer.learner = MagicMock()
    explainer.default_reject_policy = RejectPolicy.NONE

    pm = MagicMock()
    pm.explanation_orchestrator = MagicMock()
    pm.prediction_orchestrator = MagicMock()
    pm.reject_orchestrator = MagicMock()
    pm.build_plot_chain.return_value = ("legacy",)
    pm.interval_plugin_hints = {"factual": ("core.interval.legacy",)}
    pm.interval_plugin_fallbacks = {"default": ("core.interval.legacy",)}
    pm.explanation_plugin_overrides = {"factual": None}
    pm.interval_plugin_override = None
    pm.fast_interval_plugin_override = None
    pm.plot_style_override = None
    pm.interval_preferred_identifier = {"default": None, "fast": None}
    pm.telemetry_interval_sources = {"default": None, "fast": None}
    pm.interval_plugin_identifiers = {"default": None, "fast": None}
    pm.interval_context_metadata = {"default": {}, "fast": {}}
    explainer.plugin_manager = pm
    return explainer


def test_nonessential_method_deprecations_emit() -> None:
    explainer = build_stub_explainer()
    explainer.explanation_orchestrator.invoke.return_value = "ok"
    explainer.prediction_orchestrator.ensure_interval_runtime_state.return_value = None
    explainer.prediction_orchestrator.gather_interval_hints.return_value = ("hint",)

    with warns_or_raises(match="build_plot_style_chain is deprecated"):
        chain = explainer.build_plot_style_chain()
    assert chain == ("legacy",)

    with warns_or_raises(match="instantiate_plugin is deprecated"):
        _ = explainer.instantiate_plugin(MagicMock())

    with warns_or_raises(match="ensure_interval_runtime_state is deprecated"):
        explainer.ensure_interval_runtime_state()

    with warns_or_raises(match="gather_interval_hints is deprecated"):
        hints = explainer.gather_interval_hints(fast=True)
    assert hints == ("hint",)


def test_invoke_explanation_plugin_deprecation(monkeypatch: pytest.MonkeyPatch) -> None:
    explainer = build_stub_explainer()
    explainer.explanation_orchestrator.invoke.return_value = "invoke-result"

    def resolve_effective_reject_policy_test_stub(*_args, **_kwargs):
        return SimpleNamespace(policy=RejectPolicy.NONE)

    monkeypatch.setattr(
        "calibrated_explanations.core.reject.orchestrator.resolve_effective_reject_policy",
        resolve_effective_reject_policy_test_stub,
    )

    with warns_or_raises(match="invoke_explanation_plugin is deprecated"):
        result = explainer.invoke_explanation_plugin(
            mode="factual",
            x=[[0.0]],
            threshold=None,
            low_high_percentiles=None,
            bins=None,
            features_to_ignore=None,
            extras={},
        )
    assert result == "invoke-result"


def test_nonessential_alias_property_getters_emit() -> None:
    explainer = build_stub_explainer()
    assertions = [
        ("interval_plugin_hints is deprecated", lambda: explainer.interval_plugin_hints),
        ("interval_plugin_fallbacks is deprecated", lambda: explainer.interval_plugin_fallbacks),
        (
            "explanation_plugin_overrides is deprecated",
            lambda: explainer.explanation_plugin_overrides,
        ),
        ("interval_plugin_override is deprecated", lambda: explainer.interval_plugin_override),
        (
            "fast_interval_plugin_override is deprecated",
            lambda: explainer.fast_interval_plugin_override,
        ),
        ("plot_style_override is deprecated", lambda: explainer.plot_style_override),
        (
            "interval_preferred_identifier is deprecated",
            lambda: explainer.interval_preferred_identifier,
        ),
        ("telemetry_interval_sources is deprecated", lambda: explainer.telemetry_interval_sources),
        (
            "interval_plugin_identifiers is deprecated",
            lambda: explainer.interval_plugin_identifiers,
        ),
        ("interval_context_metadata is deprecated", lambda: explainer.interval_context_metadata),
    ]

    observed = 0
    for match, accessor in assertions:
        with warns_or_raises(match=match):
            _ = accessor()
        observed += 1
    assert observed == len(assertions)


def test_deprecation_strict_mode_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    explainer = build_stub_explainer()
    monkeypatch.setenv("CE_DEPRECATIONS", "error")

    with pytest.raises(DeprecationWarning, match="build_plot_style_chain is deprecated"):
        _ = explainer.build_plot_style_chain()
