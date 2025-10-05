from __future__ import annotations

from typing import Any
import pytest

from calibrated_explanations.core.calibrated_explainer import (
    CalibratedExplainer,
    ConfigurationError,
)
from calibrated_explanations.plugins.builtins import LegacyFactualExplanationPlugin
from calibrated_explanations.plugins.registry import (
    clear_explanation_plugins,
    ensure_builtin_plugins,
    register_explanation_plugin,
    unregister,
)


class _RegressionOnlyFactualPlugin(LegacyFactualExplanationPlugin):
    """Legacy factual plugin constrained to regression tasks only."""

    plugin_meta = {
        **LegacyFactualExplanationPlugin.plugin_meta,
        "name": "tests.regression_only.factual",
        "capabilities": [
            "explain",
            "explanation:factual",
            "task:regression",
        ],
        "tasks": ("regression",),
        "trust": False,
    }


class _RecordingFactualPlugin(LegacyFactualExplanationPlugin):
    """Plugin that records its context to assert dependency propagation."""

    plugin_meta = {
        **LegacyFactualExplanationPlugin.plugin_meta,
        "name": "tests.recording.factual",
        "interval_dependency": "tests.interval.pref",
        "plot_dependency": "tests.plot.pref",
        "dependencies": (
            "tests.interval.pref",
            "tests.plot.pref",
        ),
        "trust": False,
    }

    def __init__(self) -> None:
        super().__init__()
        self.initialized_context = None
        self.requests: list[Any] = []

    def initialize(self, context):  # type: ignore[override]
        self.initialized_context = context
        super().initialize(context)

    def explain_batch(self, X, request):  # type: ignore[override]
        self.requests.append(request)
        return super().explain_batch(X, request)


class _LegacySchemaFactualPlugin(LegacyFactualExplanationPlugin):
    """Plugin declaring an outdated schema version for runtime checks."""

    plugin_meta = {
        **LegacyFactualExplanationPlugin.plugin_meta,
        "name": "tests.legacy_schema.factual",
        "schema_version": 0,
        "trust": False,
    }


def _make_explainer(binary_dataset, **overrides):
    from tests._helpers import get_classification_model

    (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        _y_test,
        _num_classes,
        _num_features,
        categorical_features,
        feature_names,
    ) = binary_dataset

    model, _ = get_classification_model("RF", X_prop_train, y_prop_train)
    explainer = CalibratedExplainer(
        model,
        X_cal,
        y_cal,
        mode="classification",
        feature_names=feature_names,
        categorical_features=categorical_features,
        class_labels=["No", "Yes"],
        **overrides,
    )
    return explainer, X_test


def _cleanup_plugin(plugin):
    unregister(plugin)
    clear_explanation_plugins()
    ensure_builtin_plugins()


def test_fallback_skips_incompatible_tasks(monkeypatch, binary_dataset):
    ensure_builtin_plugins()
    plugin = _RegressionOnlyFactualPlugin()
    register_explanation_plugin("tests.regression_only.factual", plugin)
    monkeypatch.setenv(
        "CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS",
        "tests.regression_only.factual",
    )

    try:
        explainer, X_test = _make_explainer(binary_dataset)
        chain = explainer._explanation_plugin_fallbacks["factual"]
        assert chain[0] == "tests.regression_only.factual"

        explanations = explainer.explain_factual(X_test)
        assert explanations is not None
        assert explainer._explanation_plugin_identifiers["factual"] == "core.explanation.factual"
    finally:
        monkeypatch.delenv("CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS", raising=False)
        _cleanup_plugin(plugin)


def test_dependency_propagation_and_context_hints(binary_dataset):
    ensure_builtin_plugins()
    plugin = _RecordingFactualPlugin()
    register_explanation_plugin("tests.recording.factual", plugin)

    try:
        explainer, X_test = _make_explainer(
            binary_dataset, factual_plugin="tests.recording.factual"
        )
        explanations = explainer.explain_factual(X_test)
        assert explanations is not None

        assert explainer._explanation_plugin_identifiers["factual"] == ("tests.recording.factual")
        assert explainer._interval_plugin_hints["factual"] == ("tests.interval.pref",)
        assert explainer._plot_plugin_fallbacks["factual"] == (
            "tests.plot.pref",
            "legacy",
        )
        runtime_plugin = explainer._explanation_plugin_instances["factual"]
        assert isinstance(runtime_plugin, _RecordingFactualPlugin)
        assert runtime_plugin.initialized_context is not None
        assert runtime_plugin.initialized_context.interval_settings["dependencies"] == (
            "tests.interval.pref",
        )
        assert runtime_plugin.initialized_context.plot_settings["fallbacks"] == (
            "tests.plot.pref",
            "legacy",
        )
        assert runtime_plugin.requests, "plugin should record at least one request"
    finally:
        _cleanup_plugin(plugin)


def test_schema_version_override_errors(binary_dataset):
    ensure_builtin_plugins()
    plugin = _LegacySchemaFactualPlugin()
    register_explanation_plugin("tests.legacy_schema.factual", plugin)

    try:
        explainer, X_test = _make_explainer(
            binary_dataset, factual_plugin="tests.legacy_schema.factual"
        )
        with pytest.raises(ConfigurationError, match="schema_version 0"):
            explainer.explain_factual(X_test)
    finally:
        _cleanup_plugin(plugin)


def test_missing_plugin_override_raises(monkeypatch, binary_dataset):
    monkeypatch.delenv("CE_EXPLANATION_PLUGIN_FACTUAL", raising=False)

    explainer, X_test = _make_explainer(binary_dataset, factual_plugin="tests.missing.plugin")
    with pytest.raises(ConfigurationError, match="not registered"):
        explainer.explain_factual(X_test)


def test_fast_mode_predict_bridge_usage(binary_dataset):
    explainer, X_test = _make_explainer(binary_dataset)
    fast_batch = explainer.explain_fast(X_test)

    assert fast_batch is not None
    monitor = explainer._bridge_monitors.get("fast")
    assert monitor is not None
    assert monitor.used, "FAST plugin should exercise the predict bridge"
    assert explainer._interval_plugin_hints["fast"] == ("core.interval.fast",)
    assert explainer._plot_plugin_fallbacks["fast"] == ("legacy",)
