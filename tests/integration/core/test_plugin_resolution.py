from __future__ import annotations

from typing import Any
import pytest
from tests.helpers.explainer_utils import make_explainer_from_dataset

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.utils.exceptions import ConfigurationError
from calibrated_explanations.plugins.builtins import LegacyFactualExplanationPlugin
from calibrated_explanations.plugins import (
    ensure_builtin_plugins,
    register_explanation_plugin,
)
from tests.helpers.plugin_utils import cleanup_plugin


class RegressionOnlyFactualPlugin(LegacyFactualExplanationPlugin):
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


class RecordingFactualPlugin(LegacyFactualExplanationPlugin):
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

    def explain_batch(self, x, request):  # type: ignore[override]
        self.requests.append(request)
        return super().explain_batch(x, request)


class LegacySchemaFactualPlugin(LegacyFactualExplanationPlugin):
    """Plugin declaring an outdated schema version for runtime checks."""

    plugin_meta = {
        **LegacyFactualExplanationPlugin.plugin_meta,
        "name": "tests.legacy_schema.factual",
        "schema_version": 0,
        "trust": False,
    }


def make_regression_explainer(regression_dataset, **overrides):
    from tests.helpers.model_utils import get_regression_model

    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _num_features,
        categorical_features,
        feature_names,
    ) = regression_dataset

    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    explainer = CalibratedExplainer(
        model,
        x_cal,
        y_cal,
        feature_names=feature_names,
        categorical_features=categorical_features,
        mode="regression",
        **overrides,
    )
    return explainer, x_test


def test_dependency_propagation_and_context_hints(binary_dataset):
    ensure_builtin_plugins()
    plugin = RecordingFactualPlugin()
    register_explanation_plugin("tests.recording.factual", plugin)

    try:
        explainer, x_test = make_explainer_from_dataset(
            binary_dataset, factual_plugin="tests.recording.factual"
        )
        explanations = explainer.explain_factual(x_test)
        assert explanations is not None

        assert explainer._interval_plugin_hints["factual"] == ("tests.interval.pref",)
        assert explainer._explanation_plugin_identifiers["factual"] == "tests.recording.factual"
    finally:
        cleanup_plugin(plugin)


def test_plugin_override_via_env_var(binary_dataset, monkeypatch):
    ensure_builtin_plugins()
    plugin = RecordingFactualPlugin()
    register_explanation_plugin("tests.recording.factual", plugin)
    monkeypatch.setenv("CE_EXPLANATION_PLUGIN_FACTUAL", "tests.recording.factual")

    try:
        explainer, x_test = make_explainer_from_dataset(binary_dataset)
        explanations = explainer.explain_factual(x_test)
        assert explainer._explanation_plugin_identifiers["factual"] == "tests.recording.factual"
        assert explainer._interval_plugin_hints["factual"] == ("tests.interval.pref",)
        fallback_chain = explainer._plot_plugin_fallbacks["factual"]
        assert fallback_chain[0] == "tests.plot.pref"
        assert fallback_chain[-1] == "legacy"
        runtime_plugin = explainer._explanation_plugin_instances["factual"]
        assert isinstance(runtime_plugin, RecordingFactualPlugin)
        assert runtime_plugin.initialized_context is not None
        assert runtime_plugin.initialized_context.interval_settings["dependencies"] == (
            "tests.interval.pref",
        )
        fallbacks = runtime_plugin.initialized_context.plot_settings["fallbacks"]
        assert fallbacks[0] == "tests.plot.pref"
        assert fallbacks[-1] == "legacy"
        assert runtime_plugin.requests, "plugin should record at least one request"

        telemetry = explainer.runtime_telemetry
        assert telemetry.get("proba_source") == "core.interval.legacy"
        telemetry_fallbacks = telemetry.get("plot_fallbacks")
        assert telemetry_fallbacks[0] == "tests.plot.pref"
        assert telemetry_fallbacks[-1] == "legacy"

        batch_telemetry = getattr(explanations, "telemetry", {})
        assert batch_telemetry.get("proba_source") == "core.interval.legacy"
        batch_fallbacks = batch_telemetry.get("plot_fallbacks")
        assert batch_fallbacks[0] == "tests.plot.pref"
        assert batch_fallbacks[-1] == "legacy"
    finally:
        monkeypatch.delenv("CE_EXPLANATION_PLUGIN_FACTUAL", raising=False)
        cleanup_plugin(plugin)


def test_schema_version_override_errors(binary_dataset):
    ensure_builtin_plugins()
    plugin = LegacySchemaFactualPlugin()
    register_explanation_plugin("tests.legacy_schema.factual", plugin)

    try:
        explainer, x_test = make_explainer_from_dataset(
            binary_dataset, factual_plugin="tests.legacy_schema.factual"
        )
        with pytest.raises(ConfigurationError, match="schema_version 0"):
            explainer.explain_factual(x_test)
    finally:
        cleanup_plugin(plugin)


def test_missing_plugin_override_raises(monkeypatch, binary_dataset):
    monkeypatch.delenv("CE_EXPLANATION_PLUGIN_FACTUAL", raising=False)

    explainer, x_test = make_explainer_from_dataset(
        binary_dataset, factual_plugin="tests.missing.plugin"
    )
    with pytest.raises(ConfigurationError, match="not registered"):
        explainer.explain_factual(x_test)


def test_alternative_classification_records_plot_fallbacks(binary_dataset):
    ensure_builtin_plugins()
    explainer, x_test = make_explainer_from_dataset(binary_dataset)

    alternatives = explainer.explore_alternatives(x_test[:2])
    assert alternatives is not None

    fallbacks = explainer._plot_plugin_fallbacks["alternative"]
    assert fallbacks[0] == "plot_spec.default"
    assert fallbacks[-1] == "legacy"

    telemetry = explainer.runtime_telemetry
    assert telemetry.get("mode") == "alternative"
    assert telemetry.get("plot_source") == "plot_spec.default"
    assert telemetry.get("plot_fallbacks")[0] == "plot_spec.default"

    batch_telemetry = getattr(alternatives, "telemetry", {})
    assert batch_telemetry.get("plot_source") == "plot_spec.default"
    assert batch_telemetry.get("plot_fallbacks")[0] == "plot_spec.default"


def test_alternative_regression_records_plot_fallbacks(regression_dataset):
    ensure_builtin_plugins()
    explainer, x_test = make_regression_explainer(regression_dataset)

    alternatives = explainer.explore_alternatives(x_test[:2])
    assert alternatives is not None

    fallbacks = explainer._plot_plugin_fallbacks["alternative"]
    assert fallbacks[0] == "plot_spec.default"
    assert fallbacks[-1] == "legacy"

    telemetry = explainer.runtime_telemetry
    assert telemetry.get("mode") == "alternative"
    assert telemetry.get("plot_source") == "plot_spec.default"
    assert telemetry.get("plot_fallbacks")[0] == "plot_spec.default"

    batch_telemetry = getattr(alternatives, "telemetry", {})
    assert batch_telemetry.get("plot_source") == "plot_spec.default"
    assert batch_telemetry.get("plot_fallbacks")[0] == "plot_spec.default"


def test_interval_override_missing_identifier(monkeypatch, binary_dataset):
    monkeypatch.setenv("CE_INTERVAL_PLUGIN", "tests.missing.interval")
    try:
        with pytest.raises(ConfigurationError, match="not registered"):
            make_explainer_from_dataset(binary_dataset)
    finally:
        monkeypatch.delenv("CE_INTERVAL_PLUGIN", raising=False)
