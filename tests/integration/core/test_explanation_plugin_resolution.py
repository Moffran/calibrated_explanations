"""Integration tests covering explanation plugin resolution and fallbacks."""

from __future__ import annotations

from typing import ClassVar, Optional, Tuple

import numpy as np
import pytest

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.exceptions import ConfigurationError
from calibrated_explanations.plugins.builtins import LegacyFactualExplanationPlugin
from calibrated_explanations.plugins.manager import DEFAULT_EXPLANATION_IDENTIFIERS
from calibrated_explanations.plugins.registry import (
    clear_explanation_plugins,
    ensure_builtin_plugins,
    register_explanation_plugin,
    unregister,
)

from tests._helpers import (
    get_classification_model,
    get_regression_model,
    initiate_explainer,
)


class ClassificationOnlyFactualPlugin(LegacyFactualExplanationPlugin):
    """Legacy factual adapter constrained to classification tasks."""

    plugin_meta = {
        **LegacyFactualExplanationPlugin.plugin_meta,
        "name": "tests.classification_only.factual",
        "tasks": ("classification",),
        "capabilities": [
            "explain",
            "explanation:factual",
            "task:classification",
        ],
    }

    last_initialised: ClassVar[Optional[Tuple[str, str]]] = None

    def supports_mode(self, mode: str, *, task: str) -> bool:  # pragma: no cover - delegation
        if task != "classification":
            return False
        return super().supports_mode(mode, task=task)

    def initialize(self, context):  # pragma: no cover - integration behaviour
        type(self).last_initialised = (context.task, context.mode)
        super().initialize(context)


class DependencyReportingFactualPlugin(LegacyFactualExplanationPlugin):
    """Factual adapter that records the context it was initialised with."""

    plugin_meta = {
        **LegacyFactualExplanationPlugin.plugin_meta,
        "name": "tests.dependency_reporting.factual",
        "interval_dependency": ("core.interval.fast",),
        "plot_dependency": "legacy",
        "fallbacks": ("core.explanation.factual",),
    }

    last_context: ClassVar[Optional[Tuple[str, Tuple[str, ...], Tuple[str, ...]]]] = None

    def initialize(self, context):  # pragma: no cover - integration behaviour
        type(self).last_context = (
            context.mode,
            tuple(context.interval_settings.get("dependencies", ())),
            tuple(context.plot_settings.get("fallbacks", ())),
        )
        super().initialize(context)


class FutureSchemaFactualPlugin(LegacyFactualExplanationPlugin):
    """Plugin advertising a future schema version for rejection tests."""

    plugin_meta = {
        **LegacyFactualExplanationPlugin.plugin_meta,
        "name": "tests.future_schema.factual",
        "schema_version": 999,
    }


def _cleanup_plugin(plugin) -> None:
    unregister(plugin)
    clear_explanation_plugins()
    ensure_builtin_plugins()
    ClassificationOnlyFactualPlugin.last_initialised = None
    DependencyReportingFactualPlugin.last_context = None


def _build_regression_explainer(regression_dataset):
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
    )
    return explainer, x_test


def _compare_collections(lhs, rhs):
    assert len(lhs) == len(rhs)
    for left, right in zip(lhs, rhs):
        np.testing.assert_allclose(
            left.feature_weights["predict"], right.feature_weights["predict"], rtol=1e-6
        )
        np.testing.assert_allclose(
            left.prediction["predict"], right.prediction["predict"], rtol=1e-6
        )
        if "low" in left.feature_weights:
            np.testing.assert_allclose(
                left.feature_weights["low"], right.feature_weights["low"], rtol=1e-6
            )
        if "high" in left.feature_weights:
            np.testing.assert_allclose(
                left.feature_weights["high"], right.feature_weights["high"], rtol=1e-6
            )
        if "low" in left.prediction:
            np.testing.assert_allclose(left.prediction["low"], right.prediction["low"], rtol=1e-6)
        if "high" in left.prediction:
            np.testing.assert_allclose(left.prediction["high"], right.prediction["high"], rtol=1e-6)


def test_task_filtered_plugin_falls_back(monkeypatch, regression_dataset):
    ensure_builtin_plugins()
    plugin = ClassificationOnlyFactualPlugin()
    register_explanation_plugin("tests.classification_only.factual", plugin)
    monkeypatch.setenv("CE_EXPLANATION_PLUGIN_FACTUAL", "tests.classification_only.factual")

    try:
        explainer, x_test = _build_regression_explainer(regression_dataset)
        result = explainer.explain_factual(x_test)
        assert len(result) == len(x_test)
        assert ClassificationOnlyFactualPlugin.last_initialised is None
        assert explainer._explanation_plugin_identifiers["factual"] == "core.explanation.factual"
    finally:
        monkeypatch.delenv("CE_EXPLANATION_PLUGIN_FACTUAL", raising=False)
        _cleanup_plugin(plugin)


def test_dependency_metadata_populates_context(monkeypatch, binary_dataset):
    ensure_builtin_plugins()
    plugin = DependencyReportingFactualPlugin()
    register_explanation_plugin("tests.dependency_reporting.factual", plugin)
    monkeypatch.setenv("CE_EXPLANATION_PLUGIN_FACTUAL", "tests.dependency_reporting.factual")

    try:
        (
            x_prop_train,
            y_prop_train,
            x_cal,
            y_cal,
            x_test,
            _y_test,
            _num_classes,
            _num_features,
            categorical_features,
            feature_names,
        ) = binary_dataset

        model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
        explainer = CalibratedExplainer(
            model,
            x_cal,
            y_cal,
            feature_names=feature_names,
            categorical_features=categorical_features,
            mode="classification",
            class_labels=["No", "Yes"],
        )

        result = explainer.explain_factual(x_test)
        assert len(result) == len(x_test)

        assert DependencyReportingFactualPlugin.last_context is not None
        mode, interval_deps, plot_fallbacks = DependencyReportingFactualPlugin.last_context
        assert mode == "factual"
        assert interval_deps == ("core.interval.fast",)
        assert "legacy" in plot_fallbacks

        chain = explainer._explanation_plugin_fallbacks["factual"]
        assert chain[0] == "tests.dependency_reporting.factual"
        assert "core.explanation.factual" in chain
        assert explainer._interval_plugin_hints["factual"] == ("core.interval.fast",)
        assert (
            explainer._explanation_plugin_identifiers["factual"]
            == "tests.dependency_reporting.factual"
        )
    finally:
        monkeypatch.delenv("CE_EXPLANATION_PLUGIN_FACTUAL", raising=False)
        _cleanup_plugin(plugin)


def test_future_schema_plugin_rejected(binary_dataset):
    ensure_builtin_plugins()
    plugin = FutureSchemaFactualPlugin()

    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _num_classes,
        _num_features,
        categorical_features,
        feature_names,
    ) = binary_dataset

    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    explainer = CalibratedExplainer(
        model,
        x_cal,
        y_cal,
        feature_names=feature_names,
        categorical_features=categorical_features,
        mode="classification",
        class_labels=["No", "Yes"],
        factual_plugin=plugin,
    )

    with pytest.raises(ConfigurationError, match="schema_version"):
        explainer.explain_factual(x_test)


def test_unknown_plugin_identifier_raises(monkeypatch, binary_dataset):
    """Test that unknown plugin identifiers raise ConfigurationError."""
    ensure_builtin_plugins()
    monkeypatch.setenv("CE_EXPLANATION_PLUGIN_FACTUAL", "tests.missing.plugin")
    # Patch the PluginManager's default identifiers instead of the old module-level constant
    monkeypatch.setitem(DEFAULT_EXPLANATION_IDENTIFIERS, "factual", None)

    try:
        (
            x_prop_train,
            y_prop_train,
            x_cal,
            y_cal,
            x_test,
            _y_test,
            _num_classes,
            _num_features,
            categorical_features,
            feature_names,
        ) = binary_dataset

        model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
        explainer = CalibratedExplainer(
            model,
            x_cal,
            y_cal,
            feature_names=feature_names,
            categorical_features=categorical_features,
            mode="classification",
            class_labels=["No", "Yes"],
        )

        with pytest.raises(ConfigurationError, match="tests.missing.plugin"):
            explainer.explain_factual(x_test)
    finally:
        monkeypatch.delenv("CE_EXPLANATION_PLUGIN_FACTUAL", raising=False)


def test_fast_mode_predict_bridge_and_parity(binary_dataset):
    ensure_builtin_plugins()
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _num_classes,
        _num_features,
        categorical_features,
        feature_names,
    ) = binary_dataset

    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    explainer = initiate_explainer(
        model,
        x_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="classification",
        class_labels=["No", "Yes"],
        fast=True,
    )

    plugin_collection = explainer.explain_fast(x_test)
    legacy_collection = explainer.explain_fast(x_test, _use_plugin=False)

    monitor = explainer._bridge_monitors["fast"]
    assert monitor.used
    assert "predict" in monitor.calls

    assert explainer._explanation_plugin_identifiers["fast"] == "core.explanation.fast"
    assert explainer._interval_plugin_hints["fast"] == ("core.interval.fast",)
    assert "legacy" in explainer._plot_plugin_fallbacks["fast"]

    _compare_collections(plugin_collection, legacy_collection)
