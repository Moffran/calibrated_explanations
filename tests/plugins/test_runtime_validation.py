from __future__ import annotations

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


class MisreportingFactualPlugin(LegacyFactualExplanationPlugin):
    plugin_meta = {
        **LegacyFactualExplanationPlugin.plugin_meta,
        "name": "tests.misreporting.factual",
    }

    def explain_batch(self, X, request):
        batch = super().explain_batch(X, request)
        batch.collection_metadata["mode"] = "alternative"
        return batch


class NonPredictingFactualPlugin(LegacyFactualExplanationPlugin):
    plugin_meta = {
        **LegacyFactualExplanationPlugin.plugin_meta,
        "name": "tests.nonpredicting.factual",
    }

    def explain_batch(self, X, request):
        if self._explainer is None:
            raise RuntimeError("explainer handle was not initialised")
        explanation_callable = getattr(self._explainer, self._explanation_attr)
        kwargs = {
            "threshold": request.threshold,
            "low_high_percentiles": request.low_high_percentiles,
            "bins": request.bins,
            "_use_plugin": False,
        }
        if self._mode != "fast":
            kwargs["features_to_ignore"] = request.features_to_ignore
        collection = explanation_callable(X, **kwargs)
        return collection.to_batch()


class RegressionOnlyFactualPlugin(LegacyFactualExplanationPlugin):
    plugin_meta = {
        **LegacyFactualExplanationPlugin.plugin_meta,
        "name": "tests.regression_only.factual",
        "tasks": ("regression",),
        "capabilities": [
            "explain",
            "explanation:factual",
            "task:regression",
        ],
    }


def _make_explainer(binary_dataset, **overrides):
    from tests._helpers import get_classification_model

    (
        X_prop_train,
        y_prop_train,
        X_cal,
        y_cal,
        X_test,
        _,
        _,
        _,
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


def test_invalid_batch_metadata_raises_configuration_error(binary_dataset):
    explainer, X_test = _make_explainer(
        binary_dataset,
        factual_plugin=MisreportingFactualPlugin(),
    )

    with pytest.raises(ConfigurationError, match="invalid batch"):
        explainer.explain_factual(X_test)


def test_predict_bridge_usage_enforced(binary_dataset):
    explainer, X_test = _make_explainer(
        binary_dataset,
        factual_plugin=NonPredictingFactualPlugin(),
    )

    with pytest.raises(ConfigurationError, match="predict bridge"):
        explainer.explain_factual(X_test)


def test_task_metadata_mismatch_rejected(binary_dataset):
    ensure_builtin_plugins()
    plugin = RegressionOnlyFactualPlugin()
    register_explanation_plugin("tests.regression_only", plugin)

    try:
        explainer, X_test = _make_explainer(
            binary_dataset,
            factual_plugin="tests.regression_only",
        )

        with pytest.raises(ConfigurationError, match="does not support task"):
            explainer.explain_factual(X_test)
    finally:
        unregister(plugin)
        clear_explanation_plugins()
        ensure_builtin_plugins()
