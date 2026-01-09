from __future__ import annotations

import pytest
from tests.helpers.explainer_utils import make_explainer_from_dataset

from calibrated_explanations.utils.exceptions import ConfigurationError
from calibrated_explanations.plugins.builtins import LegacyFactualExplanationPlugin
from calibrated_explanations.plugins import (
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

    def explain_batch(self, x, request):
        batch = super().explain_batch(x, request)
        batch.collection_metadata["mode"] = "alternative"
        return batch


class NonPredictingFactualPlugin(LegacyFactualExplanationPlugin):
    plugin_meta = {
        **LegacyFactualExplanationPlugin.plugin_meta,
        "name": "tests.nonpredicting.factual",
    }

    def explain_batch(self, x, request):
        if self.explainer is None:
            raise RuntimeError("explainer handle was not initialised")
        explanation_callable = getattr(self.explainer, self.explanation_attr)
        kwargs = {
            "threshold": request.threshold,
            "low_high_percentiles": request.low_high_percentiles,
            "bins": request.bins,
            "_use_plugin": False,
        }
        if self.mode != "fast":
            kwargs["features_to_ignore"] = request.features_to_ignore
        collection = explanation_callable(x, **kwargs)
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


def test_invalid_batch_metadata_raises_configuration_error(binary_dataset):
    explainer, x_test = make_explainer_from_dataset(
        binary_dataset,
        factual_plugin=MisreportingFactualPlugin(),
    )

    with pytest.raises(ConfigurationError, match="invalid batch"):
        explainer.explain_factual(x_test)


def test_predict_bridge_usage_enforced(binary_dataset):
    explainer, x_test = make_explainer_from_dataset(
        binary_dataset,
        factual_plugin=NonPredictingFactualPlugin(),
    )

    with pytest.raises(ConfigurationError, match="predict bridge"):
        explainer.explain_factual(x_test)


def test_task_metadata_mismatch_rejected(binary_dataset):
    ensure_builtin_plugins()
    plugin = RegressionOnlyFactualPlugin()
    register_explanation_plugin("tests.regression_only", plugin)

    try:
        explainer, x_test = make_explainer_from_dataset(
            binary_dataset,
            factual_plugin="tests.regression_only",
        )

        with pytest.raises(ConfigurationError, match="does not support task"):
            explainer.explain_factual(x_test)
    finally:
        unregister(plugin)
        clear_explanation_plugins()
        ensure_builtin_plugins()
