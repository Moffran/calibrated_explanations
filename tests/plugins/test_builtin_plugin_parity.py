"""Parity checks between legacy flows and plugin mediated pathways."""

from __future__ import annotations

import numpy as np

from calibrated_explanations.explanations.explanations import CalibratedExplanations
from calibrated_explanations.plugins.builtins import (
    FastExplanationPlugin,
    LegacyAlternativeExplanationPlugin,
    LegacyFactualExplanationPlugin,
    LegacyPredictBridge,
)
from calibrated_explanations.plugins.explanations import (
    ExplanationContext,
    ExplanationRequest,
)


def _make_context(explainer, mode: str) -> ExplanationContext:
    task = "classification" if "regression" not in explainer.mode else "regression"
    feature_names = tuple(getattr(explainer, "feature_names", []) or [])
    categorical_features = tuple(getattr(explainer, "categorical_features", []) or [])
    categorical_labels = getattr(explainer, "categorical_labels", {}) or {}
    discretizer = getattr(explainer, "discretizer", None)
    helper_handles = {"explainer": explainer}
    bridge = LegacyPredictBridge(explainer)
    interval_dependency = "core.interval.fast" if mode == "fast" else "core.interval.legacy"
    return ExplanationContext(
        task=task,
        mode=mode,
        feature_names=feature_names,
        categorical_features=categorical_features,
        categorical_labels=categorical_labels,
        discretizer=discretizer,
        helper_handles=helper_handles,
        predict_bridge=bridge,
        interval_settings={"plugin": interval_dependency},
        plot_settings={"style": "legacy"},
    )


def _make_request(threshold=None, low_high_percentiles=(5, 95), bins=None):
    return ExplanationRequest(
        threshold=threshold,
        low_high_percentiles=low_high_percentiles,
        bins=bins,
        features_to_ignore=(),
        extras={},
    )


def _assert_collections_equal(lhs, rhs):
    assert len(lhs) == len(rhs)
    for left, right in zip(lhs, rhs):
        np.testing.assert_allclose(
            left.feature_weights["predict"], right.feature_weights["predict"]
        )
        np.testing.assert_allclose(left.feature_weights["low"], right.feature_weights["low"])
        np.testing.assert_allclose(left.feature_weights["high"], right.feature_weights["high"])
        np.testing.assert_allclose(left.prediction["predict"], right.prediction["predict"])
        np.testing.assert_allclose(left.prediction["low"], right.prediction["low"])
        np.testing.assert_allclose(left.prediction["high"], right.prediction["high"])


def test_factual_plugin_matches_legacy(binary_dataset):
    from tests._helpers import get_classification_model, initiate_explainer

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
    explainer = initiate_explainer(
        model,
        X_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="classification",
        class_labels=["No", "Yes"],
    )

    plugin = LegacyFactualExplanationPlugin()
    plugin.initialize(_make_context(explainer, "factual"))
    batch = plugin.explain_batch(X_test, _make_request())
    plugin_collection = CalibratedExplanations.from_batch(batch)

    legacy = explainer.explain_factual(X_test)

    _assert_collections_equal(plugin_collection, legacy)


def test_alternative_plugin_matches_legacy(binary_dataset):
    from tests._helpers import get_classification_model, initiate_explainer

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
    explainer = initiate_explainer(
        model,
        X_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="classification",
        class_labels=["No", "Yes"],
    )

    plugin = LegacyAlternativeExplanationPlugin()
    plugin.initialize(_make_context(explainer, "alternative"))
    batch = plugin.explain_batch(X_test, _make_request())
    plugin_collection = CalibratedExplanations.from_batch(batch)

    legacy = explainer.explore_alternatives(X_test)

    _assert_collections_equal(plugin_collection, legacy)


def test_fast_plugin_matches_legacy(binary_dataset):
    from tests._helpers import get_classification_model, initiate_explainer

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
    explainer = initiate_explainer(
        model,
        X_cal,
        y_cal,
        feature_names,
        categorical_features,
        mode="classification",
        class_labels=["No", "Yes"],
        fast=True,
    )

    plugin = FastExplanationPlugin()
    plugin.initialize(_make_context(explainer, "fast"))
    batch = plugin.explain_batch(X_test, _make_request())
    plugin_collection = CalibratedExplanations.from_batch(batch)

    legacy = explainer.explain_fast(X_test)

    _assert_collections_equal(plugin_collection, legacy)
