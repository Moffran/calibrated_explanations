"""Integration tests for guarded audit APIs."""

import json

import pytest

from tests.helpers.explainer_utils import initiate_explainer
from tests.helpers.model_utils import get_classification_model, get_regression_model

pytestmark = pytest.mark.integration


def test_classification_guarded_audit_presence(binary_dataset):
    """Guarded factual classification explanations should expose collection audit."""
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _,
        _,
        _,
        categorical_features,
        feature_names,
    ) = binary_dataset
    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )
    explanations = cal_exp.explain_guarded_factual(x_test[:2], significance=0.1)
    audit = explanations.get_guarded_audit()
    assert audit["summary"]["n_instances"] == len(explanations)
    assert len(audit["instances"]) == len(explanations)


def test_regression_guarded_audit_presence(regression_dataset):
    """Guarded factual regression explanations should expose collection audit."""
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _,
        _,
        categorical_features,
        feature_names,
    ) = regression_dataset
    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="regression"
    )
    explanations = cal_exp.explain_guarded_factual(x_test[:2], significance=0.1)
    audit = explanations.get_guarded_audit()
    assert audit["summary"]["n_instances"] == len(explanations)
    assert len(audit["instances"]) == len(explanations)


def test_guarded_audit_collection_serialization_smoke(binary_dataset):
    """Guarded collection audit payload should be JSON-serializable."""
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _,
        _,
        _,
        categorical_features,
        feature_names,
    ) = binary_dataset
    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )
    explanations = cal_exp.explain_guarded_factual(x_test[:1], significance=0.1)
    audit = explanations.get_guarded_audit()
    assert isinstance(json.dumps(audit), str)


def test_guarded_regression_remains_callable_after_reinitialize_append_path(regression_dataset):
    """Regression guarded explain must not raise after append_cal + reinitialize.

    ``append_cal`` alone updates ``explainer.x_cal`` but does not rebuild or
    update the interval learner, so alignment should still fail until the
    supported recalibration path is completed.  Regression reinitialization uses
    ``insert_calibration`` (in-place), which previously left the orchestrator
    calibration-feature snapshot stale.  The snapshot must be refreshed by
    ``update_interval_learner`` so guarded entrypoints do not produce a false
    ``ValidationError`` after a valid recalibration.
    """
    import numpy as np  # noqa: PLC0415 - local import for clarity in integration test

    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _,
        _,
        categorical_features,
        feature_names,
    ) = regression_dataset
    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="regression"
    )

    # Reinitialize through the supported regression update path, which appends
    # the new calibration data and updates the interval learner in place.
    extra_x = np.asarray(x_cal[:3]).copy()
    extra_y = np.asarray(y_cal[:3]).copy()
    cal_exp.reinitialize(model, xs=extra_x, ys=extra_y)

    # Guarded explain must succeed without ValidationError after the update.
    explanations = cal_exp.explain_guarded_factual(x_test[:1], significance=0.5)
    assert len(explanations) == 1
