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
