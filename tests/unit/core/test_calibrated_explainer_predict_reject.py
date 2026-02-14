import types
import warnings
import numpy as np
import pytest

from tests.helpers.model_utils import get_classification_model, get_regression_model
from tests.helpers.explainer_utils import initiate_explainer
from calibrated_explanations.explanations.reject import RejectResult, RejectPolicy


def make_rr_classification():
    predict = np.array([0.6, 0.4])
    low = np.array([0.1, 0.1])
    high = np.array([0.9, 0.9])
    return RejectResult(prediction=(predict, low, high, None), policy=RejectPolicy.FLAG)


def make_rr_regression():
    predict = np.array([1.5, 2.5])
    low = np.array([1.0, 2.0])
    high = np.array([2.0, 3.0])
    return RejectResult(prediction=(predict, low, high, None), policy=RejectPolicy.FLAG)


def test_predict_with_reject_classification_formats_prediction(monkeypatch):
    # prepare tiny dataset
    from tests.helpers.dataset_utils import make_binary_dataset

    dataset = make_binary_dataset()
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _,
        _,
        categorical_features,
        feature_names,
    ) = dataset

    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    # Monkeypatch the existing reject_orchestrator.apply_policy to return our RejectResult
    rr = make_rr_classification()
    current = cal_exp.reject_orchestrator
    monkeypatch.setattr(current, "apply_policy", lambda *a, **k: rr, raising=False)

    res = cal_exp.predict(x_test, reject_policy="flag")
    assert isinstance(res, RejectResult)
    # format_classification_prediction with new_classes=None will produce integer labels
    np.testing.assert_array_equal(res.prediction.astype(float).astype(int), np.array([1, 0]))


def test_predict_with_reject_regression_formats_prediction(monkeypatch):
    from tests.helpers.dataset_utils import make_regression_dataset

    dataset = make_regression_dataset()
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _,
        no_of_features,
        categorical_features,
        feature_names,
    ) = dataset

    model, _ = get_regression_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(model, x_cal, y_cal, feature_names, categorical_features, mode="regression")

    rr = make_rr_regression()
    current = cal_exp.reject_orchestrator
    monkeypatch.setattr(current, "apply_policy", lambda *a, **k: rr, raising=False)

    res = cal_exp.predict(x_test, reject_policy="flag")
    assert isinstance(res, RejectResult)
    # format_regression_prediction without threshold returns the raw prediction array
    np.testing.assert_array_equal(res.prediction, np.array([1.5, 2.5]))


def test_predict_reject_formatting_exception_warns(monkeypatch):
    from tests.helpers.dataset_utils import make_binary_dataset
    from calibrated_explanations.core import prediction_helpers

    dataset = make_binary_dataset()
    (
        x_prop_train,
        y_prop_train,
        x_cal,
        y_cal,
        x_test,
        _y_test,
        _,
        _,
        categorical_features,
        feature_names,
    ) = dataset

    model, _ = get_classification_model("RF", x_prop_train, y_prop_train)
    cal_exp = initiate_explainer(
        model, x_cal, y_cal, feature_names, categorical_features, mode="classification"
    )

    rr = make_rr_classification()
    current = cal_exp.reject_orchestrator
    monkeypatch.setattr(current, "apply_policy", lambda *a, **k: rr, raising=False)

    # Force formatting to raise so we hit the exception handling path
    monkeypatch.setattr(prediction_helpers, "format_classification_prediction", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.warns(UserWarning):
        res = cal_exp.predict(x_test, reject_policy="flag")
    assert isinstance(res, RejectResult)
