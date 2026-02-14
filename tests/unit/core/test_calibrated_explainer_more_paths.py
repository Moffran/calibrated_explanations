import logging
import types
import numpy as np
import pytest

from tests.helpers.model_utils import get_classification_model, get_regression_model
from tests.helpers.dataset_utils import make_binary_dataset, make_regression_dataset
from tests.helpers.explainer_utils import initiate_explainer
from calibrated_explanations.explanations.reject import RejectResult, RejectPolicy


def test_predict_skip_reject_internal_returns_prediction():
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
    cal_exp = initiate_explainer(model, x_cal, y_cal, feature_names, categorical_features, mode="classification")

    # When _ce_skip_reject is True the legacy calibrated prediction path is used
    res = cal_exp.predict(x_test, _ce_skip_reject=True)
    # Expect numpy array (formatted classification labels)
    assert isinstance(res, (list, np.ndarray))


def test_predict_with_implicit_default_reject_policy_logs(monkeypatch):
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
    cal_exp = initiate_explainer(model, x_cal, y_cal, feature_names, categorical_features, mode="classification")

    # Set a non-NONE default to trigger implicit_default_used when no reject_policy provided
    cal_exp.default_reject_policy = RejectPolicy.FLAG

    records = []

    def fake_info(msg, *a, **k):
        records.append(msg)

    monkeypatch.setattr(logging.getLogger("calibrated_explanations.core.calibrated_explainer"), "info", fake_info)

    rr = RejectResult(prediction=None, policy=RejectPolicy.FLAG)
    current = cal_exp.reject_orchestrator
    monkeypatch.setattr(current, "apply_policy", lambda *a, **k: rr, raising=False)

    res = cal_exp.predict(x_test)
    assert isinstance(res, RejectResult)
    assert any("Default reject policy" in str(r) or "Default reject policy" in r for r in records)


def test_predict_rr_prediction_none_preserved(monkeypatch):
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
    cal_exp = initiate_explainer(model, x_cal, y_cal, feature_names, categorical_features, mode="classification")

    rr = RejectResult(prediction=None, policy=RejectPolicy.FLAG)
    current = cal_exp.reject_orchestrator
    monkeypatch.setattr(current, "apply_policy", lambda *a, **k: rr, raising=False)

    res = cal_exp.predict(x_test, reject_policy="flag")
    assert isinstance(res, RejectResult)
    assert res.prediction is None


def test_predict_proba_uncalibrated_regression_raises_when_threshold():
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

    # uncalibrated regression with threshold should raise ValidationError inside helper
    with pytest.raises(Exception):
        cal_exp.predict(x_test, calibrated=False, threshold=0.5)
