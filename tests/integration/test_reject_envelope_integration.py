import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split

import pytest

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy
from calibrated_explanations.explanations.reject import (
    RejectCalibratedExplanations,
    RejectResult,
)

pytestmark = pytest.mark.integration


def train_classification(n_classes=2, seed=0):
    X, y = make_classification(
        n_samples=200, n_features=6, n_informative=4, n_classes=n_classes, random_state=seed
    )
    x_proper, x_cal, y_proper, y_cal = train_test_split(X, y, test_size=0.25, random_state=seed)
    model = RandomForestClassifier(n_estimators=12, random_state=seed)
    w = WrapCalibratedExplainer(model)
    w.fit(x_proper, y_proper)
    w.calibrate(x_cal, y_cal, seed=seed)
    w.explainer.seed = seed
    return w, X[:6]


def train_regression(seed=0):
    X, y = make_regression(
        n_samples=200, n_features=6, n_informative=4, noise=5.0, random_state=seed
    )
    x_proper, x_cal, y_proper, y_cal = train_test_split(X, y, test_size=0.25, random_state=seed)
    model = BayesianRidge()
    w = WrapCalibratedExplainer(model)
    w.fit(x_proper, y_proper)
    w.calibrate(x_cal, y_cal, mode="regression", seed=seed)
    w.explainer.seed = seed
    # initialize reject learner for regression threshold tests
    w.explainer.reject_orchestrator.initialize_reject_learner(threshold=np.median(y_cal))
    return w, X[:6]


def test_classification_predict_and_explain_envelope_binary():
    w, Xq = train_classification(n_classes=2, seed=3)
    # predict
    res = w.predict(Xq, reject_policy=RejectPolicy.FLAG)
    assert isinstance(res, RejectResult)
    assert "ambiguity_mask" in (res.metadata or {})
    # explain
    res2 = w.explain_factual(Xq, reject_policy=RejectPolicy.FLAG)
    assert isinstance(res2, RejectCalibratedExplanations)
    assert res2.ambiguity_mask is not None


def test_multiclass_predict_proba_envelope():
    w, Xq = train_classification(n_classes=3, seed=4)
    res = w.predict_proba(Xq, uq_interval=False, reject_policy=RejectPolicy.FLAG)
    assert isinstance(res, RejectResult)
    proba = res.prediction
    assert proba is not None


def test_regression_predict_uq_envelope():
    w, Xq = train_regression(seed=5)
    # regression probabilistic predictions require threshold for reject machinery; use predict with uq interval
    res = w.predict(Xq, uq_interval=True, reject_policy=RejectPolicy.FLAG)
    assert isinstance(res, RejectResult)
    # legacy regression payload (proba, (low, high)) must be preserved
    pred = res.prediction
    assert isinstance(pred, tuple) and len(pred) == 2
