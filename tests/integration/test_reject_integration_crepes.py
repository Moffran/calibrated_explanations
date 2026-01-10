import pytest

# Skip this integration test if heavy deps are not installed
pytest.importorskip("crepes")
pytest.importorskip("sklearn")

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy
from calibrated_explanations.explanations.reject import RejectResult


def test_integration_reject_policy_predict_and_flag_runs_and_returns_envelope():
    X, y = make_classification(n_samples=120, n_features=6, random_state=0)
    # split into fit / calibrate / test
    X_fit, y_fit = X[:50], y[:50]
    X_cal, y_cal = X[50:90], y[50:90]
    X_test = X[90:]

    clf = DecisionTreeClassifier(random_state=0)
    w = WrapCalibratedExplainer(clf)
    w.fit(X_fit, y_fit)

    # Provide explainer-level default via calibrate (wrapper __init__ must not accept it)
    w.calibrate(X_cal, y_cal, default_reject_policy=RejectPolicy.PREDICT_AND_FLAG)

    res = w.explain_factual(X_test[:4])

    assert isinstance(res, RejectResult)
    assert hasattr(res, "policy")
    assert res.policy is not None
