import pytest

from calibrated_explanations.core.reject.policy import RejectPolicy
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
from calibrated_explanations.explanations.reject import RejectResult

# Skip this integration test if heavy deps are not installed
pytest.importorskip("crepes")
pytest.importorskip("sklearn")


def test_integration_reject_policy_predict_and_flag_runs_and_returns_envelope():
    from sklearn.datasets import make_classification
    from sklearn.tree import DecisionTreeClassifier

    X, y = make_classification(n_samples=120, n_features=6, random_state=0)
    # split into fit / calibrate / test
    X_fit, y_fit = X[:50], y[:50]
    x_cal, y_cal = X[50:90], y[50:90]
    x_test = X[90:]

    clf = DecisionTreeClassifier(random_state=0)
    w = WrapCalibratedExplainer(clf)
    w.fit(X_fit, y_fit)

    # Provide explainer-level default via calibrate (wrapper __init__ must not accept it)
    w.calibrate(x_cal, y_cal, default_reject_policy=RejectPolicy.FLAG)

    res = w.explain_factual(x_test[:4])

    assert isinstance(res, RejectResult)
    assert hasattr(res, "policy")
    assert res.policy is not None


def test_integration_reject_breakdown_is_monotonic_in_expected_directions():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    X, y = make_classification(
        n_samples=200,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=0,
    )
    X_fit, X_tmp, y_fit, y_tmp = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    x_cal, x_test, y_cal, _ = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=0, stratify=y_tmp
    )

    clf = DecisionTreeClassifier(random_state=0)
    w = WrapCalibratedExplainer(clf)
    w.fit(X_fit, y_fit)
    w.calibrate(x_cal, y_cal)

    w.explainer.seed = 1337
    w.initialize_reject_learner()

    X_slice = x_test[:10]
    ambiguity_rates = []
    uncertainty_rates = []
    for conf in (0.9, 0.95, 0.99):
        out = w.explainer.reject_orchestrator.predict_reject_breakdown(X_slice, confidence=conf)
        ambiguity_rates.append(out["ambiguity_rate"])
        uncertainty_rates.append(out["novelty_rate"])

    assert ambiguity_rates[0] <= ambiguity_rates[1] <= ambiguity_rates[2]
    assert uncertainty_rates[0] >= uncertainty_rates[1] >= uncertainty_rates[2]
