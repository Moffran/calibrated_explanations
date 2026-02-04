import importlib

import pytest

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.ce_agent_utils import (
    add_conjunctions,
    add_conjunctions_to_one,
    enforce_ce_first_and_execute,
    ensure_ce_first_wrapper,
    explain_and_summarize,
    explain_and_narrate,
    fit_and_calibrate,
    get_calibrated_predictions,
    policy_as_dict,
    probe_optional_features,
    serialize_policy,
)

from calibrated_explanations.core.exceptions import ValidationError

sklearn = pytest.importorskip("sklearn")
from sklearn.datasets import load_breast_cancer, make_regression  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402


def prep_classification():
    X, y = load_breast_cancer(return_X_y=True)
    x_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=0)
    x_cal, x_test, y_cal, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)
    return x_train, y_train, x_cal, y_cal, x_test, y_test


def prep_regression():
    X, y = make_regression(n_samples=200, n_features=6, noise=0.1, random_state=0)
    x_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=0)
    x_cal, x_test, y_cal, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)
    return x_train, y_train, x_cal, y_cal, x_test, y_test


def test_ce_presence_and_wrapper():
    model = RandomForestClassifier(random_state=0)
    wrapper = ensure_ce_first_wrapper(model)
    assert isinstance(wrapper, WrapCalibratedExplainer)


def test_ensure_ce_first_wrapper_existing():
    model = RandomForestClassifier(random_state=0)
    wrapper = WrapCalibratedExplainer(model)
    assert ensure_ce_first_wrapper(wrapper) is wrapper


def test_fit_and_calibrate_sets_state():
    x_train, y_train, x_cal, y_cal, _, _ = prep_classification()
    model = RandomForestClassifier(random_state=0)
    wrapper = ensure_ce_first_wrapper(model)
    wrapper = fit_and_calibrate(wrapper, x_train, y_train, x_cal, y_cal)
    assert wrapper.fitted is True
    assert wrapper.calibrated is True


def test_explain_and_narrate_requires_calibration():
    x_train, y_train, x_cal, y_cal, x_test, _ = prep_classification()
    model = RandomForestClassifier(random_state=0)
    wrapper = ensure_ce_first_wrapper(model)
    wrapper.fit(x_train, y_train)
    with pytest.raises(ValidationError):
        explain_and_narrate(wrapper, x_test[:1])
    wrapper.calibrate(x_cal, y_cal)
    explanations, narrative = explain_and_narrate(wrapper, x_test[:1])
    assert explanations is not None
    assert isinstance(narrative, str)


def test_explain_and_summarize_includes_conjunctions_and_uq():
    x_train, y_train, x_cal, y_cal, x_test, _ = prep_classification()
    model = RandomForestClassifier(random_state=0)
    wrapper = ensure_ce_first_wrapper(model)
    fit_and_calibrate(wrapper, x_train, y_train, x_cal, y_cal)

    payload = explain_and_summarize(
        wrapper,
        x_test[:2],
        mode="factual",
        add_conjunctions_params={"n_top_features": 2, "max_rule_size": 2},
    )

    assert "summary" in payload
    assert payload["summary"]["has_conjunctions"] is True
    assert isinstance(payload["narrative"], str)


def test_explain_and_summarize_supports_probabilistic_regression_threshold():
    x_train, y_train, x_cal, y_cal, x_test, _ = prep_regression()
    model = RandomForestRegressor(random_state=0)
    wrapper = ensure_ce_first_wrapper(model)
    fit_and_calibrate(wrapper, x_train, y_train, x_cal, y_cal)

    payload = explain_and_summarize(
        wrapper,
        x_test[:1],
        mode="factual",
        threshold=0.0,
    )

    assert "predictions" in payload
    assert "probability" in payload["predictions"]
    assert payload["summary"]["y_threshold"] is not None


def test_probabilistic_threshold_behavior():
    x_train, y_train, x_cal, y_cal, x_test, _ = prep_regression()
    model = RandomForestRegressor(random_state=0)
    wrapper = ensure_ce_first_wrapper(model)
    fit_and_calibrate(wrapper, x_train, y_train, x_cal, y_cal)
    scalar = get_calibrated_predictions(wrapper, x_test[:2], threshold=0.5)
    interval = get_calibrated_predictions(wrapper, x_test[:2], threshold=(-1.0, 1.0))
    assert "prediction" in scalar
    assert "prediction" in interval


def test_add_conjunctions_collection_and_single():
    x_train, y_train, x_cal, y_cal, x_test, _ = prep_classification()
    model = RandomForestClassifier(random_state=0)
    wrapper = ensure_ce_first_wrapper(model)
    fit_and_calibrate(wrapper, x_train, y_train, x_cal, y_cal)
    explanations = wrapper.explain_factual(x_test[:2])
    add_conjunctions(explanations, n_top_features=2, max_rule_size=2)
    add_conjunctions_to_one(explanations, 0, n_top_features=2, max_rule_size=2)
    assert explanations[0].has_conjunctive_rules is True


def test_enforce_ce_first_and_execute():
    x_train, y_train, x_cal, y_cal, x_test, _ = prep_classification()
    model = RandomForestClassifier(random_state=0)
    wrapper = ensure_ce_first_wrapper(model)
    fit_and_calibrate(wrapper, x_train, y_train, x_cal, y_cal)
    result = enforce_ce_first_and_execute(lambda w, x: w.explain_factual(x), wrapper, x_test[:1])
    assert result is not None


def test_probe_optional_features_warning():
    def fake_import(name):
        if name.startswith("crepes"):
            raise ImportError("missing")
        return importlib.import_module(name)

    report = probe_optional_features(import_module=fake_import)
    assert "warnings" in report
    assert any(
        "difficulty" in warning or "conditional" in warning for warning in report["warnings"]
    )


def test_policy_serialization_helpers():
    policy_dict = policy_as_dict()
    serialized = serialize_policy()
    assert isinstance(policy_dict, dict)
    assert '"required_class"' in serialized
