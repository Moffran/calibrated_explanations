import importlib
import types

import pytest
import numpy as np

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
    get_uncalibrated_predictions,
    policy_as_dict,
    probe_optional_features,
    serialize_policy,
    set_telemetry_hook,
    wrap_and_explain,
)

from calibrated_explanations.core.exceptions import ValidationError
from calibrated_explanations.core.exceptions import (
    ConfigurationError,
    ModelNotSupportedError,
    NotFittedError,
)

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




def test_telemetry_hook_receives_events():
    seen = []

    def hook(event):
        seen.append((event.name, dict(event.payload)))

    set_telemetry_hook(hook)
    x_train, y_train, x_cal, y_cal, _, _ = prep_classification()
    model = RandomForestClassifier(random_state=0)
    wrapper = ensure_ce_first_wrapper(model)
    fit_and_calibrate(wrapper, x_train, y_train, x_cal, y_cal)
    set_telemetry_hook(None)

    names = [name for name, _ in seen]
    assert "ce.fit_and_calibrate.start" in names
    assert "ce.fit_and_calibrate.end" in names


def test_probe_optional_features_with_find_spec_gate():
    def finder(name):
        if "crepes.extras" in name:
            return None
        return object()

    report = probe_optional_features(find_spec=finder)
    assert report["available"]["conditional/Mondrian categorizer"] is False
    assert report["available"]["difficulty estimation"] is False


def test_get_uncalibrated_predictions_without_predict_proba():
    x_train, y_train, x_cal, y_cal, x_test, _ = prep_classification()
    model = RandomForestClassifier(random_state=0)
    wrapper = ensure_ce_first_wrapper(model)
    fit_and_calibrate(wrapper, x_train, y_train, x_cal, y_cal)
    wrapper.learner = types.SimpleNamespace(
        predict=lambda x: np.zeros(len(x)),
    )

    payload = get_uncalibrated_predictions(wrapper, x_test[:2])
    assert payload["prediction"] is not None
    assert payload["probability"] is None




def test_ensure_ce_first_wrapper_raises_for_missing_library(monkeypatch):
    ce_utils = importlib.import_module("calibrated_explanations.ce_agent_utils")

    monkeypatch.setattr(
        ce_utils.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("missing")),
    )
    with pytest.raises(ConfigurationError):
        ce_utils.ensure_ce_first_wrapper(object())


def test_ensure_ce_first_wrapper_raises_for_missing_wrapper_export(monkeypatch):
    ce_utils = importlib.import_module("calibrated_explanations.ce_agent_utils")
    monkeypatch.setattr(
        ce_utils.importlib,
        "import_module",
        lambda _name: types.SimpleNamespace(),
    )
    with pytest.raises(ConfigurationError):
        ce_utils.ensure_ce_first_wrapper(object())


def test_fit_and_calibrate_raises_when_fit_or_calibrate_state_not_set(monkeypatch):
    ce_utils = importlib.import_module("calibrated_explanations.ce_agent_utils")

    class FakeWrapper:
        fitted = False
        calibrated = False

        def fit(self, _x, _y, **_kwargs):
            self.fitted = False

        def calibrate(self, _x, _y, **_kwargs):
            self.calibrated = False

    monkeypatch.setattr(ce_utils, "ensure_ce_first_wrapper", lambda w: w)
    with pytest.raises(NotFittedError):
        ce_utils.fit_and_calibrate(FakeWrapper(), [1], [1], [1], [1])

    class FakeWrapper2(FakeWrapper):
        def fit(self, _x, _y, **_kwargs):
            self.fitted = True

    with pytest.raises(ValidationError):
        ce_utils.fit_and_calibrate(
            FakeWrapper2(),
            [1],
            [1],
            [1],
            [1],
            learner={"alpha": 1},
            explainer={"beta": 2},
        )


def test_explain_and_narrate_invalid_mode_and_narrative_fallback(monkeypatch):
    ce_utils = importlib.import_module("calibrated_explanations.ce_agent_utils")
    monkeypatch.setattr(
        ce_utils,
        "enforce_ce_first_and_execute",
        lambda action, *args, **kwargs: action(*args, **kwargs),
    )

    class FakeExplanation:
        prediction = {"predict": 1.0, "low": 0.8, "high": 1.2}
        rules = {"rule": ["r1"], "weight": [0.4]}

        def to_narrative(self, **_kwargs):
            raise RuntimeError("fail narrative")

    class FakeWrapper:
        fitted = True
        calibrated = True
        learner = types.SimpleNamespace(predict_proba=lambda _x: None)

        def explain_factual(self, _x, **_kwargs):
            return [FakeExplanation()]

        def explore_alternatives(self, _x, **_kwargs):
            return [FakeExplanation()]

        def predict(self, _x, **_kwargs):
            return [1.0]

        def predict_proba(self, _x, **_kwargs):
            return [[0.1, 0.9]]

    wrapper = FakeWrapper()
    with pytest.raises(ValidationError):
        ce_utils.explain_and_narrate(wrapper, [[1.0]], mode="unsupported")

    explanations, narrative = ce_utils.explain_and_narrate(wrapper, [[1.0]], mode="factual")
    assert len(explanations) == 1
    assert "Prediction" in narrative


def test_conjunction_helpers_raise_for_unsupported_objects():
    with pytest.raises(ModelNotSupportedError):
        add_conjunctions(object())

    with pytest.raises(ModelNotSupportedError):
        add_conjunctions_to_one([object()], 0)


def test_get_calibrated_predictions_and_uncalibrated_paths(monkeypatch):
    ce_utils = importlib.import_module("calibrated_explanations.ce_agent_utils")

    class FakeWrapper:
        fitted = True
        calibrated = True
        learner = types.SimpleNamespace()

        def predict(self, _x, **_kwargs):
            return [0.2]

        def predict_proba(self, _x, **_kwargs):
            return [[0.8, 0.2]]

    monkeypatch.setattr(ce_utils, "ensure_ce_first_wrapper", lambda w: w)
    out = ce_utils.get_calibrated_predictions(FakeWrapper(), [[0.0]], threshold=0.5, calibrated=True)
    assert out["prediction"] == [0.2]
    assert out["probability"] == [[0.8, 0.2]]

    class LearnerNoPredict:
        def predict_proba(self, _x, **_kwargs):
            return [[0.5, 0.5]]

    unc = ce_utils.get_uncalibrated_predictions(types.SimpleNamespace(learner=LearnerNoPredict()), [[1]])
    assert unc["prediction"] is None
    assert unc["probability"] == [[0.5, 0.5]]


def test_wrap_and_execute_wrapper_resolution_paths(monkeypatch):
    ce_utils = importlib.import_module("calibrated_explanations.ce_agent_utils")

    class FakeWrap:
        fitted = True
        calibrated = True

        def fit(self):
            return None

        def calibrate(self):
            return None

        def predict(self):
            return None

        def predict_proba(self):
            return None

        def explain_factual(self):
            return None

        def explore_alternatives(self):
            return None

        def plot(self):
            return None

    monkeypatch.setattr(ce_utils, "_require_ce", lambda: FakeWrap)
    result = ce_utils.enforce_ce_first_and_execute(lambda wrapper=None: "ok", wrapper=FakeWrap())
    assert result == "ok"

    with pytest.raises(ModelNotSupportedError):
        ce_utils.enforce_ce_first_and_execute(lambda _w: None, object())


def test_explain_and_summarize_with_percentiles_and_no_conjunctions(monkeypatch):
    ce_utils = importlib.import_module("calibrated_explanations.ce_agent_utils")

    class FakeWrapper:
        fitted = True
        calibrated = True
        learner = types.SimpleNamespace()

        def predict(self, _x, **_kwargs):
            return [0.3]

        def explain_factual(self, _x, **_kwargs):
            exp = types.SimpleNamespace(
                prediction={"predict": 0.3, "low": 0.2, "high": 0.4},
                rules={"rule": ["r"], "weight": [0.1]},
                has_conjunctive_rules=False,
                conjunctive_rules={},
            )
            return [exp]

    monkeypatch.setattr(ce_utils, "ensure_ce_first_wrapper", lambda w: w)
    monkeypatch.setattr(
        ce_utils,
        "explain_and_narrate",
        lambda *_args, **_kwargs: (
            [
                types.SimpleNamespace(
                    prediction={"predict": 0.3, "low": 0.2, "high": 0.4},
                    rules={"rule": ["r"], "weight": [0.1]},
                    has_conjunctive_rules=False,
                    conjunctive_rules={},
                )
            ],
            "narrative",
        ),
    )

    payload = ce_utils.explain_and_summarize(
        FakeWrapper(),
        [[0.0]],
        low_high_percentiles=(5, 95),
        add_conjunctions_params=None,
    )
    assert payload["predictions"]["prediction"] == [0.3]
    assert payload["summary"]["prediction"]["predict"] == 0.3


def test_get_calibrated_predictions_without_validation_or_predict_proba(monkeypatch):
    ce_utils = importlib.import_module("calibrated_explanations.ce_agent_utils")

    class W:
        fitted = False
        calibrated = False
        learner = types.SimpleNamespace()

        def predict(self, _x, **_kwargs):
            return [1]

    monkeypatch.setattr(ce_utils, "ensure_ce_first_wrapper", lambda w: w)
    out = ce_utils.get_calibrated_predictions(W(), [[1]], calibrated=False)
    assert out["prediction"] == [1]
    assert out["probability"] is None


def test_wrap_and_explain_plot_failure_is_tolerated(monkeypatch):
    ce_utils = importlib.import_module("calibrated_explanations.ce_agent_utils")

    class Explanation:
        def plot(self):
            raise RuntimeError("plot unavailable")

    monkeypatch.setattr(
        ce_utils,
        "ensure_ce_first_wrapper",
        lambda _model: types.SimpleNamespace(),
    )
    monkeypatch.setattr(
        ce_utils,
        "fit_and_calibrate",
        lambda wrapper, *_args, **_kwargs: wrapper,
    )
    monkeypatch.setattr(
        ce_utils,
        "explain_and_narrate",
        lambda *_args, **_kwargs: ([Explanation()], "n"),
    )
    out = ce_utils.wrap_and_explain("m", [1], [1], [1], [1], [1])
    assert out["plot"] is None


def test_probe_optional_features_warns_when_missing() -> None:
    def broken_import(_name):
        raise ImportError("missing")

    with pytest.warns(UserWarning):
        report = probe_optional_features(import_module=broken_import)
    assert report["warnings"]
