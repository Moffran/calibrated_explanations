import importlib
import types

import pytest
import numpy as np

from calibrated_explanations.ce_agent_utils import (
    add_conjunctions,
    add_conjunctions_to_one,
    enforce_ce_first_and_execute,
    ensure_ce_first_wrapper,
    explain_and_summarize,
    explain_and_narrate,
    format_guarded_audit_table,
    fit_and_calibrate,
    get_calibrated_predictions,
    get_uncalibrated_predictions,
    print_guarded_audit_table,
    probe_optional_features,
    summarize_explanations,
    set_telemetry_hook,
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


def test_enforce_ce_first_and_execute():
    x_train, y_train, x_cal, y_cal, x_test, _ = prep_classification()
    model = RandomForestClassifier(random_state=0)
    wrapper = ensure_ce_first_wrapper(model)
    fit_and_calibrate(wrapper, x_train, y_train, x_cal, y_cal)
    result = enforce_ce_first_and_execute(lambda w, x: w.explain_factual(x), wrapper, x_test[:1])
    assert result is not None


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


def test_summarize_explanations_handles_none_first_item() -> None:
    class Container(list):
        low_high_percentiles = (5, 95)
        y_threshold = None

    summary = summarize_explanations(Container([None]), top_k=2)
    assert summary["prediction"] is None
    assert summary["top_rules"] == []
    assert summary["has_conjunctions"] is False


def test_explain_and_narrate_handles_scalar_probability_and_missing_interval(monkeypatch) -> None:
    ce_utils = importlib.import_module("calibrated_explanations.ce_agent_utils")

    # Unit-test internal narrative formatting behavior without wrapper enforcement.
    monkeypatch.setattr(
        ce_utils,
        "enforce_ce_first_and_execute",
        lambda action, *args, **kwargs: action(*args, **kwargs),
    )

    class Explanation:
        prediction = None

        def to_narrative(self, format="short"):
            _ = format
            return "base narrative"

    class Wrapper:
        fitted = True
        calibrated = True
        learner = types.SimpleNamespace(predict_proba=True)

        def explain_factual(self, _x, **_kwargs):
            return [Explanation()]

        def predict(self, _x, **_kwargs):
            return [0.1]

        def predict_proba(self, _x, **_kwargs):
            return 0.42

    _explanations, narrative = ce_utils.explain_and_narrate(Wrapper(), [[1.0]], mode="factual")
    assert "Calibrated probability: 0.42" in narrative
    assert "Uncertainty interval: n/a" in narrative


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
    out = ce_utils.get_calibrated_predictions(
        FakeWrapper(), [[0.0]], threshold=0.5, calibrated=True
    )
    assert out["prediction"] == [0.2]
    assert out["probability"] == [[0.8, 0.2]]

    class LearnerNoPredict:
        def predict_proba(self, _x, **_kwargs):
            return [[0.5, 0.5]]

    unc = ce_utils.get_uncalibrated_predictions(
        types.SimpleNamespace(learner=LearnerNoPredict()), [[1]]
    )
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


def test_get_calibrated_predictions_tolerates_signature_probe_failures(monkeypatch):
    ce_utils = importlib.import_module("calibrated_explanations.ce_agent_utils")

    def _broken_signature(_callable):
        raise ValueError("boom")

    monkeypatch.setattr(ce_utils.inspect, "signature", _broken_signature)
    monkeypatch.setattr(ce_utils, "ensure_ce_first_wrapper", lambda w: w)

    class FakeWrapper:
        fitted = True
        calibrated = True
        learner = types.SimpleNamespace()

        def predict(self, _x, **_kwargs):
            return [0.4]

        def predict_proba(self, _x, **_kwargs):
            return [[0.6, 0.4]]

    monkeypatch.setattr(ce_utils, "ensure_ce_first_wrapper", lambda w: w)
    out = ce_utils.get_calibrated_predictions(FakeWrapper(), [[0.0]], threshold=0.25)
    assert out["probability"] == [[0.6, 0.4]]


def test_public_validation_and_summary_paths(monkeypatch):
    ce_utils = importlib.import_module("calibrated_explanations.ce_agent_utils")
    monkeypatch.setattr(ce_utils, "ensure_ce_first_wrapper", lambda w: w)

    class NotFittedWrapper:
        fitted = False
        calibrated = True
        learner = types.SimpleNamespace()

        def predict(self, _x, **_kwargs):
            return [1]

    with pytest.raises(NotFittedError):
        ce_utils.get_calibrated_predictions(NotFittedWrapper(), [[1]], calibrated=True)

    summary = ce_utils.summarize_explanations(
        [types.SimpleNamespace(prediction=[1, 2], rules={}, has_conjunctive_rules=False)]
    )
    assert summary["prediction"] == [1, 2]


def test_explain_and_narrate_covers_no_probability_and_fallback_action(monkeypatch):
    ce_utils = importlib.import_module("calibrated_explanations.ce_agent_utils")
    monkeypatch.setattr(
        ce_utils,
        "enforce_ce_first_and_execute",
        lambda action, *args, **kwargs: action(*args, **kwargs),
    )

    class FakeExplanation:
        prediction = {"predict": 1.0, "low": 1.0, "high": 1.0}
        rules = {"rule": [], "weight": []}

    class FakeWrapper:
        fitted = True
        calibrated = True
        learner = types.SimpleNamespace()

        def explain_factual(self, _x, **_kwargs):
            return [FakeExplanation()]

        def explore_alternatives(self, _x, **_kwargs):
            return [FakeExplanation()]

        def predict(self, _x, **_kwargs):
            return [1.0]

        def predict_proba(self, _x, **_kwargs):
            return [[0.1, 0.9]]

    _explanations, narrative = ce_utils.explain_and_narrate(FakeWrapper(), [[1.0]], mode="factual")
    assert "Review the most influential features" in narrative


def test_ensure_ce_first_wrapper_rejects_missing_attrs_and_methods(monkeypatch):
    ce_utils = importlib.import_module("calibrated_explanations.ce_agent_utils")

    class FakeWrapMissingAttrs:
        def __init__(self, _model):
            pass

    monkeypatch.setattr(ce_utils, "_require_ce", lambda: FakeWrapMissingAttrs)
    with pytest.raises(ModelNotSupportedError):
        ce_utils.ensure_ce_first_wrapper(object())

    class FakeWrapMissingMethods:
        fitted = True
        calibrated = True

        def __init__(self, _model):
            pass

        def fit(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(ce_utils, "_require_ce", lambda: FakeWrapMissingMethods)
    with pytest.raises(ModelNotSupportedError):
        ce_utils.ensure_ce_first_wrapper(object())


def test_get_calibrated_predictions_covers_kwarg_filter_variants(monkeypatch):
    ce_utils = importlib.import_module("calibrated_explanations.ce_agent_utils")
    monkeypatch.setattr(ce_utils, "ensure_ce_first_wrapper", lambda w: w)

    class StrictWrapper:
        fitted = True
        calibrated = True
        learner = types.SimpleNamespace()

        def predict(self, x):
            return [len(x)]

    out = ce_utils.get_calibrated_predictions(StrictWrapper(), [[0.0]], calibrated=False, foo=1)
    assert out["prediction"] == [1]

    class KwargNamedWrapper:
        fitted = True
        calibrated = True
        learner = types.SimpleNamespace(predict_proba=True)

        def predict(self, x, kwargs=None):
            return [len(x) + (0 if kwargs is None else 1)]

        def predict_proba(self, x, kwargs=None):
            return [[0.5, 0.5]]

    out2 = ce_utils.get_calibrated_predictions(
        KwargNamedWrapper(),
        [[0.0]],
        calibrated=False,
        kwargs=1,
    )
    assert out2["prediction"] == [2]
    assert out2["probability"] == [[0.5, 0.5]]


def test_format_guarded_audit_table_with_mapping_payload():
    payload = {
        "summary": {
            "n_instances": 1,
            "intervals_tested": 2,
            "intervals_conforming": 1,
            "intervals_removed_guard": 1,
            "intervals_emitted": 1,
        },
        "instances": [
            {
                "instance_index": 0,
                "summary": {
                    "intervals_tested": 2,
                    "intervals_conforming": 1,
                    "intervals_removed_guard": 1,
                    "intervals_emitted": 1,
                },
                "intervals": [
                    {
                        "feature": 0,
                        "feature_name": "f0",
                        "lower": 0.0,
                        "upper": 1.0,
                        "p_value": 0.8,
                        "conforming": True,
                        "emitted": True,
                        "emission_reason": "emitted",
                    },
                    {
                        "feature": 1,
                        "feature_name": "f1",
                        "lower": 1.0,
                        "upper": 2.0,
                        "p_value": 0.01,
                        "conforming": False,
                        "emitted": False,
                        "emission_reason": "removed_guard",
                    },
                ],
            }
        ],
    }
    text = format_guarded_audit_table(payload)
    assert "Guarded Audit Summary" in text
    assert "removed_guard=1" in text
    assert "f0" in text
    assert "f1" in text
    assert "reason_counts:" in text
    assert "legend:" in text


def test_format_guarded_audit_table_accepts_object_with_getter():
    class AuditObj:
        def get_guarded_audit(self):
            return {
                "instance_index": 0,
                "summary": {
                    "intervals_tested": 1,
                    "intervals_conforming": 1,
                    "intervals_removed_guard": 0,
                    "intervals_emitted": 1,
                },
                "intervals": [
                    {
                        "feature": 0,
                        "feature_name": "f0",
                        "lower": -np.inf,
                        "upper": np.inf,
                        "p_value": 0.5,
                        "conforming": True,
                        "emitted": True,
                        "emission_reason": "emitted",
                    }
                ],
            }

    text = format_guarded_audit_table(AuditObj())
    assert "instances=1" in text
    assert "f0" in text


def test_print_guarded_audit_table_emits_text(capsys):
    print_guarded_audit_table(
        {
            "instance_index": 0,
            "summary": {
                "intervals_tested": 0,
                "intervals_conforming": 0,
                "intervals_removed_guard": 0,
                "intervals_emitted": 0,
            },
            "intervals": [],
        }
    )
    out = capsys.readouterr().out
    assert "Guarded Audit Summary" in out


def test_format_guarded_audit_table_rounds_bounds_for_readability():
    payload = {
        "instance_index": 0,
        "summary": {
            "intervals_tested": 1,
            "intervals_conforming": 1,
            "intervals_removed_guard": 0,
            "intervals_emitted": 1,
        },
        "intervals": [
            {
                "feature": 0,
                "feature_name": "f0",
                "lower": -np.inf,
                "upper": 5.3500001430511475,
                "p_value": 0.294712312,
                "conforming": True,
                "emitted": True,
                "emission_reason": "emitted",
            }
        ],
    }
    text = format_guarded_audit_table(payload, bound_decimals=3, pvalue_decimals=3)
    assert "(-inf, 5.35]" in text
    assert "0.295" in text


def test_format_guarded_audit_table_mrg_column_present():
    """The mrg column header and Y/N values must appear in the table."""
    payload = {
        "instance_index": 0,
        "summary": {
            "intervals_tested": 2,
            "intervals_conforming": 2,
            "intervals_removed_guard": 0,
            "intervals_emitted": 2,
        },
        "intervals": [
            {
                "feature": 0,
                "feature_name": "f0",
                "lower": 0.0,
                "upper": 4.0,
                "emitted_lower": 1.0,
                "emitted_upper": 3.0,
                "p_value": 0.9,
                "conforming": True,
                "is_merged": True,
                "emitted": True,
                "emission_reason": "emitted",
            },
            {
                "feature": 1,
                "feature_name": "f1",
                "lower": 0.0,
                "upper": 1.0,
                "emitted_lower": 0.0,
                "emitted_upper": 1.0,
                "p_value": 0.8,
                "conforming": True,
                "is_merged": False,
                "emitted": True,
                "emission_reason": "emitted",
            },
        ],
    }
    text = format_guarded_audit_table(payload)
    assert "mrg" in text
    lines = text.splitlines()
    # Find the two data rows (after the divider)
    data_lines = [ln for ln in lines if ln.startswith("   0")]
    assert len(data_lines) == 2
    # First row (merged): mrg column must be Y
    assert " Y " in data_lines[0]
    # Second row (not merged): mrg column must be N
    assert " N " in data_lines[1]


def test_format_guarded_audit_table_merged_uses_emitted_bounds():
    """A merged row must display emitted_lower/emitted_upper, not raw lower/upper."""
    payload = {
        "instance_index": 0,
        "summary": {
            "intervals_tested": 1,
            "intervals_conforming": 1,
            "intervals_removed_guard": 0,
            "intervals_emitted": 1,
        },
        "intervals": [
            {
                "feature": 0,
                "feature_name": "f0",
                "lower": 0.0,
                "upper": 4.0,
                "emitted_lower": 1.0,
                "emitted_upper": 3.0,
                "p_value": 0.9,
                "conforming": True,
                "is_merged": True,
                "emitted": True,
                "emission_reason": "emitted",
            }
        ],
    }
    text = format_guarded_audit_table(payload)
    assert "(1, 3]" in text
    assert "(0, 4]" not in text


def test_format_guarded_audit_table_non_merged_uses_raw_bounds_when_equal():
    """A non-merged row where emitted bounds equal raw bounds shows raw bounds."""
    payload = {
        "instance_index": 0,
        "summary": {
            "intervals_tested": 1,
            "intervals_conforming": 1,
            "intervals_removed_guard": 0,
            "intervals_emitted": 1,
        },
        "intervals": [
            {
                "feature": 0,
                "feature_name": "f0",
                "lower": 0.0,
                "upper": 2.0,
                "emitted_lower": 0.0,
                "emitted_upper": 2.0,
                "p_value": 0.7,
                "conforming": True,
                "is_merged": False,
                "emitted": True,
                "emission_reason": "emitted",
            }
        ],
    }
    text = format_guarded_audit_table(payload)
    assert "(0, 2]" in text


def test_format_guarded_audit_table_legend_mentions_mrg():
    """The legend must explain the mrg column."""
    payload = {
        "instance_index": 0,
        "summary": {
            "intervals_tested": 1,
            "intervals_conforming": 1,
            "intervals_removed_guard": 0,
            "intervals_emitted": 1,
        },
        "intervals": [
            {
                "feature": 0,
                "feature_name": "f0",
                "lower": 0.0,
                "upper": 1.0,
                "p_value": 0.5,
                "conforming": True,
                "is_merged": False,
                "emitted": True,
                "emission_reason": "emitted",
            }
        ],
    }
    text = format_guarded_audit_table(payload)
    assert "mrg=Y" in text
