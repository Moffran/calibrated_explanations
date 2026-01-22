import numpy as np
import pytest

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.utils.exceptions import NotFittedError, ValidationError


def make_stub_explainer():
    expl = object.__new__(CalibratedExplainer)
    # Minimal defaults to avoid attribute errors in thin delegators
    expl.initialized = False
    expl.mode = "classification"
    expl.bins = None
    expl.discretizer = None
    expl.learner = None
    expl.latest_explanation = None
    expl.init_time = 0.0
    expl.interval_summary = None
    expl.verbose = False
    expl.sample_percentiles = [25, 50, 75]
    expl.seed = 42
    expl.feature_names = None
    expl.categorical_features = None
    expl.categorical_labels = None
    expl.class_labels = None
    expl.perf_parallel = None
    return expl


def test_require_plugin_manager_raises_not_fitted():
    expl = make_stub_explainer()
    # The stub does not have a plugin manager initialized
    with pytest.raises(NotFittedError):
        expl.require_plugin_manager()


def test_is_initialized_and_deprecated_alias():
    expl = make_stub_explainer()
    expl.initialized = True
    assert expl.initialized is True
    # deprecated alias
    assert expl.is_initialized is True


def test_is_mondrian_and_is_multiclass_and_fast_aliases():
    expl = make_stub_explainer()
    expl.bins = [0, 1]
    assert expl.is_mondrian() is True
    expl.num_classes = 3
    assert expl.is_multiclass() is True
    # fast aliases
    expl.fast = True
    assert expl.fast is True
    expl.fast = False
    assert expl.fast is False


def test_enable_fast_mode_calls_initializer():
    expl = make_stub_explainer()
    # ensure fast is currently off
    expl.fast = False

    def init_fn():
        init_fn.called = True

    init_fn.called = False
    # attach public alias used by enable_fast_mode
    expl.initialize_interval_learner_for_fast_explainer = init_fn
    expl.enable_fast_mode()
    assert init_fn.called is True
    assert expl.fast is True


def test_set_mode_invalid_raises_validation_error():
    expl = make_stub_explainer()
    expl.y_cal = np.array([0, 1, 0])
    with pytest.raises(ValidationError):
        expl.set_mode("invalid_mode", initialize=False)


def test_repr_verbose_and_non_verbose_runs():
    expl = make_stub_explainer()
    expl.mode = "classification"
    expl.bins = None
    expl.discretizer = None
    expl.learner = "dummy"
    expl.verbose = False
    # should not raise
    r = repr(expl)
    assert isinstance(r, str)
    # verbose path
    expl.verbose = True

    class DummyExplanation:
        total_explain_time = 0.123

    expl.latest_explanation = DummyExplanation()
    expl.feature_names = ["a", "b"]
    expl.categorical_features = [0]
    expl.categorical_labels = {0: {0: "x"}}
    expl.class_labels = {0: "neg", 1: "pos"}
    r2 = repr(expl)
    assert "CalibratedExplainer" in r2


def test_predict_proba_uncalibrated_uq_interval_binary():
    expl = make_stub_explainer()

    class DummyLearner:
        def predict_proba(self, x):
            return np.array([[0.3, 0.7]])

    expl.learner = DummyLearner()
    expl.mode = "classification"
    proba, (low, high) = expl.predict_proba(x=np.zeros((1, 1)), calibrated=False, uq_interval=True)
    assert isinstance(proba, np.ndarray)
    assert np.allclose(low, proba[:, 1])
    assert np.allclose(high, proba[:, 1])
