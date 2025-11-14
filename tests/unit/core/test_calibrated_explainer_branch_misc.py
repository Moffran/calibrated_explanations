"""Additional branch coverage for :mod:`calibrated_explanations.core.calibrated_explainer`."""

from __future__ import annotations

import types

import numpy as np
import pytest

from calibrated_explanations.core import calibrated_explainer as explainer_module
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.explain.orchestrator import ExplanationOrchestrator
from calibrated_explanations.core.exceptions import (
    ConfigurationError,
    DataShapeError,
    ValidationError,
)
from calibrated_explanations.utils.discretizers import RegressorDiscretizer


def _make_base_explainer() -> CalibratedExplainer:
    """Return a lightweight explainer instance with minimal state."""

    explainer = CalibratedExplainer.__new__(CalibratedExplainer)
    explainer._pyproject_plots = {}
    explainer._plot_style_override = None
    explainer._plot_style_chain = ("legacy",)
    explainer._bridge_monitors = {}
    explainer.discretizer = None
    explainer.sample_percentiles = [25, 50, 75]
    explainer.x_cal = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=float)
    explainer.y_cal = np.array([0, 1], dtype=int)
    explainer._X_cal = explainer.x_cal
    explainer._feature_names = [f"f{i}" for i in range(explainer.x_cal.shape[1])]
    explainer.bins = None
    explainer.interval_learner = None
    explainer.feature_values = {i: [] for i in range(explainer.x_cal.shape[1])}
    explainer.categorical_features = []
    explainer._CalibratedExplainer__initialized = False
    explainer.mode = "classification"
    explainer.learner = object()
    explainer.difficulty_estimator = None
    explainer.predict_function = lambda x, **_: x  # type: ignore[assignment]
    # Initialize orchestrator for tests that call its methods
    explainer._explanation_orchestrator = ExplanationOrchestrator(explainer)
    return explainer


def test_build_plot_style_chain_adds_defaults(monkeypatch):
    """Test that plot chain adds default when no overrides present."""
    explainer = _make_base_explainer()
    monkeypatch.delenv("CE_PLOT_STYLE", raising=False)
    monkeypatch.delenv("CE_PLOT_STYLE_FALLBACKS", raising=False)

    chain = explainer._explanation_orchestrator._build_plot_chain()

    assert chain == ("plot_spec.default", "legacy")


def test_build_plot_style_chain_inserts_before_legacy(monkeypatch):
    """Test that plot style override is inserted before legacy."""
    explainer = _make_base_explainer()
    explainer._plot_style_override = "legacy"
    monkeypatch.delenv("CE_PLOT_STYLE", raising=False)
    monkeypatch.delenv("CE_PLOT_STYLE_FALLBACKS", raising=False)

    chain = explainer._explanation_orchestrator._build_plot_chain()

    assert chain == ("plot_spec.default", "legacy")


def test_slice_threshold_branches_exercised():
    sentinel = object()
    assert CalibratedExplainer._slice_threshold(sentinel, 0, 1, 1) is sentinel

    mismatched = [0.1, 0.2]
    assert CalibratedExplainer._slice_threshold(mismatched, 0, 1, 3) is mismatched

    data = [1, 2, 3, 4]
    assert CalibratedExplainer._slice_threshold(data, 1, 3, len(data)) == [2, 3]

    array = np.arange(5)
    result = CalibratedExplainer._slice_threshold(array, 1, 4, len(array))
    assert np.all(result == np.array([1, 2, 3]))


def test_slice_bins_handles_collections():
    assert CalibratedExplainer._slice_bins(None, 0, 1) is None

    bins = ["a", "b", "c"]
    assert CalibratedExplainer._slice_bins(bins, 1, 3) == ["b", "c"]

    array_bins = np.array([[1, 2], [3, 4], [5, 6]])
    sliced = CalibratedExplainer._slice_bins(array_bins, 0, 2)
    assert np.all(sliced == array_bins[:2])


def test_build_instance_telemetry_payload_handles_variants():
    explainer = _make_base_explainer()

    class _DummyExplanation:
        def __init__(self, payload):
            self._payload = payload

        def to_telemetry(self):
            return self._payload

    assert explainer._build_instance_telemetry_payload([]) == {}

    payload = {"foo": "bar"}
    explanations = [_DummyExplanation(payload)]
    assert explainer._build_instance_telemetry_payload(explanations) == payload

    explanations = [_DummyExplanation([1, 2, 3])]
    assert explainer._build_instance_telemetry_payload(explanations) == {}


def test_infer_explanation_mode_prefers_discretizer():
    explainer = _make_base_explainer()
    assert explainer._infer_explanation_mode() == "factual"

    data = np.array([[0.0], [1.0]])
    labels = np.array([0.0, 1.0])
    discretizer = RegressorDiscretizer(
        data,
        categorical_features=[],
        feature_names=["f0"],
        labels=labels,
        random_state=0,
    )
    explainer.discretizer = discretizer
    assert explainer._infer_explanation_mode() == "alternative"


def test_set_mode_variants(monkeypatch):
    explainer = _make_base_explainer()

    explainer._CalibratedExplainer__set_mode("classification", initialize=False)
    assert explainer.mode == "classification"
    assert explainer.num_classes == 2

    explainer._CalibratedExplainer__set_mode("regression", initialize=False)
    assert explainer.mode == "regression"
    assert explainer.num_classes == 0

    with pytest.raises(ValidationError):
        explainer._CalibratedExplainer__set_mode("unsupported", initialize=False)


def test_get_sigma_test_uses_difficulty_estimator():
    explainer = _make_base_explainer()
    values = explainer._get_sigma_test(np.zeros((3, explainer.num_features)))
    assert np.all(values == 1)

    class _Estimator:
        def apply(self, x):
            return np.full(x.shape[0], 0.42)

    explainer.difficulty_estimator = _Estimator()
    updated = explainer._get_sigma_test(np.zeros((2, explainer.num_features)))
    assert np.all(updated == 0.42)


def test_update_interval_learner_branches(monkeypatch):
    explainer = _make_base_explainer()

    explainer.is_fast = types.MethodType(lambda self: True, explainer)
    with pytest.raises(ConfigurationError):
        explainer._CalibratedExplainer__update_interval_learner(explainer.x_cal, explainer.y_cal)

    explainer.is_fast = types.MethodType(lambda self: False, explainer)

    created_instances: list[tuple] = []

    class _RecorderVA:
        def __init__(self, *args, **kwargs):
            created_instances.append((args, kwargs))

    monkeypatch.setattr(explainer_module, "VennAbers", _RecorderVA)

    explainer._CalibratedExplainer__update_interval_learner(explainer.x_cal, explainer.y_cal)
    assert created_instances and explainer._CalibratedExplainer__initialized is True

    explainer.mode = "regression"
    explainer.interval_learner = []
    with pytest.raises(ConfigurationError):
        explainer._CalibratedExplainer__update_interval_learner(explainer.x_cal, explainer.y_cal)

    class _IntervalLearner:
        def __init__(self):
            self.calls: list[tuple] = []

        def insert_calibration(self, xs, ys, bins=None):
            self.calls.append((xs, ys, bins))

    explainer.interval_learner = _IntervalLearner()
    explainer.bins = np.arange(explainer.y_cal.shape[0])
    explainer._CalibratedExplainer__update_interval_learner(
        explainer.x_cal, explainer.y_cal, bins=np.array([10, 11])
    )
    assert explainer.interval_learner.calls
    assert explainer._CalibratedExplainer__initialized is True


def test_reinitialize_updates_state(monkeypatch):
    explainer = _make_base_explainer()

    appended: list[tuple] = []

    def append_cal(self, xs, ys):
        appended.append((xs, ys))

    explainer.append_cal = types.MethodType(append_cal, explainer)
    explainer.bins = np.array([0.5, 0.6])

    checked: list[object] = []

    def fake_check_is_fitted(learner):
        checked.append(learner)

    monkeypatch.setattr(explainer_module, "check_is_fitted", fake_check_is_fitted)

    update_calls: list[tuple] = []

    def fake_update(self, xs, ys, bins=None):
        update_calls.append((self, xs, ys, bins))

    monkeypatch.setattr(
        "calibrated_explanations.core.calibration_helpers.update_interval_learner",
        fake_update,
    )

    learner = object()
    xs = np.array([[2.0, 3.0], [4.0, 5.0]])
    ys = np.array([1, 0])
    bins = np.array([7, 8])

    explainer.reinitialize(learner, xs=xs, ys=ys, bins=bins)

    assert checked == [learner]
    assert appended == [(xs, ys)]
    assert np.all(explainer.bins == np.array([0.5, 0.6, 7, 8]))
    assert update_calls and update_calls[0][0] is explainer
    assert explainer._CalibratedExplainer__initialized is True

    with pytest.raises(DataShapeError):
        explainer.reinitialize(learner, xs=xs, ys=ys, bins=np.array([1]))

    init_calls: list[tuple] = []

    def fake_init(self):
        init_calls.append((self,))

    monkeypatch.setattr(
        "calibrated_explanations.core.calibration_helpers.initialize_interval_learner",
        fake_init,
    )

    explainer.reinitialize(learner)
    assert init_calls and init_calls[0][0] is explainer
