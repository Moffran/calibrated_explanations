from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from calibrated_explanations.utils.exceptions import (
    ConfigurationError,
    DataShapeError,
    ValidationError,
)
from external_plugins.fast_explanations.pipeline import FastExplanationPipeline
from external_plugins.integrations.lime_pipeline import LimePipeline
from external_plugins.integrations.shap_pipeline import ShapPipeline


class _DummyCollection:
    def __init__(self, _explainer, x_test, threshold, bins, condition_source=None):
        self.x_test = np.asarray(x_test)
        self.threshold = threshold
        self.bins = bins
        self.condition_source = condition_source
        self.low_high_percentiles = None
        self.finalized = None

    def finalize_fast(
        self, feature_weights, feature_predict, prediction, instance_time=None, total_time=None
    ):
        self.finalized = {
            "feature_weights": feature_weights,
            "feature_predict": feature_predict,
            "prediction": prediction,
            "instance_time": instance_time,
            "total_time": total_time,
        }


class DummyPredictOrchestrator:
    def __init__(self, classes=None):
        self.classes_state = classes

    def predict_internal(self, x, **_kwargs):
        arr = np.asarray(x)
        n = len(arr)
        pred = np.full(n, 0.6, dtype=float)
        low = np.full(n, 0.4, dtype=float)
        high = np.full(n, 0.8, dtype=float)
        classes = self.classes_state if self.classes_state is not None else np.zeros(n, dtype=int)
        return pred, low, high, np.asarray(classes)


class DummyLimeExplanation:
    def __init__(self, value0=0.1, value1=0.2, proba=0.7):
        self.local_exp = {1: [(0, value0), (1, value1)]}
        self.predict_proba = [1.0 - proba, proba]


class DummyLimeExplainer:
    def explain_instance(self, _instance, predict_fn=None, num_features=None):
        assert predict_fn is not None
        assert num_features is not None
        return DummyLimeExplanation()


class _DummyShapHelper:
    def __init__(self, _explainer):
        self.enabled_state = False

    def set_enabled(self, enabled):
        self.enabled_state = bool(enabled)

    def is_enabled(self):
        return self.enabled_state

    def preload(self, num_test=None):
        if num_test is None:
            return None, None
        return lambda x, **kwargs: {"x": np.asarray(x), "kwargs": kwargs}, {"ref": True}


def make_lime_explainer(*, mode="classification", multiclass=False, mondrian=False):
    classes = np.array([1, 0], dtype=int) if multiclass else np.array([1, 1], dtype=int)
    explainer = SimpleNamespace(
        num_features=2,
        mode=mode,
        condition_source="prediction",
        prediction_orchestrator=DummyPredictOrchestrator(classes=classes),
        latest_explanation=None,
    )
    explainer.is_mondrian = lambda: mondrian
    explainer.is_multiclass = lambda: multiclass
    return explainer


def make_fast_explainer(*, mode="classification", is_fast=False, multiclass=False, mondrian=False):
    classes = np.array([2, 1], dtype=int) if multiclass else np.array([1, 1], dtype=int)
    interval_learner = {
        2: SimpleNamespace(
            predict_proba=lambda x, bins=None: np.column_stack(
                [np.full(len(np.asarray(x)), 0.2), np.full(len(np.asarray(x)), 0.8)]
            )
        )
    }
    explainer = SimpleNamespace(
        num_features=2,
        mode=mode,
        condition_source="prediction",
        prediction_orchestrator=DummyPredictOrchestrator(classes=classes),
        latest_explanation=None,
        last_explanation_mode=None,
        y_cal=np.array([0.0, 1.0]),
        scaled_y_cal=np.array([0.0, 1.0]),
        features_to_ignore=[],
        interval_learner=interval_learner,
        _perf_parallel=None,
        bins=None,
        x_cal=np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]),
    )
    explainer.fast_state = is_fast

    def _is_fast():
        return explainer.fast_state

    def _enable_fast_mode():
        explainer.fast_state = True

    explainer.is_fast = _is_fast
    explainer.enable_fast_mode = _enable_fast_mode
    explainer.is_mondrian = lambda: mondrian
    explainer.is_multiclass = lambda: multiclass
    return explainer


def test_should_toggle_shap_enabled_and_delegate_explain(monkeypatch):
    import external_plugins.integrations.shap_pipeline as shap_pipeline_mod

    monkeypatch.setattr(shap_pipeline_mod, "ShapHelper", _DummyShapHelper)
    pipeline = ShapPipeline(explainer=SimpleNamespace())

    assert pipeline.is_shap_enabled() is False
    assert pipeline.is_shap_enabled(True) is True

    out = pipeline.explain(np.array([[1.0, 2.0]]), check_additivity=False)
    assert out["x"].shape == (1, 2)
    assert out["kwargs"]["check_additivity"] is False


def test_should_raise_when_shap_dependency_missing(monkeypatch):
    pipeline = ShapPipeline(explainer=SimpleNamespace())
    monkeypatch.setattr(pipeline, "preload_shap", lambda num_test=None: (None, None))

    with pytest.raises(ConfigurationError, match="optional dependency is missing"):
        pipeline.explain(np.array([[1.0, 2.0]]))


def test_should_raise_for_lime_shape_and_mondrian_contract(monkeypatch):
    import external_plugins.integrations.lime_pipeline as lime_pipeline_mod

    monkeypatch.setattr(lime_pipeline_mod, "CalibratedExplanations", _DummyCollection)

    shape_explainer = make_lime_explainer()
    shape_pipeline = LimePipeline(shape_explainer)
    monkeypatch.setattr(shape_pipeline, "_preload_lime", lambda: (DummyLimeExplainer(), None))
    with pytest.raises(DataShapeError, match="number of features"):
        shape_pipeline.explain(np.array([[1.0, 2.0, 3.0]]))

    mondrian_explainer = make_lime_explainer(mondrian=True)
    mondrian_pipeline = LimePipeline(mondrian_explainer)
    monkeypatch.setattr(mondrian_pipeline, "_preload_lime", lambda: (DummyLimeExplainer(), None))
    with pytest.raises(ValidationError, match="bins parameter must be specified"):
        mondrian_pipeline.explain(np.array([[1.0, 2.0]]), bins=None)

    with pytest.raises(DataShapeError, match="length of the bins"):
        mondrian_pipeline.explain(np.array([[1.0, 2.0], [3.0, 4.0]]), bins=[1])


def test_should_generate_lime_explanations_and_set_latest(monkeypatch):
    import external_plugins.integrations.lime_pipeline as lime_pipeline_mod

    monkeypatch.setattr(lime_pipeline_mod, "CalibratedExplanations", _DummyCollection)
    monkeypatch.setattr(lime_pipeline_mod, "assert_threshold", lambda threshold, x: None)

    explainer = make_lime_explainer(mode="regression", multiclass=False, mondrian=False)
    pipeline = LimePipeline(explainer)
    monkeypatch.setattr(pipeline, "_preload_lime", lambda: (DummyLimeExplainer(), None))

    output = pipeline.explain(np.array([[1.0, 2.0], [3.0, 4.0]]), threshold=None)

    assert isinstance(output, _DummyCollection)
    assert output.low_high_percentiles == (5, 95)
    assert output.finalized is not None
    assert len(output.finalized["feature_weights"]["predict"]) == 2
    assert len(output.finalized["feature_predict"]["predict"]) == 2
    assert explainer.latest_explanation is output


def test_should_fail_lime_when_threshold_used_for_classification(monkeypatch):
    import external_plugins.integrations.lime_pipeline as lime_pipeline_mod

    monkeypatch.setattr(lime_pipeline_mod, "CalibratedExplanations", _DummyCollection)

    explainer = make_lime_explainer(mode="classification")
    pipeline = LimePipeline(explainer)
    monkeypatch.setattr(pipeline, "_preload_lime", lambda: (DummyLimeExplainer(), None))

    with pytest.raises(ValidationError, match="threshold parameter is only supported"):
        pipeline.explain(np.array([[1.0, 2.0]]), threshold=0.5)


def test_should_fail_lime_when_dependency_not_available(monkeypatch):
    import external_plugins.integrations.lime_pipeline as lime_pipeline_mod

    monkeypatch.setattr(lime_pipeline_mod, "CalibratedExplanations", _DummyCollection)

    explainer = make_lime_explainer()
    pipeline = LimePipeline(explainer)
    monkeypatch.setattr(pipeline, "_preload_lime", lambda: (None, None))

    with pytest.raises(ConfigurationError, match="optional dependency is missing"):
        pipeline.explain(np.array([[1.0, 2.0]]))


def test_should_raise_fast_configuration_error_when_enable_fails(monkeypatch):
    explainer = make_fast_explainer(is_fast=False)

    def _boom():
        raise RuntimeError("no fast")

    explainer.enable_fast_mode = _boom
    pipeline = FastExplanationPipeline(explainer)

    with pytest.raises(ConfigurationError, match="Fast explanations are only possible"):
        pipeline.explain(np.array([[1.0, 2.0]]))


def test_should_compute_fast_explanations_and_restore_calibration_targets(monkeypatch):
    import external_plugins.fast_explanations.pipeline as fast_pipeline_mod

    monkeypatch.setattr(fast_pipeline_mod, "CalibratedExplanations", _DummyCollection)

    def _fake_compute_feature_effects(
        _explainer,
        features_to_process,
        x_test,
        threshold,
        low_high_percentiles,
        bins,
        prediction,
        executor,
    ):
        _ = (threshold, low_high_percentiles, bins, prediction, executor)
        n = len(np.asarray(x_test))
        rows = []
        for feat in features_to_process:
            rows.append(
                (
                    feat,
                    np.full(n, 0.1 + feat),
                    np.full(n, 0.01 + feat),
                    np.full(n, 0.2 + feat),
                    np.full(n, 0.3 + feat),
                    np.full(n, 0.03 + feat),
                    np.full(n, 0.4 + feat),
                )
            )
        return rows

    monkeypatch.setattr(fast_pipeline_mod, "compute_feature_effects", _fake_compute_feature_effects)

    explainer = make_fast_explainer(mode="classification", is_fast=False, multiclass=True)
    original_y_cal = explainer.y_cal.copy()
    pipeline = FastExplanationPipeline(explainer)

    out = pipeline.explain(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert isinstance(out, _DummyCollection)
    assert out.finalized is not None
    assert "__full_probabilities__" in out.finalized["prediction"]
    assert explainer.latest_explanation is out
    assert explainer.last_explanation_mode == "fast"
    assert np.array_equal(explainer.y_cal, original_y_cal)


def test_should_validate_fast_threshold_mondrian_and_shapes(monkeypatch):
    import external_plugins.fast_explanations.pipeline as fast_pipeline_mod

    monkeypatch.setattr(fast_pipeline_mod, "CalibratedExplanations", _DummyCollection)

    explainer = make_fast_explainer(mode="classification", is_fast=True)
    pipeline = FastExplanationPipeline(explainer)
    with pytest.raises(ValidationError, match="threshold parameter is only supported"):
        pipeline.explain(np.array([[1.0, 2.0]]), threshold=0.2)

    bad_shape = make_fast_explainer(mode="classification", is_fast=True)
    bad_shape_pipeline = FastExplanationPipeline(bad_shape)
    with pytest.raises(DataShapeError, match="number of features"):
        bad_shape_pipeline.explain(np.array([[1.0, 2.0, 3.0]]))

    mondrian = make_fast_explainer(mode="classification", is_fast=True, mondrian=True)
    mondrian_pipeline = FastExplanationPipeline(mondrian)
    with pytest.raises(ValidationError, match="bins parameter must be specified"):
        mondrian_pipeline.explain(np.array([[1.0, 2.0]]), bins=None)
    with pytest.raises(DataShapeError, match="length of the bins"):
        mondrian_pipeline.explain(np.array([[1.0, 2.0], [3.0, 4.0]]), bins=[1])


def test_should_preprocess_discretize_and_compute_rule_boundaries():
    explainer = make_fast_explainer(mode="classification", is_fast=True)
    explainer.discretizer = SimpleNamespace(
        to_discretize=[0],
        mins={0: np.array([0.0, 5.0])},
        maxs={0: np.array([4.99, 9.99])},
        means={0: np.array([2.5, 7.5])},
    )
    pipeline = FastExplanationPipeline(explainer)

    pipeline.preprocess()
    assert pipeline.explainer.features_to_ignore == [0]

    x = np.array([[1.0, 10.0], [8.0, 11.0]])
    xd = pipeline.discretize(x)
    assert xd.shape == x.shape
    assert xd[0, 0] == 2.5
    assert xd[1, 0] == 7.5

    one = pipeline.rule_boundaries(np.array([1.0, 10.0]))
    assert len(one) == pipeline.explainer.num_features

    many = pipeline.rule_boundaries(x, perturbed_instances=xd)
    assert many.shape == (2, pipeline.explainer.num_features, 2)
