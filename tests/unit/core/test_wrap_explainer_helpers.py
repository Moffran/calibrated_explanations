"""Unit tests for helper and fallback logic inside WrapCalibratedExplainer."""

from __future__ import annotations

from typing import Any
from types import SimpleNamespace

import numpy as np
import pytest

import calibrated_explanations.core.wrap_explainer as wrap_module
from calibrated_explanations.utils.exceptions import DataShapeError, NotFittedError, ValidationError
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
from tests.helpers.deprecation import warns_or_raises, deprecations_error_enabled


class PredictOnlyLearner:
    """Minimal learner exposing the hooks WrapCalibratedExplainer expects."""

    def __init__(self) -> None:
        self.fitted = True

    def fit(self, x: Any | None = None, y: Any | None = None, **_: Any) -> "PredictOnlyLearner":
        return self

    def predict(self, x: Any) -> Any:
        return np.asarray(x)


class PredictProbaLearner(PredictOnlyLearner):
    def predict_proba(self, x: Any) -> Any:
        return np.asarray(x)


class RecordingPreprocessor:
    """Test double that mimics key sklearn preprocessor attributes."""

    def __init__(self) -> None:
        self.fit_called_with: list[Any] = []
        self.transform_called_with: list[Any] = []
        self.categories_ = {"cat": ("a", "b")}
        transformer_type = type("Scaler", (), {})
        self.transformers_ = [("num", transformer_type(), ["x1", "x2"])]
        self.mapping_ = {"feature": {"a": 0, "b": 1}}

    def get_mapping_snapshot(self) -> dict[str, Any]:
        return {"snap": {1, 2}}

    def fit_transform(self, x: Any) -> Any:
        self.fit_called_with.append(tuple(map(tuple, np.asarray(x))))
        return np.asarray(x) * 2

    def transform(self, x: Any) -> Any:
        self.transform_called_with.append(tuple(map(tuple, np.asarray(x))))
        return np.asarray(x) + 1

    def get_feature_names_out(self) -> list[str]:
        return ["x1", "x2"]


class ExplainerRecorder:
    def __init__(self) -> None:
        self.reinitialized_with: list[Any] = []

    def reinitialize(self, learner: Any) -> None:
        self.reinitialized_with.append(learner)


@pytest.fixture()
def wrapper() -> WrapCalibratedExplainer:
    return WrapCalibratedExplainer(PredictOnlyLearner())


def test_normalize_public_kwargs_filters_aliases(wrapper: WrapCalibratedExplainer) -> None:
    payload = {"threshold": 0.3, "alpha": (1, 99), "irrelevant": "value"}
    if deprecations_error_enabled():
        with pytest.raises(DeprecationWarning):
            wrapper._normalize_public_kwargs(payload, allowed={"threshold"})
    else:
        with warns_or_raises():
            filtered = wrapper._normalize_public_kwargs(payload, allowed={"threshold"})
        assert filtered == {"threshold": 0.3}
    assert payload["alpha"] == (1, 99)
    assert payload["irrelevant"] == "value"


def test_normalize_auto_encode_flag_variants(wrapper: WrapCalibratedExplainer) -> None:
    assert wrapper._normalize_auto_encode_flag() == "auto"
    wrapper._auto_encode = True
    assert wrapper._normalize_auto_encode_flag() == "true"
    wrapper._auto_encode = "FALSE"
    assert wrapper._normalize_auto_encode_flag() == "false"
    wrapper._auto_encode = "unexpected"
    assert wrapper._normalize_auto_encode_flag() == "auto"


def test_serialise_preprocessor_value_handles_nested_structures(
    wrapper: WrapCalibratedExplainer,
) -> None:
    class BadToList:
        def tolist(self) -> Any:  # pragma: no cover - invoked and caught
            raise ValueError("boom")

    payload = {
        "numbers": {1, 2},
        "sequence": (1, 2, 3),
        "array_like": BadToList(),
    }
    serialised = wrapper._serialise_preprocessor_value(payload)
    assert serialised == {
        "numbers": [1, 2],
        "sequence": [1, 2, 3],
        "array_like": str(payload["array_like"]),
    }


def test_extract_preprocessor_snapshot(wrapper: WrapCalibratedExplainer) -> None:
    preprocessor = RecordingPreprocessor()
    snapshot = wrapper._extract_preprocessor_snapshot(preprocessor)
    assert snapshot is not None
    assert set(snapshot) == {"custom", "categories", "transformers", "feature_names_out", "mapping"}
    assert snapshot["custom"] == {"snap": [1, 2]}
    assert snapshot["categories"] == {"cat": ["a", "b"]}
    transformers = snapshot["transformers"]
    assert transformers[0]["name"] == "num"
    assert transformers[0]["columns"] == ["x1", "x2"]


def test_build_preprocessor_metadata_with_and_without_preprocessor(
    wrapper: WrapCalibratedExplainer,
) -> None:
    assert wrapper._build_preprocessor_metadata() is None

    wrapper._preprocessor = RecordingPreprocessor()
    wrapper._auto_encode = False
    metadata = wrapper._build_preprocessor_metadata()
    assert metadata is not None
    assert metadata["auto_encode"] == "false"
    assert metadata["transformer_id"].endswith(":RecordingPreprocessor")
    assert metadata["mapping_snapshot"]["custom"] == {"snap": [1, 2]}


def test_pre_fit_preprocess_and_transform_stages(wrapper: WrapCalibratedExplainer) -> None:
    preprocessor = RecordingPreprocessor()
    wrapper._preprocessor = preprocessor

    x = np.array([[1, 2], [3, 4]])
    x_fit = wrapper._pre_fit_preprocess(x)
    assert np.array_equal(x_fit, x * 2)
    assert preprocessor.fit_called_with

    x_transformed = wrapper._pre_transform(x)
    assert np.array_equal(x_transformed, x + 1)
    assert preprocessor.transform_called_with

    x_inference = wrapper._maybe_preprocess_for_inference(x)
    assert np.array_equal(x_inference, x + 1)


def test_preprocess_failures_are_swallowed(wrapper: WrapCalibratedExplainer) -> None:
    class FailingPreprocessor:
        def fit_transform(self, x: Any) -> Any:
            raise RuntimeError("boom")

        def transform(self, x: Any) -> Any:
            raise RuntimeError("boom")

    wrapper._preprocessor = FailingPreprocessor()
    x = np.array([[1, 2]])
    x_fit = wrapper._pre_fit_preprocess(x)
    assert np.array_equal(x_fit, x)
    assert not wrapper._pre_fitted

    x_pred = wrapper._pre_transform(x)
    assert np.array_equal(x_pred, x)


def test_finalize_fit_preserves_existing_explainer(wrapper: WrapCalibratedExplainer) -> None:
    recorder = ExplainerRecorder()
    wrapper.explainer = recorder
    wrapper.fitted = False
    wrapper.calibrated = True
    wrapper.learner = PredictProbaLearner()

    wrapper._finalize_fit(reinitialize=True)

    assert wrapper.fitted is True
    assert wrapper.calibrated is True
    assert recorder.reinitialized_with == [wrapper.learner]


def test_format_proba_output_variants(wrapper: WrapCalibratedExplainer) -> None:
    matrix = np.array([[0.1, 0.9], [0.2, 0.8]])
    assert wrapper._format_proba_output(matrix, False) is matrix

    multi = np.array([[0.1, 0.3, 0.6]])
    result_multi = wrapper._format_proba_output(multi, True)
    assert np.array_equal(result_multi[0], multi)
    assert np.array_equal(result_multi[1][0], multi)

    binary = np.array([[0.4, 0.6]])
    _, intervals = wrapper._format_proba_output(binary, True)
    assert np.allclose(intervals[0], binary[:, 1])

    vector = np.array([0.1, 0.9])
    _, fallback = wrapper._format_proba_output(vector, True)
    assert np.array_equal(fallback[0], vector)


def test_predict_uncalibrated_behaviour(wrapper: WrapCalibratedExplainer) -> None:
    wrapper.fitted = True
    wrapper.calibrated = False
    with pytest.warns(UserWarning):
        prediction, (lo, hi) = wrapper.predict(np.array([1, 2]), uq_interval=True)
    assert np.array_equal(prediction, np.array([1, 2]))
    assert np.array_equal(lo, prediction)
    assert np.array_equal(hi, prediction)

    with pytest.raises(DataShapeError):
        wrapper.predict(np.array([1, 2]), threshold=0.5)


def test_predict_proba_requires_threshold_for_regression(wrapper: WrapCalibratedExplainer) -> None:
    wrapper.fitted = True
    wrapper.calibrated = True
    with pytest.raises(ValidationError):
        wrapper.predict_proba(np.array([1, 2]))

    wrapper.calibrated = False
    with pytest.raises(NotFittedError):
        wrapper.predict_proba(np.array([1, 2]), threshold=0.5)


def test_from_config_sets_perf_primitives_to_none_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("calibrated_explanations.perf.from_config", lambda _: None)
    cfg = SimpleNamespace(
        model=PredictOnlyLearner(),
        threshold=0.4,
        low_high_percentiles=(5, 95),
        preprocessor=None,
        auto_encode="auto",
        unseen_category_policy="error",
    )

    wrapper = WrapCalibratedExplainer._from_config(cfg)

    assert hasattr(wrapper, "_perf_cache")
    assert hasattr(wrapper, "_perf_parallel")
    assert wrapper._perf_cache is None
    assert wrapper._perf_parallel is None
    assert getattr(wrapper, "_cfg", None) is cfg


def test_explain_counterfactual_delegates_to_explore(wrapper: WrapCalibratedExplainer) -> None:
    sentinel = object()
    wrapper.explore_alternatives = lambda *args, **kwargs: sentinel  # type: ignore[assignment]

    assert wrapper.explain_counterfactual(np.array([[1, 2]])) is sentinel


def test_explain_fast_requires_fit_and_calibration(wrapper: WrapCalibratedExplainer) -> None:
    wrapper.fitted = False
    with pytest.raises(NotFittedError):
        wrapper.explain_fast(np.array([1, 2]))

    wrapper.fitted = True
    wrapper.calibrated = False
    with pytest.raises(NotFittedError):
        wrapper.explain_fast(np.array([1, 2]))


def test_explain_lime_invokes_underlying_explainer(
    wrapper: WrapCalibratedExplainer, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper.fitted = True
    wrapper.calibrated = True

    class RecordingExplainer:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, dict[str, Any]]] = []

        def explain_lime(self, x: Any, **kwargs: Any) -> str:
            self.calls.append((x, dict(kwargs)))
            return "lime"

    monkeypatch.setattr(wrap_module, "validate_inputs_matrix", lambda *_, **__: None)
    monkeypatch.setattr(wrap_module, "validate_param_combination", lambda *_, **__: None)
    wrapper.explainer = RecordingExplainer()

    result = wrapper.explain_lime(np.array([[1, 2]]), custom_flag=True)

    assert result == "lime"
    assert wrapper.explainer.calls  # type: ignore[union-attr]
    _, payload = wrapper.explainer.calls[-1]  # type: ignore[union-attr]
    assert payload["custom_flag"] is True
    assert "bins" in payload


def test_predict_requires_fit(wrapper: WrapCalibratedExplainer) -> None:
    wrapper.fitted = False
    with pytest.raises(NotFittedError):
        wrapper.predict(np.array([1, 2]))


def test_predict_proba_threshold_requires_calibration_when_available() -> None:
    wrapper = WrapCalibratedExplainer(PredictProbaLearner())
    wrapper.fitted = True
    wrapper.calibrated = False

    with pytest.raises(DataShapeError):
        wrapper.predict_proba(np.array([[0.1, 0.9]]), threshold=0.5)


def test_set_difficulty_estimator_delegates(wrapper: WrapCalibratedExplainer) -> None:
    class Recorder:
        def __init__(self) -> None:
            self.received: list[Any] = []

        def set_difficulty_estimator(self, estimator: Any) -> None:
            self.received.append(estimator)

    wrapper.fitted = True
    wrapper.calibrated = True
    wrapper.explainer = Recorder()

    wrapper.set_difficulty_estimator("estimator")

    assert wrapper.explainer.received == ["estimator"]  # type: ignore[union-attr]


def test_plot_uses_configured_defaults() -> None:
    class PerfFactory:
        def make_cache(self) -> object:
            return object()

        def make_parallel_executor(self, cache: object) -> tuple[str, object]:
            return ("executor", cache)

    cfg = SimpleNamespace(
        model=PredictOnlyLearner(),
        threshold=0.2,
        low_high_percentiles=(10, 90),
        preprocessor=None,
        auto_encode="auto",
        unseen_category_policy="error",
        _perf_factory=PerfFactory(),
    )
    wrapper = WrapCalibratedExplainer._from_config(cfg)

    class PlotRecorder:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, dict[str, Any]]] = []

        def plot(self, x: Any, *, threshold: float | None = None, **kwargs: Any) -> None:
            payload = dict(kwargs)
            payload["threshold"] = threshold
            self.calls.append((x, payload))

    wrapper.fitted = True
    wrapper.calibrated = True
    wrapper.mc = lambda data: np.arange(len(np.asarray(data)))
    wrapper.explainer = PlotRecorder()

    x = np.zeros((3, 1))
    wrapper.plot(x)

    assert wrapper.explainer.calls  # type: ignore[union-attr]
    _, payload = wrapper.explainer.calls[-1]  # type: ignore[union-attr]
    assert payload["threshold"] == 0.2
    assert payload["low_high_percentiles"] == (10, 90)
    assert np.array_equal(payload["bins"], np.arange(len(x)))


def test_serialise_preprocessor_value_handles_none_and_objects(
    wrapper: WrapCalibratedExplainer,
) -> None:
    assert wrapper._serialise_preprocessor_value(None) is None

    class Custom:
        pass

    custom = Custom()
    assert wrapper._serialise_preprocessor_value(custom) == str(custom)


def test_pre_fit_preprocess_without_configured_preprocessor(
    wrapper: WrapCalibratedExplainer,
) -> None:
    data = np.array([[1, 2]])
    wrapper._preprocessor = None

    assert wrapper._pre_fit_preprocess(data) is data


def test_pre_fit_preprocess_uses_two_step_transform(wrapper: WrapCalibratedExplainer) -> None:
    class TwoStep:
        def __init__(self) -> None:
            self.fit_args: list[Any] = []

        def fit(self, x: Any) -> None:
            self.fit_args.append(np.asarray(x))

        def transform(self, x: Any) -> Any:
            return np.asarray(x) + 5

    preprocessor = TwoStep()
    wrapper._preprocessor = preprocessor
    data = np.array([[1, 2]])

    transformed = wrapper._pre_fit_preprocess(data)

    assert wrapper._pre_fitted is True
    assert preprocessor.fit_args
    assert np.array_equal(transformed, data + 5)
