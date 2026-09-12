"""Unit tests for helper and fallback logic inside WrapCalibratedExplainer."""

from __future__ import annotations

from typing import Any
from types import SimpleNamespace

import numpy as np
import pytest

from calibrated_explanations.utils.exceptions import (
    ConfigurationError,
    DataShapeError,
    NotFittedError,
    ValidationError,
)
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
from tests.helpers.explainer_internals import (
    build_preprocessor_metadata,
    extract_preprocessor_snapshot,
    finalize_fit,
    format_proba_output,
    maybe_preprocess_for_inference,
    normalize_auto_encode_flag,
    normalize_public_kwargs,
    pre_fit_preprocess,
    pre_transform,
    serialise_preprocessor_value,
)


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
        return {"snap": [1, 2]}

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


def test_normalize_public_kwargs_rejects_removed_aliases(wrapper: WrapCalibratedExplainer) -> None:
    payload = {"threshold": 0.3, "alpha": (1, 99), "irrelevant": "value"}
    with pytest.raises(ConfigurationError, match="removed in v0.11.0"):
        normalize_public_kwargs(wrapper, payload, allowed={"threshold"})
    assert payload["alpha"] == (1, 99)
    assert payload["irrelevant"] == "value"


def test_normalize_auto_encode_flag_variants(wrapper: WrapCalibratedExplainer) -> None:
    assert normalize_auto_encode_flag(wrapper) == "auto"
    wrapper.auto_encode = True
    assert normalize_auto_encode_flag(wrapper) == "true"
    wrapper.auto_encode = "FALSE"
    assert normalize_auto_encode_flag(wrapper) == "false"
    wrapper.auto_encode = "unexpected"
    assert normalize_auto_encode_flag(wrapper) == "auto"


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
    serialised = serialise_preprocessor_value(wrapper, payload)
    assert serialised == {
        "numbers": [1, 2],
        "sequence": [1, 2, 3],
        "array_like": str(payload["array_like"]),
    }


def test_extract_preprocessor_snapshot(wrapper: WrapCalibratedExplainer) -> None:
    preprocessor = RecordingPreprocessor()
    snapshot = extract_preprocessor_snapshot(wrapper, preprocessor)
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
    assert build_preprocessor_metadata(wrapper) is None

    wrapper.preprocessor = RecordingPreprocessor()
    wrapper.auto_encode = False
    metadata = build_preprocessor_metadata(wrapper)
    assert metadata is not None
    assert metadata["auto_encode"] == "false"
    assert metadata["transformer_id"].endswith(":RecordingPreprocessor")
    assert metadata["mapping_snapshot"]["custom"] == {"snap": [1, 2]}


def test_pre_fit_preprocess_and_transform_stages(wrapper: WrapCalibratedExplainer) -> None:
    preprocessor = RecordingPreprocessor()
    wrapper.preprocessor = preprocessor

    x = np.array([[1, 2], [3, 4]])
    x_fit = pre_fit_preprocess(wrapper, x)
    assert np.array_equal(x_fit, x * 2)
    assert preprocessor.fit_called_with

    x_transformed = pre_transform(wrapper, x)
    assert np.array_equal(x_transformed, x + 1)
    assert preprocessor.transform_called_with

    x_inference = maybe_preprocess_for_inference(wrapper, x)
    assert np.array_equal(x_inference, x + 1)


def test_pre_fit_preprocess_raises_validation_error_on_preprocessor_failure(
    wrapper: WrapCalibratedExplainer,
) -> None:
    class FailingPreprocessor:
        def fit_transform(self, x: Any) -> Any:
            raise RuntimeError("boom")

        def transform(self, x: Any) -> Any:
            raise RuntimeError("boom")

    wrapper.preprocessor = FailingPreprocessor()
    x = np.array([[1, 2]])

    with pytest.raises(ValidationError, match="Preprocessor failed during fit"):
        pre_fit_preprocess(wrapper, x)

    # A rejected fit-preprocess call must not be recorded as fitted.
    assert not wrapper.pre_fitted


def test_pre_transform_raises_validation_error_on_preprocessor_failure(
    wrapper: WrapCalibratedExplainer,
) -> None:
    class FailingTransformPreprocessor:
        def fit_transform(self, x: Any) -> Any:
            return np.asarray(x) * 2

        def transform(self, x: Any) -> Any:
            raise RuntimeError("boom")

    wrapper.preprocessor = FailingTransformPreprocessor()
    x = np.array([[1, 2]])
    pre_fit_preprocess(wrapper, x)
    assert wrapper.pre_fitted

    with pytest.raises(ValidationError, match="Preprocessor transform failed during predict"):
        pre_transform(wrapper, x)

    # The preprocessor stays fitted; only the failed transform call is rejected.
    assert wrapper.pre_fitted


def test_finalize_fit_preserves_existing_explainer(wrapper: WrapCalibratedExplainer) -> None:
    recorder = ExplainerRecorder()
    wrapper.explainer = recorder
    wrapper.fitted = False
    wrapper.calibrated = True
    wrapper.learner = PredictProbaLearner()

    finalize_fit(wrapper, reinitialize=True)

    assert wrapper.fitted is True
    assert wrapper.calibrated is True
    assert recorder.reinitialized_with == [wrapper.learner]


def test_format_proba_output_variants(wrapper: WrapCalibratedExplainer) -> None:
    matrix = np.array([[0.1, 0.9], [0.2, 0.8]])
    assert format_proba_output(wrapper, matrix, False) is matrix

    multi = np.array([[0.1, 0.3, 0.6]])
    result_multi = format_proba_output(wrapper, multi, True)
    assert np.array_equal(result_multi[0], multi)
    assert np.array_equal(result_multi[1][0], multi)

    binary = np.array([[0.4, 0.6]])
    _, intervals = format_proba_output(wrapper, binary, True)
    assert np.allclose(intervals[0], binary[:, 1])

    vector = np.array([0.1, 0.9])
    _, fallback = format_proba_output(wrapper, vector, True)
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
    monkeypatch.setattr("calibrated_explanations.api.config._build_perf_factory", lambda _: None)
    cfg = SimpleNamespace(
        model=PredictOnlyLearner(),
        threshold=0.4,
        low_high_percentiles=(5, 95),
        preprocessor=None,
        auto_encode="auto",
        unseen_category_policy="error",
    )

    wrapper = WrapCalibratedExplainer.from_config(cfg)

    assert hasattr(wrapper, "perf_cache")
    assert hasattr(wrapper, "perf_parallel")
    assert wrapper.perf_cache is None
    assert wrapper.parallel_executor is None
    assert getattr(wrapper, "cfg", None) is cfg


def test_should_raise_attribute_error_when_explain_lime_removed(
    wrapper: WrapCalibratedExplainer,
) -> None:
    wrapper.fitted = True
    wrapper.calibrated = True

    with pytest.raises(AttributeError, match="explain_lime"):
        wrapper.explain_lime(np.array([[1, 2]]), custom_flag=True)


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

        def set_difficulty_estimator(self, estimator: Any, *, initialize: bool = True) -> None:
            self.received.append((estimator, initialize))

    wrapper.fitted = True
    wrapper.calibrated = True
    wrapper.explainer = Recorder()

    wrapper.set_difficulty_estimator("estimator")

    assert wrapper.explainer.received == [("estimator", True)]  # type: ignore[union-attr]


def test_set_difficulty_estimator_forwards_initialize_flag(
    wrapper: WrapCalibratedExplainer,
) -> None:
    class Recorder:
        def __init__(self) -> None:
            self.received: list[tuple[Any, bool]] = []

        def set_difficulty_estimator(self, estimator: Any, *, initialize: bool = True) -> None:
            self.received.append((estimator, initialize))

    wrapper.fitted = True
    wrapper.calibrated = True
    wrapper.explainer = Recorder()

    wrapper.set_difficulty_estimator("estimator", initialize=False)

    assert wrapper.explainer.received == [("estimator", False)]  # type: ignore[union-attr]


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
    wrapper = WrapCalibratedExplainer.from_config(cfg)

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
    assert serialise_preprocessor_value(wrapper, None) is None

    class Custom:
        pass

    custom = Custom()
    assert serialise_preprocessor_value(wrapper, custom) == str(custom)


def test_pre_fit_preprocess_without_configured_preprocessor(
    wrapper: WrapCalibratedExplainer,
) -> None:
    data = np.array([[1, 2]])
    wrapper.auto_encode = False
    wrapper.preprocessor = None

    assert pre_fit_preprocess(wrapper, data) is data


def test_pre_fit_preprocess_uses_two_step_transform(wrapper: WrapCalibratedExplainer) -> None:
    class TwoStep:
        def __init__(self) -> None:
            self.fit_args: list[Any] = []

        def fit(self, x: Any) -> None:
            self.fit_args.append(np.asarray(x))

        def transform(self, x: Any) -> Any:
            return np.asarray(x) + 5

    preprocessor = TwoStep()
    wrapper.preprocessor = preprocessor
    data = np.array([[1, 2]])

    transformed = pre_fit_preprocess(wrapper, data)

    assert wrapper.pre_fitted is True
    assert preprocessor.fit_args
    assert np.array_equal(transformed, data + 5)


def test_export_and_import_preprocessor_mapping_applies_when_possible(
    wrapper: WrapCalibratedExplainer,
) -> None:
    pre = RecordingPreprocessor()
    wrapper.preprocessor = pre

    exported = wrapper.export_preprocessor_mapping()
    assert exported is not None
    # recording preprocessor exposes a custom snapshot
    assert "snap" in next(iter(exported.values())) or "snap" in exported

    # Apply a new mapping and ensure it is written to the preprocessor.mapping_
    new_map = {"feature": {"a": 42}}
    wrapper.import_preprocessor_mapping(new_map)
    assert pre.mapping_ == new_map


def test_export_preprocessor_mapping_rejects_non_json_serialisable_snapshots(
    wrapper: WrapCalibratedExplainer,
) -> None:
    class NonJsonSnapshotPreprocessor:
        def get_mapping_snapshot(self) -> dict[str, Any]:
            return {"bad": {1, 2}}

    wrapper.preprocessor = NonJsonSnapshotPreprocessor()

    with pytest.raises(ValidationError, match="JSON-serialisable"):
        wrapper.export_preprocessor_mapping()


def test_import_preprocessor_mapping_rejects_non_json_serialisable_payload(
    wrapper: WrapCalibratedExplainer,
) -> None:
    with pytest.raises(ValidationError, match="JSON-serialisable"):
        wrapper.import_preprocessor_mapping({"bad": {1, 2}})


def test_pre_fit_preprocess_auto_mode_uses_builtin_encoder(
    wrapper: WrapCalibratedExplainer,
) -> None:
    wrapper.preprocessor = None
    wrapper.auto_encode = "auto"
    data = np.array([[1, "a"], [2, "b"], [3, "a"]], dtype=object)

    transformed = pre_fit_preprocess(wrapper, data)

    assert wrapper.pre_fitted is True
    assert wrapper.preprocessor is not None
    assert wrapper.preprocessor.__class__.__name__ == "BuiltinEncoder"
    assert transformed.shape == (3, 2)
    np.testing.assert_allclose(transformed[:, 0], np.array([1.0, 2.0, 3.0]))
    assert transformed.dtype.kind == "f"


def test_unseen_category_policy_error_raises_validation_error(
    wrapper: WrapCalibratedExplainer,
) -> None:
    cfg = SimpleNamespace(
        model=PredictOnlyLearner(),
        threshold=0.5,
        low_high_percentiles=(5, 95),
        preprocessor=None,
        auto_encode="auto",
        unseen_category_policy="error",
    )
    wrapper = WrapCalibratedExplainer.from_config(cfg)
    pre_fit_preprocess(wrapper, np.array([["a"], ["b"]], dtype=object))

    with pytest.raises(ValidationError, match="Unseen category encountered"):
        maybe_preprocess_for_inference(wrapper, np.array([["c"]], dtype=object))


def test_unseen_category_policy_ignore_returns_sentinel_value(
    wrapper: WrapCalibratedExplainer,
) -> None:
    cfg = SimpleNamespace(
        model=PredictOnlyLearner(),
        threshold=0.5,
        low_high_percentiles=(5, 95),
        preprocessor=None,
        auto_encode="auto",
        unseen_category_policy="ignore",
    )
    wrapper = WrapCalibratedExplainer.from_config(cfg)
    pre_fit_preprocess(wrapper, np.array([["a"], ["b"]], dtype=object))

    transformed = maybe_preprocess_for_inference(wrapper, np.array([["c"]], dtype=object))

    assert transformed.shape == (1, 1)
    assert transformed[0, 0] == -1.0


def test_non_numeric_input_without_preprocessing_raises_actionable_error(
    wrapper: WrapCalibratedExplainer,
) -> None:
    wrapper.preprocessor = None
    wrapper.auto_encode = False

    with pytest.raises(ValidationError, match="Set auto_encode='auto'"):
        pre_fit_preprocess(wrapper, np.array([["x"]], dtype=object))


def test_pre_fit_preprocess_auto_mode_respects_missing_value_policy_from_config() -> None:
    cfg = SimpleNamespace(
        model=PredictOnlyLearner(),
        threshold=0.5,
        low_high_percentiles=(5, 95),
        preprocessor=None,
        auto_encode="auto",
        unseen_category_policy="error",
        missing_value_policy="error",
    )
    wrapper = WrapCalibratedExplainer.from_config(cfg)

    with pytest.raises(ValidationError, match="Missing value encountered"):
        pre_fit_preprocess(wrapper, np.array([["a"], [None]], dtype=object))
