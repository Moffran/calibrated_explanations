"""Unit tests for helper and fallback logic inside WrapCalibratedExplainer."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from calibrated_explanations.core.exceptions import DataShapeError, NotFittedError, ValidationError
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer


class _PredictOnlyLearner:
    """Minimal learner exposing the hooks WrapCalibratedExplainer expects."""

    def __init__(self) -> None:
        self.fitted = True

    def fit(self, x: Any | None = None, y: Any | None = None, **_: Any) -> "_PredictOnlyLearner":
        return self

    def predict(self, x: Any) -> Any:
        return np.asarray(x)


class _PredictProbaLearner(_PredictOnlyLearner):
    def predict_proba(self, x: Any) -> Any:
        return np.asarray(x)


class _RecordingPreprocessor:
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


class _ExplainerRecorder:
    def __init__(self) -> None:
        self.reinitialized_with: list[Any] = []

    def reinitialize(self, learner: Any) -> None:
        self.reinitialized_with.append(learner)


@pytest.fixture()
def wrapper() -> WrapCalibratedExplainer:
    return WrapCalibratedExplainer(_PredictOnlyLearner())


def test_normalize_public_kwargs_filters_aliases(wrapper: WrapCalibratedExplainer) -> None:
    payload = {"threshold": 0.3, "alpha": (1, 99), "irrelevant": "value"}
    with pytest.deprecated_call():
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
    class _BadToList:
        def tolist(self) -> Any:  # pragma: no cover - invoked and caught
            raise ValueError("boom")

    payload = {
        "numbers": {1, 2},
        "sequence": (1, 2, 3),
        "array_like": _BadToList(),
    }
    serialised = wrapper._serialise_preprocessor_value(payload)
    assert serialised == {
        "numbers": [1, 2],
        "sequence": [1, 2, 3],
        "array_like": str(payload["array_like"]),
    }


def test_extract_preprocessor_snapshot(wrapper: WrapCalibratedExplainer) -> None:
    preprocessor = _RecordingPreprocessor()
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

    wrapper._preprocessor = _RecordingPreprocessor()
    wrapper._auto_encode = False
    metadata = wrapper._build_preprocessor_metadata()
    assert metadata is not None
    assert metadata["auto_encode"] == "false"
    assert metadata["transformer_id"].endswith(":_RecordingPreprocessor")
    assert metadata["mapping_snapshot"]["custom"] == {"snap": [1, 2]}


def test_pre_fit_preprocess_and_transform_stages(wrapper: WrapCalibratedExplainer) -> None:
    preprocessor = _RecordingPreprocessor()
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
    class _FailingPreprocessor:
        def fit_transform(self, x: Any) -> Any:
            raise RuntimeError("boom")

        def transform(self, x: Any) -> Any:
            raise RuntimeError("boom")

    wrapper._preprocessor = _FailingPreprocessor()
    x = np.array([[1, 2]])
    x_fit = wrapper._pre_fit_preprocess(x)
    assert np.array_equal(x_fit, x)
    assert not wrapper._pre_fitted

    x_pred = wrapper._pre_transform(x)
    assert np.array_equal(x_pred, x)


def test_finalize_fit_preserves_existing_explainer(wrapper: WrapCalibratedExplainer) -> None:
    recorder = _ExplainerRecorder()
    wrapper.explainer = recorder
    wrapper.fitted = False
    wrapper.calibrated = True
    wrapper.learner = _PredictProbaLearner()

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
