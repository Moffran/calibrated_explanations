"""Additional behavioural tests for :mod:`calibrated_explainer`."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys
from typing import Iterable

import numpy as np
import pytest

from calibrated_explanations.core.explain.feature_task import (
    assign_threshold as normalize_threshold,
)
from calibrated_explanations.core.explain.feature_task import assign_weight
from calibrated_explanations.core.exceptions import (
    DataShapeError,
    NotFittedError,
    ValidationError,
)


@pytest.fixture()
def fake_pandas(monkeypatch):
    """Register lightweight stand-ins for the ``pandas`` modules used by ``safe_isinstance``."""

    class FakeSeries:
        def __init__(self, values: Iterable[float]):
            self._values = np.array(list(values))
            self.iloc = FakeILoc(self)

        def __len__(self) -> int:  # pragma: no cover - defensive completeness
            return len(self._values)

        def __getitem__(self, item):
            return self._values[item]

        def to_numpy(self):
            return np.array(self._values)

    class FakeILoc:
        def __init__(self, series: FakeSeries):
            self._series = series

        def __getitem__(self, key):
            return FakeSeries(self._series._values[key])

    class FakeDataFrame:
        def __init__(self, values: Iterable[Iterable[float]]):
            self.values = np.array(list(values))

    class FakeCategorical:  # pragma: no cover - required for safe_isinstance lookups
        pass

    pandas_mod = ModuleType("pandas")
    pandas_core = ModuleType("pandas.core")
    pandas_series = ModuleType("pandas.core.series")
    pandas_series.Series = FakeSeries
    pandas_frame = ModuleType("pandas.core.frame")
    pandas_frame.DataFrame = FakeDataFrame
    pandas_arrays = ModuleType("pandas.core.arrays")
    pandas_categorical = ModuleType("pandas.core.arrays.categorical")
    pandas_categorical.Categorical = FakeCategorical

    pandas_mod.core = pandas_core
    pandas_core.series = pandas_series
    pandas_core.frame = pandas_frame
    pandas_core.arrays = pandas_arrays
    pandas_arrays.categorical = pandas_categorical

    monkeypatch.setitem(sys.modules, "pandas", pandas_mod)
    monkeypatch.setitem(sys.modules, "pandas.core", pandas_core)
    monkeypatch.setitem(sys.modules, "pandas.core.series", pandas_series)
    monkeypatch.setitem(sys.modules, "pandas.core.frame", pandas_frame)
    monkeypatch.setitem(sys.modules, "pandas.core.arrays", pandas_arrays)
    monkeypatch.setitem(sys.modules, "pandas.core.arrays.categorical", pandas_categorical)

    return SimpleNamespace(Series=FakeSeries, DataFrame=FakeDataFrame)


def test_slice_helpers_support_multiple_input_types(fake_pandas):
    """Test threshold and bins slicing with multiple input types.

    Tests call explain module functions directly.
    """
    from calibrated_explanations.core.explain._helpers import slice_threshold, slice_bins

    scalar = slice_threshold(3.14, 0, 1, 2)
    assert scalar == 3.14

    sequence = [1, 2, 3]
    assert slice_threshold(sequence, 0, 2, 3) == [1, 2]
    assert slice_threshold(sequence, 0, 1, 4) is sequence

    array = np.array([4, 5, 6])
    np.testing.assert_array_equal(slice_threshold(array, 1, 3, 3), np.array([5, 6]))

    series = fake_pandas.Series([7, 8, 9])
    sliced_series = slice_threshold(series, 1, 3, 3)
    np.testing.assert_array_equal(sliced_series.to_numpy(), np.array([8, 9]))

    assert slice_bins(None, 0, 2) is None
    bins = np.array([0.1, 0.2, 0.3])
    np.testing.assert_array_equal(slice_bins(bins, 1, 3), np.array([0.2, 0.3]))
    pandas_bins = fake_pandas.Series([10, 11, 12])
    np.testing.assert_array_equal(slice_bins(pandas_bins, 0, 2), np.array([10, 11]))


def test_assign_threshold_and_weight_behaviour():
    assert normalize_threshold(None) is None
    empty_numeric = normalize_threshold([0.1, 0.2])
    assert empty_numeric.shape == (0,)
    empty_tuple = normalize_threshold([(0.1, 0.2)])
    assert empty_tuple.dtype == object  # tuples preserved via object dtype

    assert assign_weight(0.2, 0.6) == pytest.approx(0.4)
    assert assign_weight([0.1, 0.3], [0.4, 0.9]) == [
        pytest.approx(0.3),
        pytest.approx(0.6),
    ]


def test_set_difficulty_estimator_enforces_fitted_contract(explainer_factory):
    """Test that set_difficulty_estimator validates and updates difficulty estimation."""
    explainer = explainer_factory()

    class DummyEstimator:
        def __init__(self, fitted: bool, value: float = 1.0):
            self.fitted = fitted
            self.value = value

        def apply(self, x: np.ndarray) -> np.ndarray:
            return np.full(x.shape[0], self.value)

    # Should raise NotFittedError for unfitted estimator
    with pytest.raises(NotFittedError):
        explainer.set_difficulty_estimator(DummyEstimator(fitted=False))

    # Setting to None with initialize=False should result in unit difficulty
    explainer.set_difficulty_estimator(None, initialize=False)
    np.testing.assert_array_equal(explainer._get_sigma_test(np.ones((3, 2))), np.ones(3))

    # Setting a fitted estimator with initialize=False should use its values
    estimator = DummyEstimator(fitted=True, value=2.5)
    explainer.set_difficulty_estimator(estimator, initialize=False)
    np.testing.assert_array_equal(explainer._get_sigma_test(np.zeros((4, 2))), np.full(4, 2.5))

    # Setting with initialize=False should keep the estimator without triggering initialization
    explainer.set_difficulty_estimator(estimator, initialize=False)
    # Verify difficulty estimator is set
    assert explainer.difficulty_estimator is estimator
    # Verify sigma test uses the estimator's values
    np.testing.assert_array_equal(explainer._get_sigma_test(np.ones((2, 2))), np.full(2, 2.5))


def test_private_set_mode_updates_state(explainer_factory):
    """Test that __set_mode updates mode and num_classes correctly without initialization."""
    explainer = explainer_factory()

    explainer.y_cal = np.array([0, 1, 1, 2])
    explainer._CalibratedExplainer__set_mode("classification", initialize=False)
    assert explainer.num_classes == 3
    assert explainer.mode == "classification"

    explainer._CalibratedExplainer__set_mode("regression", initialize=False)
    assert explainer.num_classes == 0
    assert explainer.mode == "regression"

    with pytest.raises(ValidationError):
        explainer._CalibratedExplainer__set_mode("unsupported", initialize=False)


def test_runtime_metadata_helpers_return_copies(explainer_factory):
    explainer = explainer_factory()
    explainer._last_telemetry = {"source": "initial"}
    explainer.set_preprocessor_metadata({"scaler": "std"})
    telemetry = explainer.runtime_telemetry
    telemetry["source"] = "mutated"
    assert explainer._last_telemetry["source"] == "initial"

    metadata = explainer.preprocessor_metadata
    metadata["scaler"] = "modified"
    assert explainer._preprocessor_metadata["scaler"] == "std"

    explainer.set_preprocessor_metadata(None)
    assert explainer.preprocessor_metadata is None


def test_calibration_setters_handle_dataframe_inputs(explainer_factory, fake_pandas):
    explainer = explainer_factory()
    explainer._categorical_value_counts_cache = {}
    explainer._numeric_sorted_cache = {}
    explainer._calibration_summary_shape = (1, 1)

    df = fake_pandas.DataFrame([[1.0, 2.0], [3.0, 4.0]])
    explainer.x_cal = df
    np.testing.assert_array_equal(explainer.x_cal, np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert explainer._categorical_value_counts_cache is None

    target_df = fake_pandas.DataFrame([[5.0], [6.0]])
    explainer.y_cal = target_df
    np.testing.assert_array_equal(explainer.y_cal, np.array([[5.0], [6.0]]))

    explainer.append_cal(np.array([[7.0, 8.0]]), np.array([[9.0]]))
    assert explainer.y_cal.shape[0] == 3

    with pytest.raises(DataShapeError):
        explainer.append_cal(np.array([[1.0, 2.0, 3.0]]), np.array([1.0]))


def test_lime_and_shap_flags_toggle(explainer_factory):
    explainer = explainer_factory()
    assert explainer._is_lime_enabled() is False
    assert explainer._is_lime_enabled(True) is True
    assert explainer._is_lime_enabled() is True

    assert explainer._is_shap_enabled() is False
    assert explainer._is_shap_enabled(True) is True
    assert explainer._is_shap_enabled() is True
