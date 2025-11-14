"""Additional behavioural tests for :mod:`calibrated_explainer`."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys
from typing import Iterable

import numpy as np
import pytest

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.exceptions import (
    DataShapeError,
    NotFittedError,
    ValidationError,
)
from calibrated_explanations.integrations import LimeHelper, ShapHelper


@pytest.fixture()
def fake_pandas(monkeypatch):
    """Register lightweight stand-ins for the ``pandas`` modules used by ``safe_isinstance``."""

    class _FakeSeries:
        def __init__(self, values: Iterable[float]):
            self._values = np.array(list(values))
            self.iloc = _FakeILoc(self)

        def __len__(self) -> int:  # pragma: no cover - defensive completeness
            return len(self._values)

        def __getitem__(self, item):
            return self._values[item]

        def to_numpy(self):
            return np.array(self._values)

    class _FakeILoc:
        def __init__(self, series: _FakeSeries):
            self._series = series

        def __getitem__(self, key):
            return _FakeSeries(self._series._values[key])

    class _FakeDataFrame:
        def __init__(self, values: Iterable[Iterable[float]]):
            self.values = np.array(list(values))

    class _FakeCategorical:  # pragma: no cover - required for safe_isinstance lookups
        pass

    pandas_mod = ModuleType("pandas")
    pandas_core = ModuleType("pandas.core")
    pandas_series = ModuleType("pandas.core.series")
    pandas_series.Series = _FakeSeries
    pandas_frame = ModuleType("pandas.core.frame")
    pandas_frame.DataFrame = _FakeDataFrame
    pandas_arrays = ModuleType("pandas.core.arrays")
    pandas_categorical = ModuleType("pandas.core.arrays.categorical")
    pandas_categorical.Categorical = _FakeCategorical

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

    return SimpleNamespace(Series=_FakeSeries, DataFrame=_FakeDataFrame)


def _make_minimal_explainer(num_features: int = 2) -> CalibratedExplainer:
    """Construct an explainer instance without triggering the heavy ``__init__``."""

    explainer = CalibratedExplainer.__new__(CalibratedExplainer)
    base_x = np.arange(num_features * 2, dtype=float).reshape(2, num_features)
    explainer._X_cal = base_x.copy()
    explainer._y_cal = np.array([0, 1])
    explainer.num_classes = 2
    explainer.mode = "classification"
    explainer.sample_percentiles = [25, 50, 75]
    explainer.categorical_features = np.array([], dtype=int)
    explainer.features_to_ignore = []
    explainer._feature_names = [f"f{i}" for i in range(num_features)]
    explainer.discretizer = SimpleNamespace(to_discretize=[], mins={}, maxs={}, means={})
    explainer.feature_values = {}
    explainer.feature_frequencies = {}
    explainer._categorical_value_counts_cache = None
    explainer._numeric_sorted_cache = None
    explainer._calibration_summary_shape = None
    explainer._last_telemetry = {"source": "initial"}
    explainer._preprocessor_metadata = {"scaler": "std"}
    explainer.seed = 0
    explainer.rng = np.random.default_rng(0)
    explainer.bins = None
    explainer._CalibratedExplainer__fast = False
    explainer._CalibratedExplainer__initialized = False
    explainer._lime_helper = LimeHelper(explainer)
    explainer._shap_helper = ShapHelper(explainer)
    # Initialize the prediction orchestrator (Phase 4: Interval Registry)
    from calibrated_explanations.core.prediction import PredictionOrchestrator
    explainer._prediction_orchestrator = PredictionOrchestrator(explainer)
    return explainer


def test_slice_helpers_support_multiple_input_types(fake_pandas):
    """Test threshold and bins slicing with multiple input types (Phase 5).
    
    Phase 5 consolidation: Tests call explain module functions directly.
    """
    from calibrated_explanations.core.explain._helpers import slice_threshold, slice_bins

    scalar = slice_threshold(3.14, 0, 1, 2)
    assert scalar == 3.14

    sequence = [1, 2, 3]
    assert slice_threshold(sequence, 0, 2, 3) == [1, 2]
    assert slice_threshold(sequence, 0, 1, 4) is sequence

    array = np.array([4, 5, 6])
    np.testing.assert_array_equal(
        slice_threshold(array, 1, 3, 3), np.array([5, 6])
    )

    series = fake_pandas.Series([7, 8, 9])
    sliced_series = slice_threshold(series, 1, 3, 3)
    np.testing.assert_array_equal(sliced_series.to_numpy(), np.array([8, 9]))

    assert slice_bins(None, 0, 2) is None
    bins = np.array([0.1, 0.2, 0.3])
    np.testing.assert_array_equal(slice_bins(bins, 1, 3), np.array([0.2, 0.3]))
    pandas_bins = fake_pandas.Series([10, 11, 12])
    np.testing.assert_array_equal(
        slice_bins(pandas_bins, 0, 2), np.array([10, 11])
    )


def test_assign_threshold_and_weight_behaviour():
    explainer = _make_minimal_explainer()

    assert explainer.assign_threshold(None) is None
    empty_numeric = explainer.assign_threshold([0.1, 0.2])
    assert empty_numeric.shape == (0,)
    empty_tuple = explainer.assign_threshold([(0.1, 0.2)])
    assert empty_tuple.dtype == object  # tuples preserved via object dtype

    assert explainer._assign_weight(0.2, 0.6) == pytest.approx(0.4)
    assert explainer._assign_weight([0.1, 0.3], [0.4, 0.9]) == [
        pytest.approx(0.3),
        pytest.approx(0.6),
    ]


def test_numeric_sampling_helpers_respect_calibration_distribution():
    explainer = _make_minimal_explainer()
    explainer.sample_percentiles = [10, 50, 90]
    explainer.x_cal = np.array(
        [
            [0.1, 1.0],
            [0.6, 1.5],
            [0.8, 2.0],
            [1.2, 2.5],
        ]
    )

    greater = explainer._CalibratedExplainer__get_greater_values(0, 0.5)
    assert np.all(greater >= 0.6)

    lesser = explainer._CalibratedExplainer__get_lesser_values(0, 0.7)
    assert np.all(lesser <= 0.6)

    covered = explainer._CalibratedExplainer__get_covered_values(0, 0.6, 1.0)
    assert np.all((covered >= 0.6) & (covered <= 1.0))


def test_set_difficulty_estimator_enforces_fitted_contract():
    """Test that set_difficulty_estimator validates and updates difficulty estimation."""
    explainer = _make_minimal_explainer()

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


def test_private_set_mode_updates_state():
    """Test that __set_mode updates mode and num_classes correctly without initialization."""
    explainer = _make_minimal_explainer()

    explainer.y_cal = np.array([0, 1, 1, 2])
    explainer._CalibratedExplainer__set_mode("classification", initialize=False)
    assert explainer.num_classes == 3
    assert explainer.mode == "classification"

    explainer._CalibratedExplainer__set_mode("regression", initialize=False)
    assert explainer.num_classes == 0
    assert explainer.mode == "regression"

    with pytest.raises(ValidationError):
        explainer._CalibratedExplainer__set_mode("unsupported", initialize=False)


def test_runtime_metadata_helpers_return_copies():
    explainer = _make_minimal_explainer()
    telemetry = explainer.runtime_telemetry
    telemetry["source"] = "mutated"
    assert explainer._last_telemetry["source"] == "initial"

    metadata = explainer.preprocessor_metadata
    metadata["scaler"] = "modified"
    assert explainer._preprocessor_metadata["scaler"] == "std"

    explainer.set_preprocessor_metadata(None)
    assert explainer.preprocessor_metadata is None


def test_calibration_setters_handle_dataframe_inputs(fake_pandas):
    explainer = _make_minimal_explainer()
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


def test_set_discretizer_instantiates_expected_strategy(monkeypatch):
    explainer = _make_minimal_explainer()

    class StubEntropy:
        instances = 0

        def __init__(self, x_cal, not_to_discretize, feature_names, labels=None, random_state=None):
            StubEntropy.instances += 1
            self.args = {
                "x_cal_shape": np.shape(x_cal),
                "ignored": tuple(not_to_discretize),
                "feature_names": tuple(feature_names),
            }
            self.to_discretize = [f for f in range(x_cal.shape[1]) if f not in not_to_discretize]
            if not self.to_discretize:
                self.to_discretize = [0]
            self.mins = {f: np.array([0.0, 0.5]) for f in range(x_cal.shape[1])}
            self.maxs = {f: np.array([0.5, 1.0]) for f in range(x_cal.shape[1])}
            self.means = {f: np.array([0.1, 0.9]) for f in range(x_cal.shape[1])}

    class StubRegressor(StubEntropy):
        pass

    monkeypatch.setattr(
        "calibrated_explanations.core.calibrated_explainer.BinaryEntropyDiscretizer",
        StubEntropy,
    )
    monkeypatch.setattr(
        "calibrated_explanations.core.calibrated_explainer.BinaryRegressorDiscretizer",
        StubRegressor,
    )
    monkeypatch.setattr(
        "calibrated_explanations.core.calibrated_explainer.EntropyDiscretizer",
        StubEntropy,
    )
    monkeypatch.setattr(
        "calibrated_explanations.core.calibrated_explainer.RegressorDiscretizer",
        StubRegressor,
    )

    explainer.set_discretizer(None)
    assert isinstance(explainer.discretizer, StubEntropy)
    assert StubEntropy.instances == 1
    discretized = explainer._discretize(np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert discretized.shape == (2, 2)
    assert set(np.unique(discretized[:, explainer.discretizer.to_discretize[0]])) <= {0.1, 0.9}

    explainer.set_discretizer("binaryEntropy")
    assert StubEntropy.instances == 1  # Re-using the existing instance

    with pytest.raises(ValidationError):
        explainer.set_discretizer("invalid")


def test_lime_and_shap_flags_toggle():
    explainer = _make_minimal_explainer()
    assert explainer._is_lime_enabled() is False
    assert explainer._is_lime_enabled(True) is True
    assert explainer._is_lime_enabled() is True

    assert explainer._is_shap_enabled() is False
    assert explainer._is_shap_enabled(True) is True
    assert explainer._is_shap_enabled() is True
