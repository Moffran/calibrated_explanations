"""Regression tests validating ADR-002 exception parity in calibration flows."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from calibrated_explanations.calibration import IntervalRegressor, VennAbers
from calibrated_explanations.core import ConfigurationError, DataShapeError


class MockModel:
    """Mock model supporting predict_proba and basic attributes."""

    def __init__(self):
        self.is_fitted = True
        self.rng = np.random.default_rng(42)

    def predict(self, x):
        """Return dummy predictions."""
        return self.rng.random(len(x))

    def predict_proba(self, x):
        """Return dummy probabilities."""
        return self.rng.random((len(x), 2))


def test_venn_abers_mondrian_without_bins_raises_configuration_error():
    """VennAbers.predict_proba with Mondrian but missing bins should raise ConfigurationError."""
    # Setup: Mondrian-aware VennAbers
    rng = np.random.default_rng(42)
    x_cal = rng.standard_normal((50, 3))
    y_cal = rng.integers(0, 2, 50)
    bins = np.repeat([1, 2], 25)  # Mondrian categories

    model = MockModel()
    calibrator = VennAbers(x_cal, y_cal, model, bins=bins)

    # Test: predict_proba without bins should raise ConfigurationError
    x_test = rng.standard_normal((10, 3))
    with pytest.raises(ConfigurationError, match="bins must be provided if Mondrian"):
        calibrator.predict_proba(x_test, bins=None)

    # Verify details are attached
    try:
        calibrator.predict_proba(x_test, bins=None)
    except ConfigurationError as e:
        assert e.details is not None
        assert "context" in e.details
        assert e.details["context"] == "predict_proba"


def test_interval_regressor_bins_mismatch_raises_data_shape_error():
    """IntervalRegressor with mismatched bins should raise DataShapeError."""
    rng = np.random.default_rng(42)
    x_cal = rng.standard_normal((50, 3))
    y_cal = rng.standard_normal(50)
    model = LinearRegression()
    model.fit(x_cal, y_cal)

    # Wrap model in a mock CalibratedExplainer to satisfy IntervalRegressor
    bins = np.repeat([1, 2], 25)

    class MockExplainer:
        def __init__(self, model):
            self.model = model
            self.bins = bins
            self.y_cal = y_cal
            self.x_cal = x_cal
            self.seed = 42
            self.difficulty_estimator = None

        def predict_calibration(self):
            return self.model.predict(self.x_cal)

        def _get_sigma_test(self, x):
            return np.ones(len(x))

        def predict_function(self, x):
            return self.model.predict(x)

    explainer = MockExplainer(model)
    regressor = IntervalRegressor(explainer)
    regressor.insert_calibration(x_cal, y_cal, bins=bins)

    # Test: test bins with different length should raise DataShapeError
    x_test = rng.standard_normal((10, 3))
    bins_test = np.array([1, 2])  # Too short
    with pytest.raises(DataShapeError, match="length of test bins"):
        regressor.predict_probability(x_test, 0.5, bins=bins_test)


def test_interval_regressor_inconsistent_bins_raises_configuration_error():
    """IntervalRegressor mixing bins and no-bins raises ConfigurationError."""
    rng = np.random.default_rng(42)
    x_cal = rng.standard_normal((50, 3))
    y_cal = rng.standard_normal(50)
    model = LinearRegression()
    model.fit(x_cal, y_cal)

    # Wrap model in a mock CalibratedExplainer to satisfy IntervalRegressor
    class MockExplainer:
        def __init__(self, model):
            self.model = model
            self.bins = None
            self.y_cal = y_cal
            self.x_cal = x_cal
            self.seed = 42
            self.difficulty_estimator = None

        def predict_calibration(self):
            return self.model.predict(self.x_cal)

        def _get_sigma_test(self, x):
            return np.ones(len(x))

        def predict_function(self, x):
            return self.model.predict(x)

    explainer = MockExplainer(model)
    regressor = IntervalRegressor(explainer)
    regressor.insert_calibration(x_cal, y_cal, bins=None)

    # Add calibration with bins should raise ConfigurationError
    x_cal_2 = rng.standard_normal((25, 3))
    y_cal_2 = rng.standard_normal(25)
    bins_2 = np.repeat([1, 2], 12)
    bins_2 = np.append(bins_2, [1])  # Pad to match length

    with pytest.raises(
        ConfigurationError, match="Cannot mix calibration instances with and without bins"
    ):
        regressor.insert_calibration(x_cal_2, y_cal_2, bins=bins_2)

    # Verify details are attached
    try:
        regressor.insert_calibration(x_cal_2, y_cal_2, bins=bins_2)
    except ConfigurationError as e:
        assert e.details is not None
        assert "requirement" in e.details
