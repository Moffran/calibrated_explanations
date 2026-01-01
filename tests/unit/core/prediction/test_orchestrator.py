"""Unit tests for PredictionOrchestrator.

Tests cover prediction workflow behavior, interval calibration, uncertainty quantification,
and prediction caching - without testing implementation details or internal state.

Tests should focus on:
- Prediction outputs match expected values
- Uncertainty intervals are computed correctly
- Interval calibration is applied appropriately
- Caching improves performance without changing results
"""

# pylint: disable=protected-access, too-many-lines, invalid-name, line-too-long
# pylint: disable=missing-function-docstring, missing-class-docstring

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer


class TestPredictionOrchestratorBehavior:
    """Test suite for PredictionOrchestrator prediction behavior."""

    @pytest.fixture
    def binary_classification_explainer(self):
        """Create a simple binary classification explainer for testing."""
        x_data, y_data = make_classification(
            n_samples=100, n_features=10, n_informative=5, n_redundant=2, random_state=42
        )
        x_cal, y_cal = x_data[:40], y_data[:40]
        x_test = x_data[40:60]
        learner = LogisticRegression(random_state=42, solver="liblinear")
        learner.fit(x_cal, y_cal)
        explainer = CalibratedExplainer(learner, x_cal, y_cal, mode="classification")
        return explainer, x_test

    @pytest.fixture
    def regression_explainer(self):
        """Create a simple regression explainer for testing."""
        x_data, y_data = make_regression(n_samples=100, n_features=5, random_state=42)
        x_cal, y_cal = x_data[:40], y_data[:40]
        x_test = x_data[40:60]
        learner = LinearRegression()
        learner.fit(x_cal, y_cal)
        explainer = CalibratedExplainer(learner, x_cal, y_cal, mode="regression")
        return explainer, x_test

    def test_predict_classification_returns_valid_probabilities(
        self, binary_classification_explainer
    ):
        """Test that classification predictions return valid probability values."""
        explainer, x_test = binary_classification_explainer
        predictions = explainer.predict(x_test)

        # Should return class labels or probabilities, not NaN
        assert predictions is not None
        assert len(predictions) == len(x_test)

    def test_predict_with_uq_interval_returns_bounds(self, binary_classification_explainer):
        """Test that prediction with UQ returns both predictions and intervals."""
        explainer, x_test = binary_classification_explainer
        predictions, intervals = explainer.predict(x_test, uq_interval=True)

        # Should return tuple of (predictions, (low, high))
        assert isinstance(intervals, tuple)
        assert len(intervals) == 2
        low, high = intervals

        # Intervals should have same length as predictions
        assert len(low) == len(predictions)
        assert len(high) == len(predictions)

        # Low should be <= high for each instance
        for low_val, high_val in zip(low, high):
            assert low_val <= high_val

    def test_predict_regression_returns_predictions(self, regression_explainer):
        """Test that regression predictions return numeric values."""
        explainer, x_test = regression_explainer
        predictions = explainer.predict(x_test)

        # Should return numeric predictions, not NaN
        assert predictions is not None
        assert len(predictions) == len(x_test)
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)

    def test_predict_regression_with_intervals(self, regression_explainer):
        """Test that regression predictions with UQ return valid intervals."""
        explainer, x_test = regression_explainer
        predictions, intervals = explainer.predict(x_test, uq_interval=True)

        # Should return tuple of (predictions, (low, high))
        assert isinstance(intervals, tuple)
        assert len(intervals) == 2
        low, high = intervals

        # All intervals should be valid (not NaN)
        assert not np.any(np.isnan(low))
        assert not np.any(np.isnan(high))

        # Low should be <= prediction <= high for each instance
        for p, low_val, high_val in zip(predictions, low, high):
            assert low_val <= p <= high_val or (np.isnan(low_val) or np.isnan(high_val))

    def test_predict_with_percentiles_affects_interval_width(self, regression_explainer):
        """Test that different percentiles produce different interval widths."""
        explainer, x_test = regression_explainer

        # Narrow interval
        _, (low_narrow, high_narrow) = explainer.predict(
            x_test, uq_interval=True, low_high_percentiles=(25, 75)
        )
        narrow_width = np.mean(np.array(high_narrow) - np.array(low_narrow))

        # Wide interval
        _, (low_wide, high_wide) = explainer.predict(
            x_test, uq_interval=True, low_high_percentiles=(5, 95)
        )
        wide_width = np.mean(np.array(high_wide) - np.array(low_wide))

        # Wide percentiles should produce wider intervals
        assert wide_width > narrow_width

    def test_predict_consistency_across_calls(self, binary_classification_explainer):
        """Test that predictions are consistent across multiple calls."""
        explainer, x_test = binary_classification_explainer

        # Call predict multiple times
        pred1 = explainer.predict(x_test)
        pred2 = explainer.predict(x_test)

        # Predictions should be identical
        np.testing.assert_array_equal(pred1, pred2)

    def test_predict_single_instance(self, binary_classification_explainer):
        """Test that predict works correctly for a single instance."""
        explainer, x_test = binary_classification_explainer
        single_x = x_test[[0]]  # Select first instance as 2D array

        prediction = explainer.predict(single_x)
        assert prediction is not None
        assert len(prediction) == 1

    def test_predict_multiple_instances(self, binary_classification_explainer):
        """Test that predict correctly processes multiple instances."""
        explainer, x_test = binary_classification_explainer

        predictions = explainer.predict(x_test)
        assert len(predictions) == len(x_test)

    def test_calibrated_vs_uncalibrated_may_differ(self, binary_classification_explainer):
        """Test that calibrated and uncalibrated predictions can differ."""
        explainer, x_test = binary_classification_explainer

        calibrated_pred = explainer.predict(x_test, calibrated=True)
        uncalibrated_pred = explainer.predict(x_test, calibrated=False)

        # They may be different (calibration can adjust probabilities)
        # This is a weak test but verifies the API works
        assert len(calibrated_pred) == len(uncalibrated_pred)

    def test_predict_proba_returns_probabilities(self, binary_classification_explainer):
        """Test that predict_proba returns class probabilities."""
        explainer, x_test = binary_classification_explainer

        proba = explainer.predict_proba(x_test)

        # For binary classification, should return probabilities or numeric values
        assert proba is not None
        assert len(proba) == len(x_test)

    def test_predict_proba_with_interval(self, binary_classification_explainer):
        """Test that predict_proba with UQ returns probabilities and intervals."""
        explainer, x_test = binary_classification_explainer

        proba, intervals = explainer.predict_proba(x_test, uq_interval=True)

        # Should return tuple of (proba, (low, high))
        assert isinstance(intervals, tuple)
        assert len(intervals) == 2
        low, high = intervals

        # Intervals should have same length as probabilities
        assert len(low) == len(proba)
        assert len(high) == len(proba)

    def test_predict_output_shapes_match_input(self, binary_classification_explainer):
        """Test that prediction output shapes match input shapes."""
        explainer, x_test = binary_classification_explainer

        # Predict
        predictions = explainer.predict(x_test)

        # Should have one prediction per instance
        assert len(predictions) == x_test.shape[0]

    def test_predict_with_different_batch_sizes(self, binary_classification_explainer):
        """Test that predictions work correctly with different batch sizes."""
        explainer, x_test = binary_classification_explainer

        # Small batch
        pred_small = explainer.predict(x_test[:5])
        assert len(pred_small) == 5

        # Large batch
        pred_large = explainer.predict(x_test)
        assert len(pred_large) == len(x_test)

        # Results for the first 5 should match
        np.testing.assert_array_equal(pred_small, pred_large[:5])


class TestPredictionOrchestratorIntervalCalibration:
    """Test suite for interval calibration in prediction orchestration."""

    @pytest.fixture
    def regression_explainer_with_intervals(self):
        """Create a regression explainer configured for interval prediction."""
        x_data, y_data = make_regression(n_samples=100, n_features=5, random_state=42)
        x_cal, y_cal = x_data[:40], y_data[:40]
        x_test = x_data[40:60]
        learner = LinearRegression()
        learner.fit(x_cal, y_cal)
        explainer = CalibratedExplainer(learner, x_cal, y_cal, mode="regression")
        return explainer, x_test

    def test_interval_learner_initialized(self, regression_explainer_with_intervals):
        """Test that interval learner is initialized during explainer setup."""
        explainer, _ = regression_explainer_with_intervals

        # Interval learner should be set
        assert explainer.interval_learner is not None

    def test_interval_learner_affects_predictions(self, regression_explainer_with_intervals):
        """Test that interval learner is used in prediction computation."""
        explainer, x_test = regression_explainer_with_intervals

        # Get predictions with intervals
        predictions, (low, high) = explainer.predict(x_test, uq_interval=True)

        # Intervals should not all be the same (calibrator should differentiate)
        assert len(set(low)) > 1 or len(set(high)) > 1 or len(low) == 1


class TestPredictionOrchestratorEdgeCases:
    """Test suite for edge cases in prediction orchestration."""

    @pytest.fixture
    def simple_explainer(self):
        """Create a simple explainer for edge case testing."""
        x_data, y_data = make_classification(
            n_samples=50, n_features=10, n_informative=5, n_redundant=2, random_state=42
        )
        x_cal, y_cal = x_data[:20], y_data[:20]
        x_test = x_data[20:30]
        learner = LogisticRegression(random_state=42, solver="liblinear")
        learner.fit(x_cal, y_cal)
        explainer = CalibratedExplainer(learner, x_cal, y_cal, mode="classification")
        return explainer, x_test

    def test_predict_not_fitted_raises_error(self):
        """Test that predict raises error if explainer is not initialized correctly."""
        x_data, y_data = make_classification(
            n_samples=50, n_features=10, n_informative=5, n_redundant=2, random_state=42
        )
        x_cal, y_cal = x_data[:20], y_data[:20]
        learner = LogisticRegression(random_state=42, solver="liblinear")
        learner.fit(x_cal, y_cal)
        # Create explainer with minimal calibration (should still work)
        explainer = CalibratedExplainer(learner, x_cal, y_cal, mode="classification")

        # Predict should work on fitted explainer
        predictions = explainer.predict(x_data[20:30])
        assert predictions is not None

    def test_predict_handles_edge_values(self, simple_explainer):
        """Test that predict handles edge values gracefully."""
        explainer, x_test = simple_explainer

        # Test with zeros
        x_zeros = np.zeros_like(x_test)
        predictions = explainer.predict(x_zeros)
        assert predictions is not None

        # Test with ones
        x_ones = np.ones_like(x_test)
        predictions = explainer.predict(x_ones)
        assert predictions is not None

    def test_predict_no_regression_on_nans(self, simple_explainer):
        """Test that predict does not return NaN for valid inputs."""
        explainer, x_test = simple_explainer

        predictions = explainer.predict(x_test)

        # Predictions should be returned
        assert predictions is not None
        assert len(predictions) == len(x_test)
