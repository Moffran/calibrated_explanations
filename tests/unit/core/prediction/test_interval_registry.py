"""Unit tests for IntervalRegistry (Phase 4: Interval Management Extraction).

Tests cover interval learner lifecycle, initialization, updates, and sigma computation.
"""

# pylint: disable=protected-access, too-many-lines, invalid-name, line-too-long
# pylint: disable=missing-function-docstring, missing-class-docstring

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer


class TestIntervalRegistry:
    """Test suite for IntervalRegistry class."""

    @pytest.fixture
    def binary_explainer(self):
        """Create a simple binary classification explainer for testing."""
        x_data, y_data = make_classification(
            n_samples=100, n_features=5, n_informative=3, random_state=42
        )
        x_cal, y_cal = x_data[:40], y_data[:40]
        learner = LogisticRegression(random_state=42)
        learner.fit(x_cal, y_cal)
        return CalibratedExplainer(learner, x_cal, y_cal, mode="classification")

    def test_registry_initialization(self, binary_explainer):
        """Test that IntervalRegistry is properly initialized with explainer reference."""
        registry = binary_explainer._prediction_orchestrator._interval_registry
        assert registry is not None
        assert registry.explainer is binary_explainer

    def test_interval_learner_property_get(self, binary_explainer):
        """Test that interval_learner property getter works through registry."""
        # After explainer initialization, interval_learner should be set
        assert binary_explainer.interval_learner is not None
        # Verify it's accessible through the registry
        registry = binary_explainer._prediction_orchestrator._interval_registry
        assert registry.interval_learner is not None

    def test_interval_learner_property_set(self, binary_explainer):
        """Test that interval_learner property setter works through registry."""
        original_learner = binary_explainer.interval_learner
        mock_learner = "mock_learner"
        binary_explainer.interval_learner = mock_learner
        assert binary_explainer.interval_learner == mock_learner
        # Restore
        binary_explainer.interval_learner = original_learner

    def test_get_sigma_test_with_no_difficulty_estimator(self, binary_explainer):
        """Test _get_sigma_test returns unit vector when no difficulty estimator is set."""
        x_test = np.zeros((3, binary_explainer.num_features))
        sigma = binary_explainer._get_sigma_test(x_test)
        expected = np.ones(3)
        np.testing.assert_array_equal(sigma, expected)

    def test_get_sigma_test_with_difficulty_estimator(self, binary_explainer):
        """Test _get_sigma_test uses difficulty estimator when available."""

        class MockDifficultyEstimator:
            """Mock difficulty estimator for testing."""

            def apply(self, x_input):
                """Return constant difficulty values."""
                return np.full(x_input.shape[0], 2.5)

        binary_explainer.difficulty_estimator = MockDifficultyEstimator()
        x_test = np.zeros((4, binary_explainer.num_features))
        sigma = binary_explainer._get_sigma_test(x_test)
        expected = np.full(4, 2.5)
        np.testing.assert_array_equal(sigma, expected)

    def test_backward_compatible_get_sigma_test_delegation(self, binary_explainer):
        """Test that backward-compatible _get_sigma_test method delegates correctly."""
        x_test = np.zeros((2, binary_explainer.num_features))
        # Should delegate to registry
        sigma = binary_explainer._get_sigma_test(x_test)
        registry = binary_explainer._prediction_orchestrator._interval_registry
        registry_sigma = registry.get_sigma_test(x_test)
        np.testing.assert_array_equal(sigma, registry_sigma)

    def test_registry_maintains_explainer_reference(self, binary_explainer):
        """Test that registry maintains correct reference to explainer."""
        registry = binary_explainer._prediction_orchestrator._interval_registry
        assert registry.explainer is binary_explainer
        assert registry.explainer.mode == "classification"
        assert registry.explainer.num_features == binary_explainer.num_features

