"""Tests for explain helpers module.

This module tests the helper functions that were previously called as private
methods on CalibratedExplainer. Tests are refactored to test behavior through
the public explain module API rather than implementation details.
"""

import numpy as np
import pytest

from calibrated_explanations.core.explain import _helpers as helpers


class TestSliceThreshold:
    """Test behavior of slice_threshold helper function."""

    def test_slice_threshold_preserves_scalar(self):
        """Scalar values should be passed through unchanged."""
        scalar = 3.14
        result = helpers.slice_threshold(scalar, 0, 1, 2)
        assert result == scalar

    def test_slice_threshold_preserves_none(self):
        """None values should be passed through unchanged."""
        result = helpers.slice_threshold(None, 0, 1, 1)
        assert result is None

    def test_slice_threshold_slices_list(self):
        """Lists should be sliced correctly."""
        sequence = [1, 2, 3]
        result = helpers.slice_threshold(sequence, 0, 2, 3)
        assert result == [1, 2]

    def test_slice_threshold_preserves_list_when_range_mismatch(self):
        """Lists should be returned unchanged when range doesn't match total."""
        sequence = [1, 2, 3]
        result = helpers.slice_threshold(sequence, 0, 1, 4)
        assert result is sequence

    def test_slice_threshold_slices_numpy_array(self):
        """NumPy arrays should be sliced correctly."""
        array = np.array([4, 5, 6])
        result = helpers.slice_threshold(array, 1, 3, 3)
        np.testing.assert_array_equal(result, np.array([5, 6]))

    def test_slice_threshold_handles_pandas_series(self):
        """Pandas Series should be sliced correctly."""
        pytest.importorskip("pandas")
        import pandas as pd

        series = pd.Series([7, 8, 9])
        result = helpers.slice_threshold(series, 1, 3, 3)
        np.testing.assert_array_equal(result.to_numpy(), np.array([8, 9]))

    def test_slice_threshold_with_full_range(self):
        """Slicing full range should preserve array."""
        array = np.array([1, 2, 3, 4])
        result = helpers.slice_threshold(array, 0, 4, 4)
        np.testing.assert_array_equal(result, array)

    def test_slice_threshold_with_middle_range(self):
        """Slicing middle range should work correctly."""
        array = np.array([1, 2, 3, 4, 5])
        result = helpers.slice_threshold(array, 1, 4, 5)
        np.testing.assert_array_equal(result, np.array([2, 3, 4]))


class TestSliceBins:
    """Test behavior of slice_bins helper function."""

    def test_slice_bins_preserves_none(self):
        """None values should pass through unchanged."""
        result = helpers.slice_bins(None, 0, 1)
        assert result is None

    def test_slice_bins_slices_numpy_array(self):
        """NumPy arrays should be sliced correctly."""
        bins = np.array([0.1, 0.2, 0.3])
        result = helpers.slice_bins(bins, 1, 3)
        np.testing.assert_array_equal(result, np.array([0.2, 0.3]))

    def test_slice_bins_slices_list(self):
        """Lists should be sliced correctly."""
        bins = ["a", "b", "c"]
        result = helpers.slice_bins(bins, 1, 3)
        assert result == ["b", "c"]

    def test_slice_bins_slices_multidimensional_array(self):
        """Multidimensional arrays should be sliced correctly."""
        array_bins = np.array([[1, 2], [3, 4], [5, 6]])
        result = helpers.slice_bins(array_bins, 0, 2)
        np.testing.assert_array_equal(result, array_bins[:2])

    def test_slice_bins_with_pandas_series(self):
        """Pandas Series should be sliced correctly."""
        pytest.importorskip("pandas")
        import pandas as pd

        bins = pd.Series([10, 11, 12])
        result = helpers.slice_bins(bins, 0, 2)
        np.testing.assert_array_equal(result, np.array([10, 11]))

    def test_slice_bins_full_range(self):
        """Slicing full range should preserve array."""
        bins = np.array([1, 2, 3])
        result = helpers.slice_bins(bins, 0, 3)
        np.testing.assert_array_equal(result, bins)

    def test_slice_bins_single_element(self):
        """Slicing single element should work correctly."""
        bins = np.array([1, 2, 3])
        result = helpers.slice_bins(bins, 1, 2)
        np.testing.assert_array_equal(result, np.array([2]))


class TestComputeWeightDelta:
    """Test behavior of compute_weight_delta helper function."""

    def test_compute_weight_delta_scalar_vs_array(self):
        """Scalar baseline vs array perturbed should compute delta correctly."""
        result = helpers.compute_weight_delta(1.0, np.array([0.5, 1.5]))
        np.testing.assert_array_almost_equal(result, np.array([0.5, -0.5]))

    def test_compute_weight_delta_matching_shapes(self):
        """Arrays with matching shapes should compute delta correctly."""
        base = np.array([2.0, 3.0])
        pert = np.array([1.0, 5.0])
        result = helpers.compute_weight_delta(base, pert)
        np.testing.assert_array_almost_equal(result, np.array([1.0, -2.0]))

    def test_compute_weight_delta_all_zeros(self):
        """Zero baseline and perturbed should result in zeros."""
        result = helpers.compute_weight_delta(0.0, np.array([0.0, 0.0]))
        np.testing.assert_array_almost_equal(result, np.array([0.0, 0.0]))

    def test_compute_weight_delta_same_values(self):
        """Same baseline and perturbed values should result in zeros."""
        base = np.array([1.0, 2.0])
        pert = np.array([1.0, 2.0])
        result = helpers.compute_weight_delta(base, pert)
        np.testing.assert_array_almost_equal(result, np.array([0.0, 0.0]))

    def test_compute_weight_delta_single_value(self):
        """Single value arrays should work correctly."""
        result = helpers.compute_weight_delta(1.0, np.array([2.0]))
        np.testing.assert_array_almost_equal(result, np.array([-1.0]))

    def test_compute_weight_delta_negative_values(self):
        """Negative values should work correctly."""
        base = np.array([-1.0, -2.0])
        pert = np.array([-3.0, -1.0])
        result = helpers.compute_weight_delta(base, pert)
        np.testing.assert_array_almost_equal(result, np.array([2.0, -1.0]))

    def test_compute_weight_delta_large_values(self):
        """Large values should work correctly."""
        result = helpers.compute_weight_delta(1e6, np.array([1e6 + 1e5, 1e6 - 1e5]))
        np.testing.assert_array_almost_equal(result, np.array([-1e5, 1e5]))
