# pylint: disable=line-too-long, unused-import
"""
This module contains unit tests for the `uniform_perturbation` function from the
`calibrated_explanations.utils.perturbation` module.
The `uniform_perturbation` function is tested with various inputs to ensure its
correctness and robustness. The tests cover basic functionality, edge cases, and
performance with large inputs.
Tests included:
- `test_uniform_perturbation_basic`: Tests the function with a basic input array.
- `test_uniform_perturbation_severity_zero`: Tests the function with zero severity.
- `test_uniform_perturbation_severity_high`: Tests the function with high severity.
- `test_uniform_perturbation_negative_values`: Tests the function with negative values in the input array.
- `test_uniform_perturbation_large_column`: Tests the function with a large input array.
"""

import numpy as np
import pytest
from calibrated_explanations.utils.perturbation import (
    categorical_perturbation,
    gaussian_perturbation,
    perturb_dataset,
    uniform_perturbation,
)


def test_categorical_perturbation():
    """Test categorical_perturbation with basic input."""
    column = np.array(["a", "b", "c", "d", "e"])
    num_permutations = 5
    perturbed_column = categorical_perturbation(column, num_permutations)
    assert len(perturbed_column) == len(column)
    assert not np.array_equal(perturbed_column, column)


def test_categorical_perturbation_fallback_swaps_values():
    """Ensure the deterministic fallback swaps values when permutations keep order."""

    class StaticPermutationRNG:
        """Return the same ordering regardless of the permutation request."""

        def permutation(self, values):
            return np.array(values, copy=True)

    column = np.array([0, 1, 2, 3])
    perturbed_column = categorical_perturbation(
        column, num_permutations=2, rng=StaticPermutationRNG()
    )
    assert not np.array_equal(perturbed_column, column)
    assert sorted(perturbed_column) == sorted(column)


def test_gaussian_perturbation_basic():
    """Test gaussian_perturbation with basic input."""
    column = np.array([1, 2, 3, 4, 5])
    severity = 0.1
    perturbed_column = gaussian_perturbation(column, severity)
    assert len(perturbed_column) == len(column)
    assert not np.array_equal(perturbed_column, column)


# def test_gaussian_perturbation_severity_zero():
#     """Test gaussian_perturbation with zero severity."""
#     column = np.array([1, 2, 3, 4, 5])
#     severity = 0
#     perturbed_column = gaussian_perturbation(column, severity)
#     assert len(perturbed_column) == len(column)
#     assert np.array_equal(perturbed_column, column)


def test_gaussian_perturbation_high_severity():
    """Test gaussian_perturbation with high severity."""
    column = np.array([1, 2, 3, 4, 5])
    severity = 10
    perturbed_column = gaussian_perturbation(column, severity)
    assert len(perturbed_column) == len(column)
    assert not np.array_equal(perturbed_column, column)


def test_gaussian_perturbation_severity_zero_returns_mean():
    """Zero severity should collapse the column to its mean value."""

    column = np.array([1.0, 2.0, 3.0])
    result = gaussian_perturbation(column, severity=0.0, rng=np.random.default_rng(0))
    assert np.allclose(result, np.full_like(column, column.mean()))


def test_uniform_perturbation_basic():
    """Test uniform_perturbation with basic input."""
    column = np.array([1, 2, 3, 4, 5])
    severity = 0.1
    perturbed_column = uniform_perturbation(column, severity)
    assert len(perturbed_column) == len(column)
    assert not np.array_equal(perturbed_column, column)


def test_uniform_perturbation_severity_zero():
    """Test uniform_perturbation with zero severity."""
    column = np.array([1, 2, 3, 4, 5])
    severity = 0.0
    perturbed_column = uniform_perturbation(column, severity)
    assert np.array_equal(perturbed_column, column)


def test_uniform_perturbation_severity_high():
    """Test uniform_perturbation with high severity."""
    column = np.array([1, 2, 3, 4, 5])
    severity = 1.0
    perturbed_column = uniform_perturbation(column, severity)
    assert len(perturbed_column) == len(column)
    assert not np.array_equal(perturbed_column, column)


def test_uniform_perturbation_negative_values():
    """Test uniform_perturbation with negative values in the column."""
    column = np.array([-1, -2, -3, -4, -5])
    severity = 0.1
    perturbed_column = uniform_perturbation(column, severity)
    assert len(perturbed_column) == len(column)
    assert not np.array_equal(perturbed_column, column)


def test_uniform_perturbation_large_column():
    """Test uniform_perturbation with a large column."""
    rng = np.random.default_rng()
    column = rng.random(1000)
    severity = 0.1
    perturbed_column = uniform_perturbation(column, severity)
    assert len(perturbed_column) == len(column)
    assert not np.array_equal(perturbed_column, column)


def test_gaussian_perturbation_severity_high():
    """Test gaussian_perturbation with high severity."""
    column = np.array([1, 2, 3, 4, 5])
    severity = 1.0
    perturbed_column = gaussian_perturbation(column, severity)
    assert len(perturbed_column) == len(column)
    assert not np.array_equal(perturbed_column, column)


def test_gaussian_perturbation_negative_values():
    """Test gaussian_perturbation with negative values in the column."""
    column = np.array([-1, -2, -3, -4, -5])
    severity = 0.1
    perturbed_column = gaussian_perturbation(column, severity)
    assert len(perturbed_column) == len(column)
    assert not np.array_equal(perturbed_column, column)


def test_gaussian_perturbation_large_column():
    """Test gaussian_perturbation with a large column."""
    rng = np.random.default_rng()
    column = rng.random(1000)
    severity = 0.1
    perturbed_column = gaussian_perturbation(column, severity)
    assert len(perturbed_column) == len(column)
    assert not np.array_equal(perturbed_column, column)


class _IdentityPermutationRng:
    """Deterministic RNG that always returns the input array unchanged."""

    def permutation(self, column):
        return column


def test_categorical_perturbation_fallback_swaps_first_two_values():
    """When permutations fail, the fallback should swap two entries if possible."""

    column = np.array(["cat", "dog", "mouse"])
    rng = _IdentityPermutationRng()

    result = categorical_perturbation(column, num_permutations=2, rng=rng)

    assert not np.array_equal(result, column)
    assert list(result) == ["dog", "cat", "mouse"]


def test_categorical_perturbation_fallback_returns_copy_for_constant_column():
    """The fallback should still return a copy even when no swap is possible."""

    column = np.array(["same", "same", "same"])
    rng = _IdentityPermutationRng()

    result = categorical_perturbation(column, num_permutations=3, rng=rng)

    assert np.array_equal(result, column)
    assert result is not column


def test_perturb_dataset_rejects_unknown_noise_type():
    """`perturb_dataset` should raise for unsupported noise types."""

    x_cal = np.array([[0.0, 1.0]])
    y_cal = np.array([1])

    with pytest.raises(ValueError, match="Noise type must be either 'uniform' or 'gaussian'."):
        perturb_dataset(x_cal, y_cal, categorical_features=[], noise_type="laplace")


class _RecordingGaussianRng:
    """RNG stub that records gaussian noise requests."""

    def __init__(self):
        self.calls = []

    def normal(self, *, loc, scale, size):
        self.calls.append((loc, scale, size))
        return np.zeros(size)


def test_perturb_dataset_uses_gaussian_noise_path():
    """The gaussian branch should delegate to `gaussian_perturbation`."""

    x_cal = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_cal = np.array([0, 1])
    rng = _RecordingGaussianRng()

    perturbed_x, scaled_x, scaled_y, factor = perturb_dataset(
        x_cal,
        y_cal,
        categorical_features=[],
        noise_type="gaussian",
        scale_factor=2,
        severity=0.5,
        rng=rng,
    )

    # Each feature column should have been collapsed to its mean by the zero noise stub.
    expected_column_means = scaled_x.mean(axis=0)
    assert np.allclose(perturbed_x, expected_column_means)
    assert np.array_equal(scaled_x, np.tile(x_cal, (2, 1)))
    assert np.array_equal(scaled_y, np.tile(y_cal, 2))
    assert factor == 2
    assert rng.calls and all(call[0] == 0 for call in rng.calls)


def test_perturb_dataset_uniform_with_categorical_features():
    """The perturbation pipeline should permute categorical features and add noise to numeric ones."""

    x_cal = np.array([[0, 0.1], [1, 0.2]], dtype=float)
    y_cal = np.array([0, 1])
    perturbed_x, scaled_x, scaled_y, factor = perturb_dataset(
        x_cal,
        y_cal,
        categorical_features=[0],
        noise_type="uniform",
        scale_factor=3,
        severity=0.4,
        seed=1234,
    )
    assert perturbed_x.shape == (6, 2)
    assert scaled_x.shape == (6, 2)
    assert scaled_y.shape == (6,)
    assert factor == 3
    # Ensure categorical feature is permuted but values preserved
    assert set(perturbed_x[:, 0]) == {0, 1}
    # The numeric column should differ from the scaled copy due to uniform noise
    assert not np.allclose(perturbed_x[:, 1], scaled_x[:, 1])


def test_perturb_dataset_gaussian_reproducible_with_seed():
    """Supplying the same seed should make gaussian perturbation reproducible."""

    x_cal = np.array([[0.0, 1.0], [1.0, 2.0]])
    y_cal = np.array([0.0, 1.0])
    first_run = perturb_dataset(
        x_cal,
        y_cal,
        categorical_features=[],
        noise_type="gaussian",
        scale_factor=2,
        severity=0.3,
        seed=42,
    )
    second_run = perturb_dataset(
        x_cal,
        y_cal,
        categorical_features=[],
        noise_type="gaussian",
        scale_factor=2,
        severity=0.3,
        seed=42,
    )
    assert np.allclose(first_run[0], second_run[0])


def test_perturb_dataset_invalid_noise_type():
    """An informative error should be raised for unsupported noise types."""

    x_cal = np.array([[0.0], [1.0]])
    y_cal = np.array([0, 1])
    with pytest.raises(ValueError):
        perturb_dataset(
            x_cal,
            y_cal,
            categorical_features=[],
            noise_type="laplace",
            severity=0.1,
        )
