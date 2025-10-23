import numpy as np
import pytest

from calibrated_explanations.utils.perturbation import (
    categorical_perturbation,
    gaussian_perturbation,
    perturb_dataset,
    uniform_perturbation,
)


class _TrackingPermutationRng:
    """RNG stub that returns a deterministic permutation and records calls."""

    def __init__(self):
        self.calls = 0

    def permutation(self, values):
        self.calls += 1
        if self.calls == 1:
            return np.array(values, copy=True)
        return np.array(list(reversed(values)))


class _IdentityPermutationRng:
    """RNG stub that always returns the input unchanged."""

    def permutation(self, values):
        return np.array(values, copy=True)


class _DeterministicUniformRng:
    """RNG stub that records uniform requests and returns predictable draws."""

    def __init__(self):
        self.permutation_inputs = []
        self.uniform_calls = []

    def permutation(self, values):
        self.permutation_inputs.append(np.array(values, copy=True))
        return np.array(list(reversed(values)))

    def uniform(self, *args, **kwargs):
        low = kwargs.get("low", args[0] if args else 0.0)
        high = kwargs.get("high", args[1] if len(args) > 1 else None)
        size = kwargs.get("size", args[2] if len(args) > 2 else None)
        self.uniform_calls.append((low, high, size))
        return np.arange(size, dtype=float)


class _RecordingGaussianRng:
    """RNG stub that records gaussian requests and returns zero noise."""

    def __init__(self):
        self.calls = []

    def normal(self, *, loc, scale, size):
        self.calls.append((loc, scale, size))
        return np.zeros(size)


def test_categorical_perturbation_generates_new_permutation():
    column = np.array(["a", "b", "c", "d"])
    rng = np.random.default_rng(123)

    result = categorical_perturbation(column, num_permutations=3, rng=rng)

    assert result.shape == column.shape
    assert not np.array_equal(result, column)
    assert set(result) == set(column)


def test_categorical_perturbation_casts_float_attempts():
    column = np.array([0, 1, 2, 3])
    rng = _TrackingPermutationRng()

    result = categorical_perturbation(column, num_permutations=2.6, rng=rng)

    assert rng.calls == 2  # ``num_permutations`` should be cast to an int.
    assert np.array_equal(result, np.array([3, 2, 1, 0]))


def test_categorical_perturbation_fallback_swaps_values_when_permutations_identical():
    column = np.array(["cat", "dog", "mouse"])
    rng = _IdentityPermutationRng()

    result = categorical_perturbation(column, num_permutations=0, rng=rng)

    assert list(result) == ["dog", "cat", "mouse"]


def test_categorical_perturbation_returns_copy_for_single_unique_value():
    column = np.array(["same", "same", "same"])
    rng = _IdentityPermutationRng()

    result = categorical_perturbation(column, num_permutations=5, rng=rng)

    assert np.array_equal(result, column)
    assert result is not column


def test_gaussian_perturbation_zero_severity_returns_constant_mean():
    column = np.array([1.0, 2.0, 3.0])
    rng = np.random.default_rng(0)

    result = gaussian_perturbation(column, severity=0.0, rng=rng)

    assert np.allclose(result, np.full_like(column, column.mean()))


def test_gaussian_perturbation_matches_expected_draws():
    column = np.array([0.0, 1.0, 2.0, 3.0])
    severity = 0.5
    expected_rng = np.random.default_rng(42)
    expected_noise = expected_rng.normal(loc=0, scale=column.std() * severity, size=len(column))
    expected = column.mean() + expected_noise

    actual_rng = np.random.default_rng(42)
    result = gaussian_perturbation(column, severity=severity, rng=actual_rng)

    assert np.allclose(result, expected)


def test_uniform_perturbation_matches_expected_draws():
    column = np.array([10.0, 12.0, 14.0])
    severity = 0.25
    expected_rng = np.random.default_rng(24)
    perturbation = expected_rng.uniform(
        low=-(column.max() - column.min()) * severity,
        high=(column.max() - column.min()) * severity,
        size=len(column),
    )
    expected = column + perturbation

    actual_rng = np.random.default_rng(24)
    result = uniform_perturbation(column, severity=severity, rng=actual_rng)

    assert np.allclose(result, expected)


def test_perturb_dataset_uniform_with_categorical_features_and_rng_stub():
    x_cal = np.array([[0, 0.1], [1, 0.2]], dtype=float)
    y_cal = np.array([0, 1])
    rng = _DeterministicUniformRng()

    perturbed_x, scaled_x, scaled_y, factor = perturb_dataset(
        x_cal,
        y_cal,
        categorical_features=[0],
        noise_type="uniform",
        scale_factor=3,
        severity=0.4,
        rng=rng,
    )

    assert factor == 3
    assert perturbed_x.shape == (6, 2)
    assert np.array_equal(scaled_x, np.tile(x_cal, (3, 1)))
    assert np.array_equal(scaled_y, np.tile(y_cal, 3))
    # The categorical column should have been permuted using the stub.
    assert rng.permutation_inputs and not np.array_equal(
        rng.permutation_inputs[0], perturbed_x[:, 0]
    )
    assert set(perturbed_x[:, 0]) == {0.0, 1.0}
    # The numeric column should reflect the deterministic uniform draws.
    np.testing.assert_allclose(perturbed_x[:, 1] - scaled_x[:, 1], np.arange(6, dtype=float))
    low, high, size = rng.uniform_calls[0]
    assert size == 6
    assert pytest.approx(low) == -0.04
    assert pytest.approx(high) == 0.04


def test_perturb_dataset_accepts_none_categorical_features():
    x_cal = np.array([[0.0, 1.0]])
    y_cal = np.array([1.0])

    perturbed_x, scaled_x, scaled_y, factor = perturb_dataset(
        x_cal,
        y_cal,
        categorical_features=None,
        noise_type="uniform",
        scale_factor=2,
        severity=0.0,
        seed=123,
    )

    assert factor == 2
    assert np.array_equal(perturbed_x, scaled_x)
    assert np.array_equal(scaled_y, np.tile(y_cal, 2))


def test_perturb_dataset_seed_reproducible_for_uniform_noise():
    x_cal = np.array([[0.0, 1.0], [1.0, 2.0]])
    y_cal = np.array([0.0, 1.0])

    first = perturb_dataset(
        x_cal,
        y_cal,
        categorical_features=[],
        noise_type="uniform",
        scale_factor=2,
        severity=0.3,
        seed=2024,
    )
    second = perturb_dataset(
        x_cal,
        y_cal,
        categorical_features=[],
        noise_type="uniform",
        scale_factor=2,
        severity=0.3,
        seed=2024,
    )

    for idx in range(3):
        assert np.allclose(first[idx], second[idx])
    assert first[3] == second[3]


def test_perturb_dataset_gaussian_uses_provided_rng():
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

    assert factor == 2
    assert np.array_equal(scaled_x, np.tile(x_cal, (2, 1)))
    assert np.array_equal(scaled_y, np.tile(y_cal, 2))
    expected_means = np.tile(scaled_x.mean(axis=0), (scaled_x.shape[0], 1))
    assert np.allclose(perturbed_x, expected_means)
    assert rng.calls == [(0, scaled_x[:, 0].std() * 0.5, 4), (0, scaled_x[:, 1].std() * 0.5, 4)]


def test_perturb_dataset_rejects_invalid_noise_type():
    with pytest.raises(ValueError, match="Noise type must be either 'uniform' or 'gaussian'."):
        perturb_dataset(
            np.array([[0.0]]),
            np.array([0.0]),
            categorical_features=[],
            noise_type="laplace",
            scale_factor=1,
            severity=0.1,
        )
