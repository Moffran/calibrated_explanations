import numpy as np
import pytest

from calibrated_explanations.utils import perturbation


class _IdentityRNG:
    """Deterministic RNG that always returns the original input."""

    def permutation(self, column):
        return np.array(column, copy=True)

    def normal(self, loc, scale, size):  # pragma: no cover - only used in gaussian tests
        return np.zeros(size)

    def uniform(self, low, high, size):  # pragma: no cover - only used in uniform tests
        return np.zeros(size)


def test_categorical_perturbation_respects_rng_and_returns_copy():
    column = np.array(["red", "green", "blue", "yellow"])
    expected = np.random.default_rng(42).permutation(column)
    result = perturbation.categorical_perturbation(
        column, num_permutations=1, rng=np.random.default_rng(42)
    )

    # Ensure the operation is deterministic with the provided RNG and leaves the input untouched.
    assert np.array_equal(result, expected)
    assert np.array_equal(column, np.array(["red", "green", "blue", "yellow"]))
    assert not np.shares_memory(result, column)


def test_categorical_perturbation_identity_rng_triggers_swap_fallback():
    column = np.array([1, 2, 3, 4])
    result = perturbation.categorical_perturbation(column, num_permutations=0, rng=_IdentityRNG())

    # The degenerate RNG should force the fallback swap branch.
    assert np.array_equal(result, np.array([2, 1, 3, 4]))
    assert not np.shares_memory(result, column)


def test_categorical_perturbation_identity_rng_handles_constant_column():
    column = np.array([7, 7, 7])
    result = perturbation.categorical_perturbation(column, rng=_IdentityRNG())

    # When the column has no variability we still expect a defensive copy.
    assert np.array_equal(result, column)
    assert not np.shares_memory(result, column)


def test_gaussian_perturbation_matches_numpy_rng():
    column = np.array([1.0, 2.0, 3.0, 4.0])
    severity = 0.3
    seed = 17
    original_copy = column.copy()

    expected_rng = np.random.default_rng(seed)
    expected = column.mean() + expected_rng.normal(
        loc=0, scale=column.std() * severity, size=len(column)
    )

    result = perturbation.gaussian_perturbation(column, severity, rng=np.random.default_rng(seed))

    assert np.allclose(result, expected)
    assert np.array_equal(column, original_copy)


def test_uniform_perturbation_matches_numpy_rng():
    column = np.array([0.0, 2.0, 4.0, 6.0])
    severity = 0.25
    seed = 91
    original_copy = column.copy()

    expected_rng = np.random.default_rng(seed)
    original_range = column.max() - column.min()
    expected = column + expected_rng.uniform(
        low=-original_range * severity, high=original_range * severity, size=len(column)
    )

    result = perturbation.uniform_perturbation(column, severity, rng=np.random.default_rng(seed))

    assert np.allclose(result, expected)
    assert np.array_equal(column, original_copy)


def test_perturb_dataset_uniform_with_seed_and_categorical_features():
    x_cal = np.array(
        [
            [0, 0.1],
            [1, 0.3],
            [2, 0.5],
        ]
    )
    y_cal = np.array([1, 0, 1])
    scale_factor = 3
    severity = 0.4
    seed = 11

    perturbed_a, scaled_x_a, scaled_y_a, factor_a = perturbation.perturb_dataset(
        x_cal,
        y_cal,
        categorical_features=[0],
        noise_type="uniform",
        scale_factor=scale_factor,
        severity=severity,
        seed=seed,
    )
    perturbed_b, scaled_x_b, scaled_y_b, factor_b = perturbation.perturb_dataset(
        x_cal,
        y_cal,
        categorical_features=[0],
        noise_type="uniform",
        scale_factor=scale_factor,
        severity=severity,
        seed=seed,
    )

    # Deterministic seeding should reproduce the same perturbations.
    assert np.array_equal(perturbed_a, perturbed_b)
    assert np.array_equal(scaled_x_a, scaled_x_b)
    assert np.array_equal(scaled_y_a, scaled_y_b)
    assert factor_a == factor_b == scale_factor

    expected_scaled_x = np.tile(x_cal, (scale_factor, 1))
    expected_scaled_y = np.tile(y_cal, scale_factor)
    assert np.array_equal(scaled_x_a, expected_scaled_x)
    assert np.array_equal(scaled_y_a, expected_scaled_y)

    # Recreate the expected perturbations by mirroring the module logic with the same RNG.
    rng = np.random.default_rng(seed)
    expected_perturbed = expected_scaled_x.copy()

    # Categorical feature (index 0)
    cat_column = expected_perturbed[:, 0]
    attempts = max(1, int(5))
    fallback_result = cat_column.copy()
    for _ in range(attempts):
        candidate = rng.permutation(cat_column)
        if not np.array_equal(candidate, cat_column):
            fallback_result = candidate
            break
    else:
        if cat_column.size > 1 and len(np.unique(cat_column)) > 1:
            fallback_result = cat_column.copy()
            fallback_result[0], fallback_result[1] = fallback_result[1], fallback_result[0]
        else:
            fallback_result = cat_column.copy()
    expected_perturbed[:, 0] = fallback_result

    # Numeric feature (index 1)
    num_column = expected_scaled_x[:, 1]
    original_range = num_column.max() - num_column.min()
    noise = rng.uniform(
        low=-original_range * severity, high=original_range * severity, size=len(num_column)
    )
    expected_perturbed[:, 1] = num_column + noise

    assert np.allclose(perturbed_a, expected_perturbed)


def test_perturb_dataset_gaussian_with_custom_rng():
    x_cal = np.array(
        [
            [0.0, 1.0],
            [2.0, 3.0],
        ]
    )
    y_cal = np.array([0, 1])
    scale_factor = 2
    severity = 0.5
    rng_seed = 2024

    rng = np.random.default_rng(rng_seed)
    perturbed, scaled_x, scaled_y, factor = perturbation.perturb_dataset(
        x_cal,
        y_cal,
        categorical_features=None,
        noise_type="gaussian",
        scale_factor=scale_factor,
        severity=severity,
        rng=rng,
    )

    expected_scaled_x = np.tile(x_cal, (scale_factor, 1))
    expected_scaled_y = np.tile(y_cal, scale_factor)

    assert np.array_equal(scaled_x, expected_scaled_x)
    assert np.array_equal(scaled_y, expected_scaled_y)
    assert factor == scale_factor

    manual_rng = np.random.default_rng(rng_seed)
    expected = np.empty_like(expected_scaled_x)
    for f in range(expected_scaled_x.shape[1]):
        column = expected_scaled_x[:, f]
        mean = column.mean()
        std = column.std()
        noise = manual_rng.normal(loc=0, scale=std * severity, size=len(column))
        expected[:, f] = mean + noise

    assert np.allclose(perturbed, expected)


def test_perturb_dataset_rejects_unknown_noise_type():
    from calibrated_explanations.core.exceptions import ValidationError

    x_cal = np.zeros((2, 2))
    y_cal = np.zeros(2)

    with pytest.raises(ValidationError, match="Noise type must be either 'uniform' or 'gaussian'."):
        perturbation.perturb_dataset(x_cal, y_cal, noise_type="laplace")
