from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.utils import perturbation


class StaticRNG:
    """Return the input unchanged to force categorical fallback paths."""

    def permutation(self, values):
        return np.array(values)


def test_categorical_perturbation_swaps_when_permutation_same():
    column = np.array(["a", "b", "c", "d"])
    result = perturbation.categorical_perturbation(column, rng=StaticRNG())
    assert set(result) == set(column)
    assert not np.array_equal(result, column)


def test_perturb_dataset_uniform_with_seed():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0.1, 0.2])

    perturbed, scaled_x, scaled_y, factor = perturbation.perturb_dataset(
        x,
        y,
        categorical_features=[],
        noise_type="uniform",
        scale_factor=2,
        severity=0.0,
        seed=42,
    )

    expected = np.tile(x, (2, 1))
    assert factor == 2
    assert np.array_equal(perturbed, expected)
    assert np.array_equal(scaled_x, expected)
    assert np.array_equal(scaled_y, np.tile(y, 2))


def test_perturb_dataset_invalid_noise_type():
    from calibrated_explanations.utils.exceptions import ValidationError

    x = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0])

    with pytest.raises(ValidationError, match="Noise type must be either"):
        perturbation.perturb_dataset(x, y, noise_type="unknown")
