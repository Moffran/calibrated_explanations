from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.core.exceptions import ValidationError
from calibrated_explanations.utils.perturbation import (
    categorical_perturbation,
    perturb_dataset,
)


def test_categorical_perturbation_fallback_warns_and_swaps(monkeypatch: pytest.MonkeyPatch) -> None:
    class DegenerateRng:
        def permutation(self, arr):
            return np.asarray(arr)

    monkeypatch.delenv("CE_EXPLANATION_PLUGIN_FACTUAL_FALLBACKS", raising=False)
    with pytest.warns(UserWarning, match="deterministic swap"):
        out = categorical_perturbation(
            np.asarray([1, 2, 3]), num_permutations=2, rng=DegenerateRng()
        )
    assert out.tolist() != [1, 2, 3]


def test_perturb_dataset_rejects_invalid_noise_type() -> None:
    x_cal = np.asarray([[1.0, 0.0], [2.0, 1.0]])
    y_cal = np.asarray([0, 1])
    with pytest.raises(ValidationError):
        perturb_dataset(x_cal, y_cal, noise_type="invalid", scale_factor=2)


def test_perturb_dataset_uniform_and_gaussian_paths() -> None:
    x_cal = np.asarray([[1.0, 0.0], [2.0, 1.0]])
    y_cal = np.asarray([0, 1])
    categorical = [1]

    out_uniform = perturb_dataset(
        x_cal,
        y_cal,
        categorical_features=categorical,
        noise_type="uniform",
        scale_factor=3,
        severity=0.2,
        seed=7,
    )
    assert out_uniform[0].shape == (6, 2)
    assert out_uniform[1].shape == (6, 2)
    assert out_uniform[2].shape == (6,)
    assert out_uniform[3] == 3

    out_gauss = perturb_dataset(
        x_cal,
        y_cal,
        categorical_features=categorical,
        noise_type="gaussian",
        scale_factor=2,
        severity=0.2,
        rng=np.random.default_rng(9),
    )
    assert out_gauss[0].shape == (4, 2)
    assert out_gauss[3] == 2
