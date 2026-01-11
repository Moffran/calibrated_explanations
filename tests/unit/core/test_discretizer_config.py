from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.core.discretizer_config import (
    instantiate_discretizer,
    validate_discretizer_choice,
)
from calibrated_explanations.utils.exceptions import ValidationError


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("regression", "binaryRegressor"),
        ("classification", "binaryEntropy"),
    ],
)
def test_validate_discretizer_choice__should_default_when_none(mode, expected):
    assert validate_discretizer_choice(None, mode=mode) == expected


def test_validate_discretizer_choice__should_raise_when_invalid_for_regression():
    with pytest.raises(ValidationError, match="discretizer must be"):
        validate_discretizer_choice("entropy", mode="regression")


def test_validate_discretizer_choice__should_raise_when_invalid_for_classification():
    with pytest.raises(ValidationError, match="discretizer must be"):
        validate_discretizer_choice("regressor", mode="classification")


def test_instantiate_discretizer__should_raise_when_condition_source_invalid():
    x_cal = np.zeros((2, 2))
    features_to_ignore = np.asarray([], dtype=int)

    with pytest.raises(ValidationError, match="condition_source must be"):
        instantiate_discretizer(
            "binaryEntropy",
            x_cal=x_cal,
            features_to_ignore=features_to_ignore,
            feature_names=None,
            y_cal=None,
            seed=0,
            current_discretizer=None,
            condition_source="bad",
        )


def test_instantiate_discretizer__should_raise_when_unknown_discretizer_name():
    x_cal = np.zeros((2, 2))
    features_to_ignore = np.asarray([], dtype=int)

    with pytest.raises(ValidationError, match="Unknown discretizer"):
        instantiate_discretizer(
            "does-not-exist",
            x_cal=x_cal,
            features_to_ignore=features_to_ignore,
            feature_names=None,
            y_cal=None,
            seed=0,
            current_discretizer=None,
        )
