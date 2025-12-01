from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.utils import (
    BinaryEntropyDiscretizer,
    BinaryRegressorDiscretizer,
    EntropyDiscretizer,
    RegressorDiscretizer,
)


def test_entropy_discretizer_discretizes_and_preserves_categoricals():
    data = np.array(
        [
            [0.1, 1.0, 0.0],
            [0.2, 0.8, 1.0],
            [0.3, 0.6, 0.0],
            [0.4, 0.4, 1.0],
        ]
    )
    labels = np.array([0, 1, 0, 1])
    feature_names = ("feat_a", "feat_b", "category")

    discretizer = EntropyDiscretizer(
        data,
        categorical_features=[2],
        feature_names=feature_names,
        labels=labels,
        random_state=0,
    )

    transformed = discretizer.discretize(data.copy())
    assert transformed.shape == data.shape
    # Continuous columns should be integer bucket identifiers (even if stored as floats).
    assert np.allclose(transformed[:, 0], transformed[:, 0].astype(int))
    assert np.allclose(transformed[:, 1], transformed[:, 1].astype(int))
    # Categorical features are untouched by the discretizer.
    assert np.allclose(transformed[:, 2], data[:, 2])
    # Names and min/max metadata are populated for downstream explanations.
    assert 0 in discretizer.names
    assert discretizer.names[0][0].startswith("feat_a")
    assert discretizer.mins[0][0] <= discretizer.maxs[0][-1]


def test_binary_entropy_discretizer_requires_labels():
    from calibrated_explanations.core.exceptions import ValidationError
    data = np.array([[0.1], [0.2]])
    feature_names = ("feat",)
    with pytest.raises(ValidationError):
        BinaryEntropyDiscretizer(
            data,
            categorical_features=[],
            feature_names=feature_names,
            labels=None,
            random_state=0,
        )


def test_binary_entropy_discretizer_falls_back_to_median_split():
    data = np.ones((4, 1)) * 0.5  # Constant feature triggers median fallback.
    labels = np.array([0, 1, 0, 1])
    feature_names = ("feat",)

    discretizer = BinaryEntropyDiscretizer(
        data,
        categorical_features=[],
        feature_names=feature_names,
        labels=labels,
        random_state=0,
    )

    transformed = discretizer.discretize(data.copy())
    assert transformed.shape == data.shape
    assert np.all(transformed == 0)  # Median fallback collapses to a single bin.
    assert discretizer.mins[0][0] == pytest.approx(0.5)
    assert discretizer.maxs[0][-1] == pytest.approx(0.5)


def test_regressor_discretizer_discretize_vector_input():
    data = np.array(
        [
            [0.1, 1.0],
            [0.2, 0.8],
            [0.3, 0.6],
            [0.4, 0.4],
        ]
    )
    labels = np.array([0.5, 0.6, 0.7, 0.8])
    feature_names = ("feat_a", "feat_b")

    discretizer = RegressorDiscretizer(
        data,
        categorical_features=[],
        feature_names=feature_names,
        labels=labels,
        random_state=0,
    )

    row = data[0].copy()
    discretized_row = discretizer.discretize(row)
    assert discretized_row.shape == row.shape
    assert discretized_row.dtype == row.dtype
    # 2-D arrays should also be supported without raising.
    discretized_matrix = discretizer.discretize(data.copy())
    assert discretized_matrix.shape == data.shape


def test_binary_regressor_discretizer_requires_labels():
    from calibrated_explanations.core.exceptions import ValidationError
    data = np.array([[0.1], [0.2]])
    feature_names = ("feat",)

    with pytest.raises(ValidationError):
        BinaryRegressorDiscretizer(
            data,
            categorical_features=[],
            feature_names=feature_names,
            labels=None,
            random_state=0,
        )


def test_binary_regressor_discretizer_handles_constant_feature():
    data = np.full((4, 1), 0.25)
    labels = np.array([0.1, 0.2, 0.3, 0.4])
    feature_names = ("feat",)

    discretizer = BinaryRegressorDiscretizer(
        data,
        categorical_features=[],
        feature_names=feature_names,
        labels=labels,
        random_state=0,
    )

    transformed = discretizer.discretize(data.copy())
    assert transformed.shape == data.shape
    assert np.all(transformed == 0)
    assert discretizer.mins[0][0] == pytest.approx(0.25)
    assert discretizer.maxs[0][-1] == pytest.approx(0.25)
