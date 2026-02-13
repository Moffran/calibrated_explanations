from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.utils import (
    BinaryEntropyDiscretizer,
    BinaryRegressorDiscretizer,
    EntropyDiscretizer,
    RegressorDiscretizer,
)




def test_binary_entropy_discretizer_requires_labels():
    from calibrated_explanations.utils.exceptions import ValidationError

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




def test_binary_regressor_discretizer_requires_labels():
    from calibrated_explanations.utils.exceptions import ValidationError

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


def test_entropy_discretizer_is_picklable():
    import pickle

    rng = np.random.default_rng(42)
    data = rng.random((100, 5))
    categorical_features = [0]
    feature_names = [f"f{i}" for i in range(5)]
    labels = rng.integers(0, 2, 100)

    discretizer = EntropyDiscretizer(data, categorical_features, feature_names, labels=labels)

    pickled = pickle.dumps(discretizer)
    unpickled = pickle.loads(pickled)

    assert unpickled is not None
    assert isinstance(unpickled, EntropyDiscretizer)


def test_regressor_discretizer_is_picklable():
    import pickle

    rng = np.random.default_rng(42)
    data = rng.random((100, 5))
    categorical_features = [0]
    feature_names = [f"f{i}" for i in range(5)]
    labels = rng.random(100)

    discretizer = RegressorDiscretizer(data, categorical_features, feature_names, labels=labels)

    pickled = pickle.dumps(discretizer)
    unpickled = pickle.loads(pickled)

    assert unpickled is not None
    assert isinstance(unpickled, RegressorDiscretizer)
