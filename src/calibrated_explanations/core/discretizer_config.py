"""Discretization configuration and setup utilities."""

from typing import Any, Dict, Optional

import numpy as np

from ..utils.discretizers import (
    BinaryEntropyDiscretizer,
    BinaryRegressorDiscretizer,
    EntropyDiscretizer,
    RegressorDiscretizer,
)
from ..utils.helper import immutable_array
from .exceptions import ValidationError
from .explain._computation import discretize as _discretize_func


def validate_discretizer_choice(discretizer: Any, mode: str) -> str:
    """Validate that the discretizer choice is appropriate for the mode.

    Parameters
    ----------
    discretizer : str or None
        The discretizer choice.
    mode : str
        Either 'classification' or 'regression'.

    Returns
    -------
    str
        The validated (and potentially defaulted) discretizer name.

    Raises
    ------
    ValidationError
        If the discretizer is invalid for the given mode.
    """
    if discretizer is None:
        return "binaryRegressor" if "regression" in mode else "binaryEntropy"

    if "regression" in mode:
        if discretizer not in {"regressor", "binaryRegressor", None}:
            raise ValidationError(
                "The discretizer must be 'binaryRegressor' (default for factuals) or 'regressor' (default for alternatives) for regression."
            )
    else:
        if discretizer not in {"entropy", "binaryEntropy", None}:
            raise ValidationError(
                "The discretizer must be 'binaryEntropy' (default for factuals) or 'entropy' (default for alternatives) for classification."
            )

    return discretizer


def instantiate_discretizer(
    discretizer_name: str,
    x_cal: np.ndarray,
    features_to_ignore: np.ndarray,
    feature_names: Optional[list],
    y_cal: Optional[np.ndarray],
    seed: int,
    current_discretizer: Optional[Any] = None,
) -> Any:
    """Instantiate or return cached discretizer if already correct type.

    Parameters
    ----------
    discretizer_name : str
        Name of the discretizer ('binaryEntropy', 'entropy', 'binaryRegressor', 'regressor').
    x_cal : np.ndarray
        Calibration input data.
    features_to_ignore : np.ndarray
        Indices of features to skip during discretization.
    feature_names : list or None
        Human-readable feature names.
    y_cal : np.ndarray or None
        Calibration target data.
    seed : int
        Random seed for reproducibility.
    current_discretizer : Any, optional
        The current discretizer instance (used to check if reinitialize is needed).

    Returns
    -------
    Any
        The discretizer instance.
    """
    if discretizer_name == "binaryEntropy":
        if isinstance(current_discretizer, BinaryEntropyDiscretizer):
            return current_discretizer
        return BinaryEntropyDiscretizer(
            x_cal, features_to_ignore, feature_names, labels=y_cal, random_state=seed
        )
    elif discretizer_name == "binaryRegressor":
        if isinstance(current_discretizer, BinaryRegressorDiscretizer):
            return current_discretizer
        return BinaryRegressorDiscretizer(
            x_cal, features_to_ignore, feature_names, labels=y_cal, random_state=seed
        )
    elif discretizer_name == "entropy":
        if isinstance(current_discretizer, EntropyDiscretizer):
            return current_discretizer
        return EntropyDiscretizer(
            x_cal, features_to_ignore, feature_names, labels=y_cal, random_state=seed
        )
    elif discretizer_name == "regressor":
        if isinstance(current_discretizer, RegressorDiscretizer):
            return current_discretizer
        return RegressorDiscretizer(
            x_cal, features_to_ignore, feature_names, labels=y_cal, random_state=seed
        )
    else:
        raise ValidationError(f"Unknown discretizer: {discretizer_name}")


def setup_discretized_data(
    explainer_instance: Any,
    discretizer: Any,
    x_cal: np.ndarray,
    num_features: int,
) -> Dict[int, list]:
    """Discretize calibration data and build feature value/frequency caches.

    Parameters
    ----------
    explainer_instance : CalibratedExplainer
        The explainer instance (needed for discretize function call).
    discretizer : Any
        The instantiated discretizer.
    x_cal : np.ndarray
        Calibration input data.
    num_features : int
        Number of features.

    Returns
    -------
    dict
        Dictionary mapping feature index to (values, frequencies) tuple.
    """
    discretized_X_cal = _discretize_func(explainer_instance, immutable_array(x_cal))

    feature_data = {}
    for feature in range(num_features):
        assert discretized_X_cal is not None
        column = discretized_X_cal[:, feature]
        feature_count: Dict[Any, int] = {}
        for item in column:
            feature_count[item] = feature_count.get(item, 0) + 1
        values, frequencies = map(list, zip(*(sorted(feature_count.items()))))

        feature_data[feature] = {
            "values": values,
            "frequencies": np.array(frequencies) / float(sum(frequencies)),
        }

    return feature_data, discretized_X_cal


__all__ = [
    "validate_discretizer_choice",
    "instantiate_discretizer",
    "setup_discretized_data",
]
