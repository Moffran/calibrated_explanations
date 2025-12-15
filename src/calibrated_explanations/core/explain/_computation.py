"""Computational functions for explain feature tasks.

This module contains the core computation logic for per-feature aggregation
and weight calculation used by all explain executors.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

from ...utils import concatenate_thresholds
from ...utils.helper import assign_threshold as normalize_threshold
from ...utils.int_utils import as_int_array
from ..exceptions import CalibratedError
from .feature_task import (
    FeatureTaskResult,
    assign_weight_scalar,
)
from .feature_task import (
    _feature_task as feature_task,
)

if TYPE_CHECKING:
    from ...explanations import CalibratedExplanations
    from ..calibrated_explainer import CalibratedExplainer

# Type alias for explain_predict_step results
ExplainPredictStepResult = Tuple[
    np.ndarray,  # predict
    np.ndarray,  # low
    np.ndarray,  # high
    Dict[str, Any],  # prediction
    np.ndarray,  # perturbed_feature
    Any,  # rule_boundaries
    Dict[int, Any],  # lesser_values
    Dict[int, Any],  # greater_values
    Dict[int, Any],  # covered_values
    np.ndarray,  # x_cal
    Any,  # perturbed_threshold
    Optional[np.ndarray],  # perturbed_bins
    np.ndarray,  # perturbed_x
    np.ndarray,  # perturbed_class
]


def discretize(explainer: CalibratedExplainer, data: np.ndarray) -> np.ndarray:
    """Apply the discretizer to the data sample.

    For new data samples and missing values, the nearest bin is used.
    This function extracts the discretization logic from CalibratedExplainer._discretize()
    to make it available to the explain subpackage without circular dependencies.

    Parameters
    ----------
    explainer : CalibratedExplainer
        The explainer instance containing discretizer configuration.
    data : np.ndarray
        The data sample to discretize.

    Returns
    -------
    np.ndarray
        The discretized data sample.
    """
    # pylint: disable=invalid-name
    data = np.array(data, copy=True)  # Ensure data is a numpy array
    if explainer.discretizer is None:
        return data
    for f in explainer.discretizer.to_discretize:
        bins = np.concatenate(([-np.inf], explainer.discretizer.mins[f][1:], [np.inf]))
        bin_indices = np.digitize(data[:, f], bins, right=True) - 1
        means = np.asarray(explainer.discretizer.means[f])
        bin_indices = np.clip(bin_indices, 0, len(means) - 1)
        data[:, f] = means[bin_indices]
    return data


def rule_boundaries(
    explainer: CalibratedExplainer,
    instances: np.ndarray,
    perturbed_instances: Optional[np.ndarray] = None,
) -> Any:
    """Extract the rule boundaries for a set of instances.

    This function extracts the rule boundary extraction logic from
    CalibratedExplainer.rule_boundaries() to make it available to the explain
    subpackage without circular dependencies.

    Parameters
    ----------
    explainer : CalibratedExplainer
        The explainer instance containing discretizer configuration.
    instances : array-like
        The instances to extract boundaries for.
    perturbed_instances : array-like, optional
        Discretized versions of instances. Defaults to None.

    Returns
    -------
    array-like
        Min and max values for each feature for each instance.
        Shape: (num_instances, num_features, 2) for multiple instances
        or (num_features, 2) for a single instance (as list).
    """
    # pylint: disable=invalid-name
    instances = np.array(instances)  # Ensure instances is a numpy array

    # If no discretizer, return trivial boundaries (min=max=original value)
    if explainer.discretizer is None:
        if len(instances.shape) == 1:
            # Single instance: return array of shape (num_features, 2)
            return np.array([[val, val] for val in instances])
        else:
            # Multiple instances: return array of shape (num_instances, num_features, 2)
            return np.array([[[val, val] for val in instance] for instance in instances])

    # backwards compatibility
    if len(instances.shape) == 1:
        min_max = []
        if perturbed_instances is None:
            perturbed_instances = discretize(explainer, instances.reshape(1, -1))
        for f in range(explainer.num_features):
            if f not in explainer.discretizer.to_discretize:
                min_max.append([instances[f], instances[f]])
            else:
                bins = np.concatenate(([-np.inf], explainer.discretizer.mins[f][1:], [np.inf]))
                min_max.append(
                    [
                        explainer.discretizer.mins[f][
                            np.digitize(perturbed_instances[0, f], bins, right=True) - 1
                        ],
                        explainer.discretizer.maxs[f][
                            np.digitize(perturbed_instances[0, f], bins, right=True) - 1
                        ],
                    ]
                )
        return np.array(min_max)

    if perturbed_instances is None:
        perturbed_instances = discretize(explainer, instances)
    else:
        perturbed_instances = np.array(
            perturbed_instances
        )  # Ensure perturbed_instances is a numpy array

    all_min_max = []
    for instance, perturbed_instance in zip(instances, perturbed_instances):
        min_max = []
        for f in range(explainer.num_features):
            if f not in explainer.discretizer.to_discretize:
                min_max.append([instance[f], instance[f]])
            else:
                bins = np.concatenate(([-np.inf], explainer.discretizer.mins[f][1:], [np.inf]))
                min_max.append(
                    [
                        explainer.discretizer.mins[f][
                            np.digitize(perturbed_instance[f], bins, right=True) - 1
                        ],
                        explainer.discretizer.maxs[f][
                            np.digitize(perturbed_instance[f], bins, right=True) - 1
                        ],
                    ]
                )
        all_min_max.append(min_max)
    return np.array(all_min_max)


def get_greater_values(
    explainer: CalibratedExplainer,
    feature_index: int,
    threshold: float,
) -> np.ndarray:
    """Get sampled values above threshold for a numeric feature.

    Samples percentiles from calibration data for values exceeding the threshold.
    Used during perturbation planning to establish rule boundaries.

    Parameters
    ----------
    explainer : CalibratedExplainer
        The parent explainer instance with calibration data.
    feature_index : int
        Index of the feature.
    threshold : float
        The threshold value.

    Returns
    -------
    np.ndarray
        Array of sampled percentile values above the threshold, or empty array
        if no calibration samples exceed the threshold.
    """
    if not np.any(explainer.x_cal[:, feature_index] > threshold):
        return np.array([])
    candidates = np.percentile(
        explainer.x_cal[explainer.x_cal[:, feature_index] > threshold, feature_index],
        explainer.sample_percentiles,
    )
    return candidates


def get_lesser_values(
    explainer: CalibratedExplainer,
    feature_index: int,
    threshold: float,
) -> np.ndarray:
    """Get sampled values below threshold for a numeric feature.

    Samples percentiles from calibration data for values below the threshold.
    Used during perturbation planning to establish rule boundaries.

    Parameters
    ----------
    explainer : CalibratedExplainer
        The parent explainer instance with calibration data.
    feature_index : int
        Index of the feature.
    threshold : float
        The threshold value.

    Returns
    -------
    np.ndarray
        Array of sampled percentile values below the threshold, or empty array
        if no calibration samples are below the threshold.
    """
    if not np.any(explainer.x_cal[:, feature_index] < threshold):
        return np.array([])
    candidates = np.percentile(
        explainer.x_cal[explainer.x_cal[:, feature_index] < threshold, feature_index],
        explainer.sample_percentiles,
    )
    return candidates


def get_covered_values(
    explainer: CalibratedExplainer,
    feature_index: int,
    lower_threshold: float,
    upper_threshold: float,
) -> np.ndarray:
    """Get sampled values within an interval for a numeric feature.

    Samples percentiles from calibration data for values within the given interval.
    Used during perturbation planning to establish rule boundaries.

    Parameters
    ----------
    explainer : CalibratedExplainer
        The parent explainer instance with calibration data.
    feature_index : int
        Index of the feature.
    lower_threshold : float
        The lower bound of the interval (inclusive).
    upper_threshold : float
        The upper bound of the interval (inclusive).

    Returns
    -------
    np.ndarray
        Array of sampled percentile values within the interval, or empty array
        if no calibration samples fall in the interval.
    """
    covered = np.where(
        (explainer.x_cal[:, feature_index] >= lower_threshold)
        & (explainer.x_cal[:, feature_index] <= upper_threshold)
    )[0]
    if len(covered) == 0:
        return np.array([])
    candidates = np.percentile(
        explainer.x_cal[covered, feature_index],
        explainer.sample_percentiles,
    )
    return candidates


def initialize_explanation(
    explainer: CalibratedExplainer,
    x: np.ndarray,  # pylint: disable=invalid-name
    low_high_percentiles: Tuple[int, int],
    threshold: Any,
    bins: Any,
    features_to_ignore: Any,
) -> "CalibratedExplanations":
    """Initialize a CalibratedExplanations object for explanation.

    Delegates to the prediction_helpers module which contains the
    authoritative initialization logic.

    Parameters
    ----------
    explainer : CalibratedExplainer
        The parent explainer instance.
    x : np.ndarray
        Input data to explain.
    low_high_percentiles : tuple of int
        Low and high percentiles for interval calibration.
    threshold : Any
        Optional threshold for regression explanations.
    bins : Any
        Optional binned representations.
    features_to_ignore : Any
        Feature indices to exclude from explanation.

    Returns
    -------
    CalibratedExplanations
        Initialized explanation object.
    """
    from ..prediction_helpers import (
        initialize_explanation as _init_expl,  # pylint: disable=import-outside-toplevel
    )

    return _init_expl(explainer, x, low_high_percentiles, threshold, bins, features_to_ignore)


def explain_predict_step(
    explainer: CalibratedExplainer,
    x: np.ndarray,  # pylint: disable=invalid-name
    threshold: Any,
    low_high_percentiles: Tuple[int, int],
    bins: Any,
    features_to_ignore: Any,
    *,
    features_to_ignore_per_instance: Any | None = None,
) -> ExplainPredictStepResult:
    """Execute the baseline prediction and perturbation planning step.

    This is the first phase of explanation generation that:
    1. Predicts on original test instances
    2. Determines rule boundaries for numerical features
    3. Plans perturbation sets (lesser/greater/covered values)

    This is the authoritative implementation consolidated from prediction_helpers.

    Parameters
    ----------
    explainer : CalibratedExplainer
        The parent explainer instance.
    x : np.ndarray
        Input data (already validated, 2D array).
    threshold : Any
        Optional threshold for regression.
    low_high_percentiles : tuple of int
        Low and high percentiles for intervals.
    bins : Any
        Optional binned representations.
    features_to_ignore : Any
        Feature indices to exclude.

    Returns
    -------
    ExplainPredictStepResult
        Tuple containing predictions, perturbation metadata, and rule boundaries.
    """
    import logging  # pylint: disable=import-outside-toplevel

    from ..prediction_helpers import assert_threshold  # pylint: disable=import-outside-toplevel

    if features_to_ignore is None:
        features_to_ignore = ()

    if bins is not None:
        bins = np.asarray(bins)
    n_instances = x.shape[0]
    num_features = explainer.num_features
    ignore_mask = np.zeros((n_instances, num_features), dtype=bool)
    ignore_indices = as_int_array(features_to_ignore)
    if ignore_indices.size:
        ignore_mask[:, ignore_indices] = True
    if isinstance(features_to_ignore_per_instance, Iterable) and not isinstance(
        features_to_ignore_per_instance, (str, bytes)
    ):
        for idx, inst_mask in enumerate(features_to_ignore_per_instance):
            if idx >= n_instances:
                break
            inst_indices = as_int_array(inst_mask)
            if inst_indices.size:
                ignore_mask[idx, inst_indices] = True

    x_cal = explainer.x_cal
    base_predict, base_low, base_high, predicted_class = explainer._predict(  # pylint: disable=protected-access
        x, threshold=threshold, low_high_percentiles=low_high_percentiles, bins=bins
    )

    prediction = {
        "predict": base_predict,
        "low": base_low,
        "high": base_high,
        "classes": (predicted_class if explainer.is_multiclass() else np.ones(base_predict.shape)),
    }
    if explainer.mode == "classification":  # store full calibrated probability matrix
        try:  # pragma: no cover - defensive
            if explainer.is_multiclass():
                if explainer.is_fast():
                    full_probs = explainer.interval_learner[  # pylint: disable=protected-access
                        explainer.num_features
                    ].predict_proba(x, bins=bins)
                else:
                    full_probs = explainer.interval_learner.predict_proba(  # pylint: disable=protected-access
                        x, bins=bins
                    )
            else:  # binary
                if explainer.is_fast():
                    full_probs = explainer.interval_learner[  # pylint: disable=protected-access
                        explainer.num_features
                    ].predict_proba(x, bins=bins)
                else:
                    full_probs = explainer.interval_learner.predict_proba(  # pylint: disable=protected-access
                        x, bins=bins
                    )
            prediction["__full_probabilities__"] = full_probs
        except (AttributeError, CalibratedError) as exc:
            logging.getLogger("calibrated_explanations").debug(
                "Failed to compute full calibrated probabilities: %s", exc
            )

    x.flags.writeable = False
    assert_threshold(threshold, x)
    perturbed_threshold = normalize_threshold(threshold)
    # Optimization: Use lists for accumulation to avoid O(N^2) concatenation
    perturbed_bins_list = []
    perturbed_x_list = []
    perturbed_feature_list = []
    perturbed_class_list = []

    x_perturbed = discretize(explainer, x)
    rule_boundaries_result = rule_boundaries(explainer, x, x_perturbed)

    lesser_values: dict[int, Any] = {}
    greater_values: dict[int, Any] = {}
    covered_values: dict[int, Any] = {}

    categorical_features = set(getattr(explainer, "categorical_features", ()))

    for f in range(explainer.num_features):
        active_indices = np.where(~ignore_mask[:, f])[0]
        if active_indices.size == 0:
            continue
        if f in categorical_features:
            feature_values = explainer.feature_values[f]
            x_copy = np.array(x, copy=True)
            for value in feature_values:
                x_local = x_copy[active_indices, :]
                x_local[:, f] = value
                perturbed_x_list.append(x_local)
                perturbed_feature_list.append([(f, int(i), value, None) for i in active_indices])

                if bins is not None:
                    selected_bins = (
                        bins[active_indices]
                        if len(active_indices) > 1
                        else [bins[active_indices[0]]]
                    )
                    perturbed_bins_list.append(selected_bins)

                p_class = prediction.get("predict", np.zeros_like(active_indices))[active_indices]
                if not hasattr(p_class, "__len__"):
                    p_class = np.full(len(active_indices), p_class)
                perturbed_class_list.append(p_class)

                perturbed_threshold = concatenate_thresholds(
                    perturbed_threshold, threshold, active_indices
                )
        else:
            x_copy = np.array(x, copy=True)
            feature_values = np.unique(np.array(x_cal[:, f]))
            lower_boundary = rule_boundaries_result[:, f, 0]
            upper_boundary = rule_boundaries_result[:, f, 1]
            for i in range(len(x)):
                lower_boundary[i] = (
                    lower_boundary[i] if np.any(feature_values < lower_boundary[i]) else -np.inf
                )
                upper_boundary[i] = (
                    upper_boundary[i] if np.any(feature_values > upper_boundary[i]) else np.inf
                )

            lesser_values[f] = {}
            greater_values[f] = {}
            covered_values[f] = {}

            active_lower = lower_boundary[active_indices]
            active_upper = upper_boundary[active_indices]

            for j, val in enumerate(np.unique(active_lower)):
                lesser_values[f][j] = (
                    np.unique(get_lesser_values(explainer, f, val)),
                    val,
                )
                indices = np.where(lower_boundary == val)[0]
                indices = np.intersect1d(indices, active_indices, assume_unique=False)
                if len(indices) == 0:
                    continue
                for value in lesser_values[f][j][0]:
                    x_local = x_copy[indices, :]
                    x_local[:, f] = value
                    perturbed_x_list.append(x_local)
                    perturbed_feature_list.append([(f, int(i), j, True) for i in indices])

                    if bins is not None:
                        selected_bins = bins[indices] if len(indices) > 1 else [bins[indices[0]]]
                        perturbed_bins_list.append(selected_bins)

                    perturbed_class_list.append(prediction["classes"][indices])

                    perturbed_threshold = concatenate_thresholds(
                        perturbed_threshold, threshold, indices
                    )
            for j, val in enumerate(np.unique(active_upper)):
                greater_values[f][j] = (
                    np.unique(get_greater_values(explainer, f, val)),
                    val,
                )
                indices = np.where(upper_boundary == val)[0]
                indices = np.intersect1d(indices, active_indices, assume_unique=False)
                if len(indices) == 0:
                    continue
                for value in greater_values[f][j][0]:
                    x_local = x_copy[indices, :]
                    x_local[:, f] = value
                    perturbed_x_list.append(x_local)
                    perturbed_feature_list.append([(f, int(i), j, False) for i in indices])

                    if bins is not None:
                        selected_bins = bins[indices] if len(indices) > 1 else [bins[indices[0]]]
                        perturbed_bins_list.append(selected_bins)

                    perturbed_class_list.append(prediction["classes"][indices])

                    perturbed_threshold = concatenate_thresholds(
                        perturbed_threshold, threshold, indices
                    )
            for i in active_indices:
                covered_values[f][int(i)] = (
                    get_covered_values(explainer, f, lower_boundary[i], upper_boundary[i]),
                    (lower_boundary[i], upper_boundary[i]),
                )
                for value in covered_values[f][int(i)][0]:
                    x_local = x_copy[int(i), :]
                    x_local[f] = value
                    perturbed_x_list.append(x_local.reshape(1, -1))
                    perturbed_feature_list.append([(f, int(i), int(i), None)])

                    if bins is not None:
                        perturbed_bins_list.append([bins[int(i)]])

                    perturbed_class_list.append([prediction["classes"][int(i)]])

                    if threshold is not None and isinstance(threshold, (list, np.ndarray)):
                        if isinstance(threshold[0], tuple) and len(perturbed_threshold) == 0:
                            perturbed_threshold = [threshold[int(i)]]
                        else:
                            perturbed_threshold = np.concatenate(
                                (perturbed_threshold, [threshold[int(i)]])
                            )

    # Finalize arrays from lists
    if perturbed_x_list:
        perturbed_x = np.concatenate(perturbed_x_list)
        perturbed_feature = np.concatenate(perturbed_feature_list)
        perturbed_class = np.concatenate(perturbed_class_list)
        if bins is not None:
            perturbed_bins = np.concatenate(perturbed_bins_list)
        else:
            perturbed_bins = None
    else:
        perturbed_x = np.empty((0, explainer.num_features))
        perturbed_feature = np.empty((0, 4))
        perturbed_class = np.empty((0,), dtype=int)
        perturbed_bins = np.empty((0,)) if bins is not None else None

    if (
        threshold is not None
        and isinstance(threshold, (list, np.ndarray))
        and len(threshold) > 0
        and isinstance(threshold[0], tuple)
    ):
        perturbed_threshold = [tuple(pair) for pair in perturbed_threshold]
    predict, low, high, _ = explainer._predict(  # pylint: disable=protected-access
        perturbed_x,
        threshold=perturbed_threshold,
        low_high_percentiles=low_high_percentiles,
        classes=perturbed_class,
        bins=perturbed_bins,
    )
    predict = np.array(predict)
    low = np.array(low)
    high = np.array(high)
    return (
        predict,
        low,
        high,
        prediction,
        perturbed_feature,
        rule_boundaries_result,
        lesser_values,
        greater_values,
        covered_values,
        x_cal,
        perturbed_threshold,
        perturbed_bins,
        perturbed_x,
        perturbed_class,
    )


__all__ = [
    "ExplainPredictStepResult",
    "FeatureTaskResult",
    "assign_weight_scalar",
    "discretize",
    "explain_predict_step",
    "feature_task",
    "get_covered_values",
    "get_greater_values",
    "get_lesser_values",
    "initialize_explanation",
    "rule_boundaries",
]
