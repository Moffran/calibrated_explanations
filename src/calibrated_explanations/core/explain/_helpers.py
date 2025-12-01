"""Shared helper functions for explain executors.

This module extracts reusable utilities that are common across sequential,
feature-parallel, and instance-parallel explain implementations.
"""

# pylint: disable=invalid-name, protected-access

from __future__ import annotations

import contextlib
from functools import partial
from typing import TYPE_CHECKING, Any, List, Mapping, Sequence, Tuple

import numpy as np

from ...utils import safe_isinstance
from ..prediction_helpers import initialize_explanation as _ih
from ._computation import explain_predict_step  # Re-export for backward compatibility
from .feature_task import FeatureTaskResult

if TYPE_CHECKING:
    from ..calibrated_explainer import CalibratedExplainer


def validate_and_prepare_input(explainer: CalibratedExplainer, x: Any) -> np.ndarray:
    """Validate and prepare input data for explanation.

    Ensures that input is in the correct shape and format:
    - Converts pandas DataFrames to arrays
    - Reshapes 1D arrays to 2D
    - Validates feature count matches calibration data

    Parameters
    ----------
    explainer : CalibratedExplainer
        The parent explainer instance.
    x : array-like
        Input data, either 1D (single instance) or 2D (batch).

    Returns
    -------
    np.ndarray
        Validated input as 2D array (n_samples, n_features).

    Raises
    ------
    DataShapeError
        If number of features doesn't match calibration data.
    """
    from ..prediction_helpers import validate_and_prepare_input as _vapi

    return _vapi(explainer, x)


def slice_threshold(threshold: Any, start: int, stop: int, total_len: int) -> Any:
    """Return the portion of *threshold* covering ``[start, stop)``.

    Handles scalar, array-like, and pandas Series thresholds appropriately.
    """
    if threshold is None or np.isscalar(threshold):
        return threshold
    try:
        length = len(threshold)
    except TypeError:
        return threshold
    if length != total_len:
        return threshold
    if safe_isinstance(threshold, "pandas.core.series.Series"):
        return threshold.iloc[start:stop]
    sliced = threshold[start:stop]
    if isinstance(threshold, (np.ndarray, list)):
        return sliced
    return sliced


def slice_bins(bins: Any, start: int, stop: int) -> Any:
    """Return the subset of *bins* covering ``[start, stop)``.

    Handles pandas Series and array-like bins.
    """
    if bins is None:
        return None
    if safe_isinstance(bins, "pandas.core.series.Series"):
        return bins.iloc[start:stop].to_numpy()
    return bins[start:stop]


def merge_ignore_features(
    explainer: CalibratedExplainer,
    features_to_ignore: Any,
) -> np.ndarray:
    """Merge explainer default and request-specific features to ignore.

    Returns
    -------
    np.ndarray
        Combined array of feature indices to ignore.
    """
    if features_to_ignore is None:
        return np.asarray(explainer.features_to_ignore, dtype=int)
    return np.asarray(np.union1d(explainer.features_to_ignore, features_to_ignore), dtype=int)


def initialize_explanation(
    explainer: CalibratedExplainer,
    x: np.ndarray,
    low_high_percentiles: Tuple[float, float],
    threshold: Any,
    bins: Any,
    features_to_ignore: np.ndarray,
):
    """Initialize a CalibratedExplanations container.

    Delegates to the prediction_helpers module which contains the
    authoritative initialization logic.
    """
    return _ih(explainer, x, low_high_percentiles, threshold, bins, features_to_ignore)


def compute_feature_effects(
    explainer: CalibratedExplainer,
    features_to_process: Sequence[int],
    x: np.ndarray,
    threshold: Any,
    low_high_percentiles: Tuple[float, float],
    bins: Any,
    prediction: Mapping[str, Any],
    executor: Any | None,
) -> List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Compute feature perturbation results, optionally in parallel.

    Mirrors the original CalibratedExplainer._compute_feature_effects behaviour
    as a free function to avoid explain-specific logic living on the class.
    """
    worker = partial(
        feature_effect_for_index,
        explainer,
        x=x,
        threshold=threshold,
        low_high_percentiles=low_high_percentiles,
        bins=bins,
        baseline_prediction=prediction,
    )

    if executor is None:
        return [worker(f_idx) for f_idx in features_to_process]

    work_items = max(len(features_to_process), 1) * max(x.shape[0], 1)
    return executor.map(worker, features_to_process, work_items=work_items)


def merge_feature_result(
    result: FeatureTaskResult,
    weights_predict: np.ndarray,
    weights_low: np.ndarray,
    weights_high: np.ndarray,
    predict_matrix: np.ndarray,
    low_matrix: np.ndarray,
    high_matrix: np.ndarray,
    rule_values: List[dict],
    instance_binned: List[dict],
    rule_boundaries: np.ndarray,
) -> None:
    """Merge a per-feature aggregation result into shared buffers.

    Ported from CalibratedExplainer._merge_feature_result.
    """
    (
        feature_index,
        feature_weights_predict,
        feature_weights_low,
        feature_weights_high,
        feature_predict_values,
        feature_low_values,
        feature_high_values,
        rule_values_entries,
        binned_entries,
        lower_update,
        upper_update,
    ) = result

    weights_predict[:, feature_index] = feature_weights_predict
    weights_low[:, feature_index] = feature_weights_low
    weights_high[:, feature_index] = feature_weights_high
    predict_matrix[:, feature_index] = feature_predict_values
    low_matrix[:, feature_index] = feature_low_values
    high_matrix[:, feature_index] = feature_high_values

    if lower_update is not None:
        rule_boundaries[:, feature_index, 0] = lower_update
    if upper_update is not None:
        rule_boundaries[:, feature_index, 1] = upper_update

    for inst, value in enumerate(rule_values_entries):
        if value is not None:
            rule_values[inst][feature_index] = value

    for inst, entry in enumerate(binned_entries):
        if entry is None:
            continue
        (
            predict_row,
            low_row,
            high_row,
            current_bin,
            counts_row,
            fractions_row,
        ) = entry
        instance_binned[inst]["predict"][feature_index] = predict_row
        instance_binned[inst]["low"][feature_index] = low_row
        instance_binned[inst]["high"][feature_index] = high_row
        instance_binned[inst]["current_bin"][feature_index] = current_bin
        instance_binned[inst]["counts"][feature_index] = counts_row
        instance_binned[inst]["fractions"][feature_index] = fractions_row


def compute_weight_delta(baseline, perturbed) -> np.ndarray:
    """Return the contribution weight delta between *baseline* and *perturbed*."""
    baseline_arr = np.asarray(baseline)
    perturbed_arr = np.asarray(perturbed)

    if baseline_arr.shape == ():
        return np.asarray(baseline_arr - perturbed_arr, dtype=float)

    if baseline_arr.shape != perturbed_arr.shape:
        with contextlib.suppress(ValueError):
            baseline_arr = np.broadcast_to(baseline_arr, perturbed_arr.shape)

    try:
        return np.asarray(baseline_arr - perturbed_arr, dtype=float)
    except (TypeError, ValueError):
        # Fallback to element-wise assignment via explainer semantics
        baseline_flat = np.asarray(baseline, dtype=object).reshape(-1)
        perturbed_flat = np.asarray(perturbed, dtype=object).reshape(-1)
        deltas = np.empty_like(perturbed_flat, dtype=float)
        for idx, (pert_value, base_value) in enumerate(zip(perturbed_flat, baseline_flat)):
            # Use scalar helper from module-level implementation
            delta_value = base_value - pert_value
            delta_array = np.asarray(delta_value, dtype=float).reshape(-1)
            deltas[idx] = float(delta_array[0])
        return deltas.reshape(perturbed_arr.shape)


def feature_effect_for_index(
    explainer: CalibratedExplainer,
    f_idx: int,
    *,
    x: np.ndarray,
    threshold: Any,
    low_high_percentiles: Tuple[float, float],
    bins: Any,
    baseline_prediction: Mapping[str, Any],
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute feature-level contributions for a single feature index."""
    local_predict, local_low, local_high, _ = explainer._predict(
        x,
        threshold=threshold,
        low_high_percentiles=low_high_percentiles,
        bins=bins,
        feature=f_idx,
    )

    baseline_predict = baseline_prediction["predict"]
    delta_predict = compute_weight_delta(baseline_predict, local_predict)
    delta_low = compute_weight_delta(baseline_predict, local_low)
    delta_high = compute_weight_delta(baseline_predict, local_high)

    weights_low = np.minimum(delta_low, delta_high)
    weights_high = np.maximum(delta_low, delta_high)

    return (
        f_idx,
        np.asarray(delta_predict),
        np.asarray(weights_low),
        np.asarray(weights_high),
        np.asarray(local_predict),
        np.asarray(local_low),
        np.asarray(local_high),
    )


__all__ = [
    "compute_feature_effects",
    "compute_weight_delta",
    "explain_predict_step",
    "feature_effect_for_index",
    "initialize_explanation",
    "merge_feature_result",
    "merge_ignore_features",
    "slice_bins",
    "slice_threshold",
    "validate_and_prepare_input",
]
