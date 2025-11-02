"""Computational functions for explain feature tasks.

This module contains the core computation logic for per-feature aggregation
and weight calculation used by all explain plugins.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ...utils.helper import safe_mean

# Type alias for feature task results
FeatureTaskResult = Tuple[
    int,  # feature_index
    np.ndarray,  # weights_predict
    np.ndarray,  # weights_low
    np.ndarray,  # weights_high
    np.ndarray,  # predict_matrix
    np.ndarray,  # low_matrix
    np.ndarray,  # high_matrix
    List[Any],  # rule_values_result
    List[Any],  # binned_result
    Optional[np.ndarray],  # lower_update
    Optional[np.ndarray],  # upper_update
]


def assign_weight_scalar(instance_predict: Any, prediction: Any) -> float:
    """Return the scalar delta between *prediction* and *instance_predict*."""
    if np.isscalar(prediction):
        try:
            return float(prediction - instance_predict)
        except TypeError:
            return float(
                np.asarray(prediction, dtype=float) - np.asarray(instance_predict, dtype=float)
            )

    base_arr = np.asarray(prediction)
    inst_arr = np.asarray(instance_predict)
    try:
        diff = base_arr - inst_arr
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        diff = np.asarray(base_arr, dtype=float) - np.asarray(inst_arr, dtype=float)
    flat = np.asarray(diff, dtype=float).reshape(-1)
    if flat.size == 0:
        return 0.0
    return float(flat[0])


def feature_task(args: Tuple[Any, ...]) -> FeatureTaskResult:
    """Execute the per-feature aggregation logic for explain operations.

    This function processes a single feature across all instances, computing:
    - Feature importance weights (predict, low, high)
    - Perturbed prediction matrices
    - Rule boundaries and binned results

    Parameters
    ----------
    args : tuple
        Packed arguments containing:
        - feature_index: int
        - x_column: array of feature values for all instances
        - predict, low, high: baseline predictions
        - baseline_predict: reference predictions for weight calculation
        - features_to_ignore, categorical_features: feature metadata
        - feature_values: mapping of categorical values
        - feature_indices: indices into perturbed_feature array
        - perturbed_feature: array of perturbation metadata
        - lower_boundary, upper_boundary: rule boundaries
        - lesser_feature, greater_feature, covered_feature: value mappings
        - value_counts_cache: cached categorical value counts
        - numeric_sorted_values: sorted calibration values
        - x_cal_column: calibration data for this feature

    Returns
    -------
    FeatureTaskResult
        Tuple containing feature weights, predictions, rule values, and binned results
    """
    (
        feature_index,
        x_column,
        predict,
        low,
        high,
        baseline_predict,
        features_to_ignore,
        categorical_features,
        feature_values,
        feature_indices,
        perturbed_feature,
        lower_boundary,
        upper_boundary,
        lesser_feature,
        greater_feature,
        covered_feature,
        value_counts_cache,
        numeric_sorted_values,
        x_cal_column,
    ) = args

    n_instances = int(len(x_column))
    weights_predict = np.zeros(n_instances, dtype=float)
    weights_low = np.zeros(n_instances, dtype=float)
    weights_high = np.zeros(n_instances, dtype=float)
    predict_matrix = np.zeros(n_instances, dtype=float)
    low_matrix = np.zeros(n_instances, dtype=float)
    high_matrix = np.zeros(n_instances, dtype=float)
    rule_values_result: List[Any] = [None] * n_instances
    binned_result: List[Any] = [None] * n_instances
    lower_update: Optional[np.ndarray] = None
    upper_update: Optional[np.ndarray] = None

    features_to_ignore_set: Set[int] = set(features_to_ignore)
    categorical_features_set: Set[int] = set(categorical_features)

    feature_values_list = feature_values[feature_index]
    feature_values_list = (
        feature_values_list
        if isinstance(feature_values_list, (list, tuple, np.ndarray))
        else list(feature_values_list)
    )

    # Handle ignored features
    if feature_index in features_to_ignore_set:
        for i in range(n_instances):
            rule_values_result[i] = (feature_values_list, x_column[i], x_column[i])
            binned_result[i] = (
                predict[i],
                low[i],
                high[i],
                -1,
                np.array([], dtype=float),
                np.array([], dtype=float),
            )
        return (
            feature_index,
            weights_predict,
            weights_low,
            weights_high,
            predict_matrix,
            low_matrix,
            high_matrix,
            rule_values_result,
            binned_result,
            lower_update,
            upper_update,
        )

    # Handle features with no perturbations
    if feature_indices is None or len(feature_indices) == 0:
        for i in range(n_instances):
            rule_values_result[i] = (feature_values_list, x_column[i], x_column[i])
            binned_result[i] = (
                predict[i],
                low[i],
                high[i],
                -1,
                np.array([], dtype=float),
                np.array([], dtype=float),
            )
        return (
            feature_index,
            weights_predict,
            weights_low,
            weights_high,
            predict_matrix,
            low_matrix,
            high_matrix,
            rule_values_result,
            binned_result,
            lower_update,
            upper_update,
        )

    # Extract perturbation results for this feature
    feature_slice = np.asarray(perturbed_feature[feature_indices])
    feature_predict_local = np.asarray(predict[feature_indices])
    feature_low_local = np.asarray(low[feature_indices])
    feature_high_local = np.asarray(high[feature_indices])
    feature_instances = feature_slice[:, 1].astype(int)
    unique_instances = np.unique(feature_instances)

    # CATEGORICAL FEATURE PROCESSING
    if feature_index in categorical_features_set:
        feature_values_array = np.asarray(feature_values_list, dtype=object)
        num_feature_values = int(feature_values_array.size)
        value_counts_cache = value_counts_cache or {}
        counts_template = (
            np.array(
                [value_counts_cache.get(val, 0) for val in feature_values_list],
                dtype=float,
            )
            if num_feature_values
            else np.zeros((0,), dtype=float)
        )

        if num_feature_values == 0:
            for inst in unique_instances:
                i = int(inst)
                rule_values_result[i] = (feature_values_list, x_column[i], x_column[i])
                binned_result[i] = (
                    np.zeros((0,), dtype=float),
                    np.zeros((0,), dtype=float),
                    np.zeros((0,), dtype=float),
                    -1,
                    counts_template.copy(),
                    np.zeros((0,), dtype=float),
                )
            for idx in range(n_instances):
                if rule_values_result[idx] is None:
                    rule_values_result[idx] = (
                        feature_values_list,
                        x_column[idx],
                        x_column[idx],
                    )
                if binned_result[idx] is None:
                    binned_result[idx] = (
                        np.zeros((0,), dtype=float),
                        np.zeros((0,), dtype=float),
                        np.zeros((0,), dtype=float),
                        -1,
                        counts_template.copy(),
                        np.zeros((0,), dtype=float),
                    )
            return (
                feature_index,
                weights_predict,
                weights_low,
                weights_high,
                predict_matrix,
                low_matrix,
                high_matrix,
                rule_values_result,
                binned_result,
                lower_update,
                upper_update,
            )

        # Build value-to-index mapping
        value_to_index = {val: idx for idx, val in enumerate(feature_values_list)}
        value_indices = np.array(
            [value_to_index.get(row[2], -1) for row in feature_slice], dtype=int
        )
        valid_mask = value_indices >= 0
        feature_instances_local = feature_instances

        # Accumulate predictions by (instance, value) pairs
        sums_shape = (n_instances, num_feature_values)
        predict_sums = np.zeros(sums_shape, dtype=float)
        low_sums = np.zeros(sums_shape, dtype=float)
        high_sums = np.zeros(sums_shape, dtype=float)
        combo_counts = np.zeros(sums_shape, dtype=float)

        if np.any(valid_mask):
            np.add.at(
                predict_sums,
                (feature_instances_local[valid_mask], value_indices[valid_mask]),
                np.asarray(feature_predict_local[valid_mask], dtype=float),
            )
            np.add.at(
                low_sums,
                (feature_instances_local[valid_mask], value_indices[valid_mask]),
                np.asarray(feature_low_local[valid_mask], dtype=float),
            )
            np.add.at(
                high_sums,
                (feature_instances_local[valid_mask], value_indices[valid_mask]),
                np.asarray(feature_high_local[valid_mask], dtype=float),
            )
            np.add.at(
                combo_counts,
                (feature_instances_local[valid_mask], value_indices[valid_mask]),
                1,
            )

        # Compute averages
        with np.errstate(divide="ignore", invalid="ignore"):
            average_matrix = np.divide(
                predict_sums,
                combo_counts,
                out=np.zeros_like(predict_sums),
                where=combo_counts > 0,
            )
            low_matrix_local = np.divide(
                low_sums,
                combo_counts,
                out=np.zeros_like(low_sums),
                where=combo_counts > 0,
            )
            high_matrix_local = np.divide(
                high_sums,
                combo_counts,
                out=np.zeros_like(high_sums),
                where=combo_counts > 0,
            )

        # Determine current bin for each instance
        current_bins = np.full(n_instances, -1, dtype=int)
        if value_to_index:
            current_bins = np.array(
                [value_to_index.get(val, -1) for val in np.asarray(x_column)],
                dtype=int,
            )

        # Compute weights for each instance
        for inst in unique_instances:
            i = int(inst)
            avg_row = np.array(average_matrix[i], copy=True)
            low_row = np.array(low_matrix_local[i], copy=True)
            high_row = np.array(high_matrix_local[i], copy=True)
            current_bin = current_bins[i]

            # Exclude current value from weight calculation
            mask = np.ones(num_feature_values, dtype=bool)
            if 0 <= current_bin < num_feature_values:
                mask[current_bin] = False
            uncovered = np.nonzero(mask)[0]

            counts = counts_template.copy()
            counts_uncovered = counts[mask]
            total_counts = counts_uncovered.sum() if uncovered.size else 0
            fractions = (
                counts_uncovered / total_counts
                if uncovered.size and total_counts
                else np.zeros(uncovered.size, dtype=float)
            )

            rule_values_result[i] = (feature_values_list, x_column[i], x_column[i])
            binned_result[i] = (
                avg_row,
                low_row,
                high_row,
                current_bin,
                counts,
                fractions,
            )

            if uncovered.size == 0:
                continue

            # Compute feature weights
            predict_matrix[i] = safe_mean(avg_row[mask])
            low_matrix[i] = safe_mean(low_row[mask])
            high_matrix[i] = safe_mean(high_row[mask])
            base_val = baseline_predict[i]
            weights_predict[i] = assign_weight_scalar(predict_matrix[i], base_val)
            tmp_low = assign_weight_scalar(low_matrix[i], base_val)
            tmp_high = assign_weight_scalar(high_matrix[i], base_val)
            weights_low[i] = np.min([tmp_low, tmp_high])
            weights_high[i] = np.max([tmp_low, tmp_high])

    # NUMERIC FEATURE PROCESSING
    else:
        slice_bins = np.array(feature_slice[:, 2], dtype=int)
        slice_flags = np.asarray(feature_slice[:, 3], dtype=object)

        # Group perturbations by (instance, bin, flag)
        numeric_grouped: Dict[Tuple[int, int, Any], np.ndarray] = {}
        for rel_idx, inst in enumerate(feature_instances):
            key = (int(inst), int(slice_bins[rel_idx]), slice_flags[rel_idx])
            numeric_grouped.setdefault(key, []).append(rel_idx)
        for key, rel_list in list(numeric_grouped.items()):
            numeric_grouped[key] = np.asarray(rel_list, dtype=int)

        # Get sorted calibration values
        if numeric_sorted_values is None:
            feature_values_numeric = np.unique(np.asarray(x_cal_column))
            sorted_cal = np.sort(feature_values_numeric)
        else:
            sorted_cal = np.asarray(numeric_sorted_values)
            feature_values_numeric = np.unique(sorted_cal)

        # Update rule boundaries
        lower_boundary = np.asarray(lower_boundary, dtype=float)
        upper_boundary = np.asarray(upper_boundary, dtype=float)
        if feature_values_numeric.size:
            min_val = np.min(feature_values_numeric)
            max_val = np.max(feature_values_numeric)
            lower_boundary = np.where(min_val < lower_boundary, lower_boundary, -np.inf)
            upper_boundary = np.where(max_val > upper_boundary, upper_boundary, np.inf)
        lower_update = lower_boundary.copy()
        upper_update = upper_boundary.copy()

        # Initialize per-instance bins
        avg_predict_map: Dict[int, np.ndarray] = {}
        low_predict_map: Dict[int, np.ndarray] = {}
        high_predict_map: Dict[int, np.ndarray] = {}
        counts_map: Dict[int, np.ndarray] = {}
        rule_value_map: Dict[int, List[np.ndarray]] = {}

        for i in range(n_instances):
            num_bins = 1 + (1 if lower_boundary[i] != -np.inf else 0)
            num_bins += 1 if upper_boundary[i] != np.inf else 0
            avg_predict_map[i] = np.zeros(num_bins)
            low_predict_map[i] = np.zeros(num_bins)
            high_predict_map[i] = np.zeros(num_bins)
            counts_map[i] = np.zeros(num_bins)
            rule_value_map[i] = []

        bin_value = np.zeros(n_instances, dtype=int)
        current_bin = -np.ones(n_instances, dtype=int)

        # Pre-compute boundary caches
        unique_lower, lower_inverse = np.unique(lower_boundary, return_inverse=True)
        unique_upper, upper_inverse = np.unique(upper_boundary, return_inverse=True)
        lower_groups = {
            idx: np.flatnonzero(lower_inverse == idx) for idx in range(unique_lower.size)
        }
        upper_groups = {
            idx: np.flatnonzero(upper_inverse == idx) for idx in range(unique_upper.size)
        }

        lower_cache = {
            val: 0 if val == -np.inf else int(np.searchsorted(sorted_cal, val, side="left"))
            for val in unique_lower
        }
        upper_cache = {
            val: 0
            if val == np.inf
            else int(sorted_cal.size - np.searchsorted(sorted_cal, val, side="right"))
            for val in unique_upper
        }

        bounds_matrix = np.column_stack((lower_boundary, upper_boundary))
        unique_bounds, _ = np.unique(bounds_matrix, axis=0, return_inverse=True)
        between_cache: Dict[int, int] = {}
        for idx_bound, (lower_b, upper_b) in enumerate(unique_bounds):
            left = (
                0 if lower_b == -np.inf else int(np.searchsorted(sorted_cal, lower_b, side="left"))
            )
            right = (
                sorted_cal.size
                if upper_b == np.inf
                else int(np.searchsorted(sorted_cal, upper_b, side="right"))
            )
            between_cache[idx_bound] = right - left

        lesser_feature = lesser_feature or {}
        greater_feature = greater_feature or {}
        covered_feature = covered_feature or {}

        # Process lesser values (below lower boundary)
        for j, val in enumerate(unique_lower):
            values_tuple = lesser_feature.get(j)
            if not values_tuple or getattr(values_tuple[0], "size", 0) == 0:
                continue
            for idx in lower_groups.get(j, []):
                inst = int(idx)
                rel_indices = numeric_grouped.get((inst, j, True), np.empty((0,), dtype=int))
                avg_predict_map[inst][bin_value[inst]] = (
                    safe_mean(feature_predict_local[rel_indices]) if rel_indices.size else 0
                )
                low_predict_map[inst][bin_value[inst]] = (
                    safe_mean(feature_low_local[rel_indices]) if rel_indices.size else 0
                )
                high_predict_map[inst][bin_value[inst]] = (
                    safe_mean(feature_high_local[rel_indices]) if rel_indices.size else 0
                )
                counts_map[inst][bin_value[inst]] = lower_cache.get(val, 0)
                rule_value_map[inst].append(values_tuple[0])
                bin_value[inst] += 1

        # Process greater values (above upper boundary)
        for j, val in enumerate(unique_upper):
            values_tuple = greater_feature.get(j)
            if not values_tuple or getattr(values_tuple[0], "size", 0) == 0:
                continue
            for idx in upper_groups.get(j, []):
                inst = int(idx)
                rel_indices = numeric_grouped.get((inst, j, False), np.empty((0,), dtype=int))
                avg_predict_map[inst][bin_value[inst]] = (
                    safe_mean(feature_predict_local[rel_indices]) if rel_indices.size else 0
                )
                low_predict_map[inst][bin_value[inst]] = (
                    safe_mean(feature_low_local[rel_indices]) if rel_indices.size else 0
                )
                high_predict_map[inst][bin_value[inst]] = (
                    safe_mean(feature_high_local[rel_indices]) if rel_indices.size else 0
                )
                counts_map[inst][bin_value[inst]] = upper_cache.get(val, 0)
                rule_value_map[inst].append(values_tuple[0])
                bin_value[inst] += 1

        # Process covered values (within boundaries)
        for inst in range(n_instances):
            current_index = bin_value[inst]
            for j in range(unique_bounds.shape[0]):
                rel_indices = numeric_grouped.get((inst, j, None), np.empty((0,), dtype=int))
                avg_predict_map[inst][current_index] = (
                    safe_mean(feature_predict_local[rel_indices]) if rel_indices.size else 0
                )
                low_predict_map[inst][current_index] = (
                    safe_mean(feature_low_local[rel_indices]) if rel_indices.size else 0
                )
                high_predict_map[inst][current_index] = (
                    safe_mean(feature_high_local[rel_indices]) if rel_indices.size else 0
                )
                counts_map[inst][current_index] = between_cache.get(j, 0)
                rule_entry = covered_feature.get(j)
                if rule_entry is None:
                    rule_entry = covered_feature.get(inst)
                rule_value_map[inst].append(
                    rule_entry[0] if rule_entry is not None else np.array([])
                )
                current_bin[inst] = current_index

        # Compute weights for numeric features
        for inst in range(n_instances):
            rule_values_result[inst] = (
                rule_value_map[inst],
                x_column[inst],
                x_column[inst],
            )
            mask = np.ones_like(avg_predict_map[inst], dtype=bool)
            if 0 <= current_bin[inst] < mask.size:
                mask[current_bin[inst]] = False
            uncovered = np.nonzero(mask)[0]
            counts_uncovered = counts_map[inst][mask]
            total_counts = counts_uncovered.sum() if uncovered.size else 0
            fractions = (
                counts_uncovered / total_counts
                if uncovered.size and total_counts
                else np.zeros(uncovered.size, dtype=float)
            )
            binned_result[inst] = (
                avg_predict_map[inst],
                low_predict_map[inst],
                high_predict_map[inst],
                current_bin[inst],
                counts_map[inst],
                fractions,
            )
            if uncovered.size == 0:
                continue
            predict_matrix[inst] = safe_mean(avg_predict_map[inst][mask])
            low_matrix[inst] = safe_mean(low_predict_map[inst][mask])
            high_matrix[inst] = safe_mean(high_predict_map[inst][mask])
            base_val = baseline_predict[inst]
            weights_predict[inst] = assign_weight_scalar(predict_matrix[inst], base_val)
            tmp_low = assign_weight_scalar(low_matrix[inst], base_val)
            tmp_high = assign_weight_scalar(high_matrix[inst], base_val)
            weights_low[inst] = np.min([tmp_low, tmp_high])
            weights_high[inst] = np.max([tmp_low, tmp_high])

    # Fill in any missing results
    for idx in range(n_instances):
        if rule_values_result[idx] is None:
            rule_values_result[idx] = (feature_values_list, x_column[idx], x_column[idx])
        if binned_result[idx] is None:
            binned_result[idx] = (
                np.zeros((0,), dtype=float),
                np.zeros((0,), dtype=float),
                np.zeros((0,), dtype=float),
                -1,
                np.array([], dtype=float),
                np.array([], dtype=float),
            )

    return (
        feature_index,
        weights_predict,
        weights_low,
        weights_high,
        predict_matrix,
        low_matrix,
        high_matrix,
        rule_values_result,
        binned_result,
        lower_update,
        upper_update,
    )


__all__ = [
    "FeatureTaskResult",
    "assign_weight_scalar",
    "feature_task",
]
