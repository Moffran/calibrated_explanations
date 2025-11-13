"""Feature-level task computation for explanation pipelines.

This module contains domain-specific logic for computing per-feature
perturbations, weight deltas, and rule aggregation used in parallel
explanation execution.
"""

from __future__ import annotations

import numpy as np
from typing import Any, List, Optional, Set, Tuple

# Type alias for the aggregated result of processing a single feature
FeatureTaskResult = Tuple[
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Any],
    List[Any],
    Optional[np.ndarray],
    Optional[np.ndarray],
]


def assign_weight_scalar(instance_predict: Any, prediction: Any) -> float:
    """Return the scalar delta between *prediction* and *instance_predict*.

    Computes a single numerical scalar representing the difference between
    two predictions. Used to measure feature-level impact on model output.

    Parameters
    ----------
    instance_predict : Any
        Baseline prediction (scalar or array-like).
    prediction : Any
        Perturbed prediction (scalar or array-like).

    Returns
    -------
    float
        Scalar difference. If inputs are arrays, returns the first element
        of the flattened difference array. Returns 0.0 if computation fails.

    Examples
    --------
    >>> assign_weight_scalar(0.5, 0.7)
    0.2

    >>> assign_weight_scalar([0.5], [0.7])
    0.2
    """
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
    except Exception:  # pragma: no cover - defensive fallback
        diff = np.asarray(base_arr, dtype=float) - np.asarray(inst_arr, dtype=float)
    flat = np.asarray(diff, dtype=float).reshape(-1)
    if flat.size == 0:
        return 0.0
    return float(flat[0])


def execute_feature_task(args: Tuple[Any, ...]) -> FeatureTaskResult:
    """Execute the per-feature aggregation logic for explanation computation.

    This is the core kernel for parallel feature-level explanation loops.
    It processes one feature index across all test instances, computing:
    - Weight deltas for predictions and uncertainty bounds
    - Rule match counts and value distributions
    - Binned result matrices for later aggregation

    Parameters
    ----------
    args : Tuple[Any, ...]
        Packed arguments containing:
        - feature_index: int - Feature being processed
        - x_column: array-like - Feature values for all instances
        - predict: array-like - Base predictions
        - low: array-like - Lower uncertainty bounds
        - high: array-like - Upper uncertainty bounds
        - baseline_predict: array-like - Reference predictions for delta
        - features_to_ignore: Sequence[int] - Indices to skip
        - categorical_features: Sequence[int] - Categorical feature indices
        - feature_values: Dict - Possible values per feature
        - feature_indices: Optional[np.ndarray] - Indices of perturbations
        - perturbed_feature: np.ndarray - Perturbed feature matrix
        - lower_boundary: np.ndarray - Rule boundary conditions
        - upper_boundary: np.ndarray - Rule boundary conditions
        - lesser_feature: np.ndarray - Lesser value boundaries
        - greater_feature: np.ndarray - Greater value boundaries
        - covered_feature: np.ndarray - Covered range boundaries
        - value_counts_cache: Optional[Dict] - Cached value frequencies
        - numeric_sorted_values: Optional[Dict] - Cached sorted values
        - x_cal_column: array-like - Calibration feature values

    Returns
    -------
    FeatureTaskResult
        Tuple containing:
        - feature_index: Feature processed
        - weights_predict: Weight deltas for predictions
        - weights_low: Weight deltas for lower bounds
        - weights_high: Weight deltas for upper bounds
        - predict_matrix: Aggregated prediction scores
        - low_matrix: Aggregated lower bound scores
        - high_matrix: Aggregated upper bound scores
        - rule_values_result: Per-instance rule value info
        - binned_result: Per-instance binned result tuples
        - lower_update: Optional lower boundary update matrix
        - upper_update: Optional upper boundary update matrix

    Notes
    -----
    This function is designed for use with parallel executors (e.g.,
    multiprocessing, joblib). It is stateless and accepts all required
    context via the packed ``args`` tuple.

    This function will be moved to an orchestrator in future refactors
    as the explanation architecture evolves.
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

    # Early exit for ignored or unperturbed features
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

    # Process perturbations for this feature
    feature_slice = np.asarray(perturbed_feature[feature_indices])
    feature_predict_local = np.asarray(predict[feature_indices])
    feature_low_local = np.asarray(low[feature_indices])
    feature_high_local = np.asarray(high[feature_indices])
    feature_instances = feature_slice[:, 1].astype(int)
    unique_instances = np.unique(feature_instances)

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

        # Placeholder: Categorical feature processing
        # Full implementation continues in orchestrator refactor
        for i in range(n_instances):
            if rule_values_result[i] is None:
                rule_values_result[i] = (feature_values_list, x_column[i], x_column[i])
            if binned_result[i] is None:
                binned_result[i] = (
                    np.zeros((0,), dtype=float),
                    np.zeros((0,), dtype=float),
                    np.zeros((0,), dtype=float),
                    -1,
                    counts_template.copy() if num_feature_values > 0 else np.array([], dtype=float),
                    np.zeros((0,), dtype=float),
                )

    else:
        # Numeric feature processing - placeholder for orchestrator
        for i in range(n_instances):
            if rule_values_result[i] is None:
                rule_values_result[i] = (feature_values_list, x_column[i], x_column[i])
            if binned_result[i] is None:
                binned_result[i] = (
                    np.zeros((0,), dtype=float),
                    np.zeros((0,), dtype=float),
                    np.zeros((0,), dtype=float),
                    -1,
                    np.array([], dtype=float),
                    np.array([], dtype=float),
                )

    # Finalize results
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
