"""Shared data structures for explain executor system.

This module defines request/response contracts used by sequential, feature-parallel,
and instance-parallel explain executors per the plugin decomposition strategy.
"""

from __future__ import annotations

import contextlib
from collections import defaultdict
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Access to protected explainer internals is intentional in this module
# Some names (e.g. dataclass fields) follow existing project API and are
# intentionally short; suppress style warnings for those cases.
# pylint: disable=protected-access,invalid-name


def build_feature_tasks(
    explainer,
    x_input: np.ndarray,
    perturbed_feature,
    x_cal,
    features_to_ignore_array: np.ndarray,
    config: "ExplainConfig",
    rule_boundaries: np.ndarray,
    lesser_values: Mapping | Any,
    greater_values: Mapping | Any,
    covered_values: Mapping | Any,
    predict: Any,
    low: Any,
    high: Any,
    baseline_predict: Any,
) -> List[Tuple[Any, ...]]:  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
    """Build the per-feature task tuples used by the explain executors.

    This consolidates the duplicate logic from sequential and feature-parallel
    plugins so both can reuse a single, tested implementation.
    """
    perturbed_feature = np.asarray(perturbed_feature, dtype=object)
    x_cal_np = np.asarray(x_cal)

    # Build feature index map for quick lookup
    if perturbed_feature.size:
        feature_ids = perturbed_feature[:, 0].astype(int)
        feature_index_lists: Dict[int, List[int]] = defaultdict(list)
        for idx, fid in enumerate(feature_ids):
            feature_index_lists[int(fid)].append(idx)
        feature_index_map = {
            fid: np.asarray(indices, dtype=int) for fid, indices in feature_index_lists.items()
        }
    else:
        feature_index_map = {}

    # Get calibration summaries for fast lookups
    categorical_value_counts, numeric_sorted_cache = explainer._get_calibration_summaries(x_cal_np)

    features_to_ignore_set = set(features_to_ignore_array.tolist())
    features_to_ignore_tuple = tuple(int(f) for f in features_to_ignore_set)
    categorical_features_tuple = tuple(int(f) for f in config.categorical_features)
    feature_values_all = config.feature_values

    # Build feature tasks
    feature_tasks: List[Tuple[Any, ...]] = []

    for feature_idx in range(config.num_features):
        feature_indices = feature_index_map.get(feature_idx)
        lower_boundary = np.array(rule_boundaries[:, feature_idx, 0], copy=True)
        upper_boundary = np.array(rule_boundaries[:, feature_idx, 1], copy=True)

        # Extract feature-specific value mappings
        if isinstance(lesser_values, Mapping):
            lesser_feature = lesser_values.get(feature_idx, {})
        else:
            lesser_feature = {}
            with contextlib.suppress(IndexError, KeyError, TypeError):
                lesser_feature = lesser_values[feature_idx]

        if isinstance(greater_values, Mapping):
            greater_feature = greater_values.get(feature_idx, {})
        else:
            greater_feature = {}
            with contextlib.suppress(IndexError, KeyError, TypeError):
                greater_feature = greater_values[feature_idx]

        if isinstance(covered_values, Mapping):
            covered_feature = covered_values.get(feature_idx, {})
        else:
            covered_feature = {}
            with contextlib.suppress(IndexError, KeyError, TypeError):
                covered_feature = covered_values[feature_idx]

        value_counts_cache = categorical_value_counts.get(int(feature_idx), {})
        numeric_sorted_values = numeric_sorted_cache.get(feature_idx)

        # Extract feature columns safely
        if (
            isinstance(x_cal_np, np.ndarray)
            and x_cal_np.ndim >= 2
            and x_cal_np.shape[0]
            and feature_idx < x_cal_np.shape[1]
        ):
            x_cal_column = np.asarray(x_cal_np[:, feature_idx])
        else:
            x_cal_column = np.empty((0,))
        x_column = np.asarray(x_input[:, feature_idx])

        feature_tasks.append(
            (
                feature_idx,
                x_column,
                predict,
                low,
                high,
                baseline_predict,
                features_to_ignore_tuple,
                categorical_features_tuple,
                feature_values_all,
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
            )
        )

    return feature_tasks


def finalize_explanation(
    explanation,
    weights_predict,
    weights_low,
    weights_high,
    predict_matrix,
    low_matrix,
    high_matrix,
    rule_values,
    instance_binned,
    rule_boundaries,
    prediction,
    instance_start_time: float,
    total_start_time: float,
    explainer,
):  # pylint: disable=too-many-arguments,too-many-locals
    """Aggregate buffers into the final explanation and update explainer state.

    Returns the finalized explanation object.
    """
    # Some callers pass rule_boundaries but this function does not use it;
    # reference it to satisfy linters.
    _ = rule_boundaries
    n_instances = weights_predict.shape[0]

    binned_predict: Dict[str, List[Any]] = {
        "predict": [],
        "low": [],
        "high": [],
        "current_bin": [],
        "rule_values": [],
        "counts": [],
        "fractions": [],
    }
    feature_weights: Dict[str, List[np.ndarray]] = {"predict": [], "low": [], "high": []}
    feature_predict: Dict[str, List[np.ndarray]] = {"predict": [], "low": [], "high": []}

    for i in range(n_instances):
        binned_predict["predict"].append(instance_binned[i]["predict"])
        binned_predict["low"].append(instance_binned[i]["low"])
        binned_predict["high"].append(instance_binned[i]["high"])
        binned_predict["current_bin"].append(instance_binned[i]["current_bin"])
        binned_predict["rule_values"].append(rule_values[i])
        binned_predict["counts"].append(instance_binned[i]["counts"])
        binned_predict["fractions"].append(instance_binned[i]["fractions"])

        feature_weights["predict"].append(weights_predict[i].copy())
        feature_weights["low"].append(weights_low[i].copy())
        feature_weights["high"].append(weights_high[i].copy())

        feature_predict["predict"].append(predict_matrix[i].copy())
        feature_predict["low"].append(low_matrix[i].copy())
        feature_predict["high"].append(high_matrix[i].copy())

    elapsed_time = time() - instance_start_time
    list_instance_time = [elapsed_time / n_instances for _ in range(n_instances)]
    total_time = time() - total_start_time

    explanation = explanation.finalize(
        binned_predict,
        feature_weights,
        feature_predict,
        prediction,
        instance_time=list_instance_time,
        total_time=total_time,
    )

    # Attach per-instance feature ignore information when provided by the
    # FAST-based feature filter. This is stored on the explainer by the
    # execution wrapper and propagated here to the explanation container.
    per_instance_ignore = getattr(explainer, "_feature_filter_per_instance_ignore", None)
    if per_instance_ignore is not None:
        with contextlib.suppress(Exception):
            explanation.features_to_ignore_per_instance = per_instance_ignore
    # Clear transient state to avoid accidental reuse across subsequent runs.
    with contextlib.suppress(AttributeError):
        delattr(explainer, "_feature_filter_per_instance_ignore")

    # Update explainer state
    explainer.latest_explanation = explanation
    explainer._last_explanation_mode = explainer._infer_explanation_mode()

    return explanation


@dataclass(frozen=True)
class ExplainRequest:
    """Immutable request context for explain operations.

    Contains all input data and configuration needed by explain executors.
    Designed to be passed uniformly across sequential and parallel implementations.
    """

    # Input data
    x: np.ndarray
    """Test instances to explain (n_instances, n_features)."""

    # Configuration parameters
    threshold: Optional[Any]
    """Threshold for binary classification or regression intervals."""

    low_high_percentiles: Tuple[float, float]
    """Percentile bounds for uncertainty intervals (default: (5, 95))."""

    bins: Optional[Any]
    """Difficulty bins for conditional calibration."""

    features_to_ignore: np.ndarray
    """Array of feature indices to skip during perturbation."""
    extras: Any | None = None
    """Opaque extras forwarded from the explanation request (unused by executors)."""
    features_to_ignore_per_instance: Any | None = None
    """Optional per-instance feature ignore masks."""

    # Control flags
    use_plugin: bool = True
    """Whether to invoke plugin registry (internal)."""

    skip_instance_parallel: bool = False
    """Prevent recursive instance parallelism (internal)."""

    # Slice context for instance parallelism
    instance_slice: Optional[Tuple[int, int]] = None
    """(start, stop) indices when processing instance chunk."""


@dataclass
class ExplainResponse:
    """Mutable response payload from explain operations.

    Contains all computed artifacts needed to construct CalibratedExplanations.
    Structured to support incremental assembly by parallel workers.
    """

    # Core prediction results
    predict: np.ndarray
    """Baseline predictions for test instances."""

    low: np.ndarray
    """Lower bound predictions."""

    high: np.ndarray
    """Upper bound predictions."""

    prediction: Dict[str, Any]
    """Full prediction metadata from _explain_predict_step."""

    # Feature perturbation results
    weights_predict: np.ndarray
    """Feature importance weights (n_instances, n_features)."""

    weights_low: np.ndarray
    """Lower bound feature weights."""

    weights_high: np.ndarray
    """Upper bound feature weights."""

    predict_matrix: np.ndarray
    """Per-feature perturbed predictions."""

    low_matrix: np.ndarray
    """Per-feature lower bounds."""

    high_matrix: np.ndarray
    """Per-feature upper bounds."""

    # Perturbation metadata
    perturbed_feature: List[Tuple[int, int, Any, Any]]
    """List of (feature_idx, instance_idx, value, boundary) tuples."""

    rule_boundaries: np.ndarray
    """Rule boundary arrays (n_instances, n_features, 2)."""

    lesser_values: Dict[int, Dict[int, Tuple[np.ndarray, float]]]
    """Mapping of feature -> boundary_idx -> (values, boundary)."""

    greater_values: Dict[int, Dict[int, Tuple[np.ndarray, float]]]
    """Mapping of feature -> boundary_idx -> (values, boundary)."""

    covered_values: Dict[int, Dict[int, Tuple[np.ndarray, float, float]]]
    """Mapping of feature -> boundary_idx -> (values, lower, upper)."""

    x_cal: np.ndarray
    """Calibration data reference."""

    # Timing metadata
    instance_time: float = 0.0
    """Time to explain individual instances (seconds)."""

    total_time: float = 0.0
    """Total explanation time (seconds)."""


@dataclass
class ExplainConfig:
    """Configuration context for explain executor selection and execution.

    Encapsulates executor settings, parallelism granularity, and explainer state
    needed for plugin dispatch logic.
    """

    executor: Optional[Any] = None
    """ParallelExecutor instance if parallel execution is enabled."""

    granularity: str = "feature"
    """Parallelism granularity: 'feature', 'instance', or 'none'."""

    min_instances_for_parallel: int = 8
    """Minimum instances required to trigger instance parallelism."""

    chunk_size: int = 100
    """Instance chunk size for parallel processing."""

    # Explainer state references
    num_features: int = 0
    """Number of features in the dataset."""

    features_to_ignore_default: Sequence[int] = field(default_factory=list)
    """Default set of features to ignore from explainer configuration."""

    categorical_features: Sequence[int] = field(default_factory=list)
    """Indices of categorical features."""

    feature_values: Mapping[int, np.ndarray] = field(default_factory=dict)
    """Mapping of categorical feature index to unique values."""

    mode: str = "classification"
    """Task mode: 'classification' or 'regression'."""


__all__ = [
    "ExplainConfig",
    "ExplainRequest",
    "ExplainResponse",
    "build_feature_tasks",
    "finalize_explanation",
]
