"""Shared data structures for explain plugin system.

This module defines request/response contracts used by sequential, feature-parallel,
and instance-parallel explain plugins per the plugin decomposition strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ExplainRequest:
    """Immutable request context for explain operations.
    
    Contains all input data and configuration needed by explain plugins.
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
    """Configuration context for explain plugin selection and execution.
    
    Encapsulates executor settings, parallelism granularity, and explainer state
    needed for plugin dispatch logic.
    """

    executor: Optional[Any] = None
    """ParallelExecutor instance if parallel execution is enabled."""
    
    granularity: str = "feature"
    """Parallelism granularity: 'feature', 'instance', or 'none'."""
    
    min_instances_for_parallel: int = 4
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
]
