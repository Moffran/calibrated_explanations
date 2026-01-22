"""Public helpers that mirror the internal explain utilities."""

from __future__ import annotations

from typing import Any

from . import _helpers as _impl

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
    "impl",
]


# Public alias to the underlying implementation module for testing hooks.
impl = _impl


def slice_threshold(threshold: Any, start: int, stop: int, total_len: int) -> Any:
    """Return the portion of *threshold* covering ``[start, stop)``.

    Handles scalar, array-like, and pandas Series thresholds appropriately.
    """
    return _impl.slice_threshold(threshold, start, stop, total_len)


def slice_bins(bins: Any, start: int, stop: int) -> Any:
    """Return the subset of *bins* covering ``[start, stop)``.

    Handles pandas Series and array-like bins.
    """
    return _impl.slice_bins(bins, start, stop)


def compute_weight_delta(baseline: Any, perturbed: Any) -> Any:
    """Compute the weight delta between baseline and perturbed predictions."""
    return _impl.compute_weight_delta(baseline, perturbed)


def merge_feature_result(
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Merge feature results into a single result object."""
    return _impl.merge_feature_result(*args, **kwargs)


def compute_feature_effects(
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Compute feature effects for a set of instances."""
    return _impl.compute_feature_effects(*args, **kwargs)


def merge_ignore_features(
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Merge explainer default and request-specific features to ignore."""
    return _impl.merge_ignore_features(*args, **kwargs)


def initialize_explanation(
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Initialize an explanation object."""
    return _impl.initialize_explanation(*args, **kwargs)


def explain_predict_step(
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Perform a single prediction step for explanation."""
    return _impl.explain_predict_step(*args, **kwargs)


def feature_effect_for_index(
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Compute feature effect for a specific feature index."""
    return _impl.feature_effect_for_index(*args, **kwargs)


def validate_and_prepare_input(
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Validate and prepare input data for explanation."""
    return _impl.validate_and_prepare_input(*args, **kwargs)
