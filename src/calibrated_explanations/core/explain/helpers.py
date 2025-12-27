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
]


def slice_threshold(threshold: Any, start: int, stop: int, total_len: int) -> Any:
    return _impl.slice_threshold(threshold, start, stop, total_len)


def slice_bins(bins: Any, start: int, stop: int) -> Any:
    return _impl.slice_bins(bins, start, stop)


def compute_weight_delta(baseline: Any, perturbed: Any) -> Any:
    return _impl.compute_weight_delta(baseline, perturbed)


def merge_feature_result(
    *args: Any,
    **kwargs: Any,
) -> Any:
    return _impl.merge_feature_result(*args, **kwargs)


def compute_feature_effects(
    *args: Any,
    **kwargs: Any,
) -> Any:
    return _impl.compute_feature_effects(*args, **kwargs)


def merge_ignore_features(
    *args: Any,
    **kwargs: Any,
) -> Any:
    return _impl.merge_ignore_features(*args, **kwargs)


def initialize_explanation(
    *args: Any,
    **kwargs: Any,
) -> Any:
    return _impl.initialize_explanation(*args, **kwargs)


def explain_predict_step(
    *args: Any,
    **kwargs: Any,
) -> Any:
    return _impl.explain_predict_step(*args, **kwargs)


def feature_effect_for_index(
    *args: Any,
    **kwargs: Any,
) -> Any:
    return _impl.feature_effect_for_index(*args, **kwargs)


def validate_and_prepare_input(
    *args: Any,
    **kwargs: Any,
) -> Any:
    return _impl.validate_and_prepare_input(*args, **kwargs)
