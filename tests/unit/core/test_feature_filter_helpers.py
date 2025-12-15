"""Unit tests for FAST-based feature filtering helpers."""

from __future__ import annotations

import numpy as np

from calibrated_explanations.core.explain._feature_filter import (
    FeatureFilterConfig,
    FeatureFilterResult,
    compute_filtered_features_to_ignore,
)
from calibrated_explanations.explanations.explanations import CalibratedExplanations


class _DummyExplainer:
    """Minimal explainer stub for constructing CalibratedExplanations."""

    def __init__(self, num_features: int) -> None:
        self.x_cal = np.zeros((1, num_features))
        self.y_cal = np.zeros(1)
        self.num_features = num_features


class _DummyFastExplanation:
    """Minimal FAST explanation stub exposing feature_weights['predict']."""

    def __init__(self, weights: np.ndarray) -> None:
        self.feature_weights = {"predict": np.asarray(weights, dtype=float)}


def _make_fast_collection(weight_rows: list[np.ndarray]) -> CalibratedExplanations:
    """Helper constructing a FAST-style CalibratedExplanations container."""
    n_instances = len(weight_rows)
    num_features = int(weight_rows[0].shape[0]) if weight_rows else 0
    explainer = _DummyExplainer(num_features)
    x = np.zeros((n_instances, num_features))
    collection = CalibratedExplanations(explainer, x, None, None)
    collection.explanations = [_DummyFastExplanation(row) for row in weight_rows]
    return collection


def test_compute_filtered_features_to_ignore_keeps_at_most_top_k_per_instance_no_base_ignore() -> (
    None
):
    """Per-instance keep-set must be <= top_k when there is no baseline ignore."""
    weights = [
        np.array([10.0, 0.1, 0.2, 0.0]),
        np.array([0.0, 5.0, 0.3, 0.0]),
    ]
    collection = _make_fast_collection(weights)
    num_features = 4

    cfg = FeatureFilterConfig(enabled=True, per_instance_top_k=2)
    result: FeatureFilterResult = compute_filtered_features_to_ignore(
        collection, num_features=num_features, base_ignore=np.array([], dtype=int), config=cfg
    )
    # One per-instance mask per explanation
    assert len(result.per_instance_ignore) == len(weights)

    # Each instance should keep at most top_k features.
    for ignore_arr in result.per_instance_ignore:
        ignore_set = set(ignore_arr.tolist())
        kept = [f for f in range(num_features) if f not in ignore_set]
        assert len(kept) <= cfg.per_instance_top_k

    # Global keep-set must cover all per-instance keeps.
    global_ignore_set = set(result.global_ignore.tolist())
    global_keep_set = set(range(num_features)) - global_ignore_set
    for ignore_arr in result.per_instance_ignore:
        per_keep = set(range(num_features)) - set(ignore_arr.tolist())
        assert per_keep.issubset(global_keep_set)


def test_compute_filtered_features_to_ignore_respects_top_k_with_base_ignore() -> None:
    """Per-instance keep-set must be <= top_k even when some features are baseline-ignored."""
    weights = [
        np.array([10.0, 0.1, 0.2, 0.0]),
        np.array([0.0, 5.0, 3.0, 0.0]),
    ]
    collection = _make_fast_collection(weights)
    num_features = 4

    # Feature 1 is always ignored by the explainer; allow filter to pick up to 2 among {0,2,3}.
    base_ignore = np.array([1], dtype=int)
    base_ignore_set = set(base_ignore.tolist())
    cfg = FeatureFilterConfig(enabled=True, per_instance_top_k=2)
    result: FeatureFilterResult = compute_filtered_features_to_ignore(
        collection, num_features=num_features, base_ignore=base_ignore, config=cfg
    )

    # Baseline ignore must be present in all per-instance and global masks.
    global_ignore_set = set(result.global_ignore.tolist())
    assert base_ignore_set.issubset(global_ignore_set)

    for ignore_arr in result.per_instance_ignore:
        ignore_set = set(ignore_arr.tolist())
        assert base_ignore_set.issubset(ignore_set)
        kept = [f for f in range(num_features) if f not in ignore_set]
        assert len(kept) <= cfg.per_instance_top_k

    # Global keep-set must cover all per-instance keeps.
    global_keep_set = set(range(num_features)) - global_ignore_set
    for ignore_arr in result.per_instance_ignore:
        per_keep = set(range(num_features)) - set(ignore_arr.tolist())
        assert per_keep.issubset(global_keep_set)
