"""Internal helpers for FAST-based feature filtering.

This module implements the per-batch, per-instance feature filtering logic
used by execution plugins. It does NOT touch CalibratedExplainer directly;
callers pass in the relevant explanation containers and configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np

from ...explanations.explanations import CalibratedExplanations


@dataclass
class FeatureFilterConfig:
    """Configuration for internal FAST-based feature filtering."""

    enabled: bool = False
    per_instance_top_k: int = 8

    @classmethod
    def from_base_and_env(cls, base: "FeatureFilterConfig | None" = None) -> "FeatureFilterConfig":
        """Merge CE_FEATURE_FILTER overrides with an optional base configuration."""
        cfg = FeatureFilterConfig(
            **(base.__dict__ if base is not None else {})
        )  # shallow copy to avoid mutating base

        raw = os.getenv("CE_FEATURE_FILTER")
        if not raw:
            return cfg

        tokens = [segment.strip() for segment in str(raw).split(",") if segment.strip()]
        if not tokens:
            return cfg

        # Simple forms: "on"/"off"/"1"/"0"
        if len(tokens) == 1 and tokens[0].lower() in {"1", "true", "on"}:
            cfg.enabled = True
            return cfg
        if len(tokens) == 1 and tokens[0].lower() in {"0", "false", "off"}:
            cfg.enabled = False
            return cfg

        for token in tokens:
            lowered = token.lower()
            if lowered in {"0", "off", "false"}:
                cfg.enabled = False
                continue
            if lowered in {"1", "on", "true", "enable"}:
                cfg.enabled = True
                continue
            if token.startswith("top_k="):
                value_str = token.split("=", 1)[1].strip()
                normalized = value_str.lstrip("+-")
                if normalized.isdigit():
                    cfg.per_instance_top_k = max(1, int(value_str))
                continue

        return cfg


@dataclass
class FeatureFilterResult:
    """Result of FAST-based feature filtering for a batch.

    Attributes
    ----------
    global_ignore:
        Features to ignore for all instances when running the expensive
        factual/alternative explain path. This includes both the baseline
        ignore set (explainer + user) and any additional features filtered
        out by FAST that are not needed for any instance in the batch.
    per_instance_ignore:
        A list with one entry per instance in ``fast_explanations``. Each
        entry is an array of feature indices that are ignored for that
        specific instance (baseline ignore plus per-instance filtering).
    """

    global_ignore: np.ndarray
    per_instance_ignore: List[np.ndarray]


def _safe_len_feature_weights(explanations: CalibratedExplanations) -> int:
    """Return the number of features inferred from the first explanation."""
    if not explanations.explanations:
        return 0
    first = explanations.explanations[0]
    weights = getattr(first, "feature_weights", None) or {}
    predict = weights.get("predict")
    if predict is None:
        return 0
    arr = np.asarray(predict)
    if arr.ndim == 0:
        return 1
    return int(arr.shape[-1])


def compute_filtered_features_to_ignore(
    fast_explanations: CalibratedExplanations,
    *,
    num_features: int | None,
    base_ignore: np.ndarray,
    config: FeatureFilterConfig,
) -> FeatureFilterResult:
    """Return per-instance and global feature indices to ignore based on FAST.

    Parameters
    ----------
    fast_explanations : CalibratedExplanations
        Collection of FAST explanations for the current batch.
    num_features : int or None
        Total number of features for the explainer; when None it is inferred.
    base_ignore : np.ndarray
        Baseline features to ignore (constants + user-specified).
    config : FeatureFilterConfig
        Active feature-filter configuration.

    Returns
    -------
    FeatureFilterResult
        ``FeatureFilterResult`` with:
        - ``global_ignore``: indices ignored for all instances (compute-time).
        - ``per_instance_ignore``: per-instance ignore arrays, length equal
          to ``len(fast_explanations.explanations)``.
    """
    # Normalise baseline ignore set and instance count so that even when
    # filtering is effectively disabled we can return a consistent shape.
    base_ignore_arr = (
        np.asarray(base_ignore, dtype=int) if base_ignore is not None else np.array([], dtype=int)
    )
    num_instances = len(fast_explanations.explanations)

    if not config.enabled or num_instances == 0:
        per_instance = [base_ignore_arr.copy() for _ in range(num_instances)]
        return FeatureFilterResult(
            global_ignore=base_ignore_arr.copy(),
            per_instance_ignore=per_instance,
        )

    inferred_num_features = _safe_len_feature_weights(fast_explanations)
    if num_features is None:
        num_features = inferred_num_features
    else:
        num_features = int(num_features)
    if num_features <= 0 or inferred_num_features == 0:
        per_instance = [base_ignore_arr.copy() for _ in range(num_instances)]
        return FeatureFilterResult(
            global_ignore=base_ignore_arr.copy(),
            per_instance_ignore=per_instance,
        )

    base_ignore_set = {int(f) for f in base_ignore_arr.tolist() if 0 <= int(f) < num_features}
    all_features = set(range(num_features))
    per_instance_ignore: list[np.ndarray] = []
    per_instance_keep: list[set[int]] = []

    per_instance_top_k = max(1, int(config.per_instance_top_k))

    # Compute per-instance keep/ignore sets based on absolute FAST weights.
    for exp in fast_explanations.explanations:
        weights_mapping = getattr(exp, "feature_weights", None)
        if not isinstance(weights_mapping, dict):
            # No usable weights; fall back to baseline ignore only for this instance.
            ignore_set = sorted(base_ignore_set)
            per_instance_ignore.append(np.asarray(ignore_set, dtype=int))
            per_instance_keep.append(all_features - base_ignore_set)
            continue
        predict_weights = weights_mapping.get("predict")
        if predict_weights is None:
            ignore_set = sorted(base_ignore_set)
            per_instance_ignore.append(np.asarray(ignore_set, dtype=int))
            per_instance_keep.append(all_features - base_ignore_set)
            continue

        weights_arr = np.asarray(predict_weights, dtype=float).reshape(-1)
        if weights_arr.size == 0:
            ignore_set = sorted(base_ignore_set)
            per_instance_ignore.append(np.asarray(ignore_set, dtype=int))
            per_instance_keep.append(all_features - base_ignore_set)
            continue

        # Align to num_features by truncation or padding with zeros as needed.
        if weights_arr.size < num_features:
            padded = np.zeros(num_features, dtype=float)
            padded[: weights_arr.size] = weights_arr
            weights_arr = padded
        elif weights_arr.size > num_features:
            weights_arr = weights_arr[:num_features]

        candidates_for_filter = all_features - base_ignore_set
        if not candidates_for_filter:
            ignore_set = sorted(base_ignore_set)
            per_instance_ignore.append(np.asarray(ignore_set, dtype=int))
            per_instance_keep.append(set())
            continue

        candidate_indices = np.asarray(sorted(candidates_for_filter), dtype=int)
        candidate_scores = np.abs(weights_arr[candidate_indices])

        k = min(per_instance_top_k, candidate_indices.size)
        ranking = np.argsort(candidate_scores)
        top_indices = candidate_indices[ranking[-k:]]
        keep_features = {int(f) for f in top_indices.tolist()}
        extra_ignore = candidates_for_filter - keep_features

        per_instance_keep.append(keep_features)
        ignore_total = sorted(set(extra_ignore) | base_ignore_set)
        per_instance_ignore.append(np.asarray(ignore_total, dtype=int))

    # Derive a conservative global ignore set: skip only features that are not
    # needed by any instance in the batch (beyond the baseline ignore).
    candidates_for_filter = all_features - base_ignore_set
    if not candidates_for_filter:
        global_ignore = np.asarray(sorted(base_ignore_set), dtype=int)
    else:
        union_keep: set[int] = set()
        for keep_set in per_instance_keep:
            union_keep.update(int(f) for f in keep_set)
        extra_global_ignore = candidates_for_filter - union_keep
        final_ignore = sorted(set(extra_global_ignore) | base_ignore_set)
        global_ignore = np.asarray(final_ignore, dtype=int)

    return FeatureFilterResult(global_ignore=global_ignore, per_instance_ignore=per_instance_ignore)


__all__ = ["FeatureFilterConfig", "FeatureFilterResult", "compute_filtered_features_to_ignore"]
