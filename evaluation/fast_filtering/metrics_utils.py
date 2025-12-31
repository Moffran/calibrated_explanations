"""Metric utilities for fast-filtering evaluations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class OverlapMetrics:
    """Overlap metrics for a single instance."""

    jaccard_topk: float
    topk_inclusion: float
    spearman_rank: float | None
    kept_feature_count: int


def _rank_from_weights(weights: np.ndarray) -> np.ndarray:
    order = np.argsort(np.abs(weights))
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(weights) + 1)
    return ranks


def compute_overlap_metrics(
    *,
    baseline_weights: np.ndarray,
    filtered_weights: np.ndarray,
    baseline_topk: Iterable[int],
    filtered_keep: Iterable[int],
) -> OverlapMetrics:
    """Compute overlap metrics for a single instance."""
    baseline_set = set(int(i) for i in baseline_topk)
    filtered_set = set(int(i) for i in filtered_keep)

    union = baseline_set | filtered_set
    intersection = baseline_set & filtered_set
    jaccard = len(intersection) / len(union) if union else 1.0

    topk = len(baseline_set) if baseline_set else 0
    topk_inclusion = len(intersection) / topk if topk else 1.0

    spearman = None
    if len(intersection) > 1:
        baseline_ranks = _rank_from_weights(baseline_weights)
        filtered_ranks = _rank_from_weights(filtered_weights)
        idx = np.array(sorted(intersection), dtype=int)
        baseline_subset = baseline_ranks[idx]
        filtered_subset = filtered_ranks[idx]
        spearman = float(np.corrcoef(baseline_subset, filtered_subset)[0, 1])

    return OverlapMetrics(
        jaccard_topk=float(jaccard),
        topk_inclusion=float(topk_inclusion),
        spearman_rank=spearman,
        kept_feature_count=len(filtered_set),
    )


def summarize_metrics(metrics: Iterable[OverlapMetrics]) -> dict:
    """Aggregate overlap metrics across instances."""
    metrics = list(metrics)
    if not metrics:
        return {}

    def _collect(attr: str):
        values = [getattr(m, attr) for m in metrics if getattr(m, attr) is not None]
        return np.array(values, dtype=float)

    summary = {}
    for attr in ("jaccard_topk", "topk_inclusion", "kept_feature_count"):
        values = _collect(attr)
        if values.size:
            summary[attr] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
            }

    spearman_values = _collect("spearman_rank")
    if spearman_values.size:
        summary["spearman_rank"] = {
            "mean": float(spearman_values.mean()),
            "median": float(np.median(spearman_values)),
            "std": float(spearman_values.std(ddof=0)),
            "min": float(spearman_values.min()),
            "max": float(spearman_values.max()),
        }

    return summary
