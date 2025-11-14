"""Calibration summary computation and caching utilities.

This module provides functions for computing and caching statistical summaries
of calibration data, including categorical value counts and sorted numeric values.
These summaries are used during explanation generation for efficient feature analysis.

Part of Phase 6: Refactor Calibration Functionality (ADR-001).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..calibrated_explainer import CalibratedExplainer


def invalidate_calibration_summaries(explainer: CalibratedExplainer) -> None:
    """Drop cached calibration summaries used during explanation.

    This function is called whenever calibration data is modified to ensure
    that cached summaries are recomputed on the next access.

    Parameters
    ----------
    explainer : CalibratedExplainer
        The explainer instance whose calibration summary caches should be cleared.
    """
    explainer._categorical_value_counts_cache = None
    explainer._numeric_sorted_cache = None
    explainer._calibration_summary_shape = None


def get_calibration_summaries(
    explainer: CalibratedExplainer, x_cal_np: Optional[np.ndarray] = None
) -> Tuple[Dict[int, Dict[Any, int]], Dict[int, np.ndarray]]:
    """Return cached categorical counts and sorted numeric calibration values.

    Computes and caches statistical summaries of the calibration data:
    - For categorical features: unique value counts
    - For numeric features: sorted arrays of values

    These summaries enable efficient feature analysis during explanation generation.
    The cache is invalidated when calibration data changes.

    Parameters
    ----------
    explainer : CalibratedExplainer
        The explainer instance holding calibration data.
    x_cal_np : np.ndarray, optional
        Optional pre-computed calibration data array. If None, uses explainer.x_cal.

    Returns
    -------
    Tuple[Dict[int, Dict[Any, int]], Dict[int, np.ndarray]]
        A tuple containing:
        - categorical_value_counts: Mapping from feature index to (value -> count) dict
        - numeric_sorted_cache: Mapping from feature index to sorted value array

    Notes
    -----
    Results are cached in the explainer instance. Cache is invalidated via
    :func:`invalidate_calibration_summaries` when calibration data changes.
    """
    if x_cal_np is None:
        x_cal_np = np.asarray(explainer.x_cal)

    shape = getattr(x_cal_np, "shape", None)

    # Check if cache is still valid
    if (
        explainer._categorical_value_counts_cache is None
        or explainer._numeric_sorted_cache is None
        or explainer._calibration_summary_shape != shape
    ):
        categorical_value_counts: Dict[int, Dict[Any, int]] = {}
        numeric_sorted_cache: Dict[int, np.ndarray] = {}

        if x_cal_np.size:
            categorical_features = tuple(int(f) for f in explainer.categorical_features)

            # Compute value counts for categorical features
            for f_cat in categorical_features:
                unique_vals, unique_counts = np.unique(x_cal_np[:, f_cat], return_counts=True)
                categorical_value_counts[int(f_cat)] = {
                    val: int(cnt)
                    for val, cnt in zip(unique_vals.tolist(), unique_counts.tolist())
                }

            # Sort numeric feature values for later use
            numeric_features = [
                f for f in range(explainer.num_features) if f not in categorical_features
            ]
            for f_num in numeric_features:
                numeric_sorted_cache[f_num] = np.sort(np.asarray(x_cal_np[:, f_num]))

        # Store cache
        explainer._categorical_value_counts_cache = categorical_value_counts
        explainer._numeric_sorted_cache = numeric_sorted_cache
        explainer._calibration_summary_shape = shape

    assert explainer._categorical_value_counts_cache is not None
    assert explainer._numeric_sorted_cache is not None

    return explainer._categorical_value_counts_cache, explainer._numeric_sorted_cache
