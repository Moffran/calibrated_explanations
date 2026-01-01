"""Calibration summary computation and caching utilities.

This module provides functions for computing and caching statistical summaries
of calibration data, including categorical value counts and sorted numeric values.
These summaries are used during explanation generation for efficient feature analysis.

Caching is delegated to the shared ExplanationCacheFacade (ADR-001, ADR-003).
Instance-level caches are retained for backward compatibility but superseded by
the shared cache layer when the facade is available.

Part of ADR-001: Core Decomposition Boundaries (Stage 1a).
Part of ADR-003: Caching Strategy.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from calibrated_explanations.core import CalibratedExplainer


def invalidate_calibration_summaries(explainer: CalibratedExplainer) -> None:
    """Drop cached calibration summaries used during explanation.

    This function is called whenever calibration data is modified to ensure
    that cached summaries are recomputed on the next access. Invalidation is
    delegated to the shared cache facade (ADR-003) if available; instance-level
    caches are cleared for backward compatibility.

    Parameters
    ----------
    explainer : CalibratedExplainer
        The explainer instance whose calibration summary caches should be cleared.
    """
    # Invalidate shared cache if available
    cache_facade = getattr(explainer, "_explanation_cache", None)
    if cache_facade is not None:
        cache_facade.invalidate_all()

    # Maintain backward compatibility by clearing instance-level caches
    explainer.categorical_value_counts_cache = None
    explainer.numeric_sorted_cache = None
    explainer.calibration_summary_shape = None


def _get_calibration_data_hash(x_cal_np: np.ndarray) -> str:
    """Compute a stable hash of calibration data shape/dtype for cache keys.

    Parameters
    ----------
    x_cal_np : np.ndarray
        The calibration feature matrix.

    Returns
    -------
    str
        Hex digest of blake2b hash over shape, dtype, and size.
    """
    h = hashlib.blake2b()
    h.update(str(x_cal_np.shape).encode())
    h.update(str(x_cal_np.dtype).encode())
    h.update(str(x_cal_np.nbytes).encode())
    return h.hexdigest()[:16]


def get_calibration_summaries(
    explainer: CalibratedExplainer, x_cal_np: Optional[np.ndarray] = None
) -> Tuple[Dict[int, Dict[Any, int]], Dict[int, np.ndarray]]:
    """Return cached categorical counts and sorted numeric calibration values.

    Computes and caches statistical summaries of the calibration data:
    - For categorical features: unique value counts
    - For numeric features: sorted arrays of values

    These summaries enable efficient feature analysis during explanation generation.

    Caching Strategy (ADR-003):
    - If ExplanationCacheFacade is available, uses shared cache layer with versioning
    - Falls back to instance-level cache for backward compatibility
    - Cache is invalidated when calibration data changes via invalidate_calibration_summaries()

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
    Results are cached. Cache is invalidated via :func:`invalidate_calibration_summaries`
    when calibration data changes.
    """
    if x_cal_np is None:
        x_cal_np = np.asarray(explainer.x_cal)

    shape = getattr(x_cal_np, "shape", None)
    x_cal_hash = _get_calibration_data_hash(x_cal_np)
    explainer_id = str(id(explainer))

    # Try shared cache first (ADR-003)
    cache_facade = getattr(explainer, "_explanation_cache", None)
    if cache_facade is not None:
        cached = cache_facade.get_calibration_summaries(
            explainer_id=explainer_id, x_cal_hash=x_cal_hash
        )
        if cached is not None:
            cat_counts, num_sorted = cached
            return cat_counts, num_sorted

    # Check instance-level cache for backward compatibility
    if (
        explainer.categorical_value_counts_cache is not None
        and explainer.numeric_sorted_cache is not None
        and explainer.calibration_summary_shape == shape
    ):
        return explainer.categorical_value_counts_cache, explainer.numeric_sorted_cache

    # Compute summaries
    categorical_value_counts: Dict[int, Dict[Any, int]] = {}
    numeric_sorted_cache: Dict[int, np.ndarray] = {}

    if x_cal_np.size:
        categorical_features = tuple(int(f) for f in explainer.categorical_features)

        # Compute value counts for categorical features
        for f_cat in categorical_features:
            unique_vals, unique_counts = np.unique(x_cal_np[:, f_cat], return_counts=True)
            categorical_value_counts[int(f_cat)] = {
                val: int(cnt)
                for val, cnt in zip(unique_vals.tolist(), unique_counts.tolist(), strict=False)
            }

        # Sort numeric feature values for later use
        numeric_features = [
            f for f in range(explainer.num_features) if f not in categorical_features
        ]
        for f_num in numeric_features:
            numeric_sorted_cache[f_num] = np.sort(np.asarray(x_cal_np[:, f_num]))

    # Store in shared cache if available
    if cache_facade is not None:
        cache_facade.set_calibration_summaries(
            explainer_id=explainer_id,
            x_cal_hash=x_cal_hash,
            categorical_counts=categorical_value_counts,
            numeric_sorted=numeric_sorted_cache,
        )

    # Maintain instance-level cache for backward compatibility
    explainer.categorical_value_counts_cache = categorical_value_counts
    explainer.numeric_sorted_cache = numeric_sorted_cache
    explainer.calibration_summary_shape = shape

    assert explainer.categorical_value_counts_cache is not None
    assert explainer.numeric_sorted_cache is not None

    return explainer.categorical_value_counts_cache, explainer.numeric_sorted_cache
