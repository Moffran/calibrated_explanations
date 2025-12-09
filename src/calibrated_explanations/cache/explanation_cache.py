"""Explanation-layer caching facade for explanation artifacts.

This module provides a thin wrapper over CalibratorCache that exposes
explanation-specific stages (calibration summaries, attribution tensors,
feature name caches) while delegating to the canonical cache layer.

Part of ADR-001 (boundary layers) and ADR-003 (caching strategy).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .cache import CalibratorCache

logger = logging.getLogger(__name__)


class ExplanationCacheFacade:
    """Thin wrapper over CalibratorCache for explanation-specific stages.

    This facade defines and routes explain-stage artifacts (calibration summaries,
    attribution tensors, feature names) to the shared cache layer with appropriate
    namespacing and versioning, keeping explain logic grouped while preserving
    ADR-001's physical boundary and ADR-003's unified cache semantics.

    Attributes
    ----------
    cache : CalibratorCache or None
        The underlying shared cache instance (None if caching is disabled).
    """

    # Stage identifiers for explanation artifacts
    STAGE_CALIBRATION_SUMMARIES = "explain:calibration_summaries"
    STAGE_FEATURE_NAMES = "explain:feature_names"
    STAGE_ATTRIBUTION_TENSORS = "explain:attribution_tensors"

    def __init__(self, cache: CalibratorCache[Any] | None) -> None:
        """Initialize the facade with a CalibratorCache instance.

        Parameters
        ----------
        cache : CalibratorCache or None
            The shared cache instance, or None if caching is disabled.
        """
        self.cache = cache

    @property
    def enabled(self) -> bool:
        """Return True when the underlying cache is enabled."""
        return self.cache is not None and self.cache.enabled

    def get_calibration_summaries(
        self, *, explainer_id: str, x_cal_hash: str
    ) -> Optional[Tuple[Dict[int, Dict[Any, int]], Dict[int, np.ndarray]]]:
        """Retrieve cached calibration summaries (categorical counts + sorted numerics).

        Parameters
        ----------
        explainer_id : str
            Unique identifier for the explainer instance.
        x_cal_hash : str
            Hash of the calibration feature matrix shape/dtype.

        Returns
        -------
        tuple of (dict, dict) or None
            Tuple of (categorical_counts, numeric_sorted) if cached and not expired,
            None otherwise.
        """
        if self.cache is None:
            return None
        return self.cache.get(
            stage=self.STAGE_CALIBRATION_SUMMARIES,
            parts=(explainer_id, x_cal_hash),
        )

    def set_calibration_summaries(
        self,
        *,
        explainer_id: str,
        x_cal_hash: str,
        categorical_counts: Dict[int, Dict[Any, int]],
        numeric_sorted: Dict[int, np.ndarray],
    ) -> None:
        """Store computed calibration summaries in the cache.

        Parameters
        ----------
        explainer_id : str
            Unique identifier for the explainer instance.
        x_cal_hash : str
            Hash of the calibration feature matrix shape/dtype.
        categorical_counts : dict
            Mapping from feature index to dict of value counts.
        numeric_sorted : dict
            Mapping from feature index to sorted numeric values.
        """
        if self.cache is None:
            return
        self.cache.set(
            stage=self.STAGE_CALIBRATION_SUMMARIES,
            parts=(explainer_id, x_cal_hash),
            value=(categorical_counts, numeric_sorted),
        )

    def compute_calibration_summaries(
        self,
        *,
        explainer_id: str,
        x_cal_hash: str,
        compute_fn: Callable[[], Tuple[Dict[int, Dict[Any, int]], Dict[int, np.ndarray]]],
    ) -> Tuple[Dict[int, Dict[Any, int]], Dict[int, np.ndarray]]:
        """Return cached summaries or compute and cache them.

        Parameters
        ----------
        explainer_id : str
            Unique identifier for the explainer instance.
        x_cal_hash : str
            Hash of the calibration feature matrix shape/dtype.
        compute_fn : callable
            Function that returns (categorical_counts, numeric_sorted).

        Returns
        -------
        tuple of (dict, dict)
            Calibration summaries.
        """
        if self.cache is None:
            return compute_fn()
        return self.cache.compute(
            stage=self.STAGE_CALIBRATION_SUMMARIES,
            parts=(explainer_id, x_cal_hash),
            fn=compute_fn,
        )

    def get_feature_names_cache(self, *, explainer_id: str) -> Optional[Tuple[str, ...]]:
        """Retrieve cached feature names if available.

        Parameters
        ----------
        explainer_id : str
            Unique identifier for the explainer instance.

        Returns
        -------
        tuple of str or None
            Cached feature names, or None.
        """
        if self.cache is None:
            return None
        return self.cache.get(
            stage=self.STAGE_FEATURE_NAMES,
            parts=(explainer_id,),
        )

    def set_feature_names_cache(self, *, explainer_id: str, feature_names: Tuple[str, ...]) -> None:
        """Store feature names in the cache for quick reuse.

        Parameters
        ----------
        explainer_id : str
            Unique identifier for the explainer instance.
        feature_names : tuple of str
            The feature names to cache.
        """
        if self.cache is None:
            return
        self.cache.set(
            stage=self.STAGE_FEATURE_NAMES,
            parts=(explainer_id,),
            value=feature_names,
        )

    def invalidate_all(self) -> None:
        """Manually invalidate all explanation-layer caches."""
        if self.cache is not None:
            self.cache.flush()
            logger.debug("Invalidated all explanation-layer caches")

    def reset_version(self, new_version: str) -> None:
        """Reset the cache version to invalidate explanation artifacts.

        Parameters
        ----------
        new_version : str
            New version tag (e.g., "explain_v2").
        """
        if self.cache is not None:
            self.cache.reset_version(new_version)


__all__ = ["ExplanationCacheFacade"]
