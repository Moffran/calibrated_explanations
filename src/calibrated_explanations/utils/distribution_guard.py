# pylint: disable=too-many-arguments, too-many-positional-arguments, line-too-long
"""In-distribution guard for calibrated explanations.

Provides :class:`InDistributionGuard` – a KNN-based conformal non-conformity
filter used by the guarded explanation methods to prune out-of-distribution
perturbations before they influence explanation generation.

The guard implements the standard conformal anomaly detection framework
(Shafer & Vovk, 2008):

    alpha(x) = distance from x to its k-th nearest calibration neighbour

    p_value(x) = |{cal_i : alpha(cal_i) >= alpha(x)}| / n_cal

An instance is considered *conforming* when ``p_value(x) >= significance``.
"""

from __future__ import annotations

import numpy as np


class InDistributionGuard:
    """Test whether perturbed instances conform to the calibration distribution.

    Uses a KNN-based non-conformity measure: the distance from a test point
    to its k-th nearest calibration neighbour is its non-conformity score.
    A conformal p-value is then derived by comparing that score against the
    empirical distribution of calibration scores (leave-one-out on the cal
    set).  Instances with ``p_value < significance`` are flagged as OOD.

    Parameters
    ----------
    x_cal : array-like of shape (n_cal, n_features)
        Calibration feature matrix.
    n_neighbors : int, default=5
        Number of nearest neighbours used for the non-conformity measure.
    metric : str or callable, default='euclidean'
        Distance metric passed to ``sklearn.neighbors.NearestNeighbors``.
    leaf_strategy : {'median', 'percentiles'}, default='median'
        How to pick representative values from a discretiser leaf when
        testing conformity of an entire bin.

        ``'median'``
            Use only the median of the calibration samples in the leaf.
            Fastest; recommended for production use.

        ``'percentiles'``
            Use the 25th, 50th, and 75th percentile values.
            More thorough but ~3× slower per leaf.

    normalize : bool, default=True
        Apply per-feature min-max normalisation before computing distances
        to prevent features with large numeric ranges from dominating the
        metric.
    """

    def __init__(
        self,
        x_cal: np.ndarray,
        *,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        leaf_strategy: str = "median",
        normalize: bool = True,
    ) -> None:
        self.x_cal = np.asarray(x_cal, dtype=float)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.leaf_strategy = leaf_strategy
        self.normalize = normalize
        self._scale_min: np.ndarray | None = None
        self._scale_range: np.ndarray | None = None
        self._fit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit(self) -> None:
        """Fit KNN on calibration set and precompute per-point scores."""
        from sklearn.neighbors import NearestNeighbors  # local to avoid import at module level

        x = self._scale(self.x_cal)
        n_cal = x.shape[0]
        # k+1 so the point itself (distance 0) can be excluded from the k-th neighbour
        k_actual = min(self.n_neighbors + 1, n_cal)
        self._knn = NearestNeighbors(
            n_neighbors=k_actual, metric=self.metric, algorithm="auto"
        ).fit(x)

        dists, _ = self._knn.kneighbors(x)
        # LOO non-conformity: distance to the k-th *other* calibration point.
        # Index 0 is self (dist 0); the k-th other point is at column k (1-based).
        score_col = min(self.n_neighbors, k_actual - 1)
        self._cal_scores = dists[:, score_col]

    def _scale(self, x: np.ndarray) -> np.ndarray:
        """Return (optionally) min-max normalised version of *x*."""
        if not self.normalize:
            return x
        if self._scale_min is None:
            self._scale_min = self.x_cal.min(axis=0)
            self._scale_range = np.clip(
                self.x_cal.max(axis=0) - self._scale_min, 1e-12, None
            )
        return (x - self._scale_min) / self._scale_range

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def nonconformity_scores(self, x_test: np.ndarray) -> np.ndarray:
        """Compute raw non-conformity scores for a batch of test instances.

        Parameters
        ----------
        x_test : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Distance to the k-th nearest calibration neighbour.
        """
        x = self._scale(np.asarray(x_test, dtype=float))
        k_actual = self._knn.n_neighbors
        # For test instances, we do NOT skip the zeroth column (no self-neighbour)
        score_col = min(self.n_neighbors - 1, k_actual - 1)
        dists, _ = self._knn.kneighbors(x)
        return dists[:, score_col]

    def p_values(self, x_test: np.ndarray) -> np.ndarray:
        """Return conformal p-values for each test instance.

        ``p_value(x) = |{cal_i : alpha(cal_i) >= alpha(x)}| / n_cal``

        Parameters
        ----------
        x_test : array-like of shape (n_samples, n_features)

        Returns
        -------
        p_vals : np.ndarray of shape (n_samples,)
        """
        test_scores = self.nonconformity_scores(np.asarray(x_test, dtype=float))
        # Vectorised comparison: (n_cal, 1) >= (1, n_test) -> (n_cal, n_test)
        return np.mean(
            self._cal_scores[:, np.newaxis] >= test_scores[np.newaxis, :], axis=0
        )

    def is_conforming(
        self, x_test: np.ndarray, significance: float = 0.1
    ) -> np.ndarray:
        """Return a boolean mask indicating which test instances are in-distribution.

        Parameters
        ----------
        x_test : array-like of shape (n_samples, n_features)
        significance : float, default=0.1
            Instances with ``p_value < significance`` are flagged as OOD.

        Returns
        -------
        conforming : np.ndarray of bool, shape (n_samples,)
        """
        return self.p_values(np.asarray(x_test, dtype=float)) >= significance

    def leaf_conforming_fraction(
        self,
        feature_idx: int,
        lower: float,
        upper: float,
        x_instance: np.ndarray,
        significance: float = 0.1,
    ) -> tuple[float, np.ndarray]:
        """Test a single discretiser leaf for in-distribution conformity.

        Picks representative feature values from the calibration samples that
        fall inside the leaf ``(lower, upper]``, builds perturbed instances by
        swapping only ``feature_idx`` while keeping all other features at their
        original values, then tests conformity.

        Parameters
        ----------
        feature_idx : int
            Feature being varied.
        lower : float
            Lower bound of the leaf (may be ``-np.inf``).
        upper : float
            Upper bound of the leaf (may be ``np.inf``).
        x_instance : np.ndarray of shape (n_features,)
            Original instance; all features except *feature_idx* stay fixed.
        significance : float

        Returns
        -------
        fraction : float
            Fraction of representative values whose perturbed instances are
            conforming.  ``0.0`` when the leaf has no calibration samples.
        representative_values : np.ndarray
            The candidate values sampled from the leaf.
        """
        feat_vals = self.x_cal[:, feature_idx]

        if lower == -np.inf and upper == np.inf:
            in_leaf = np.ones(len(feat_vals), dtype=bool)
        elif lower == -np.inf:
            in_leaf = feat_vals <= upper
        elif upper == np.inf:
            in_leaf = feat_vals > lower
        else:
            in_leaf = (feat_vals > lower) & (feat_vals <= upper)

        cal_in_leaf = feat_vals[in_leaf]
        if cal_in_leaf.size == 0:
            return 0.0, np.array([])

        if self.leaf_strategy == "median":
            candidates = np.array([np.median(cal_in_leaf)])
        else:  # percentiles
            candidates = np.unique(np.percentile(cal_in_leaf, [25, 50, 75]))

        x_pert = np.tile(x_instance, (len(candidates), 1))
        x_pert[:, feature_idx] = candidates
        conforming_mask = self.is_conforming(x_pert, significance)
        return float(conforming_mask.mean()), candidates
