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
A larger ``significance`` yields a stricter test (fewer instances accepted).
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
    A larger ``significance`` yields a stricter test (fewer instances accepted).

    Parameters
    ----------
    x_cal : array-like of shape (n_cal, n_features)
        Calibration feature matrix.
    n_neighbors : int, default=5
        Number of nearest neighbours used for the non-conformity measure.
    metric : str or callable, default='euclidean'
        Distance metric passed to ``sklearn.neighbors.NearestNeighbors``.
    categorical_features : sequence of int, optional
        Column indices that should be treated as categorical.

        These columns are one-hot encoded before computing KNN distances.
        This avoids imposing an arbitrary geometry on integer-encoded
        category IDs.

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
        normalize: bool = True,
        categorical_features: np.ndarray | list[int] | tuple[int, ...] | None = None,
    ) -> None:
        self.x_cal = np.asarray(x_cal)
        if int(n_neighbors) < 1:
            raise ValueError("n_neighbors must be >= 1")
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.normalize = normalize
        requested_categorical = (
            tuple(int(v) for v in categorical_features)
            if categorical_features is not None
            else tuple()
        )
        n_features = int(self.x_cal.shape[1])
        cat = sorted({i for i in requested_categorical if 0 <= i < n_features})
        self.categorical_features = tuple(cat)
        cat_set = set(cat)
        self._numeric_features = tuple(i for i in range(n_features) if i not in cat_set)
        self._scale_min: np.ndarray | None = None
        self._scale_range: np.ndarray | None = None
        self._ohe = None
        self._fit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit(self) -> None:
        """Fit KNN on calibration set and precompute per-point scores."""
        from sklearn.neighbors import NearestNeighbors  # local to avoid import at module level

        x = self._preprocess(self.x_cal)
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

    def _preprocess(self, x: np.ndarray) -> np.ndarray:
        """Preprocess data into numeric vectors suitable for distance computations."""
        x_arr = np.asarray(x)
        if x_arr.ndim != 2:
            raise ValueError("x must be a 2D array")
        if x_arr.shape[1] != self.x_cal.shape[1]:
            raise ValueError(
                "x must have the same number of features as x_cal "
                f"(got {x_arr.shape[1]} vs {self.x_cal.shape[1]})"
            )

        if not self.categorical_features:
            return self._scale(np.asarray(x_arr, dtype=float))

        num = list(self._numeric_features)
        cat = list(self.categorical_features)

        if num:
            x_num = self._scale(np.asarray(x_arr[:, num], dtype=float))
        else:
            x_num = np.zeros((x_arr.shape[0], 0), dtype=float)

        if cat:
            from sklearn.preprocessing import OneHotEncoder  # local import

            x_cat_raw = np.asarray(x_arr[:, cat], dtype=object)
            if self._ohe is None:
                try:
                    self._ohe = OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        dtype=float,
                    )
                except TypeError:  # pragma: no cover
                    self._ohe = OneHotEncoder(
                        handle_unknown="ignore",
                        sparse=False,
                        dtype=float,
                    )
                self._ohe.fit(x_cat_raw)
            x_cat = self._ohe.transform(x_cat_raw)
        else:
            x_cat = np.zeros((x_arr.shape[0], 0), dtype=float)

        return np.hstack([x_num, x_cat])

    def _scale(self, x: np.ndarray) -> np.ndarray:
        """Return (optionally) min-max normalised version of *x*."""
        if not self.normalize:
            return x
        if x.size == 0:
            return x
        if self._scale_min is None:
            if self._numeric_features:
                x_cal = np.asarray(self.x_cal[:, self._numeric_features], dtype=float)
            else:
                x_cal = np.asarray(self.x_cal, dtype=float)
            self._scale_min = x_cal.min(axis=0)
            self._scale_range = np.clip(x_cal.max(axis=0) - self._scale_min, 1e-12, None)
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
        x = self._preprocess(x_test)
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
        test_scores = self.nonconformity_scores(x_test)
        # Vectorised comparison: (n_cal, 1) >= (1, n_test) -> (n_cal, n_test)
        return np.mean(self._cal_scores[:, np.newaxis] >= test_scores[np.newaxis, :], axis=0)

    def is_conforming(self, x_test: np.ndarray, significance: float = 0.1) -> np.ndarray:
        """Return a boolean mask indicating which test instances are in-distribution.

        Parameters
        ----------
        x_test : array-like of shape (n_samples, n_features)
        significance : float, default=0.1
            Acceptable false-OOD rate.  Instances with
            ``p_value < significance`` are flagged as OOD.  A larger
            value yields a stricter test (fewer instances accepted).

        Returns
        -------
        conforming : np.ndarray of bool, shape (n_samples,)
        """
        if not 0 < float(significance) < 1:
            raise ValueError("significance must be strictly between 0 and 1")
        return self.p_values(x_test) >= significance
