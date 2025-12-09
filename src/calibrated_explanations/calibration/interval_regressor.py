# pylint: disable=invalid-name, line-too-long, too-many-instance-attributes
"""Interval regression helpers built on conformal calibration.

Exposes `IntervalRegressor`, which combines conformal predictive systems
and Venn-Abers scaling to deliver calibrated probabilities and intervals.

Part of ADR-001: Core Decomposition Boundaries (Stage 1a).
"""

import numbers
from functools import singledispatchmethod

import crepes
import numpy as np

from calibrated_explanations.core import ConfigurationError, DataShapeError

from ..utils import safe_first_element
from .venn_abers import VennAbers


class IntervalRegressor:
    """Estimate predictive intervals using calibrated explanations."""

    def __init__(self, calibrated_explainer):
        """Initialize the interval regressor with a calibrated explainer.

        Parameters
        ----------
        calibrated_explainer : CalibratedExplainer
            A fitted and calibrated explainer instance. The regressor extracts calibration
            data (features, targets, predictions, bins) from this explainer and builds
            conformal predictive systems for interval and probability estimation.

        Notes
        -----
        This initializer:
        - Extracts calibration predictions and computes residuals
        - Fits a conformal predictive system using the calibration data
        - Pre-splits calibration data for probabilistic (thresholded) regression
        - Initializes Venn-Abers calibrator for probability refinement
        """
        self.ce = calibrated_explainer
        self._bins_storage = None
        self._bins_size = 0

        # Normalize bins to 1D if provided
        initial_bins = calibrated_explainer.bins
        if initial_bins is not None:
            initial_bins = np.asarray(initial_bins)
            if initial_bins.ndim != 1:
                initial_bins = initial_bins.reshape(-1)
            self.bins = np.array(initial_bins, copy=True)
        else:
            self.bins = None

        self.model = self

        initial_y_cal_hat = np.asarray(self.ce.predict_calibration())
        if initial_y_cal_hat.ndim != 1:
            initial_y_cal_hat = initial_y_cal_hat.reshape(-1)
        self._y_cal_hat_storage = np.array(initial_y_cal_hat, copy=True)
        self._y_cal_hat_size = self._y_cal_hat_storage.shape[0]

        initial_residuals = np.asarray(self.ce.y_cal)
        if initial_residuals.ndim != 1:
            initial_residuals = initial_residuals.reshape(-1)
        initial_residuals = initial_residuals - self.y_cal_hat
        self._residual_cal_storage = np.array(initial_residuals, copy=True)
        self._residual_cal_size = self._residual_cal_storage.shape[0]

        sigma_cal = np.asarray(self.ce._get_sigma_test(x=self.ce.x_cal))  # pylint: disable=protected-access
        if sigma_cal.ndim != 1:
            sigma_cal = sigma_cal.reshape(-1)
        self._sigma_cal_storage = np.array(sigma_cal, copy=True)
        self._sigma_cal_size = self._sigma_cal_storage.shape[0]
        cps = crepes.ConformalPredictiveSystem()
        if self.ce.difficulty_estimator is not None:
            cps.fit(
                residuals=self.residual_cal,
                sigmas=self.sigma_cal,
                bins=self.bins,
                seed=self.ce.seed,
            )
        else:
            cps.fit(residuals=self.residual_cal, bins=self.bins, seed=self.ce.seed)
        self.cps = cps
        self.venn_abers = None
        self.proba_cal = None
        self.y_threshold = None
        self.current_y_threshold = None
        self.split = {}
        self.pre_fit_for_probabilistic()

    def _append_calibration_buffer(self, name, values):
        """Append new calibration values to the dynamic storage backing ``name``."""
        values = np.asarray(values)
        if values.size == 0:
            return
        if values.ndim != 1:
            values = values.reshape(-1)

        storage_attr = f"_{name}_storage"
        size_attr = f"_{name}_size"

        storage = getattr(self, storage_attr)
        size = getattr(self, size_attr)
        values = values.astype(storage.dtype, copy=False)

        storage = self._ensure_capacity(storage, size, values.size)
        storage[size : size + values.size] = values

        setattr(self, storage_attr, storage)
        setattr(self, size_attr, size + values.size)

    def _append_bins(self, values):
        """Append Mondrian bin assignments while reusing allocated storage."""
        values = np.asarray(values)
        if values.size == 0:
            return
        if values.ndim != 1:
            values = values.reshape(-1)

        if self._bins_storage is None:
            self._bins_storage = np.array(values, copy=True)
            self._bins_size = values.size
            return

        values = values.astype(self._bins_storage.dtype, copy=False)
        storage = self._ensure_capacity(self._bins_storage, self._bins_size, values.size)
        storage[self._bins_size : self._bins_size + values.size] = values
        self._bins_storage = storage
        self._bins_size += values.size

    @staticmethod
    def _ensure_capacity(storage, size, additional):
        """Grow ``storage`` so that ``size + additional`` entries fit without reallocation."""
        required = size + additional
        capacity = storage.shape[0]
        if capacity >= required:
            return storage

        new_capacity = max(required, max(1, capacity * 2))
        new_storage = np.empty(new_capacity, dtype=storage.dtype)
        if size:
            new_storage[:size] = storage[:size]
        return new_storage

    # pylint: disable=too-many-locals
    def predict_probability(self, x, y_threshold, bins=None):
        """Predict probabilistic regression probabilities with confidence intervals.

        Probabilistic regression (also called thresholded regression in the architecture layer)
        converts regression predictions into calibrated probabilities for a threshold event.
        This method returns the calibrated probability that y <= y_threshold, along with
        confidence intervals around that probability.

        Parameters
        ----------
        x
            x is a numpy.ndarray containing the instance objects for which we want to predict the
            probability.
        y_threshold
            The threshold value to evaluate. Returns the probability P(y <= y_threshold).
        bins
            array-like of shape (n_samples,), default=None
            Mondrian categories

        Returns
        -------
            four values: proba (y <= y_threshold), lower bound, upper bound, and None.
        """
        if bins is not None and self.bins is None:
            raise ConfigurationError(
                "Calibration bins must be assigned.",
                details={
                    "context": "predict",
                    "requirement": "calibration with bins or no test bins",
                },
            )

        n_samples = x.shape[0]
        normalized_bins = None
        if bins is not None:
            candidate = np.asarray(bins)
            if candidate.ndim == 0:
                candidate = np.repeat(candidate, n_samples)
            elif candidate.ndim > 1:
                candidate = candidate.reshape(-1)
            if candidate.shape[0] != n_samples:
                raise DataShapeError(
                    f"length of test bins ({candidate.shape[0]}) does not match number of test instances ({n_samples}).",
                    details={"bins_length": candidate.shape[0], "n_samples": n_samples},
                )  # pylint: disable=line-too-long
            normalized_bins = candidate.tolist()

        iter_bins = normalized_bins if normalized_bins is not None else [None] * n_samples

        self.y_threshold = y_threshold
        if np.isscalar(self.y_threshold) or isinstance(self.y_threshold, tuple):
            self.current_y_threshold = self.y_threshold
            self.compute_proba_cal(self.y_threshold)
            proba, low, high = self.split["va"].predict_proba(
                x,
                output_interval=True,
                bins=normalized_bins if normalized_bins is not None else None,
            )
            return proba[:, 1], low, high, None

        bins = iter_bins
        interval = np.zeros((x.shape[0], 2))
        proba = np.zeros(x.shape[0])
        for i, _ in enumerate(proba):
            self.current_y_threshold = self.y_threshold[i]
            self.compute_proba_cal(self.y_threshold[i])
            p, low, high = self.split["va"].predict_proba(
                x[i, :].reshape(1, -1), output_interval=True, bins=[bins[i]]
            )
            p = safe_first_element(p, col=1)
            low = safe_first_element(low)
            high = safe_first_element(high)
            proba[i] = p
            interval[i, :] = np.array([low, high])
        return proba, interval[:, 0], interval[:, 1], None

    def predict_uncertainty(self, x, low_high_percentiles, bins=None):
        """Predict the uncertainty of a given set of instances using a `ConformalPredictiveSystem`.

        Parameters
        ----------
        x
            x is a numpy array containing the instance objects for which we want to predict the
            uncertainty.
        low_high_percentiles
            The `low_high_percentiles` parameter is a list containing two values. The first value
            represents the lower percentile and the second value represents the higher percentile. These
            percentiles are used to calculate the prediction interval for the uncertainty estimation. If the
            first value is set to -np.inf (negative infinity), the interval will be one-sided and upper-bounded
            and if the second value is np.inf (infinity), the interval will be one-sided and lower-bounded.
        bins
            array-like of shape (n_samples,), default=None
            Mondrian categories

        Returns
        -------
            four values: median, lower bound, upper bound, and None.
        """
        y_test_hat = self.ce.predict_function(x).reshape(-1, 1)

        sigma_test = self.ce._get_sigma_test(x=x)  # pylint: disable=protected-access
        low = [low_high_percentiles[0], 50] if low_high_percentiles[0] != -np.inf else [50, 50]
        high = [low_high_percentiles[1], 50] if low_high_percentiles[1] != np.inf else [50, 50]

        interval = self.cps.predict(
            y_hat=y_test_hat,
            sigmas=sigma_test,
            lower_percentiles=low,
            higher_percentiles=high,
            bins=bins,
        )
        y_test_hat = (interval[:, 1] + interval[:, 3]) / 2  # The median
        return (
            y_test_hat,
            interval[:, 0]
            if low_high_percentiles[0] != -np.inf
            else np.tile(np.array([np.min(self.ce.y_cal)]), len(interval)),
            interval[:, 2]
            if low_high_percentiles[1] != np.inf
            else np.tile(np.array([np.max(self.ce.y_cal)]), len(interval)),
            None,
        )

    def predict_proba(self, x, bins=None):
        """Predict the probabilities for being below the y_threshold (for float threshold) or below the lower bound and above the upper bound (for tuple threshold).

        Parameters
        ----------
        x
            The x parameter is the input data for which you want to predict the probabilities. It
            should be a numpy array or a pandas DataFrame containing the features of the test data.
        bins
            array-like of shape (n_samples,), default=None
            Mondrian categories

        Returns
        -------
            a numpy array of shape (n_samples, 2), where each row represents the predicted probabilities
            for being above or below the y_threshold. The first column represents the probability of the
            negative class (1-proba) and the second column represents the probability of the positive class (proba).
        """
        y_test_hat = self.ce.predict_function(x).reshape(-1, 1)

        sigma_test = self.ce._get_sigma_test(x=x)  # pylint: disable=protected-access
        if isinstance(self.current_y_threshold, tuple):
            proba_lower = self.cps.predict(
                y_hat=y_test_hat, sigmas=sigma_test, y=self.current_y_threshold[0], bins=bins
            )
            proba_upper = self.cps.predict(
                y_hat=y_test_hat, sigmas=sigma_test, y=self.current_y_threshold[1], bins=bins
            )
            proba = proba_upper - proba_lower
        else:
            proba = self.cps.predict(
                y_hat=y_test_hat, sigmas=sigma_test, y=self.current_y_threshold, bins=bins
            )
        return np.array([[1 - proba[i], proba[i]] for i in range(len(proba))])

    def pre_fit_for_probabilistic(self):
        """Split the calibration set into two parts.

        The first part is used to fit the `ConformalPredictiveSystem` and the second part is used to
        calculate the probability calibration for a given threshold (at prediction time).
        """
        n = len(self.ce.y_cal)
        rng = np.random.default_rng(self.ce.seed)
        cal_parts = rng.permutation(n).tolist()
        self.split["parts"] = [cal_parts[: n // 2], cal_parts[n // 2 :]]
        cal_cps = self.split["parts"][0]
        self.split["cps"] = crepes.ConformalPredictiveSystem()
        if self.bins is None:
            self.split["cps"].fit(
                residuals=self.residual_cal[cal_cps],
                sigmas=self.sigma_cal[cal_cps],
                seed=self.ce.seed,
            )
        else:
            self.split["cps"].fit(
                residuals=self.residual_cal[cal_cps],
                sigmas=self.sigma_cal[cal_cps],
                bins=self.bins[cal_cps],
                seed=self.ce.seed,
            )

    @singledispatchmethod
    def compute_proba_cal(self, y_threshold):
        """Validate threshold types before computing probability calibration.

        Parameters
        ----------
            y_threshold : float or tuple
                Threshold defining the calibration target event.
        """
        raise TypeError("y_threshold must be a float or a tuple.")

    @compute_proba_cal.register(numbers.Real)
    def _(self, y_threshold: numbers.Real):
        """Compute the probability calibration for a scalar threshold.

        Parameters
        ----------
            y_threshold : float
                Threshold value that defines the calibration target.
        """
        cal_va = self.split["parts"][1]
        bins = None if self.bins is None else self.bins[cal_va]
        proba = self.split["cps"].predict(
            y_hat=self.y_cal_hat[cal_va], y=y_threshold, sigmas=self.sigma_cal[cal_va], bins=bins
        )
        self.split["proba"] = np.array([[1 - proba[i], proba[i]] for i in range(len(proba))])
        self.split["va"] = VennAbers(
            None,
            (self.ce.y_cal[cal_va] <= y_threshold).astype(int),
            self,
            bins=bins,
            cprobs=self.split["proba"],
        )

    @compute_proba_cal.register(tuple)
    def _(self, y_threshold: tuple):
        """Compute the probability calibration for an interval threshold.

        Parameters
        ----------
            y_threshold : tuple
                Lower and upper bounds that define the calibration target interval.
        """
        cal_va = self.split["parts"][1]
        bins = None if self.bins is None else self.bins[cal_va]
        proba_lower = self.split["cps"].predict(
            y_hat=self.y_cal_hat[cal_va], y=y_threshold[0], sigmas=self.sigma_cal[cal_va], bins=bins
        )
        proba_upper = self.split["cps"].predict(
            y_hat=self.y_cal_hat[cal_va], y=y_threshold[1], sigmas=self.sigma_cal[cal_va], bins=bins
        )
        proba = proba_upper - proba_lower
        self.split["proba"] = np.array([[1 - proba[i], proba[i]] for i in range(len(proba))])
        self.split["va"] = VennAbers(
            None,
            (
                (y_threshold[0] < self.ce.y_cal[cal_va]) & (self.ce.y_cal[cal_va] <= y_threshold[1])
            ).astype(int),
            self,
            bins=bins,
            cprobs=self.split["proba"],
        )

    def insert_calibration(self, xs, ys, bins=None):
        """Insert calibration instances while preserving the conformal splits.

        Parameters
        ----------
            xs : ndarray
                New calibration features.
            ys : ndarray
                New calibration targets.
            bins : array-like, optional
                Mondrian categories associated with the new instances.
        """
        num_add = len(ys)  # number of new instances
        if num_add % 2 != 0:  # is odd?
            parts = self.split["parts"]
            small_part = int(np.argmin([len(parts[0]), len(parts[1])]))
            large_part = int(np.argmax([len(parts[0]), len(parts[1])]))
            large_part = 1 if small_part == large_part else large_part
        else:  # divide equally
            small_part = 0
            large_part = 1
        small_idx = list(range(0, num_add, 2))  # if odd, one more to the smaller part
        large_idx = list(range(1, num_add, 2))

        # Update split indices
        if len(small_idx) > 0:
            self.split["parts"][small_part].extend(
                [len(self.residual_cal) - 1 + i for i in small_idx]
            )
        if len(large_idx) > 0:
            self.split["parts"][large_part].extend(
                [len(self.residual_cal) - 1 + i for i in large_idx]
            )

        # Update y_hat, residuals, and sigma
        y_cal_hat = self.ce.predict_function(xs)
        residuals = ys - y_cal_hat
        sigmas = self.ce._get_sigma_test(x=xs)  # pylint: disable=protected-access
        self._append_calibration_buffer("y_cal_hat", y_cal_hat)
        self._append_calibration_buffer("residual_cal", residuals)
        self._append_calibration_buffer("sigma_cal", sigmas)

        # Update bins
        if bins is not None:
            if self.bins is None:
                raise ConfigurationError(
                    "Cannot mix calibration instances with and without bins.",
                    details={
                        "context": "add_calibration_instances",
                        "requirement": "consistent bin usage",
                    },
                )
            if len(bins) != len(ys):
                raise DataShapeError(
                    f"length of bins ({len(bins)}) does not match number of added instances ({len(ys)}).",
                    details={"bins_length": len(bins), "n_instances": len(ys)},
                )  # pylint: disable=line-too-long
            self._append_bins(bins)

        if small_part == 0 or len(large_idx) > 0:  # add to cps calibration
            cps_idx = small_idx if small_part == 0 else large_idx
            # Update split cps
            if bins is None:
                alphas = self.split["cps"].alphas
                indices = np.searchsorted(alphas, residuals[cps_idx])
                self.split["cps"].alphas = np.insert(alphas, indices, residuals[cps_idx])
            else:
                for b in np.unique(bins):
                    alphas = self.split["cps"].alphas[1][b]
                    res = residuals[cps_idx]
                    indices = np.searchsorted(alphas, res[bins == b])
                    self.split["cps"].alphas[1][b] = np.insert(alphas, indices, res[bins == b])

        # Update cps
        if bins is None:
            alphas = self.cps.alphas
            indices = np.searchsorted(alphas, residuals)
            self.cps.alphas = np.insert(alphas, indices, residuals)
        else:
            for b in np.unique(bins):
                alphas = self.cps.alphas[1][b]
                indices = np.searchsorted(alphas, residuals[bins == b])
                self.cps.alphas[1][b] = np.insert(alphas, indices, residuals[bins == b])

    @property
    def y_cal_hat(self):
        """Predicted calibration targets."""
        return self._y_cal_hat_storage[: self._y_cal_hat_size]

    @property
    def residual_cal(self):
        """Calibration residuals reused by conformal updates."""
        return self._residual_cal_storage[: self._residual_cal_size]

    @property
    def sigma_cal(self):
        """Calibration difficulty estimates."""
        return self._sigma_cal_storage[: self._sigma_cal_size]

    @property
    def bins(self):
        """Return the Mondrian bins associated with the calibration data."""
        if self._bins_storage is None:
            return None
        return self._bins_storage[: self._bins_size]

    @bins.setter
    def bins(self, value):
        """Assign Mondrian bin categories to the calibration data.

        Parameters
        ----------
        value : array-like or None
            Mondrian categories for each calibration instance. If None, clears bin assignments.
            Otherwise, must be a 1D array matching the number of calibration instances.
        """
        if value is None:
            self._bins_storage = None
            self._bins_size = 0
            return

        value = np.asarray(value)
        if value.ndim != 1:
            value = value.reshape(-1)
        self._bins_storage = np.array(value, copy=True)
        self._bins_size = self._bins_storage.shape[0]
