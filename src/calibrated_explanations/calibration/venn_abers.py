# ruff: noqa: N999
# pylint: disable=unknown-option-value
# pylint: disable=invalid-name, line-too-long, too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, fixme
"""Venn-Abers calibration utilities for post-processing model probabilities.

Wraps the `venn_abers` package to offer Mondrian-aware calibration
integrated with the calibrated explanations toolkit.

Part of ADR-001: Core Decomposition Boundaries (Stage 1a).
"""

import warnings

import numpy as np
import venn_abers as va

from ..core.exceptions import ConfigurationError
from ..utils import convert_targets_to_numeric


class VennAbers:
    """Calibrate probabilistic predictions with the Venn-Abers method.

    Parameters
    ----------
        x_cal : array-like
            Calibration feature set used to fit the post-hoc model.
        y_cal : array-like
            Calibration target values.
        learner : object
            Estimator exposing a `predict_proba` method.
        bins : array-like, optional
            Mondrian categories associated with the calibration data.
        cprobs : array-like, optional
            Pre-computed calibration probabilities.
        difficulty_estimator : callable, optional
            Callable that scores sample difficulty.
        predict_function : callable, optional
            Custom probability function overriding `learner.predict_proba`.

    Attributes
    ----------
        de : callable or None
            Difficulty estimator applied to inputs.
        learner : object
            Base estimator used for predictions.
        x_cal : array-like
            Calibration feature set.
        ctargets : ndarray
            Numeric calibration targets.
        cprobs : ndarray
            Calibration probabilities for each sample.
        bins : array-like or None
            Mondrian categories used during calibration.
        va : dict or venn_abers.VennAbers
            Underlying Venn-Abers models fitted per class or bin.
    """

    def __init__(
        self,
        x_cal,
        y_cal,
        learner,
        bins=None,
        cprobs=None,
        difficulty_estimator=None,
        predict_function=None,
    ):
        """Initialize the VennAbers calibrator.

        Parameters
        ----------
            x_cal : array-like
                Calibration feature set.
            y_cal : array-like
                Calibration target values.
            learner : object
                Estimator exposing a `predict_proba` method.
            bins : array-like, optional
                Mondrian categories associated with calibration data.
            cprobs : array-like, optional
                Pre-computed calibration probabilities.
            difficulty_estimator : callable, optional
                Callable that scores sample difficulty.
            predict_function : callable, optional
                Custom function used instead of `learner.predict_proba`.
        """
        self.y_cal_numeric, self.label_map = convert_targets_to_numeric(y_cal)
        self.original_labels = y_cal

        self.de = difficulty_estimator
        self.learner = learner
        self._predict_proba = (
            predict_function if predict_function is not None else learner.predict_proba
        )
        self.x_cal = x_cal
        self.__is_multiclass = len(np.unique(self.y_cal_numeric)) > 2

        cprobs = self.__predict_proba_with_difficulty(x_cal) if cprobs is None else cprobs
        self.cprobs = cprobs
        self.bins = bins

        self.ctargets = self.y_cal_numeric

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if self.is_mondrian():
            self.va = {}
            if self.is_multiclass():
                tmp_probs = np.zeros((cprobs.shape[0], 2))
                for c in np.unique(self.ctargets):
                    self.va[c] = {}
                    tmp_probs[:, 0] = 1 - cprobs[:, c]
                    tmp_probs[:, 1] = cprobs[:, c]
                    for b in np.unique(self.bins):
                        va_class_bin = va.VennAbers()
                        va_class_bin.fit(
                            tmp_probs[self.bins == b, :],
                            np.multiply(c == self.ctargets[self.bins == b], 1),
                            precision=4,
                        )
                        self.va[c][b] = va_class_bin
            else:
                for b in np.unique(self.bins):
                    va_bin = va.VennAbers()
                    va_bin.fit(
                        cprobs[self.bins == b, :], self.ctargets[self.bins == b], precision=4
                    )
                    self.va[b] = va_bin
        elif self.is_multiclass():
            self.va = {}
            tmp_probs = np.zeros((cprobs.shape[0], 2))
            for c in np.unique(self.ctargets):
                tmp_probs[:, 0] = 1 - cprobs[:, c]
                tmp_probs[:, 1] = cprobs[:, c]
                va_class = va.VennAbers()
                va_class.fit(tmp_probs, np.multiply(c == self.ctargets, 1), precision=4)
                self.va[c] = va_class
        else:
            self.va = va.VennAbers()
            self.va.fit(cprobs, self.ctargets, precision=4)
        warnings.filterwarnings("default", category=RuntimeWarning)

    def __predict_proba_with_difficulty(self, x, bins=None):
        """Augment raw probabilities with optional difficulty adjustments."""
        if "bins" in self._predict_proba.__code__.co_varnames:
            probs = self._predict_proba(x, bins=bins)
        else:
            probs = self._predict_proba(x)
        if self.de is not None:
            difficulty = self.de.apply(x)
            # method = logit_based_scaling_list
            method = exponent_scaling_list
            # method = sigmoid_scaling_list
            if self.is_multiclass():
                probs_tmp = method(probs, difficulty)
            else:
                probs_tmp = method(probs, np.repeat(difficulty, 2).reshape(-1, 2))
            probs = np.array([np.asarray(tmp) for tmp in probs_tmp])
        return probs

    def predict(self, x, bins=None):
        """Predict the class of the test samples.

        Parameters
        ----------
            x (n_test_samples, n_features): Test samples.
            bins (array-like of shape (n_samples,), optional): Mondrian categories.

        Returns
        -------
            ndarray: Predicted classes based on the regularized VennAbers probabilities.
                If multiclass, the predicted class is 1 if the prediction from the underlying model is the same after calibration and 0 otherwise.
        """
        if self.is_multiclass():
            tmp, _ = self.predict_proba(x, bins=bins)
            return np.asarray(np.round(tmp[:, 1]))
        tmp = self.predict_proba(x, bins=bins)[:, 1]
        return np.asarray(np.round(tmp))

    # pylint: disable=too-many-locals, too-many-branches
    def predict_proba(self, x, output_interval=False, classes=None, bins=None):
        """Predict the probabilities of the test samples, optionally outputting the VennAbers interval.

        Parameters
        ----------
            x (n_test_samples, n_features): Test samples.
            output_interval (bool, optional): If true, the VennAbers intervals are outputted. Defaults to False.
            classes (array-like, optional): A list of predicted classes. Defaults to None.
            bins (array-like of shape (n_samples,), optional): Mondrian categories.

        Returns
        -------
            ndarray: Regularized VennAbers probabilities for the test samples.
            If output_interval is true, the VennAbers intervals are also returned:
                low (n_test_samples,): Lower bounds of the VennAbers interval for each test sample.
                high (n_test_samples,): Upper bounds of the VennAbers interval for each test sample.
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        tprobs = self.__predict_proba_with_difficulty(x, bins=bins)
        p0p1 = np.zeros((tprobs.shape[0], 2))
        va_proba = np.zeros(tprobs.shape)

        if self.is_multiclass():
            low, high = np.zeros(tprobs.shape), np.zeros(tprobs.shape)
            tmp_probs = np.zeros((tprobs.shape[0], 2))
            for c, va_class in self.va.items():
                tmp_probs[:, 0] = 1 - tprobs[:, c]
                tmp_probs[:, 1] = tprobs[:, c]
                if self.is_mondrian():
                    if bins is None:
                        raise ConfigurationError("Mondrian calibration: bins must be provided if Mondrian.", details={"context": "predict_proba", "requirement": "bins parameter"})
                    for b, va_class_bin in va_class.items():
                        p0p1[bins == b, :] = va_class_bin.predict_proba(tmp_probs[bins == b, :])[1]
                else:
                    p0p1 = va_class.predict_proba(tmp_probs)[1]
                low[:, c], high[:, c] = p0p1[:, 0], p0p1[:, 1]
                tmp = high[:, c] / (1 - low[:, c] + high[:, c])
                va_proba[:, c] = tmp
            # TODO: Surprisingly, probability normalization is needed, needs looking into
            row_sums = va_proba.sum(axis=1, keepdims=True)
            # Guard against divide-by-zero for degenerate rows.
            safe_row_sums = np.where(row_sums == 0, 1.0, row_sums)
            va_proba = va_proba / safe_row_sums
            low = low / safe_row_sums
            high = high / safe_row_sums
            if classes is not None:
                if type(classes) not in (list, np.ndarray):
                    classes = [classes]
                if output_interval:
                    return (
                        np.asarray(va_proba),
                        [low[i, c] for i, c in enumerate(classes)],
                        [high[i, c] for i, c in enumerate(classes)],
                        classes,
                    )
                return np.asarray(va_proba), classes
            classes = np.argmax(va_proba, axis=1)
            if output_interval:
                return np.asarray(va_proba), low, high, classes
            return np.asarray(va_proba), classes

        if self.is_mondrian():
            if bins is None:
                raise ConfigurationError("Mondrian calibration: bins must be provided if Mondrian.", details={"context": "predict_proba", "requirement": "bins parameter"})
            for b, va_bin in self.va.items():
                p0p1[bins == b, :] = va_bin.predict_proba(tprobs[bins == b, :])[1]
        else:
            _, p0p1 = self.va.predict_proba(tprobs)
        low, high = p0p1[:, 0], p0p1[:, 1]
        tmp = high / (1 - low + high)
        va_proba[:, 0] = 1 - tmp
        va_proba[:, 1] = tmp
        # binary
        warnings.filterwarnings("default", category=RuntimeWarning)
        if output_interval:
            return np.asarray(va_proba), low, high
        return np.asarray(va_proba)

    def is_multiclass(self) -> bool:
        """Return true if the problem is multiclass.

        Returns
        -------
            bool: True if more than two classes.
        """
        return self.__is_multiclass

    def is_mondrian(self) -> bool:
        """Return true if Mondrian categories are used.

        Returns
        -------
            bool: True if Mondrian.
        """
        return self.bins is not None


def exponent_scaling_list(probs, difficulties, beta=5):
    """
    Exponentially scale a list of probabilities towards 0/1 for low difficulty, and towards 0.5 for high difficulty.

    Parameters
    ----------
        probs (list of float): List of predicted probabilities (between 0 and 1).
        difficulties (list of float): List of difficulties (0 = easy, 1 = hard).
        beta (float): Scaling factor to control the effect of difficulty (default is 5).

    Returns
    -------
        list of float: Scaled probabilities.
    """
    scaled_probs = []
    for p, difficulty in zip(probs, difficulties):
        if p[0] < 0.5:
            scaled_p = p ** (1 + beta * (1 - difficulty))
        else:
            scaled_p = 1 - (1 - p) ** (1 + beta * (1 - difficulty))

        final_scaled_p = (1 - difficulty) * scaled_p + difficulty * 0.5
        final_scaled_p = final_scaled_p / np.sum(final_scaled_p)
        scaled_probs.append(final_scaled_p)

    return scaled_probs
