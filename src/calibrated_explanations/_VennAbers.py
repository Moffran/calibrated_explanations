# pylint: disable=unknown-option-value
# pylint: disable=invalid-name, line-too-long, too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, fixme
"""
This module contains the VennAbers class for calibrating model predictions using the Venn-Abers method.

Classes
-------
    VennAbers: A class to calibrate the predictions of a model using the Venn-Abers method.
"""

import warnings

import numpy as np
import venn_abers as va

from .utils.helper import convert_targets_to_numeric


class VennAbers:
    """
    A class to calibrate the predictions of a model using the Venn-Abers method.

    Attributes
    ----------
        de (callable): A difficulty estimator function.
        learner (object): A machine learning model with a `predict_proba` method.
        X_cal (array-like): Calibration feature set.
        ctargets (array-like): Calibration target values.
        __is_multiclass (bool): Indicates if the problem is multiclass.
        cprobs (array-like): Calibration probabilities.
        bins (array-like): Mondrian categories for calibration.
        va (dict or object): Venn-Abers model(s) for calibration.
    Methods
    -------
        __init__(X_cal, y_cal, learner, bins=None, cprobs=None, difficulty_estimator=None):
            Initializes the VennAbers class with calibration data and model.
        __predict_proba_with_difficulty(X, bins=None):
            Predicts probabilities with difficulty adjustment.
        predict(X_test, bins=None):
            Predicts the class of the test samples.
        predict_proba(X_test, output_interval=False, classes=None, bins=None):
            Predicts the probabilities of the test samples, optionally outputting the Venn-ABERS interval.
        is_multiclass() -> bool:
            Returns true if the problem is multiclass.
        is_mondrian() -> bool:
            Returns true if Mondrian categories are used.
    """

    def __init__(
        self,
        X_cal,
        y_cal,
        learner,
        bins=None,
        cprobs=None,
        difficulty_estimator=None,
        predict_function=None,
    ):
        """Initialize the VennAbers class with calibration data and model.

        Parameters
        ----------
            X_cal (array-like): Calibration feature set.
            y_cal (array-like): Calibration target values.
            learner (object): A machine learning model with a `predict_proba` method.
            bins (array-like, optional): Mondrian categories for calibration. Defaults to None.
            cprobs (array-like, optional): Calibration probabilities. Defaults to None.
            difficulty_estimator (callable, optional): A difficulty estimator function. Defaults to None.
            predict_function (callable, optional): A predict_proba function. Defaults to None.
        """
        self.y_cal_numeric, self.label_map = convert_targets_to_numeric(y_cal)
        self.original_labels = y_cal

        self.de = difficulty_estimator
        self.learner = learner
        self._predict_proba = (
            predict_function if predict_function is not None else learner.predict_proba
        )
        self.X_cal = X_cal
        self.__is_multiclass = len(np.unique(self.y_cal_numeric)) > 2

        cprobs = self.__predict_proba_with_difficulty(X_cal) if cprobs is None else cprobs
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

    def __predict_proba_with_difficulty(self, X, bins=None):
        if "bins" in self._predict_proba.__code__.co_varnames:
            probs = self._predict_proba(X, bins=bins)
        else:
            probs = self._predict_proba(X)
        if self.de is not None:
            difficulty = self.de.apply(X)
            # method = logit_based_scaling_list
            method = exponent_scaling_list
            # method = sigmoid_scaling_list
            if self.is_multiclass():
                probs_tmp = method(probs, difficulty)
            else:
                probs_tmp = method(probs, np.repeat(difficulty, 2).reshape(-1, 2))
            probs = np.array([np.asarray(tmp) for tmp in probs_tmp])
        return probs

    def predict(self, X_test, bins=None):
        """Predict the class of the test samples.

        Parameters
        ----------
            X_test (n_test_samples, n_features): Test samples.
            bins (array-like of shape (n_samples,), optional): Mondrian categories.

        Returns
        -------
            ndarray: Predicted classes based on the regularized VennAbers probabilities.
                If multiclass, the predicted class is 1 if the prediction from the underlying model is the same after calibration and 0 otherwise.
        """
        if self.is_multiclass():
            tmp, _ = self.predict_proba(X_test, bins=bins)
            return np.asarray(np.round(tmp[:, 1]))
        tmp = self.predict_proba(X_test, bins=bins)[:, 1]
        return np.asarray(np.round(tmp))

    # pylint: disable=too-many-locals, too-many-branches
    def predict_proba(self, X_test, output_interval=False, classes=None, bins=None):
        """Predict the probabilities of the test samples, optionally outputting the VennAbers interval.

        Parameters
        ----------
            X_test (n_test_samples, n_features): Test samples.
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
        tprobs = self.__predict_proba_with_difficulty(X_test, bins=bins)
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
                        raise ValueError("bins must be provided if Mondrian")
                    for b, va_class_bin in va_class.items():
                        p0p1[bins == b, :] = va_class_bin.predict_proba(tmp_probs[bins == b, :])[1]
                else:
                    p0p1 = va_class.predict_proba(tmp_probs)[1]
                low[:, c], high[:, c] = p0p1[:, 0], p0p1[:, 1]
                tmp = high[:, c] / (1 - low[:, c] + high[:, c])
                va_proba[:, c] = tmp
            # TODO: Surprisingly, probability normalization is needed, needs looking into
            for i in range(va_proba.shape[0]):
                low[i] = low[i] / np.sum(va_proba[i, :])
                high[i] = high[i] / np.sum(va_proba[i, :])
                va_proba[i, :] = va_proba[i, :] / np.sum(va_proba[i, :])
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
                raise ValueError("bins must be provided if Mondrian")
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
