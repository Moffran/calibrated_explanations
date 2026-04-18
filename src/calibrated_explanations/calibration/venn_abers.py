# ruff: noqa: N999
# pylint: disable=unknown-option-value
# pylint: disable=invalid-name, line-too-long, too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, fixme
"""Venn-Abers calibration utilities for post-processing model probabilities.

Wraps the `venn_abers` package to offer Mondrian-aware calibration
integrated with the calibrated explanations toolkit.

Part of ADR-001: Core Decomposition Boundaries (Stage 1a).
"""

import base64
import hashlib
import pickle  # nosec B403 - deserialization is restricted to trusted, checksum-validated state
import warnings
from typing import Any, Mapping

import numpy as np
import venn_abers as va

from ..core.prediction.interval_summary import IntervalSummary, coerce_interval_summary
from ..utils import convert_targets_to_numeric
from ..utils.exceptions import ConfigurationError


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
            # Alternative difficulty scaling methods are available for experimentation:
            #   method = logit_based_scaling_list
            #   method = sigmoid_scaling_list
            # By default, exponent_scaling_list is used. To try a different scaling method,
            # uncomment the corresponding line above. See documentation for details.
            method = exponent_scaling_list
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
    def predict_proba(
        self,
        x,
        output_interval=False,
        classes=None,
        bins=None,
        interval_summary=None,
        normalize=True,
    ):
        """Predict the probabilities of the test samples, optionally outputting the VennAbers interval.

        Parameters
        ----------
            x (n_test_samples, n_features): Test samples.
            output_interval (bool, optional): If true, the VennAbers intervals are outputted. Defaults to False.
            classes (array-like, optional): A list of predicted classes. Defaults to None.
            bins (array-like of shape (n_samples,), optional): Mondrian categories.
            interval_summary (IntervalSummary or str, optional): Strategy for selecting the
                point estimate from the interval bounds. Defaults to regularized mean.
            normalize (bool, optional): If True (default), apply two-step normalization to
                multiclass OvR outputs. Step 1 (coherence): adjusts each upper bound to
                enforce h_c + sum_{k!=c} l_k = 1 for all c, derived from the IS-c scenario
                probability axiom; lower bounds are preserved. Step 2 (simplex): normalizes
                point estimates to sum to 1. Set to False to obtain raw pre-normalization
                outputs for diagnostic purposes. Only affects the multiclass branch; ignored
                for binary classification.

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
        interval_summary = coerce_interval_summary(interval_summary)

        if self.is_multiclass():
            low, high = np.zeros(tprobs.shape), np.zeros(tprobs.shape)
            tmp_probs = np.zeros((tprobs.shape[0], 2))
            for c, va_class in self.va.items():
                tmp_probs[:, 0] = 1 - tprobs[:, c]
                tmp_probs[:, 1] = tprobs[:, c]
                if self.is_mondrian():
                    if bins is None:
                        raise ConfigurationError(
                            "Mondrian calibration: bins must be provided if Mondrian.",
                            details={"context": "predict_proba", "requirement": "bins parameter"},
                        )
                    for b, va_class_bin in va_class.items():
                        p0p1[bins == b, :] = va_class_bin.predict_proba(tmp_probs[bins == b, :])[1]
                else:
                    p0p1 = va_class.predict_proba(tmp_probs)[1]
                low[:, c], high[:, c] = p0p1[:, 0], p0p1[:, 1]
                va_proba[:, c] = self._select_interval_summary(
                    low[:, c], high[:, c], interval_summary
                )
            if normalize:
                # Step 1: Coherence normalization — enforce h_c + sum_{k!=c} l_k = 1 for all c.
                # Derived from the IS-c scenario probability axiom: l_c (NOT-c scenario) is
                # preserved; h_c is set to 1 - S_l + l_c where S_l = sum of all lower bounds.
                s_low = low.sum(axis=1, keepdims=True)
                high = np.clip(1.0 - s_low + low, low, 1.0)
                # Step 2: Recompute point estimates from coherence-normalized bounds.
                for c in range(tprobs.shape[1]):
                    va_proba[:, c] = self._select_interval_summary(
                        low[:, c], high[:, c], interval_summary
                    )
                # Step 3: Simplex normalization only for IS options where Σ p_c = 1 is expected.
                # LOWER and UPPER are not expected to sum to 1 (they represent conservative /
                # optimistic bounds, not a probability distribution).
                if interval_summary in (IntervalSummary.MEAN, IntervalSummary.REGULARIZED_MEAN):
                    row_sums = va_proba.sum(axis=1, keepdims=True)
                    safe_row_sums = np.where(row_sums == 0, 1.0, row_sums)
                    va_proba = va_proba / safe_row_sums
            if classes is not None:
                if type(classes) not in (list, np.ndarray):
                    classes = [classes]
                if output_interval:
                    return (
                        np.asarray(va_proba),
                        np.array([low[i, c] for i, c in enumerate(classes)]),
                        np.array([high[i, c] for i, c in enumerate(classes)]),
                        classes,
                    )
                return np.asarray(va_proba), classes
            classes = np.argmax(va_proba, axis=1)
            if output_interval:
                return np.asarray(va_proba), low, high, classes
            return np.asarray(va_proba), classes

        if self.is_mondrian():
            if bins is None:
                raise ConfigurationError(
                    "Mondrian calibration: bins must be provided if Mondrian.",
                    details={"context": "predict_proba", "requirement": "bins parameter"},
                )
            for b, va_bin in self.va.items():
                p0p1[bins == b, :] = va_bin.predict_proba(tprobs[bins == b, :])[1]
        else:
            _, p0p1 = self.va.predict_proba(tprobs)
        low, high = p0p1[:, 0], p0p1[:, 1]
        tmp = self._select_interval_summary(low, high, interval_summary)
        va_proba[:, 0] = 1 - tmp
        va_proba[:, 1] = tmp
        # binary
        warnings.filterwarnings("default", category=RuntimeWarning)
        if output_interval:
            return np.asarray(va_proba), low, high
        return np.asarray(va_proba)

    @staticmethod
    def _select_interval_summary(low, high, summary: IntervalSummary):
        """Select a point estimate from Venn-Abers interval bounds."""
        if summary is IntervalSummary.MEAN:
            return (low + high) / 2
        if summary is IntervalSummary.LOWER:
            return low
        if summary is IntervalSummary.UPPER:
            return high
        return high / (1 - low + high)

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

    def to_primitive(self) -> dict[str, Any]:
        """Serialize the calibrator into a JSON-safe primitive payload."""
        payload_bytes = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        payload_b64 = base64.b64encode(payload_bytes).decode("ascii")
        return {
            "schema_version": 1,
            "calibrator_type": "venn_abers",
            "parameters": {
                "is_multiclass": bool(self.is_multiclass()),
                "is_mondrian": bool(self.is_mondrian()),
            },
            "checksums": {
                "sha256": hashlib.sha256(payload_bytes).hexdigest(),
            },
            "payload": {
                "pickle_b64": payload_b64,
            },
        }

    @classmethod
    def from_primitive(cls, payload: Mapping[str, object]) -> "VennAbers":
        """Rehydrate a calibrator from a primitive payload."""
        schema_version = payload.get("schema_version")
        if schema_version != 1:
            raise ConfigurationError(
                "Unsupported VennAbers schema_version. Supported versions: [1].",
                details={"schema_version": schema_version, "supported_versions": [1]},
            )
        calibrator_type = payload.get("calibrator_type")
        if calibrator_type != "venn_abers":
            raise ConfigurationError(
                "Invalid calibrator_type for VennAbers payload.",
                details={"calibrator_type": calibrator_type, "expected": "venn_abers"},
            )
        payload_section = payload.get("payload")
        if not isinstance(payload_section, Mapping):
            raise ConfigurationError(
                "VennAbers primitive payload is missing 'payload' mapping.",
                details={"field": "payload"},
            )
        pickle_b64 = payload_section.get("pickle_b64")
        if not isinstance(pickle_b64, str):
            raise ConfigurationError(
                "VennAbers primitive payload is missing 'pickle_b64'.",
                details={"field": "payload.pickle_b64"},
            )
        payload_bytes = base64.b64decode(pickle_b64.encode("ascii"))
        checksums = payload.get("checksums")
        if not isinstance(checksums, Mapping):
            raise ConfigurationError(
                "VennAbers primitive payload is missing checksum metadata.",
                details={"field": "checksums"},
            )
        expected_sha = checksums.get("sha256")
        actual_sha = hashlib.sha256(payload_bytes).hexdigest()
        if not isinstance(expected_sha, str) or expected_sha != actual_sha:
            raise ConfigurationError(
                "VennAbers primitive checksum validation failed.",
                details={"expected_sha256": expected_sha, "actual_sha256": actual_sha},
            )
        restored = pickle.loads(payload_bytes)  # noqa: S301  # nosec B301 - trusted, checksum-validated payload
        if not isinstance(restored, cls):
            raise ConfigurationError(
                "VennAbers primitive payload restored unexpected object type.",
                details={"restored_type": type(restored).__name__},
            )
        return restored


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
    for p, difficulty in zip(probs, difficulties, strict=False):
        if p[0] < 0.5:
            scaled_p = p ** (1 + beta * (1 - difficulty))
        else:
            scaled_p = 1 - (1 - p) ** (1 + beta * (1 - difficulty))

        final_scaled_p = (1 - difficulty) * scaled_p + difficulty * 0.5
        final_scaled_p = final_scaled_p / np.sum(final_scaled_p)
        scaled_probs.append(final_scaled_p)

    return scaled_probs
