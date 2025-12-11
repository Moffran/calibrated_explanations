"""Orchestration layer for reject learner initialization and inference."""

from __future__ import annotations

from typing import Any

import numpy as np
from crepes import ConformalClassifier
from crepes.extras import hinge

from ..exceptions import ValidationError


class RejectOrchestrator:
    """Coordinate reject learner lifecycle and predictions."""

    def __init__(self, explainer: Any) -> None:
        self.explainer = explainer

    def initialize_reject_learner(self, calibration_set=None, threshold=None):
        """Initialize the reject learner with a threshold value."""
        if calibration_set is None:
            x_cal, y_cal = self.explainer.x_cal, self.explainer.y_cal
        elif calibration_set is tuple:
            x_cal, y_cal = calibration_set
        else:
            x_cal, y_cal = calibration_set[0], calibration_set[1]
        self.explainer.reject_threshold = None
        if self.explainer.mode in "regression":
            proba_1, _, _, _ = self.explainer.interval_learner.predict_probability(
                x_cal, y_threshold=threshold, bins=self.explainer.bins
            )
            proba = np.array([[1 - proba_1[i], proba_1[i]] for i in range(len(proba_1))])
            classes = (y_cal < threshold).astype(int)
            self.explainer.reject_threshold = threshold
        elif self.explainer.is_multiclass():  # pylint: disable=protected-access
            proba, classes = self.explainer.interval_learner.predict_proba(
                x_cal, bins=self.explainer.bins
            )
            proba = np.array([[1 - proba[i, c], proba[i, c]] for i, c in enumerate(classes)])
            classes = (classes == y_cal).astype(int)
        else:
            proba = self.explainer.interval_learner.predict_proba(x_cal, bins=self.explainer.bins)
            classes = y_cal
        alphas_cal = hinge(proba, np.unique(classes), classes)
        self.explainer.reject_learner = ConformalClassifier().fit(alphas=alphas_cal, bins=classes)
        return self.explainer.reject_learner

    def predict_reject(self, x, bins=None, confidence=0.95):
        """Predict whether to reject the explanations for the test data."""
        if bins is not None:
            bins = np.asarray(bins)
        if self.explainer.mode in "regression":
            if self.explainer.reject_threshold is None:
                raise ValidationError(
                    "The reject learner is only available for regression with a threshold."
                )
            proba_1, _, _, _ = self.explainer.interval_learner.predict_probability(
                x, y_threshold=self.explainer.reject_threshold, bins=bins
            )
            proba = np.array([[1 - proba_1[i], proba_1[i]] for i in range(len(proba_1))])
            classes = [0, 1]
        elif self.explainer.is_multiclass():  # pylint: disable=protected-access
            proba, classes = self.explainer.interval_learner.predict_proba(x, bins=bins)
            proba = np.array([[1 - proba[i, c], proba[i, c]] for i, c in enumerate(classes)])
            classes = [0, 1]
        else:
            proba = self.explainer.interval_learner.predict_proba(x, bins=bins)
            classes = np.unique(self.explainer.y_cal)
        alphas_test = hinge(proba)

        prediction_set = np.array(
            [
                self.explainer.reject_learner.predict_set(
                    alphas_test, np.full(len(alphas_test), classes[c]), confidence=confidence
                )[:, c]
                for c in range(len(classes))
            ]
        ).T
        singleton = np.sum(np.sum(prediction_set, axis=1) == 1)
        empty = np.sum(np.sum(prediction_set, axis=1) == 0)
        n = len(x)

        epsilon = 1 - confidence
        error_rate = (n * epsilon - empty) / singleton
        reject_rate = 1 - singleton / n

        rejected = np.sum(prediction_set, axis=1) != 1
        return rejected, error_rate, reject_rate
