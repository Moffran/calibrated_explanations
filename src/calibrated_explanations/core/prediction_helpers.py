"""Phase 1A prediction helper delegators.

This module introduces *thin* wrapper functions around existing private
methods of ``CalibratedExplainer``. It is an intermediate, mechanical step
that allows future extractions without touching behavior now. Tests will
exercise these wrappers to lock in semantics before moving logic bodies.
"""

from __future__ import annotations

import logging
import warnings as _warnings
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple, Union, cast

import numpy as np

from ..explanations import CalibratedExplanations
from .exceptions import (
    ValidationError,
    DataShapeError,
)
from ..utils.helper import assert_threshold, safe_isinstance
from .explain._computation import explain_predict_step

# Local typing protocol to avoid importing CalibratedExplainer and creating cycles.
# Captures just the members used by these helpers.
ThresholdLike = Union[
    float,
    Tuple[float, float],
    Sequence[Tuple[float, float]],
    np.ndarray,
]


class _ExplainerProtocol(Protocol):
    """Structural subset of ``CalibratedExplainer`` used by helper functions."""

    num_features: int
    mode: str
    x_cal: np.ndarray
    interval_learner: Any

    def _is_mondrian(self) -> bool:
        """Return True when a Mondrian (per-bin) calibration is active."""
        ...

    def is_multiclass(self) -> bool:
        """Return True when the underlying task involves more than two classes."""
        ...

    def is_fast(self) -> bool:
        """Return True when the specialized fast explainer path is available."""
        ...

    def _predict(
        self,
        x: np.ndarray,
        *,
        threshold: Optional[ThresholdLike] = ...,  # noqa: D401
        low_high_percentiles: Tuple[int, int] = ...,
        classes: Optional[Sequence[int]] = ...,
        bins: Optional[np.ndarray] = ...,
        feature: Optional[int] = ...,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute calibrated predictions and interval bounds."""
        ...

    def _discretize(self, x: np.ndarray) -> np.ndarray:
        """Transform inputs into discretized representations when needed."""
        ...

    def rule_boundaries(self, x: np.ndarray, x_perturbed: np.ndarray) -> Any:
        """Return rule boundary metadata for categorical perturbations."""
        ...


# NOTE: We intentionally avoid importing CalibratedExplainer for type-only usage to
# prevent cyclical import complexity during the gradual split.


def validate_and_prepare_input(explainer: _ExplainerProtocol, x: Any) -> np.ndarray:
    """Validate and prepare input data (extracted logic).

    Mechanical move from ``CalibratedExplainer._validate_and_prepare_input``.
    """
    if safe_isinstance(x, "pandas.core.frame.DataFrame"):
        x = x.values  # pragma: no cover - passthrough
    if len(x.shape) == 1:  # noqa: PLR2004
        x = x.reshape(1, -1)
    if x.shape[1] != explainer.num_features:
        raise DataShapeError("Number of features must match calibration data")
    return cast(np.ndarray, np.asarray(x))


def initialize_explanation(
    explainer: _ExplainerProtocol,
    x: np.ndarray,
    low_high_percentiles: Tuple[int, int],
    threshold: Optional[ThresholdLike],
    bins: Optional[np.ndarray],
    features_to_ignore: Optional[Sequence[int]],
) -> CalibratedExplanations:
    """Initialize explanation object (extracted logic)."""
    if explainer._is_mondrian():  # noqa: SLF001
        if bins is None:
            raise ValidationError("Bins required for Mondrian explanations")
        if len(bins) != len(x):  # pragma: no cover - defensive
            raise DataShapeError("The length of bins must match the number of added instances.")
    explanation = CalibratedExplanations(explainer, x, threshold, bins, features_to_ignore)
    if threshold is not None:
        if "regression" not in explainer.mode:
            raise ValidationError(
                "The threshold parameter is only supported for mode='regression'."
            )
        if isinstance(threshold, (list, np.ndarray)) and isinstance(threshold[0], tuple):
            _warnings.warn(
                "Having a list of interval thresholds (i.e. a list of tuples) is likely going to be very slow. Consider using a single interval threshold for all instances.",
                stacklevel=2,
            )
        assert_threshold(threshold, x)
    elif "regression" in explainer.mode:
        explanation.low_high_percentiles = low_high_percentiles
    return explanation


def predict_internal(
    explainer: _ExplainerProtocol,
    x: np.ndarray,
    threshold: Optional[ThresholdLike] = None,
    low_high_percentiles: Tuple[int, int] = (5, 95),
    classes: Optional[Sequence[int]] = None,
    bins: Optional[np.ndarray] = None,
    feature: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the internal prediction logic (mechanically moved)."""
    # (Body kept inside calibrated_explainer for now to limit patch size) -- placeholder stub if future isolation needed
    return explainer._predict(  # noqa: SLF001
        x,
        threshold=threshold,
        low_high_percentiles=low_high_percentiles,
        classes=classes,
        bins=bins,
        feature=feature,
    )


__all__ = [
    "validate_and_prepare_input",
    "initialize_explanation",
    "predict_internal",
    "explain_predict_step",
    "format_regression_prediction",
    "format_classification_prediction",
    "handle_uncalibrated_regression_prediction",
    "handle_uncalibrated_classification_prediction",
]


def format_regression_prediction(
    predict: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    threshold: Optional[ThresholdLike] = None,
    uq_interval: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """Format regression predictions with optional thresholds and intervals.

    Parameters
    ----------
    predict : np.ndarray
        The predicted values.
    low : np.ndarray
        Lower bounds of prediction intervals.
    high : np.ndarray
        Upper bounds of prediction intervals.
    threshold : float, int, array-like, optional
        Threshold for probabilistic regression. If provided, returns probabilities.
    uq_interval : bool, default=False
        Whether to return uncertainty intervals.

    Returns
    -------
    predictions or (predictions, (low, high))
        Formatted predictions with optional intervals.
    """
    if threshold is None:
        return (predict, (low, high)) if uq_interval else predict

    # Thresholded prediction - convert to probability labels
    def get_label(prob_val, thresh):
        if np.isscalar(thresh):
            return f"y_hat <= {thresh}" if prob_val >= 0.5 else f"y_hat > {thresh}"
        if isinstance(thresh, tuple):
            return (
                f"{thresh[0]} < y_hat <= {thresh[1]}"
                if prob_val >= 0.5
                else f"y_hat <= {thresh[0]} || y_hat > {thresh[1]}"
            )
        return "Error in format_regression_prediction.get_label()"

    if np.isscalar(threshold) or isinstance(threshold, tuple):
        new_classes = [get_label(predict[i], threshold) for i in range(len(predict))]
    else:
        new_classes = [get_label(predict[i], threshold[i]) for i in range(len(predict))]

    return (new_classes, (low, high)) if uq_interval else new_classes


def format_classification_prediction(
    predict: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    new_classes: Optional[np.ndarray],
    is_multiclass_val: bool,
    label_map: Optional[dict] = None,
    class_labels: Optional[np.ndarray] = None,
    uq_interval: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """Format classification predictions with optional class label mapping and intervals.

    Parameters
    ----------
    predict : np.ndarray
        The predicted probabilities.
    low : np.ndarray
        Lower bounds of prediction intervals.
    high : np.ndarray
        Upper bounds of prediction intervals.
    new_classes : np.ndarray or None
        Predicted class indices or None.
    is_multiclass_val : bool
        Whether this is a multiclass problem.
    label_map : dict, optional
        Mapping from numeric class indices to labels.
    class_labels : array-like, optional
        Human-readable class labels.
    uq_interval : bool, default=False
        Whether to return uncertainty intervals.

    Returns
    -------
    predictions or (predictions, (low, high))
        Formatted predictions with optional intervals.
    """
    if new_classes is None:
        new_classes = (predict >= 0.5).astype(int)

    if label_map is not None or class_labels is not None:
        new_classes = np.array([class_labels[c] for c in new_classes])

    return (new_classes, (low, high)) if uq_interval else new_classes


def handle_uncalibrated_regression_prediction(
    learner,
    x: np.ndarray,
    threshold: Optional[ThresholdLike] = None,
    uq_interval: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """Handle uncalibrated regression prediction.

    Parameters
    ----------
    learner : object
        The underlying predictive learner.
    x : np.ndarray
        Input data.
    threshold : float, int, array-like, optional
        Threshold for regression tasks (not allowed for uncalibrated).
    uq_interval : bool, default=False
        Whether to return uncertainty intervals.

    Returns
    -------
    predictions or (predictions, (low, high))
        Uncalibrated predictions.

    Raises
    ------
    ValidationError
        If threshold is provided.
    """
    if threshold is not None:
        raise ValidationError(
            "A thresholded prediction is not possible for uncalibrated predictions."
        )

    predict = learner.predict(x)
    return (predict, (predict, predict)) if uq_interval else predict


def handle_uncalibrated_classification_prediction(
    learner,
    x: np.ndarray,
    threshold: Optional[ThresholdLike] = None,
    uq_interval: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """Handle uncalibrated classification prediction.

    Parameters
    ----------
    learner : object
        The underlying predictive learner.
    x : np.ndarray
        Input data.
    threshold : float, int, array-like, optional
        Threshold (not allowed for classification).
    uq_interval : bool, default=False
        Whether to return uncertainty intervals.

    Returns
    -------
    predictions or (predictions, (low, high))
        Uncalibrated class predictions (not probabilities).

    Raises
    ------
    ValidationError
        If threshold is provided.
    """
    if threshold is not None:
        raise ValidationError(
            "A thresholded prediction is not possible for uncalibrated learners."
        )

    # Use learner.predict() to get class predictions, not probabilities
    predictions = learner.predict(x)
    if uq_interval:
        # For intervals, use the same predictions as bounds (no uncertainty)
        return predictions, (predictions, predictions)
    return predictions
