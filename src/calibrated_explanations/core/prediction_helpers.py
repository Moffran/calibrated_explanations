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

# Local typing protocol to avoid importing CalibratedExplainer and creating cycles.
# Captures just the members used by these helpers.
ThresholdLike = Union[
    float,
    Tuple[float, float],
    Sequence[Tuple[float, float]],
    np.ndarray,
]


class _ExplainerProtocol(Protocol):
    num_features: int
    mode: str
    X_cal: np.ndarray
    interval_learner: Any

    def _is_mondrian(self) -> bool: ...

    def is_multiclass(self) -> bool: ...

    def is_fast(self) -> bool: ...

    def _predict(
        self,
        X_test: np.ndarray,
        *,
        threshold: Optional[ThresholdLike] = ...,  # noqa: D401
        low_high_percentiles: Tuple[int, int] = ...,
        classes: Optional[Sequence[int]] = ...,
        bins: Optional[np.ndarray] = ...,
        feature: Optional[int] = ...,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...

    def assign_threshold(self, threshold: Optional[ThresholdLike]) -> Any: ...

    def _discretize(self, X: np.ndarray) -> np.ndarray: ...

    def rule_boundaries(self, X: np.ndarray, X_perturbed: np.ndarray) -> Any: ...


# NOTE: We intentionally avoid importing CalibratedExplainer for type-only usage to
# prevent cyclical import complexity during the gradual split.


def validate_and_prepare_input(explainer: _ExplainerProtocol, X_test: Any) -> np.ndarray:
    """Validate and prepare input data (extracted logic).

    Mechanical move from ``CalibratedExplainer._validate_and_prepare_input``.
    """
    if safe_isinstance(X_test, "pandas.core.frame.DataFrame"):
        X_test = X_test.values  # pragma: no cover - passthrough
    if len(X_test.shape) == 1:  # noqa: PLR2004
        X_test = X_test.reshape(1, -1)
    if X_test.shape[1] != explainer.num_features:
        raise DataShapeError("Number of features must match calibration data")
    return cast(np.ndarray, np.asarray(X_test))


def initialize_explanation(
    explainer: _ExplainerProtocol,
    X_test: np.ndarray,
    low_high_percentiles: Tuple[int, int],
    threshold: Optional[ThresholdLike],
    bins: Optional[np.ndarray],
    features_to_ignore: Optional[Sequence[int]],
) -> CalibratedExplanations:
    """Initialize explanation object (extracted logic)."""
    if explainer._is_mondrian():  # noqa: SLF001
        if bins is None:
            raise ValidationError("Bins required for Mondrian explanations")
        if len(bins) != len(X_test):  # pragma: no cover - defensive
            raise DataShapeError("The length of bins must match the number of added instances.")
    explanation = CalibratedExplanations(explainer, X_test, threshold, bins, features_to_ignore)
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
        assert_threshold(threshold, X_test)
    elif "regression" in explainer.mode:
        explanation.low_high_percentiles = low_high_percentiles
    return explanation


def predict_internal(
    explainer: _ExplainerProtocol,
    X_test: np.ndarray,
    threshold: Optional[ThresholdLike] = None,
    low_high_percentiles: Tuple[int, int] = (5, 95),
    classes: Optional[Sequence[int]] = None,
    bins: Optional[np.ndarray] = None,
    feature: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Internal prediction logic (mechanically moved)."""
    # (Body kept inside calibrated_explainer for now to limit patch size) -- placeholder stub if future isolation needed
    return explainer._predict(  # noqa: SLF001
        X_test,
        threshold=threshold,
        low_high_percentiles=low_high_percentiles,
        classes=classes,
        bins=bins,
        feature=feature,
    )


def explain_predict_step(
    explainer: _ExplainerProtocol,
    X_test: np.ndarray,
    threshold: Optional[ThresholdLike],
    low_high_percentiles: Tuple[int, int],
    bins: Optional[np.ndarray],
    features_to_ignore: Optional[Sequence[int]],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
    np.ndarray,
    Any,
    Dict[int, Any],
    Dict[int, Any],
    Dict[int, Any],
    np.ndarray,
    Any,
    Optional[np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    """Execute the initial prediction step for explanation (mechanically moved)."""
    X_cal = explainer.X_cal
    predict, low, high, predicted_class = explainer._predict(  # noqa: SLF001
        X_test, threshold=threshold, low_high_percentiles=low_high_percentiles, bins=bins
    )

    prediction = {
        "predict": predict,
        "low": low,
        "high": high,
        "classes": (predicted_class if explainer.is_multiclass() else np.ones(predict.shape)),
    }
    if explainer.mode == "classification":  # store full calibrated probability matrix
        try:  # pragma: no cover - defensive
            if explainer.is_multiclass():
                if explainer.is_fast():
                    full_probs = explainer.interval_learner[  # noqa: SLF001
                        explainer.num_features
                    ].predict_proba(X_test, bins=bins)
                else:
                    full_probs = explainer.interval_learner.predict_proba(  # noqa: SLF001
                        X_test, bins=bins
                    )
            else:  # binary
                if explainer.is_fast():
                    full_probs = explainer.interval_learner[  # noqa: SLF001
                        explainer.num_features
                    ].predict_proba(X_test, bins=bins)
                else:
                    full_probs = explainer.interval_learner.predict_proba(  # noqa: SLF001
                        X_test, bins=bins
                    )
            prediction["__full_probabilities__"] = full_probs
        except Exception as exc:  # pragma: no cover
            logging.getLogger("calibrated_explanations").debug(
                "Failed to compute full calibrated probabilities: %s", exc
            )

    X_test.flags.writeable = False
    assert_threshold(threshold, X_test)
    perturbed_threshold = explainer.assign_threshold(threshold)
    perturbed_bins = np.empty((0,)) if bins is not None else None
    perturbed_X = np.empty((0, explainer.num_features))
    perturbed_feature = np.empty((0, 4))  # (feature, instance, bin_index, is_lesser)
    perturbed_class = np.empty((0,), dtype=int)
    X_perturbed = explainer._discretize(X_test)  # noqa: SLF001
    rule_boundaries = explainer.rule_boundaries(X_test, X_perturbed)  # noqa: SLF001

    lesser_values: dict[int, Any] = {}
    greater_values: dict[int, Any] = {}
    covered_values: dict[int, Any] = {}

    return (
        predict,
        low,
        high,
        prediction,
        perturbed_feature,
        rule_boundaries,
        lesser_values,
        greater_values,
        covered_values,
        X_cal,
        perturbed_threshold,
        perturbed_bins,
        perturbed_X,
        perturbed_class,
    )


__all__ = [
    "validate_and_prepare_input",
    "initialize_explanation",
    "predict_internal",
    "explain_predict_step",
]
