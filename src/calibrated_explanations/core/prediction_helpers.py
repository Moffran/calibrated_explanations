"""Phase 1A prediction helper delegators.

This module introduces *thin* wrapper functions around existing private
methods of ``CalibratedExplainer``. It is an intermediate, mechanical step
that allows future extractions without touching behavior now. Tests will
exercise these wrappers to lock in semantics before moving logic bodies.
"""

from __future__ import annotations

import warnings as _warnings
from typing import Any

import numpy as np

from ..explanations import CalibratedExplanations
from ..utils.helper import assert_threshold, safe_isinstance

# NOTE: We intentionally avoid importing CalibratedExplainer for type-only usage to
# prevent cyclical import complexity during the gradual split.


def validate_and_prepare_input(explainer: Any, X_test):
    """Validate and prepare input data (extracted logic).

    Mechanical move from ``CalibratedExplainer._validate_and_prepare_input``.
    """
    if safe_isinstance(X_test, "pandas.core.frame.DataFrame"):
        X_test = X_test.values  # pragma: no cover - passthrough
    if len(X_test.shape) == 1:  # noqa: PLR2004
        X_test = X_test.reshape(1, -1)
    if X_test.shape[1] != explainer.num_features:
        raise ValueError("Number of features must match calibration data")
    return X_test


def initialize_explanation(
    explainer: Any,
    X_test,
    low_high_percentiles,
    threshold,
    bins,
    features_to_ignore,
):
    """Initialize explanation object (extracted logic)."""
    if explainer._is_mondrian():  # noqa: SLF001
        if bins is None:
            raise ValueError("Bins required for Mondrian explanations")
        if len(bins) != len(X_test):  # pragma: no cover - defensive
            raise ValueError("The length of bins must match the number of added instances.")
    explanation = CalibratedExplanations(explainer, X_test, threshold, bins, features_to_ignore)
    if threshold is not None:
        if "regression" not in explainer.mode:
            raise Warning("The threshold parameter is only supported for mode='regression'.")
        if isinstance(threshold, (list, np.ndarray)) and isinstance(threshold[0], tuple):
            _warnings.warn(
                "Having a list of interval thresholds (i.e. a list of tuples) is likely going to be very slow. Consider using a single interval threshold for all instances.",
                stacklevel=2,
            )
        assert_threshold(threshold, X_test)
    elif "regression" in explainer.mode:
        explanation.low_high_percentiles = low_high_percentiles
    return explanation


def explain_predict_step(
    explainer: Any,
    X_test,
    threshold,
    low_high_percentiles,
    bins,
    features_to_ignore,
):  # pragma: no cover - thin wrapper
    return explainer._explain_predict_step(
        X_test, threshold, low_high_percentiles, bins, features_to_ignore
    )


__all__ = [
    "validate_and_prepare_input",
    "initialize_explanation",
    "explain_predict_step",
]
