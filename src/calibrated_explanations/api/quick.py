"""Convenience API: quick_explain.

Provides a compact, opinionated path to fit, calibrate, and generate
factual explanations in one call using the existing wrapper/config.

Kept minimal and backward-compatible with current public surface.
"""

from __future__ import annotations

import importlib
from typing import Any, Literal


def quick_explain(
    model: Any,
    x_train: Any,
    y_train: Any,
    x_cal: Any,
    y_cal: Any,
    x: Any,
    *,
    task: Literal["classification", "regression"] | None = None,
    threshold: float | None = None,
    low_high_percentiles: tuple[int, int] | None = None,
    preprocessor: Any | None = None,
) -> Any:
    """Fit, calibrate, and explain in one step.

    Parameters
    ----------
    model : Any
        Estimator exposing fit/predict (and optionally predict_proba).
    x_train, y_train : Any
        Proper training data for the model.
    x_cal, y_cal : Any
        Calibration data for the explainer.
    x : Any
        Instances to explain.
    task : {"classification", "regression"}, optional
        Overrides automatic mode detection.
    threshold : float, optional
        Threshold for regression probability-style outputs.
    low_high_percentiles : tuple[int, int], optional
        Percentiles for interval computation (regression).
    preprocessor : Any, optional
        Optional user-supplied preprocessor (Pipeline/ColumnTransformer).

    Returns
    -------
    Any
        A CalibratedExplanations-like object from `explain_factual`.
    """
    # Import from core to avoid circular dependency (api -> core)
    # This shim allows keeping the API surface while moving implementation to core.
    module = importlib.import_module("calibrated_explanations.core.quick")
    return module.quick_explain(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_cal=x_cal,
        y_cal=y_cal,
        x=x,
        task=task,
        threshold=threshold,
        low_high_percentiles=low_high_percentiles,
        preprocessor=preprocessor,
    )


__all__ = ["quick_explain"]
