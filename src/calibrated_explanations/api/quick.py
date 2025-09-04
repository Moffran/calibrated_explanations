"""Convenience API: quick_explain.

Provides a compact, opinionated path to fit, calibrate, and generate
factual explanations in one call using the existing wrapper/config.

Kept minimal and backward-compatible with current public surface.
"""

from __future__ import annotations

from typing import Any, Literal

from ..core.wrap_explainer import WrapCalibratedExplainer
from .config import ExplainerConfig


def quick_explain(
    model: Any,
    X_train: Any,
    y_train: Any,
    X_cal: Any,
    y_cal: Any,
    X_test: Any,
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
    X_train, y_train : Any
        Proper training data for the model.
    X_cal, y_cal : Any
        Calibration data for the explainer.
    X_test : Any
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
    cfg = ExplainerConfig(
        model=model,
        low_high_percentiles=low_high_percentiles,
        threshold=threshold,
        preprocessor=preprocessor,
    )
    w = WrapCalibratedExplainer._from_config(cfg)  # private constructor by design
    w.fit(X_train, y_train)
    # Calibrate; pass explicit mode if provided
    cal_kwargs: dict[str, Any] = {}
    if task is not None:
        cal_kwargs["mode"] = task
    w.calibrate(X_cal, y_cal, **cal_kwargs)
    # Use cfg defaults implicitly for factual explanations
    return w.explain_factual(X_test)


__all__ = ["quick_explain"]
