"""Phase 1A calibration/interval-learner helper delegators.

This module contains thin wrapper functions that encapsulate calibration-related
logic from ``CalibratedExplainer`` without changing behavior. The explainer
instance is passed in and used directly to avoid re-wiring state.
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

import numpy as np

from .._interval_regressor import IntervalRegressor
from .._VennAbers import VennAbers
from ..utils.perturbation import perturb_dataset
from .exceptions import ConfigurationError

if TYPE_CHECKING:  # avoid circular import at runtime
    from .calibrated_explainer import CalibratedExplainer


def assign_threshold(explainer: "CalibratedExplainer", threshold: Any) -> Any:
    """Thin wrapper around ``CalibratedExplainer.assign_threshold``.

    Exposed as a helper for tests and future extraction stages.
    """
    return explainer.assign_threshold(threshold)


def update_interval_learner(
    explainer: "CalibratedExplainer",
    xs: np.ndarray,
    ys: np.ndarray,
    bins: Optional[np.ndarray] = None,
) -> None:
    """Mechanical move of ``CalibratedExplainer.__update_interval_learner``.

    Mirrors original semantics and exceptions exactly.
    """
    if explainer.is_fast():
        raise ConfigurationError(
            "OnlineCalibratedExplainers does not currently support fast explanations."
        )
    if explainer.mode == "classification":
        explainer.interval_learner = VennAbers(
            explainer.X_cal,
            explainer.y_cal,
            explainer.learner,
            explainer.bins,
            difficulty_estimator=explainer.difficulty_estimator,
            predict_function=explainer.predict_function,
        )
    elif "regression" in explainer.mode:
        if isinstance(explainer.interval_learner, list):
            raise ConfigurationError(
                "OnlineCalibratedExplainers does not currently support fast explanations."
            )
        # update the IntervalRegressor
        explainer.interval_learner.insert_calibration(xs, ys, bins=bins)
    explainer._CalibratedExplainer__initialized = True  # noqa: SLF001


def initialize_interval_learner(explainer: "CalibratedExplainer") -> None:
    """Mechanical move of ``CalibratedExplainer.__initialize_interval_learner``."""
    if explainer.is_fast():
        initialize_interval_learner_for_fast_explainer(explainer)
    elif explainer.mode == "classification":
        explainer.interval_learner = VennAbers(
            explainer.X_cal,
            explainer.y_cal,
            explainer.learner,
            explainer.bins,
            difficulty_estimator=explainer.difficulty_estimator,
            predict_function=explainer.predict_function,
        )
    elif "regression" in explainer.mode:
        explainer.interval_learner = IntervalRegressor(explainer)
    explainer._CalibratedExplainer__initialized = True  # noqa: SLF001


def initialize_interval_learner_for_fast_explainer(explainer: "CalibratedExplainer") -> None:
    """Mechanical move of ``CalibratedExplainer.__initialize_interval_learner_for_fast_explainer``."""
    explainer.interval_learner = []
    X_cal, y_cal, bins = explainer.X_cal, explainer.y_cal, explainer.bins
    (
        explainer.fast_X_cal,
        explainer.scaled_X_cal,
        explainer.scaled_y_cal,
        scale_factor,
    ) = perturb_dataset(
        explainer.X_cal,
        explainer.y_cal,
        explainer.categorical_features,
        noise_type=explainer._CalibratedExplainer__noise_type,  # noqa: SLF001
        scale_factor=explainer._CalibratedExplainer__scale_factor,  # noqa: SLF001
        severity=explainer._CalibratedExplainer__severity,  # noqa: SLF001
    )
    explainer.bins = (
        np.tile(explainer.bins.copy(), scale_factor) if explainer.bins is not None else None
    )
    for f in range(explainer.num_features):
        fast_X_cal = explainer.scaled_X_cal.copy()
        fast_X_cal[:, f] = explainer.fast_X_cal[:, f]
        if explainer.mode == "classification":
            explainer.interval_learner.append(
                VennAbers(
                    fast_X_cal,
                    explainer.scaled_y_cal,
                    explainer.learner,
                    explainer.bins,
                    difficulty_estimator=explainer.difficulty_estimator,
                )
            )
        elif "regression" in explainer.mode:
            explainer.X_cal = fast_X_cal
            explainer.y_cal = explainer.scaled_y_cal
            explainer.interval_learner.append(IntervalRegressor(explainer))

    explainer.X_cal, explainer.y_cal, explainer.bins = X_cal, y_cal, bins
    if explainer.mode == "classification":
        explainer.interval_learner.append(
            VennAbers(
                explainer.X_cal,
                explainer.y_cal,
                explainer.learner,
                explainer.bins,
                difficulty_estimator=explainer.difficulty_estimator,
            )
        )
    elif "regression" in explainer.mode:
        # Add a reference learner using the original calibration data last
        explainer.interval_learner.append(IntervalRegressor(explainer))


__all__ = ["assign_threshold", "initialize_interval_learner", "update_interval_learner"]
