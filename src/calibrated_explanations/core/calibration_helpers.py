"""Phase 1A calibration/interval-learner helper delegators.

This module contains thin wrapper functions that encapsulate calibration-related
logic from ``CalibratedExplainer`` without changing behavior. The explainer
instance is passed in and used directly to avoid re-wiring state.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._interval_regressor import IntervalRegressor
from .._VennAbers import VennAbers
from ..utils.perturbation import perturb_dataset
from .exceptions import ConfigurationError


def assign_threshold(explainer, threshold):
    """Thin wrapper around ``CalibratedExplainer.assign_threshold``.

    Exposed as a helper for tests and future extraction stages.
    """
    return explainer.assign_threshold(threshold)


def update_interval_learner(explainer, xs, ys, bins=None) -> None:
    """Mechanical move of ``CalibratedExplainer.__update_interval_learner``.

    Mirrors original semantics and exceptions exactly.
    """
    if explainer.is_fast():
        raise ConfigurationError("Fast explanations are not supported in this update path.")
    if explainer.mode == "classification":
        calibrator = VennAbers(
            explainer.X_cal,
            explainer.y_cal,
            explainer.learner,
            explainer.bins,
            difficulty_estimator=explainer.difficulty_estimator,
            predict_function=explainer.predict_function,
        )
        interval, _identifier = explainer._obtain_interval_calibrator(
            fast=False,
            metadata={"calibrator": calibrator},
        )
        explainer.interval_learner = interval
    elif "regression" in explainer.mode:
        if isinstance(explainer.interval_learner, list):
            raise ConfigurationError("Fast explanations are not supported in this update path.")
        # update the IntervalRegressor
        explainer.interval_learner.insert_calibration(xs, ys, bins=bins)
    explainer._CalibratedExplainer__initialized = True  # noqa: SLF001


def initialize_interval_learner(explainer) -> None:
    """Mechanical move of ``CalibratedExplainer.__initialize_interval_learner``."""

    ensure_state = getattr(explainer, "_ensure_interval_runtime_state", None)
    if callable(ensure_state):
        ensure_state()

    if explainer.is_fast():
        initialize_interval_learner_for_fast_explainer(explainer)
    elif explainer.mode == "classification":
        calibrator = VennAbers(
            explainer.X_cal,
            explainer.y_cal,
            explainer.learner,
            explainer.bins,
            difficulty_estimator=explainer.difficulty_estimator,
            predict_function=explainer.predict_function,
        )
        interval, _identifier = explainer._obtain_interval_calibrator(
            fast=False,
            metadata={"calibrator": calibrator},
        )
        explainer.interval_learner = interval
    elif "regression" in explainer.mode:
        calibrator = IntervalRegressor(explainer)
        interval, _identifier = explainer._obtain_interval_calibrator(
            fast=False,
            metadata={"calibrator": calibrator},
        )
        explainer.interval_learner = interval
    explainer._CalibratedExplainer__initialized = True  # noqa: SLF001

def initialize_interval_learner_for_fast_explainer(explainer) -> None:
    """Mechanical move of ``CalibratedExplainer.__initialize_interval_learner_for_fast_explainer``."""

    ensure_state = getattr(explainer, "_ensure_interval_runtime_state", None)
    if callable(ensure_state):
        ensure_state()

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
        seed=getattr(explainer, "seed", None),
        rng=getattr(explainer, "rng", None),
    )
    explainer.bins = (
        np.tile(explainer.bins.copy(), scale_factor) if explainer.bins is not None else None
    )
    fast_calibrators: list[Any] = []
    for f in range(explainer.num_features):
        fast_X_cal = explainer.scaled_X_cal.copy()
        fast_X_cal[:, f] = explainer.fast_X_cal[:, f]
        if explainer.mode == "classification":
            fast_calibrators.append(
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
            fast_calibrators.append(IntervalRegressor(explainer))

    explainer.X_cal, explainer.y_cal, explainer.bins = X_cal, y_cal, bins
    if explainer.mode == "classification":
        fast_calibrators.append(
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
        fast_calibrators.append(IntervalRegressor(explainer))

    interval, _identifier = explainer._obtain_interval_calibrator(
        fast=True,
        metadata={"fast_calibrators": fast_calibrators},
    )
    explainer.interval_learner = interval

__all__ = ["assign_threshold", "initialize_interval_learner", "update_interval_learner"]
