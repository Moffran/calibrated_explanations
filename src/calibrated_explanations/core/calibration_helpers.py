"""Phase 1A calibration/interval-learner helper delegators.

This module contains thin wrapper functions that encapsulate calibration-related
logic from ``CalibratedExplainer`` without changing behavior. The explainer
instance is passed in and used directly to avoid re-wiring state.
"""

from __future__ import annotations

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
        interval, _identifier = explainer._obtain_interval_calibrator(
            fast=False,
            metadata={"operation": "update"},
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
    elif explainer.mode == "classification" or "regression" in explainer.mode:
        interval, _identifier = explainer._obtain_interval_calibrator(
            fast=False,
            metadata={"operation": "initialize"},
        )
        explainer.interval_learner = interval
    explainer._CalibratedExplainer__initialized = True  # noqa: SLF001

def initialize_interval_learner_for_fast_explainer(explainer) -> None:
    """Mechanical move of ``CalibratedExplainer.__initialize_interval_learner_for_fast_explainer``."""
    ensure_state = getattr(explainer, "_ensure_interval_runtime_state", None)
    if callable(ensure_state):
        ensure_state()

    interval, _identifier = explainer._obtain_interval_calibrator(
        fast=True,
        metadata={"operation": "initialize_fast"},
    )
    explainer.interval_learner = interval

__all__ = [
    "assign_threshold",
    "initialize_interval_learner",
    "initialize_interval_learner_for_fast_explainer",
    "update_interval_learner",
]
