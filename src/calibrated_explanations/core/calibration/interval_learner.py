"""Interval learner management and calibration helpers.

This module provides functions for managing interval learners used in uncertainty
quantification and conformal prediction. It handles initialization, updates, and
threshold assignment for both standard and fast explanation modes.

Part of Phase 6: Refactor Calibration Functionality (ADR-001).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..exceptions import ConfigurationError
from ..explain.feature_task import assign_threshold as normalize_threshold

if TYPE_CHECKING:
    from ..calibrated_explainer import CalibratedExplainer


def assign_threshold(explainer: CalibratedExplainer, threshold) -> None:
    """Set the classification decision threshold.

    This is a thin wrapper kept for backward compatibility. Threshold
    normalization is now handled entirely by ``feature_task.assign_threshold``.

    Parameters
    ----------
    explainer : CalibratedExplainer
        The explainer instance to update.
    threshold : array-like or float, optional
        The new decision threshold(s).

    Returns
    -------
    None or array-like
        Normalized threshold output from ``feature_task.assign_threshold``.
    """
    _ = explainer  # preserve signature; explainer no longer used
    return normalize_threshold(threshold)


def update_interval_learner(  # pylint: disable=invalid-name
    explainer: CalibratedExplainer,
    xs,
    ys,
    bins=None,  # pylint: disable=invalid-name
) -> None:
    """Update the interval learner with new calibration data.

    This is a mechanical extraction of ``CalibratedExplainer.__update_interval_learner``
    that maintains the original behavior while enabling independent testing.

    Parameters
    ----------
    explainer : CalibratedExplainer
        The explainer instance to update.
    xs : array-like
        New calibration input data.
    ys : array-like
        New calibration target values.
    bins : array-like, optional
        Mondrian bins or categories for the new data.

    Raises
    ------
    ConfigurationError
        If the explainer is in fast mode (not supported for updates).
        If regression mode uses fast-mode sentinel (list) for interval learner.

    Notes
    -----
    - For classification mode: Obtains a fresh interval calibrator from the explainer
    - For regression mode: Delegates to the interval learner's insert_calibration method
    - Sets the explainer's __initialized flag to True after successful update
    """
    if explainer.is_fast():
        raise ConfigurationError("Fast explanations are not supported in this update path.")

    if explainer.mode == "classification":
        interval, _identifier = explainer._obtain_interval_calibrator(  # pylint: disable=protected-access
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


def initialize_interval_learner(explainer: CalibratedExplainer) -> None:
    """Initialize the interval learner for the explainer.

    This is a mechanical extraction of ``CalibratedExplainer.__initialize_interval_learner``
    that maintains the original behavior while enabling independent testing.

    Parameters
    ----------
    explainer : CalibratedExplainer
        The explainer instance to initialize.

    Notes
    -----
    - Ensures interval runtime state is valid before initialization
    - For fast mode: Delegates to initialize_interval_learner_for_fast_explainer()
    - For standard mode: Obtains interval calibrator from the explainer
    - Sets the explainer's __initialized flag to True after successful initialization

    See Also
    --------
    initialize_interval_learner_for_fast_explainer : Special initialization for fast mode
    """
    ensure_state = getattr(explainer, "_ensure_interval_runtime_state", None)
    if callable(ensure_state):
        ensure_state()

    if explainer.is_fast():
        initialize_interval_learner_for_fast_explainer(explainer)
    elif explainer.mode == "classification" or "regression" in explainer.mode:
        interval, _identifier = explainer._obtain_interval_calibrator(  # pylint: disable=protected-access
            fast=False,
            metadata={"operation": "initialize"},
        )
        explainer.interval_learner = interval

    explainer._CalibratedExplainer__initialized = True  # noqa: SLF001


def initialize_interval_learner_for_fast_explainer(explainer: CalibratedExplainer) -> None:  # pylint: disable=invalid-name
    """Initialize the interval learner in fast explanation mode.

    This is a mechanical extraction of
    ``CalibratedExplainer.__initialize_interval_learner_for_fast_explainer``
    that maintains the original behavior while enabling independent testing.

    Parameters
    ----------
    explainer : CalibratedExplainer
        The explainer instance to initialize (must be in fast mode).

    Notes
    -----
    - Ensures interval runtime state is valid before initialization
    - Obtains interval calibrator with fast=True from the explainer
    - Assigns the result to the explainer's interval_learner property
    - Sets the explainer's __initialized flag to True after successful initialization

    See Also
    --------
    initialize_interval_learner : Standard initialization for non-fast mode
    """
    ensure_state = getattr(explainer, "_ensure_interval_runtime_state", None)
    if callable(ensure_state):
        ensure_state()

    interval, _identifier = explainer._obtain_interval_calibrator(  # pylint: disable=protected-access
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
