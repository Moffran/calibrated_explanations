"""Interval learner management and calibration helpers.

This module provides functions for managing interval learners used in uncertainty
quantification and conformal prediction. It handles initialization, updates, and
threshold assignment for both standard and fast explanation modes.

Part of ADR-001: Core Decomposition Boundaries (Stage 1a).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from ..utils.exceptions import ConfigurationError
from ..utils.helper import assign_threshold as normalize_threshold
from .interval_wrappers import is_fast_interval_collection

if TYPE_CHECKING:
    from calibrated_explanations.core import CalibratedExplainer


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


_CACHED_CALIBRATOR_KEYS = ("calibrator", "fast_calibrators", "existing_fast_calibrators")


def discard_interval_calibrator(explainer: CalibratedExplainer) -> Callable[[], None]:
    """Discard the active and cached interval calibrators of an explainer.

    Interval resolution persists the last calibrator in
    ``plugin_manager.interval_context_metadata`` and the built-in interval
    plugin reuses it (or the active ``interval_learner``) when present. Call
    this before rebuilding calibration state on changed calibration data or a
    changed learner, so the rebuild cannot return a calibrator fitted on other
    data.

    Parameters
    ----------
    explainer : CalibratedExplainer
        The explainer whose calibrators are discarded.

    Returns
    -------
    Callable[[], None]
        Callback that restores the discarded calibrators, for use when the
        rebuild fails.
    """
    metadata_store = getattr(explainer.plugin_manager, "interval_context_metadata", None) or {}
    removed: dict[str, dict[str, Any]] = {}
    for key, bucket in metadata_store.items():
        if not isinstance(bucket, dict):
            continue
        popped = {name: bucket.pop(name) for name in _CACHED_CALIBRATOR_KEYS if name in bucket}
        if popped:
            removed[key] = popped
    previous = explainer.interval_learner
    explainer.interval_learner = None

    def restore() -> None:
        explainer.interval_learner = previous
        for key, popped in removed.items():
            bucket = metadata_store.get(key)
            if isinstance(bucket, dict):
                bucket.update(popped)

    return restore


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
    - The caller must already have appended ``xs``/``ys`` (and ``bins``) to the
      explainer's calibration set.
    - For classification mode: Rebuilds the interval calibrator on the full
      (appended) calibration set. The previously cached calibrator is discarded
      so the interval plugin cannot return a calibrator fitted on other data. If
      the rebuild fails, the previous calibrator is restored and the error is
      re-raised.
    - For regression mode: Delegates to the interval learner's insert_calibration method
    - Sets the explainer's _initialized flag to True after successful update
    """
    if explainer.is_fast():
        raise ConfigurationError("Fast explanations are not supported in this update path.")

    if explainer.mode == "classification":
        restore = discard_interval_calibrator(explainer)
        try:
            interval, _identifier = explainer.obtain_interval_calibrator(
                fast=False,
                metadata={"operation": "update"},
            )
        except BaseException:
            restore()
            raise
        explainer.interval_learner = interval
    elif "regression" in explainer.mode:
        if is_fast_interval_collection(explainer.interval_learner):
            raise ConfigurationError("Fast explanations are not supported in this update path.")
        # update the IntervalRegressor
        explainer.interval_learner.insert_calibration(xs, ys, bins=bins)
        # Refresh the calibration-feature snapshot so guarded entrypoints see
        # the updated x_cal rather than the stale pre-insert snapshot that was
        # recorded when the learner was originally built via obtain_interval_calibrator.
        _orch = getattr(explainer, "prediction_orchestrator", None)
        if _orch is not None and hasattr(_orch, "refresh_interval_calibration_snapshot"):
            _orch.refresh_interval_calibration_snapshot()

    explainer.initialized = True


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
    - Sets the explainer's _initialized flag to True after successful initialization

    See Also
    --------
    initialize_interval_learner_for_fast_explainer : Special initialization for fast mode
    """
    orchestrator = getattr(explainer, "prediction_orchestrator", None)
    if orchestrator is not None:
        ensure_state = getattr(orchestrator, "ensure_interval_runtime_state", None)
        if callable(ensure_state):
            ensure_state()

    if explainer.is_fast():
        initialize_interval_learner_for_fast_explainer(explainer)
    elif explainer.mode == "classification" or "regression" in explainer.mode:
        interval, _identifier = explainer.obtain_interval_calibrator(
            fast=False,
            metadata={"operation": "initialize"},
        )
        explainer.interval_learner = interval

    explainer.initialized = True


def initialize_interval_learner_for_fast_explainer(explainer: CalibratedExplainer) -> None:  # pylint: disable=invalid-name
    """Initialize the interval learner in fast explanation mode.

    This is a mechanical extraction of
    ``CalibratedExplainer.initialize_interval_learner_for_fast_explainer``
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
    - Sets the explainer's _initialized flag to True after successful initialization

    See Also
    --------
    initialize_interval_learner : Standard initialization for non-fast mode
    """
    orchestrator = getattr(explainer, "prediction_orchestrator", None)
    if orchestrator is not None:
        ensure_state = getattr(orchestrator, "ensure_interval_runtime_state", None)
        if callable(ensure_state):
            ensure_state()

    interval, _identifier = explainer.obtain_interval_calibrator(
        fast=True,
        metadata={"operation": "initialize_fast"},
    )
    explainer.interval_learner = interval
    explainer.initialized = True


__all__ = [
    "assign_threshold",
    "initialize_interval_learner",
    "initialize_interval_learner_for_fast_explainer",
    "update_interval_learner",
]
