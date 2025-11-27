"""Calibration metrics and diagnostic utilities."""

from collections import Counter

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold

from .calibration.venn_abers import VennAbers
from .exceptions import ValidationError


def compute_calibrated_confusion_matrix(
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    learner,
    bins: np.ndarray = None,
    stratified: bool = True,
) -> np.ndarray:
    """Compute a calibrated confusion matrix using cross-validation.

    Generates a confusion matrix for the calibration set using stratified cross-validation
    to avoid quadratic recalibration overhead. Uses Venn-Abers predictions to generate
    the confusion matrix.

    Parameters
    ----------
    x_cal : np.ndarray
        Calibration input data.
    y_cal : np.ndarray
        Calibration target data.
    learner : object
        The fitted predictive learner.
    bins : np.ndarray, optional
        Mondrian categories for conditional calibration. Default is None.
    stratified : bool, default=True
        If True, uses StratifiedKFold; otherwise uses KFold.

    Returns
    -------
    np.ndarray
        The calibrated confusion matrix.

    Raises
    ------
    ValidationError
        If no calibration samples are available.
    """
    y_cal = np.asarray(y_cal)
    bins = None if bins is None else np.asarray(bins)
    n_samples = len(y_cal)

    if n_samples == 0:
        raise ValidationError(
            "At least one calibration sample is required to build a confusion matrix."
        )

    cal_predicted_classes = np.empty_like(y_cal)

    # Determine the maximum feasible number of stratified folds.
    n_splits = min(10, n_samples)
    class_counts = Counter(y_cal)
    while n_splits > 1 and any(count < n_splits for count in class_counts.values()):
        n_splits -= 1

    # Single fold case: use all data without cross-validation
    if n_splits <= 1:
        va = VennAbers(x_cal, y_cal, learner, bins=bins)
        _, _, _, predict = va.predict_proba(
            x_cal,
            output_interval=True,
            bins=bins,
        )
        cal_predicted_classes[:] = predict
        return confusion_matrix(y_cal, cal_predicted_classes)

    # Multi-fold cross-validation
    if stratified and len(class_counts) > 1:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        split_iter = splitter.split(x_cal, y_cal)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        split_iter = splitter.split(x_cal)

    for train_idx, test_idx in split_iter:
        va = VennAbers(
            x_cal[train_idx],
            y_cal[train_idx],
            learner,
            bins=bins[train_idx] if bins is not None else None,
        )
        _, _, _, predict = va.predict_proba(
            x_cal[test_idx],
            output_interval=True,
            bins=bins[test_idx] if bins is not None else None,
        )
        cal_predicted_classes[test_idx] = predict

    return confusion_matrix(y_cal, cal_predicted_classes)


__all__ = ["compute_calibrated_confusion_matrix"]
