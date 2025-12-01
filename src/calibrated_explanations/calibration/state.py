"""Calibration data state management.

This module provides the data structure and access patterns for managing
calibration datasets within the explainer. It handles x_cal and y_cal
properties with proper validation and format conversion.

Part of ADR-001: Core Decomposition Boundaries (Stage 1a).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from calibrated_explanations.core import DataShapeError
from ..utils import safe_isinstance

if TYPE_CHECKING:
    from calibrated_explanations.core import CalibratedExplainer


class CalibrationState:
    """Encapsulate calibration dataset state and property access.

    This class is a thin wrapper around the explainer's calibration data,
    providing validated access to x_cal and y_cal while delegating storage
    to the explainer instance itself.
    """

    @staticmethod
    def set_x_cal(explainer: CalibratedExplainer, value: Any) -> None:
        """Set the calibration input data with validation and format conversion.

        Parameters
        ----------
        explainer : CalibratedExplainer
            The explainer instance to update.
        value : array-like of shape (n_samples, n_features)
            The new calibration input data.

        Notes
        -----
        - Converts pandas DataFrames to numpy arrays
        - Reshapes 1D arrays to (n_samples, 1)
        - Converts dict-of-dicts format to numpy arrays
        - Invalidates calibration summary caches
        """
        if safe_isinstance(value, "pandas.core.frame.DataFrame"):
            value = value.values

        if len(value.shape) == 1:
            value = value.reshape(1, -1)

        explainer._X_cal = value

        if isinstance(explainer._X_cal[0], dict):
            explainer._CalibratedExplainer__X_cal = np.array(
                [[x[f] for f in x] for x in explainer._X_cal]
            )

        # Invalidate summary caches when data changes
        from .summaries import invalidate_calibration_summaries

        invalidate_calibration_summaries(explainer)

    @staticmethod
    def get_x_cal(explainer: CalibratedExplainer) -> Any:
        """Get the calibration input data.

        Returns
        -------
        array-like
            The calibration input data. Returns the dict-converted numpy array
            if input data is dict format, otherwise returns the raw array.
        """
        return (
            explainer._CalibratedExplainer__X_cal
            if isinstance(explainer._X_cal[0], dict)
            else explainer._X_cal
        )

    @staticmethod
    def set_y_cal(explainer: CalibratedExplainer, value: Any) -> None:
        """Set the calibration target data with validation and format conversion.

        Parameters
        ----------
        explainer : CalibratedExplainer
            The explainer instance to update.
        value : array-like of shape (n_samples,)
            The new calibration target data.

        Notes
        -----
        - Converts pandas DataFrames to numpy arrays
        - Flattens 2D arrays with shape (n, 1) to 1D
        - Invalidates calibration summary caches
        """
        if safe_isinstance(value, "pandas.core.frame.DataFrame"):
            explainer._y_cal = np.asarray(value.values)
        else:
            if len(value.shape) == 2 and value.shape[1] == 1:
                value = value.ravel()
            explainer._y_cal = np.asarray(value)

        # Invalidate summary caches when data changes
        from .summaries import invalidate_calibration_summaries

        invalidate_calibration_summaries(explainer)

    @staticmethod
    def get_y_cal(explainer: CalibratedExplainer) -> Any:
        """Get the calibration target data.

        Returns
        -------
        array-like
            The calibration target data.
        """
        return explainer._y_cal

    @staticmethod
    def append_calibration(explainer: CalibratedExplainer, x: Any, y: Any) -> None:
        """Append new calibration data to existing calibration set.

        Parameters
        ----------
        explainer : CalibratedExplainer
            The explainer instance to update.
        x : array-like of shape (n_samples, n_features)
            The new calibration input data to append.
        y : array-like of shape (n_samples,)
            The new calibration target data to append.

        Raises
        ------
        DataShapeError
            If the number of features in x does not match existing calibration data.

        Notes
        -----
        Uses the property setters to ensure proper format conversion and cache invalidation.
        """
        if x.shape[1] != explainer.num_features:
            raise DataShapeError("Number of features must match existing calibration data")

        # Use property setters to handle format conversion and cache invalidation
        x_cal = CalibrationState.get_x_cal(explainer)
        y_cal = CalibrationState.get_y_cal(explainer)

        new_x = np.vstack((x_cal, x))
        new_y = np.concatenate((y_cal, y))

        CalibrationState.set_x_cal(explainer, new_x)
        CalibrationState.set_y_cal(explainer, new_y)
