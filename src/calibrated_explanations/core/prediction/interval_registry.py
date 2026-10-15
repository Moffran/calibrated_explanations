"""Interval calibrator lifecycle and state management.

This module provides the IntervalRegistry class which manages the initialization
and state of interval calibrators used for uncertainty quantification. Updates
with appended calibration data go through
``calibration.interval_learner.update_interval_learner`` (via
``CalibratedExplainer.reinitialize``), the single authoritative update path.

Part of Phase 4: Delegate Prediction Functionality (ADR-001, ADR-004).
"""

# pylint: disable=protected-access, invalid-name, import-outside-toplevel

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...calibration.interval_learner import (
    initialize_interval_learner,
    initialize_interval_learner_for_fast_explainer,
)

if TYPE_CHECKING:
    from ..calibrated_explainer import CalibratedExplainer


class IntervalRegistry:
    """Manage interval calibrator lifecycle and operations.

    This class encapsulates interval learner state and methods for initialization
    and difficulty estimation. It serves as the centralized registry for
    interval calibration within the prediction orchestrator.

    Attributes
    ----------
    explainer : CalibratedExplainer
        Back-reference to the parent explainer instance.
    interval_learner : Any
        The active interval calibrator (e.g., VennAbers, IntervalRegressor, or list for fast mode).
    """

    def __init__(self, explainer: CalibratedExplainer) -> None:
        """Initialize the interval registry with a back-reference to the explainer.

        Parameters
        ----------
        explainer : CalibratedExplainer
            The parent explainer instance.
        """
        self.explainer = explainer
        self.interval_learner: Any = None

    def get_sigma_test(self, x: np.ndarray) -> np.ndarray:
        """Return the difficulty (sigma) of the test instances.

        Uses the configured difficulty estimator if available; otherwise returns
        a unit vector indicating constant difficulty.

        Parameters
        ----------
        x : np.ndarray
            Test instances for which to estimate difficulty.

        Returns
        -------
        np.ndarray
            Difficulty estimates (sigma values) for each test instance.
        """
        if self.explainer.difficulty_estimator is None:
            return self._constant_sigma(x)
        return self.explainer.difficulty_estimator.apply(x)

    def _constant_sigma(self, x: np.ndarray, learner=None, beta=None) -> np.ndarray:
        """Return a unit difficulty vector when no estimator is configured.

        Parameters
        ----------
        x : np.ndarray
            Test instances (shape information used for output sizing).
        learner : optional
            Unused; retained for backward compatibility.
        beta : optional
            Unused; retained for backward compatibility.

        Returns
        -------
        np.ndarray
            Unit vector of ones with length matching the number of instances in x.
        """
        # pylint: disable=unused-argument
        return np.ones(x.shape[0]) if isinstance(x, (np.ndarray, list, tuple)) else np.ones(1)

    def constant_sigma(self, x: np.ndarray, learner=None, beta=None) -> np.ndarray:
        """Public wrapper returning a unit difficulty vector when no estimator is configured.

        Tests and external callers should use this public helper instead of the
        internal `_constant_sigma` protected method.
        """
        return self._constant_sigma(x, learner=learner, beta=beta)

    def initialize(self) -> None:
        """Create the interval learner backend using calibration helpers.

        This is a thin delegator that preserves the initialization sequence
        defined in calibration_helpers.
        """
        initialize_interval_learner(self.explainer)

    def initialize_for_fast_explainer(self) -> None:
        """Provision fast-path interval learners for Mondrian explanations.

        This method sets up interval calibrators optimized for the fast explanation path,
        typically creating multiple interval learners for different feature bins.
        """
        initialize_interval_learner_for_fast_explainer(self.explainer)
