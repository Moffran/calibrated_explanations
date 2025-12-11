"""Prediction bridge monitoring and instrumentation for plugins.

This module provides runtime guards and telemetry collection for tracking
which bridge methods are invoked during explanation pipeline execution.
Used for debugging, validation, and usage analytics.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

import numpy as np

from ..plugins.predict import PredictBridge


class PredictBridgeMonitor(PredictBridge):
    """Runtime guard ensuring plugins use the calibrated predict bridge.

    This class wraps an active prediction bridge and records which methods
    are called during explanation pipeline execution. Used for:
    - Validating plugin compliance with the bridge contract
    - Collecting telemetry on prediction bridge usage patterns
    - Debugging unexpected behavior during explanation generation

    Attributes
    ----------
    _bridge : PredictBridge
        The underlying bridge being monitored.
    _calls : List[str]
        Recorded method invocations ("predict", "predict_interval", "predict_proba").

    Examples
    --------
    >>> from calibrated_explanations.plugins.predict import PredictBridge
    >>> bridge = PredictBridge(...)
    >>> monitor = PredictBridgeMonitor(bridge)
    >>> monitor.predict(x, mode="decision", task="classification")
    >>> print(monitor.calls)
    ('predict',)
    """

    def __init__(self, bridge: PredictBridge) -> None:
        """Wrap the active predict bridge and start recording usage.

        Parameters
        ----------
        bridge : PredictBridge
            The prediction bridge to monitor and wrap.
        """
        self._bridge = bridge
        self._calls: List[str] = []

    def reset_usage(self) -> None:
        """Clear recorded bridge interactions.

        Call this to reset the call history before executing a new
        explanation pipeline to get clean usage metrics.
        """
        self._calls.clear()

    def predict(
        self,
        x: Any,
        *,
        mode: str,
        task: str,
        bins: Any | None = None,
    ) -> Mapping[str, Any]:
        """Forward predict calls while tagging invocation history.

        Parameters
        ----------
        x : Any
            Input features.
        mode : str
            Prediction mode (e.g., "decision", "proba").
        task : str
            Task type ("classification", "regression").
        bins : Any, optional
            Binning configuration.

        Returns
        -------
        Mapping[str, Any]
            Prediction results from the wrapped bridge.
        """
        self._calls.append("predict")
        result = self._bridge.predict(x, mode=mode, task=task, bins=bins)
        self._validate_invariants(result)
        return result

    def predict_interval(
        self,
        x: Any,
        *,
        task: str,
        bins: Any | None = None,
    ) -> Sequence[Any]:
        """Proxy interval predictions and record the access.

        Parameters
        ----------
        x : Any
            Input features.
        task : str
            Task type.
        bins : Any, optional
            Binning configuration.

        Returns
        -------
        Sequence[Any]
            Interval prediction results.
        """
        self._calls.append("predict_interval")
        result = self._bridge.predict_interval(x, task=task, bins=bins)
        # Result is typically (predict, low, high, classes) or similar tuple
        if isinstance(result, (tuple, list)) and len(result) >= 3:
            # Map tuple to dict-like structure for validation
            # Assuming standard order: predict, low, high
            payload = {
                "predict": result[0],
                "low": result[1],
                "high": result[2],
            }
            self._validate_invariants(payload)
        return result

    def _validate_invariants(self, payload: Mapping[str, Any]) -> None:
        """Enforce low <= predict <= high invariant.

        Parameters
        ----------
        payload : dict
            Prediction payload containing 'predict', 'low', 'high'.

        Raises
        ------
        ValidationError
            If the invariant is violated.
        """
        from collections.abc import Mapping

        if not isinstance(payload, Mapping):
            return

        if "predict" not in payload or "low" not in payload or "high" not in payload:
            return

        try:
            predict = np.asanyarray(payload["predict"])
            low = np.asanyarray(payload["low"])
            high = np.asanyarray(payload["high"])

            # Skip validation if any component is None (e.g. classification without interval)
            if predict.size == 0 or low.size == 0 or high.size == 0:
                return

            # Check for numeric types to avoid ufunc errors with strings/objects
            if not (
                np.issubdtype(predict.dtype, np.number)
                and np.issubdtype(low.dtype, np.number)
                and np.issubdtype(high.dtype, np.number)
            ):
                return

            # Check low <= high
            if not np.all(low <= high):
                import warnings

                warnings.warn(
                    "Prediction interval invariant violated: low > high. This indicates an issue with the underlying estimator.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            # Check low <= predict <= high
            # Note: We use a small epsilon for float comparison if needed, but strict inequality is safer for now
            epsilon = 1e-9
            if not np.all((low - epsilon <= predict) & (predict <= high + epsilon)):
                import warnings

                warnings.warn(
                    "Prediction invariant violated: predict not in [low, high]. This may indicate poor calibration or inconsistent point predictions.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        except (TypeError, ValueError):
            # Skip validation if type conversion or comparison fails
            pass

    def predict_proba(self, x: Any, bins: Any | None = None) -> Sequence[Any]:
        """Delegate ``predict_proba`` while tracking usage.

        Parameters
        ----------
        x : Any
            Input features.
        bins : Any, optional
            Binning configuration.

        Returns
        -------
        Sequence[Any]
            Probabilistic predictions from the wrapped bridge.
        """
        self._calls.append("predict_proba")
        return self._bridge.predict_proba(x, bins=bins)

    @property
    def calls(self) -> Tuple[str, ...]:
        """Return a tuple describing which bridge methods were used.

        Returns
        -------
        Tuple[str, ...]
            Immutable snapshot of recorded method names.

        Examples
        --------
        >>> monitor.calls
        ('predict', 'predict_interval', 'predict')
        """
        return tuple(self._calls)

    @property
    def used(self) -> bool:
        """Return True when the monitor observed any bridge invocation.

        Returns
        -------
        bool
            True if any predict method was called, False otherwise.
        """
        return bool(self._calls)
