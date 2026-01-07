from __future__ import annotations
"""Protocol-aware interval calibrator wrappers (ADR-013)."""
__all__ = ["FastIntervalCalibrator", "is_fast_interval_collection"]


from collections.abc import Iterable, Sequence
from typing import Any
from calibrated_explanations.core.exceptions import CalibratedError


class FastIntervalCalibrator(Sequence[Any]):
    """Wrapper providing protocol methods for FAST interval calibrator lists."""

    def __init__(self, calibrators: Sequence[Any]) -> None:
        if not calibrators:
            raise CalibratedError("FAST interval calibrators collection cannot be empty.")
        self._calibrators = tuple(calibrators)

    @property
    def calibrators(self) -> tuple[Any, ...]:
        """Return the underlying calibrator tuple."""
        return self._calibrators

    def __getitem__(self, index: int) -> Any:
        """Return the calibrator at the specified index.

        Parameters
        ----------
        index : int
            Index of the calibrator to retrieve.

        Returns
        -------
        Any
            The calibrator at the given index.
        """
        return self._calibrators[index]

    def __len__(self) -> int:
        """Return the number of calibrators in the wrapper.

        Returns
        -------
        int
            The number of calibrators.
        """
        return len(self._calibrators)

    def __iter__(self) -> Iterable[Any]:
        """Return an iterator over the calibrators.

        Returns
        -------
        Iterable[Any]
            An iterator over the calibrators.
        """
        return iter(self._calibrators)

    def _default(self) -> Any:
        return self._calibrators[-1]

    def predict_proba(
        self,
        *args,
        **kwargs,
    ) -> Any:
        return self._default().predict_proba(*args, **kwargs)

    def predict_probability(self, x: Any) -> Any:
        """Return calibrated low/high probabilities for *x*."""
        return self._default().predict_probability(x)

    def predict_uncertainty(self, x: Any) -> Any:
        """Return uncertainty estimates for *x*."""
        return self._default().predict_uncertainty(x)

    def is_multiclass(self) -> bool:
        """Return ``True`` when the calibrator handles multiclass data."""
        return self._default().is_multiclass()

    def is_mondrian(self) -> bool:
        """Return ``True`` when Mondrian binning is enabled."""
        return self._default().is_mondrian()

    def pre_fit_for_probabilistic(self, x: Any, y: Any) -> None:
        """Prepare the calibrator for probabilistic inference."""
        self._default().pre_fit_for_probabilistic(x, y)

    def compute_proba_cal(self, x: Any, y: Any, *, weights: Any | None = None) -> Any:
        """Compute probability calibration adjustments."""
        return self._default().compute_proba_cal(x, y, weights=weights)

    def insert_calibration(self, x: Any, y: Any, *, warm_start: bool = False) -> None:
        """Insert additional calibration samples."""
        self._default().insert_calibration(x, y, warm_start=warm_start)

def is_fast_interval_collection(value: Any) -> bool:
    """Return True when *value* is a FAST interval calibrator collection."""
    return isinstance(value, (FastIntervalCalibrator, list, tuple))
