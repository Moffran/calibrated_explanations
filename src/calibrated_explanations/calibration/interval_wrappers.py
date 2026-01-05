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

def is_fast_interval_collection(value: Any) -> bool:
    """Return True when *value* is a FAST interval calibrator collection."""
    return isinstance(value, (FastIntervalCalibrator, list, tuple))
