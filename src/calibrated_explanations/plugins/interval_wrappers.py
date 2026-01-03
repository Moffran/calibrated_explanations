"""Protocol-aware interval calibrator wrappers (ADR-013)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


class FastIntervalCalibrator(Sequence[Any]):
    """Wrapper providing protocol methods for FAST interval calibrator lists."""

    def __init__(self, calibrators: Sequence[Any]) -> None:
        if not calibrators:
            raise ValueError("FAST interval calibrators collection cannot be empty.")
        self._calibrators = tuple(calibrators)

    @property
    def calibrators(self) -> tuple[Any, ...]:
        """Return the underlying calibrator tuple."""
        return self._calibrators

    def __getitem__(self, index: int) -> Any:
        return self._calibrators[index]

    def __len__(self) -> int:
        return len(self._calibrators)

    def __iter__(self) -> Iterable[Any]:
        return iter(self._calibrators)

    def _default(self) -> Any:
        return self._calibrators[-1]

    def predict_proba(
        self,
        x: Any,
        *,
        output_interval: bool = False,
        classes: Sequence[Any] | None = None,
        bins: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Delegate probability predictions to the default calibrator."""
        return self._default().predict_proba(
            x,
            output_interval=output_interval,
            classes=classes,
            bins=bins,
            **kwargs,
        )

    def predict_probability(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Delegate probabilistic regression predictions."""
        return self._default().predict_probability(x, *args, **kwargs)

    def predict_uncertainty(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Delegate uncertainty estimation to the default calibrator."""
        return self._default().predict_uncertainty(x, *args, **kwargs)

    def pre_fit_for_probabilistic(self, x: Any, y: Any) -> None:
        """Delegate probabilistic pre-fit hooks when available."""
        if hasattr(self._default(), "pre_fit_for_probabilistic"):
            self._default().pre_fit_for_probabilistic(x, y)

    def compute_proba_cal(self, x: Any, y: Any, *, weights: Any | None = None) -> Any:
        """Delegate probability calibration computation when available."""
        return self._default().compute_proba_cal(x, y, weights=weights)

    def insert_calibration(self, x: Any, y: Any, *, warm_start: bool = False) -> None:
        """Delegate insert_calibration when available."""
        return self._default().insert_calibration(x, y, warm_start=warm_start)

    def is_multiclass(self) -> bool:
        """Delegate multiclass checks when available."""
        if hasattr(self._default(), "is_multiclass"):
            return bool(self._default().is_multiclass())
        return False

    def is_mondrian(self) -> bool:
        """Delegate Mondrian checks when available."""
        if hasattr(self._default(), "is_mondrian"):
            return bool(self._default().is_mondrian())
        return False


def is_fast_interval_collection(value: Any) -> bool:
    """Return True when *value* is a FAST interval calibrator collection."""
    return isinstance(value, (FastIntervalCalibrator, list, tuple))


__all__ = ["FastIntervalCalibrator", "is_fast_interval_collection"]
