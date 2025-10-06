"""Interval calibrator protocol scaffolding (ADR-013)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class IntervalCalibratorContext:
    """Frozen view of calibration artefacts provided to interval plugins."""

    learner: object
    calibration_splits: Sequence[Any]
    bins: Mapping[str, Any]
    residuals: Mapping[str, Any]
    difficulty: Mapping[str, Any]
    metadata: Mapping[str, Any]
    fast_flags: Mapping[str, Any]


@runtime_checkable
class ClassificationIntervalCalibrator(Protocol):
    """Protocol exposing the VennAbers-style classification surface."""

    def predict_proba(
        self,
        x: Any,
        *,
        output_interval: bool = False,
        classes: Sequence[Any] | None = None,
        bins: Sequence[Any] | None = None,
    ) -> Any:
        """Return calibrated probabilities or intervals for *X*."""

    def is_multiclass(self) -> bool:
        """Return ``True`` when the calibrator handles multiclass data."""

    def is_mondrian(self) -> bool:
        """Return ``True`` when Mondrian binning is enabled."""


@runtime_checkable
class RegressionIntervalCalibrator(ClassificationIntervalCalibrator, Protocol):
    """Protocol extending classification calibrators with regression helpers."""

    def predict_probability(self, x: Any) -> Any:
        """Return calibrated low/high probabilities for *X*."""

    def predict_uncertainty(self, x: Any) -> Any:
        """Return uncertainty estimates for *X*."""

    def pre_fit_for_probabilistic(self, x: Any, y: Any) -> None:
        """Prepare the calibrator for probabilistic inference."""

    def compute_proba_cal(self, x: Any, y: Any, *, weights: Any | None = None) -> Any:
        """Compute probability calibration adjustments."""

    def insert_calibration(self, x: Any, y: Any, *, warm_start: bool = False) -> None:
        """Insert additional calibration samples."""


@runtime_checkable
class IntervalCalibratorPlugin(Protocol):
    """Protocol for factories that provide interval calibrators."""

    plugin_meta: Mapping[str, Any]

    def create(
        self, context: IntervalCalibratorContext, *, fast: bool = False
    ) -> ClassificationIntervalCalibrator:
        """Return a calibrator instance for the requested execution path."""


__all__ = [
    "ClassificationIntervalCalibrator",
    "IntervalCalibratorContext",
    "IntervalCalibratorPlugin",
    "RegressionIntervalCalibrator",
]
