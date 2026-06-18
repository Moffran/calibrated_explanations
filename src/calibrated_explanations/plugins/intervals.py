"""Interval calibrator protocol scaffolding (ADR-013)."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping, Protocol, Sequence, runtime_checkable

import numpy as np

from ..utils.exceptions import ValidationError
from .base import freeze_plugin_config, thaw_plugin_config


@dataclass(frozen=True)
class IntervalCalibratorContext:
    """Frozen view of calibration artefacts provided to interval plugins.

    Metadata is exposed as an immutable mapping; use ``plugin_state`` for transient storage.
    """

    learner: object
    calibration_splits: Sequence[Any]
    bins: Mapping[str, Any]
    residuals: Mapping[str, Any]
    difficulty: Mapping[str, Any]
    metadata: Mapping[str, Any]
    fast_flags: Mapping[str, Any]
    plugin_config: Mapping[str, Any] = field(default_factory=dict)
    plugin_state: MutableMapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Post-initialization hook for the dataclass."""
        # Ensure metadata is always exposed as an immutable mapping.
        raw_metadata = self.metadata or {}
        try:
            normalized = dict(raw_metadata)
        except Exception:  # adr002_allow
            normalized = (
                {} if raw_metadata is None else dict(getattr(raw_metadata, "items", lambda: ())())
            )
        object.__setattr__(self, "metadata", MappingProxyType(normalized))
        object.__setattr__(self, "plugin_config", freeze_plugin_config(self.plugin_config))
        # Freeze mutable container fields. calibration_splits elements may be numpy
        # arrays (accepted array-payload exception); only the outer sequence is frozen.
        object.__setattr__(self, "calibration_splits", tuple(self.calibration_splits or ()))
        if self.bins is not None:
            object.__setattr__(self, "bins", MappingProxyType(dict(self.bins)))
        if self.residuals is not None:
            object.__setattr__(self, "residuals", MappingProxyType(dict(self.residuals)))
        if self.difficulty is not None:
            object.__setattr__(self, "difficulty", MappingProxyType(dict(self.difficulty)))
        if self.fast_flags is not None:
            object.__setattr__(self, "fast_flags", MappingProxyType(dict(self.fast_flags)))
        # Ensure plugin_state is mutable so plugins can store transient data.
        if not isinstance(self.plugin_state, MutableMapping):  # pragma: no cover - defensive
            object.__setattr__(self, "plugin_state", dict(self.plugin_state))  # type: ignore[arg-type]

    def __getstate__(self) -> dict:
        """Return pickle-safe state with all MappingProxyType values thawed to plain dicts."""
        return {k: thaw_plugin_config(v) for k, v in self.__dict__.items()}

    def __setstate__(self, state: dict) -> None:
        """Restore state, re-freezing metadata and plugin_config, ensuring plugin_state is mutable."""
        for key, value in state.items():
            if key == "metadata":
                value = MappingProxyType(dict(value)) if value is not None else None
            elif key == "plugin_config":
                value = freeze_plugin_config(value if value is not None else {})
            elif key == "plugin_state" and not isinstance(value, MutableMapping):
                value = dict(value) if value is not None else {}
            object.__setattr__(self, key, value)


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
        interval_summary: Any | None = None,
    ) -> Any:
        """Return calibrated probabilities or intervals for *x*."""

    def is_multiclass(self) -> bool:
        """Return ``True`` when the calibrator handles multiclass data."""

    def is_mondrian(self) -> bool:
        """Return ``True`` when Mondrian binning is enabled."""


@runtime_checkable
class RegressionIntervalCalibrator(ClassificationIntervalCalibrator, Protocol):
    """Protocol extending classification calibrators with regression helpers."""

    def predict_probability(self, x: Any, *, interval_summary: Any | None = None) -> Any:
        """Return calibrated low/high probabilities for *x*."""

    def predict_uncertainty(self, x: Any) -> Any:
        """Return uncertainty estimates for *x*."""

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


def validate_interval_calibrator_output(
    result: Any,
    context: IntervalCalibratorContext,
    *,
    identifier: str | None = None,
    output_interval: bool = False,
) -> None:
    """Validate ADR-013 interval calibrator output shape, dtype, and bounds."""
    label = identifier or "<unknown>"
    expected_rows = _expected_calibration_rows(context)
    if not output_interval and isinstance(result, (list, tuple)) and len(result) == 2:
        result = result[0]
    array = np.asanyarray(result)

    if not np.issubdtype(array.dtype, np.floating):
        raise ValidationError(
            f"Interval calibrator '{label}' returned non-floating output dtype",
            details={"identifier": label, "dtype": str(array.dtype)},
        )
    if expected_rows is not None and array.shape[:1] != (expected_rows,):
        raise ValidationError(
            f"Interval calibrator '{label}' returned output with unexpected row count",
            details={
                "identifier": label,
                "expected_rows": expected_rows,
                "actual_shape": tuple(array.shape),
            },
        )

    if output_interval:
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValidationError(
                f"Interval calibrator '{label}' interval output must have shape (n, classes, 3)",
                details={"identifier": label, "actual_shape": tuple(array.shape)},
            )
        predict = array[..., 0]
        low = array[..., 1]
        high = array[..., 2]
        if not np.all(low <= high):
            raise ValidationError(
                f"Interval calibrator '{label}' interval output violates low <= high",
                details={"identifier": label},
            )
        epsilon = 1e-9
        if not np.all((low - epsilon <= predict) & (predict <= high + epsilon)):
            raise ValidationError(
                f"Interval calibrator '{label}' interval output violates low <= predict <= high",
                details={"identifier": label},
            )
        return

    if array.ndim != 2:
        raise ValidationError(
            f"Interval calibrator '{label}' probability output must have shape (n, classes)",
            details={"identifier": label, "actual_shape": tuple(array.shape)},
        )
    if not np.all((array >= -1e-9) & (array <= 1.0 + 1e-9)):
        raise ValidationError(
            f"Interval calibrator '{label}' probability output is outside [0, 1]",
            details={"identifier": label},
        )
    row_sums = array.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValidationError(
            f"Interval calibrator '{label}' probability output rows must sum to 1",
            details={"identifier": label, "row_sums": row_sums.tolist()},
        )


def _expected_calibration_rows(context: IntervalCalibratorContext) -> int | None:
    """Return the expected calibration sample count for validation."""
    if not context.calibration_splits:
        return None
    try:
        features = context.calibration_splits[0][0]
    except (IndexError, TypeError):
        return None
    try:
        return int(len(features))
    except TypeError:
        return None


__all__ = [
    "ClassificationIntervalCalibrator",
    "IntervalCalibratorContext",
    "IntervalCalibratorPlugin",
    "RegressionIntervalCalibrator",
    "validate_interval_calibrator_output",
]
