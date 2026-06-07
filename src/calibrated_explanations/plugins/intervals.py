"""Interval calibrator protocol scaffolding (ADR-013)."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping, Protocol, Sequence, runtime_checkable

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


__all__ = [
    "ClassificationIntervalCalibrator",
    "IntervalCalibratorContext",
    "IntervalCalibratorPlugin",
    "RegressionIntervalCalibrator",
]
