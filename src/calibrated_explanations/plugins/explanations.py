"""Explanation plugin protocol and shared data structures (ADR-015)."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from collections.abc import MutableMapping as MutableMappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    runtime_checkable,
)

if TYPE_CHECKING:
    from ..explanations.explanations import CalibratedExplanations as CalibratedExplanationsType
else:
    CalibratedExplanationsType = object
from .base import ExplainerPlugin, PluginMeta
from .predict import PredictBridge


@dataclass(frozen=True)
class ExplanationContext:
    """Frozen request-independent context shared with explanation plugins."""

    task: str
    mode: str
    feature_names: Sequence[str]
    categorical_features: Sequence[int]
    categorical_labels: Mapping[int, Mapping[int, str]]
    discretizer: object
    helper_handles: Mapping[str, object]
    predict_bridge: PredictBridge
    interval_settings: Mapping[str, object]
    plot_settings: Mapping[str, object]


@dataclass(frozen=True)
class ExplanationRequest:
    """Frozen context for a specific explanation batch request."""

    threshold: Optional[object]
    low_high_percentiles: Optional[Tuple[float, float]]
    bins: Optional[object]
    features_to_ignore: Sequence[int]
    extras: Mapping[str, object]


@dataclass
class ExplanationBatch:
    """Batch payload returned by ``ExplanationPlugin.explain_batch``."""

    container_cls: Type[CalibratedExplanationsType]
    explanation_cls: Type  # CalibratedExplanation (deferred import to avoid circular dependency)
    instances: Sequence[Mapping[str, Any]]
    collection_metadata: MutableMapping[str, Any]


@runtime_checkable
class ExplanationPlugin(ExplainerPlugin, Protocol):
    """Extended protocol for explanation plugins."""

    plugin_meta: PluginMeta

    def supports_mode(self, mode: str, *, task: str) -> bool:
        """Return ``True`` when the plugin supports *mode* for *task*."""

    def initialize(self, context: ExplanationContext) -> None:
        """Initialise the plugin with immutable runtime *context*."""

    def explain_batch(self, x: Any, request: ExplanationRequest) -> ExplanationBatch:
        """Produce an :class:`ExplanationBatch` for payload *x*."""


__all__ = [
    "ExplanationBatch",
    "ExplanationContext",
    "ExplanationPlugin",
    "ExplanationRequest",
    "PluginMeta",
    "validate_explanation_batch",
]


def validate_explanation_batch(
    batch: ExplanationBatch,
    *,
    expected_mode: str | None = None,
    expected_task: str | None = None,
) -> ExplanationBatch:
    """Validate runtime contracts for ``ExplanationBatch`` payloads."""
    if not isinstance(batch, ExplanationBatch):
        raise TypeError("explanation plugins must return an ExplanationBatch instance")

    container_cls = batch.container_cls
    if not isinstance(container_cls, type):
        raise TypeError("batch.container_cls must be a class")

    def _inherits_calibrated_explanations(cls: type) -> bool:
        try:
            from ..explanations.explanations import (
                CalibratedExplanations,  # pylint: disable=import-outside-toplevel
            )

            if issubclass(cls, CalibratedExplanations):
                return True
        except (ImportError, TypeError):
            pass
        # Fall back to name-based check in case multiple module copies exist (e.g. notebooks)
        for base in getattr(cls, "__mro__", ()):
            if base is cls:
                continue
            if base.__name__ == "CalibratedExplanations":
                return True
        return False

    if not _inherits_calibrated_explanations(container_cls):
        raise TypeError("batch.container_cls must inherit from CalibratedExplanations")

    explanation_cls = batch.explanation_cls
    if not isinstance(explanation_cls, type):
        raise TypeError("batch.explanation_cls must be a class")

    def _inherits_calibrated_explanation(cls: type) -> bool:
        try:
            # Attempt direct import-based check (may fail due to circular imports)
            from ..explanations.explanation import (
                CalibratedExplanation,
            )

            if issubclass(cls, CalibratedExplanation):
                return True
        except (ImportError, TypeError):
            pass
        # Fall back to name-based check in case of circular imports or multiple module copies
        for base in getattr(cls, "__mro__", ()):
            if base is cls:
                continue
            if base.__name__ == "CalibratedExplanation":
                return True
        return False

    if not _inherits_calibrated_explanation(explanation_cls):
        raise TypeError("batch.explanation_cls must inherit from CalibratedExplanation")

    instances = batch.instances
    if not isinstance(instances, SequenceABC) or isinstance(instances, (str, bytes)):
        raise TypeError("batch.instances must be a sequence of mappings")
    for index, instance in enumerate(instances):
        if not isinstance(instance, MappingABC):
            raise TypeError(f"batch.instances[{index}] must be a mapping describing the instance")

    metadata = batch.collection_metadata
    if not isinstance(metadata, MutableMappingABC):
        from ..core.exceptions import ValidationError

        raise ValidationError(
            "batch.collection_metadata must be a mutable mapping",
            details={
                "param": "batch.collection_metadata",
                "expected_type": "MutableMapping",
                "actual_type": type(metadata).__name__,
            },
        )

    mode_hint = metadata.get("mode")
    if expected_mode is not None and mode_hint is not None and str(mode_hint) != expected_mode:
        from ..core.exceptions import ValidationError

        raise ValidationError(
            "ExplanationBatch metadata reports mode '"
            + str(mode_hint)
            + "' but runtime expected '"
            + expected_mode
            + "'",
            details={
                "param": "mode",
                "expected": expected_mode,
                "actual": str(mode_hint),
                "source": "batch.collection_metadata",
            },
        )

    task_hint = metadata.get("task")
    if expected_task is not None and task_hint is not None and str(task_hint) != expected_task:
        from ..core.exceptions import ValidationError

        raise ValidationError(
            "ExplanationBatch metadata reports task '"
            + str(task_hint)
            + "' but runtime expected '"
            + expected_task
            + "'",
            details={
                "param": "task",
                "expected": expected_task,
                "actual": str(task_hint),
                "source": "batch.collection_metadata",
            },
        )

    # Validate instance payloads for interval invariants
    for index, instance in enumerate(instances):
        prediction = instance.get("prediction")
        if isinstance(prediction, MappingABC):
            _validate_prediction_invariant(prediction, f"Instance {index} prediction")

    return batch


def _validate_prediction_invariant(payload: Mapping[str, Any], context: str) -> None:
    """Enforce low <= predict <= high invariant on prediction payload."""
    from ..core.exceptions import ValidationError

    predict = payload.get("predict")
    low = payload.get("low")
    high = payload.get("high")

    if predict is None or low is None or high is None:
        return

    try:
        # Handle scalar values (common case)
        if isinstance(predict, (int, float)) and isinstance(low, (int, float)) and isinstance(high, (int, float)):
            if not low <= high:
                raise ValidationError(
                    f"{context}: interval invariant violated (low > high)",
                    details={"low": low, "high": high},
                )
            # Use small epsilon for float comparison if needed, but strict for now
            # Allow small floating point tolerance
            epsilon = 1e-9
            if not (low - epsilon <= predict <= high + epsilon):
                raise ValidationError(
                    f"{context}: prediction invariant violated (predict not in [low, high])",
                    details={"predict": predict, "low": low, "high": high},
                )
    except (TypeError, ValueError):
        # Skip validation for non-numeric types
        pass

