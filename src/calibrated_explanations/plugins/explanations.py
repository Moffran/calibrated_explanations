"""Explanation plugin protocol and shared data structures (ADR-015)."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from collections.abc import MutableMapping as MutableMappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from typing import (
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

from ..explanations.explanation import (
    CalibratedExplanation as AbstractCalibratedExplanation,
)
from ..explanations.explanations import CalibratedExplanations
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

    container_cls: Type[CalibratedExplanations]
    explanation_cls: Type[AbstractCalibratedExplanation]
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

    def explain_batch(self, X: Any, request: ExplanationRequest) -> ExplanationBatch:
        """Produce an :class:`ExplanationBatch` for payload *X*."""


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
    if not issubclass(container_cls, CalibratedExplanations):
        raise TypeError("batch.container_cls must inherit from CalibratedExplanations")

    explanation_cls = batch.explanation_cls
    if not isinstance(explanation_cls, type):
        raise TypeError("batch.explanation_cls must be a class")
    if not issubclass(explanation_cls, AbstractCalibratedExplanation):
        raise TypeError("batch.explanation_cls must inherit from CalibratedExplanation")

    instances = batch.instances
    if not isinstance(instances, SequenceABC) or isinstance(instances, (str, bytes)):
        raise TypeError("batch.instances must be a sequence of mappings")
    for index, instance in enumerate(instances):
        if not isinstance(instance, MappingABC):
            raise TypeError(f"batch.instances[{index}] must be a mapping describing the instance")

    metadata = batch.collection_metadata
    if not isinstance(metadata, MutableMappingABC):
        raise TypeError("batch.collection_metadata must be a mutable mapping")

    mode_hint = metadata.get("mode")
    if expected_mode is not None and mode_hint is not None and str(mode_hint) != expected_mode:
        raise ValueError(
            "ExplanationBatch metadata reports mode '"
            + str(mode_hint)
            + "' but runtime expected '"
            + expected_mode
            + "'"
        )

    task_hint = metadata.get("task")
    if expected_task is not None and task_hint is not None and str(task_hint) != expected_task:
        raise ValueError(
            "ExplanationBatch metadata reports task '"
            + str(task_hint)
            + "' but runtime expected '"
            + expected_task
            + "'"
        )

    container = metadata.get("container")
    if container is not None and not isinstance(container, container_cls):
        raise TypeError("ExplanationBatch metadata 'container' has unexpected type")

    return batch
