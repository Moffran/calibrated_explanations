"""Explanation plugin protocol and shared data structures (ADR-015)."""

from __future__ import annotations

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
]

