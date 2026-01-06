"""Explanation plugin protocol and shared data structures (ADR-015)."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping as MappingABC
from collections.abc import MutableMapping as MutableMappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from types import MappingProxyType
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
import numpy as np

if TYPE_CHECKING:
    from ..explanations.explanations import CalibratedExplanations as CalibratedExplanationsType
else:
    CalibratedExplanationsType = object
from ..utils.exceptions import ValidationError
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


class ExplainerHandle:
    """Read-only wrapper exposing a constrained explainer API to plugins."""

    __slots__ = ("__explainer", "__metadata")

    def __init__(self, explainer: Any, metadata: Mapping[str, Any]) -> None:
        self.__explainer = explainer
        self.__metadata = MappingProxyType(dict(metadata))

    @property
    def num_features(self) -> int:
        """Return the number of features."""
        return self.__explainer.num_features

    @property
    def mode(self) -> str:
        """Return the prediction mode."""
        return self.__explainer.mode

    @property
    def is_multiclass(self) -> bool:
        """Return True when in multiclass mode."""
        return self.__explainer.is_multiclass()

    @property
    def class_labels(self) -> Any:
        """Return class labels."""
        return self.__explainer.class_labels

    @property
    def feature_names(self) -> Any:
        """Return feature names."""
        return self.__explainer.feature_names

    @property
    def features_to_ignore(self) -> Any:
        """Return features to ignore."""
        return self.__explainer.features_to_ignore

    @property
    def learner(self) -> Any:
        """Return the underlying learner."""
        return self.__explainer.learner

    @property
    def bins(self) -> Any:
        """Return bins configuration."""
        return self.__explainer.bins

    @property
    def preprocessor(self) -> Any:
        """Return the preprocessor if available."""
        return getattr(self.__explainer, "preprocessor", None)

    @property
    def feature_filter_config(self) -> Any:
        """Return the feature filter configuration if available."""
        return getattr(self.__explainer, "feature_filter_config", None)

    @property
    def plugin_manager(self) -> Any:
        """Return the plugin manager if available."""
        return getattr(self.__explainer, "plugin_manager", None)

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Return calibrated predictions from the underlying explainer."""
        return self.__explainer.predict(*args, **kwargs)

    def explain_factual(self, *args: Any, **kwargs: Any) -> Any:
        """Return factual explanations from the underlying explainer."""
        return self.__explainer.explain_factual(*args, **kwargs)

    def explore_alternatives(self, *args: Any, **kwargs: Any) -> Any:
        """Return alternative explanations from the underlying explainer."""
        return self.__explainer.explore_alternatives(*args, **kwargs)

    def explain_fast(self, *args: Any, **kwargs: Any) -> Any:
        """Return fast explanations from the underlying explainer."""
        return self.__explainer.explain_fast(*args, **kwargs)

    def get_metadata(self) -> Mapping[str, Any]:
        """Return immutable metadata describing the explainer and dependencies."""
        return self.__metadata

    def get_preprocessor_state(self) -> Mapping[str, Any] | None:
        """Return the preprocessor metadata snapshot when available."""
        preprocessor_meta = getattr(self.__explainer, "preprocessor_metadata", None)
        if preprocessor_meta is None:
            return None
        if isinstance(preprocessor_meta, MappingABC):
            return MappingProxyType(dict(preprocessor_meta))
        return {"value": preprocessor_meta}


@dataclass(frozen=True)
class ExplanationRequest:
    """Frozen context for a specific explanation batch request."""

    threshold: Optional[object]
    low_high_percentiles: Optional[Tuple[float, float]]
    bins: Optional[object]
    features_to_ignore: Sequence[int] | Sequence[Sequence[int]]
    extras: Mapping[str, object] = field(default_factory=dict)
    feature_filter_per_instance_ignore: Sequence[Sequence[int]] | None = None

    def __post_init__(self) -> None:
        """Freeze mutable fields such as `bins` and `extras` for safety.

        This ensures plugins cannot mutate shared request state (Mondrian
        bins in particular) by converting arrays/lists into tuples and
        mapping payloads into read-only proxies.
        """

        def _freeze_value(val: object) -> object:
            # numpy arrays -> nested tuples
            if isinstance(val, np.ndarray):
                try:
                    return tuple(_freeze_value(x) for x in val.tolist())
                except Exception:
                    return tuple(val.tolist())
            if isinstance(val, (list, tuple)):
                return tuple(_freeze_value(x) for x in val)
            # mappings -> MappingProxyType with frozen values
            if isinstance(val, MappingABC):
                return MappingProxyType({k: _freeze_value(v) for k, v in val.items()})
            return val

        # Freeze bins if present
        frozen_bins = None if self.bins is None else _freeze_value(self.bins)
        object.__setattr__(self, "bins", frozen_bins)

        # Freeze extras into an immutable mapping
        try:
            frozen_extras = (
                MappingProxyType({k: _freeze_value(v) for k, v in dict(self.extras).items()})
                if self.extras is not None
                else MappingProxyType({})
            )
        except Exception:
            frozen_extras = MappingProxyType(dict(self.extras or {}))
        object.__setattr__(self, "extras", frozen_extras)


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
    "ExplainerHandle",
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
        raise ValidationError("explanation plugins must return an ExplanationBatch instance")

    container_cls = batch.container_cls
    if not isinstance(container_cls, type):
        raise ValidationError("batch.container_cls must be a class")

    def _inherits_calibrated_explanations(cls: type) -> bool:
        with contextlib.suppress(ImportError, TypeError):
            from ..explanations.explanations import (
                CalibratedExplanations,  # pylint: disable=import-outside-toplevel
            )

            if issubclass(cls, CalibratedExplanations):
                return True
        for base in getattr(cls, "__mro__", ()):  # pragma: no cover - defensive
            if base is cls:
                continue
            if base.__name__ == "CalibratedExplanations":
                return True
        return False

    if not _inherits_calibrated_explanations(container_cls):
        raise ValidationError("batch.container_cls must inherit from CalibratedExplanations")

    explanation_cls = batch.explanation_cls
    if not isinstance(explanation_cls, type):
        raise ValidationError("batch.explanation_cls must be a class")

    def _inherits_calibrated_explanation(cls: type) -> bool:
        with contextlib.suppress(ImportError, TypeError):
            from ..explanations.explanation import (
                CalibratedExplanation,
            )

            if issubclass(cls, CalibratedExplanation):
                return True
        for base in getattr(cls, "__mro__", ()):  # pragma: no cover - defensive
            if base is cls:
                continue
            if base.__name__ == "CalibratedExplanation":
                return True
        return False

    if not _inherits_calibrated_explanation(explanation_cls):
        raise ValidationError("batch.explanation_cls must inherit from CalibratedExplanation")

    instances = batch.instances
    if not isinstance(instances, SequenceABC) or isinstance(instances, (str, bytes)):
        raise ValidationError("batch.instances must be a sequence of mappings")
    for index, instance in enumerate(instances):
        if not isinstance(instance, MappingABC):
            raise ValidationError(
                f"batch.instances[{index}] must be a mapping describing the instance"
            )

    metadata = batch.collection_metadata
    if not isinstance(metadata, MutableMappingABC):
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

    for index, instance in enumerate(instances):
        prediction = instance.get("prediction")
        if isinstance(prediction, MappingABC):
            _validate_prediction_invariant(prediction, f"Instance {index} prediction")

    return batch


def _validate_prediction_invariant(payload: Mapping[str, Any], context: str) -> None:
    """Enforce low <= predict <= high invariant on prediction payload."""
    import numpy as np

    predict = payload.get("predict")
    low = payload.get("low")
    high = payload.get("high")

    if predict is None or low is None or high is None:
        return

    with contextlib.suppress(TypeError, ValueError):
        # Convert to numpy arrays for uniform handling
        predict_arr = np.asanyarray(predict)
        low_arr = np.asanyarray(low)
        high_arr = np.asanyarray(high)

        # Skip if any are empty
        if predict_arr.size == 0 or low_arr.size == 0 or high_arr.size == 0:
            return

        # Check for numeric types
        if not (
            np.issubdtype(predict_arr.dtype, np.number)
            and np.issubdtype(low_arr.dtype, np.number)
            and np.issubdtype(high_arr.dtype, np.number)
        ):
            return

        # Check low <= high
        if not np.all(low_arr <= high_arr):
            import warnings

            warnings.warn(
                f"{context}: interval invariant violated (low > high)",
                UserWarning,
                stacklevel=2,
            )

        # Check low <= predict <= high
        # Allow small floating point tolerance
        epsilon = 1e-9
        if not np.all((low_arr - epsilon <= predict_arr) & (predict_arr <= high_arr + epsilon)):
            import warnings

            warnings.warn(
                f"{context}: prediction invariant violated (predict not in [low, high])",
                UserWarning,
                stacklevel=2,
            )
