# pylint: disable=unknown-option-value, too-many-arguments
# pylint: disable=too-many-lines, too-many-public-methods, invalid-name, too-many-positional-arguments, line-too-long
"""Containers for storing, exporting, and visualising calibrated explanations.

This module implements :class:`CalibratedExplanations`, a container that
holds per-instance explanation objects (factual, alternative, fast) and
provides helpers for exporting, iterating and aggregating explanation
collections.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
import tracemalloc
import warnings
from collections.abc import Sequence as ABCSequence
from copy import copy, deepcopy
from dataclasses import dataclass
from itertools import permutations
from time import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy as np

from ..core.prediction_helpers import validate_and_prepare_input
from ..utils import EntropyDiscretizer, RegressorDiscretizer, prepare_for_saving
from ..utils.exceptions import ValidationError
from ..utils.helper import calculate_metrics
from .adapters import legacy_to_domain
from .explanation import AlternativeExplanation, FactualExplanation, FastExplanation
from .models import Explanation as DomainExplanation

_LOGGER = logging.getLogger(__name__)


def _plot_alternative_dict(*args, **kwargs):
    """Lazy wrapper to avoid importing plotting dependencies at module import time."""
    from ..plotting import _plot_alternative_dict as _impl

    return _impl(*args, **kwargs)


def _plot_probabilistic_dict(*args, **kwargs):
    """Lazy wrapper to avoid importing plotting dependencies at module import time."""
    from ..plotting import _plot_probabilistic_dict as _impl

    return _impl(*args, **kwargs)


def get_multiclass_config():
    """Lazy wrapper to avoid importing plotting dependencies at module import time."""
    from ..plotting import get_multiclass_config as _impl

    return _impl()


@dataclass(frozen=True)
class ExportedExplanationCollection:
    """Lightweight representation of exported explanations plus collection metadata."""

    metadata: Mapping[str, Any]
    explanations: Sequence[DomainExplanation]

    def __getstate__(self):
        """Get state for pickling.

        Returns
        -------
        dict
            The state dictionary.
        """
        # Convert mappingproxy to dict for pickling
        return dict(self.__dict__)


@dataclass(frozen=True)
class ExportedMultiClassExplanationCollection:
    """Exported multiclass explanations grouped by instance and class index."""

    metadata: Mapping[str, Any]
    explanations_by_instance: Sequence[Mapping[int, DomainExplanation]]

    @property
    def explanations(self) -> Sequence[DomainExplanation]:
        """Return flattened exported explanations for backward-compatible access."""
        flattened: list[DomainExplanation] = []
        for per_instance in self.explanations_by_instance:
            flattened.extend(per_instance.values())
        return tuple(flattened)

    def __getstate__(self):
        """Get state for pickling.

        Returns
        -------
        dict
            The state dictionary.
        """
        return dict(self.__dict__)


def _jsonify(value: Any) -> Any:
    """Convert numpy objects and arrays into JSON-serialisable primitives."""
    if isinstance(value, np.ndarray):
        return [_jsonify(item) for item in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, np.generic):  # numpy scalars
        return value.item()
    if callable(value):
        return str(value)
    return value


class CalibratedExplanations:  # pylint: disable=too-many-instance-attributes
    """A class for storing and visualizing calibrated explanations.

    This class is created by :class:`.CalibratedExplainer` and provides methods for managing
    and accessing explanations for test instances.
    """

    def __init__(
        self,
        calibrated_explainer,
        x,
        y_threshold,
        bins,
        features_to_ignore=None,
        *,
        condition_source: str = "prediction",
    ) -> None:
        """Initialize the explanation collection for a calibrated explainer.

        Parameters
        ----------
        calibrated_explainer : CalibratedExplainer
            The calibrated explainer object.
        x : array-like
            The test data.
        y_threshold : float or tuple
            The threshold for regression explanations.
        bins : array-like
            The bins for conditional explanations.
        """
        if condition_source not in {"observed", "prediction"}:
            raise ValidationError(
                "condition_source must be 'observed' or 'prediction'",
                details={"param": "condition_source", "value": condition_source},
            )

        self.calibrated_explainer: FrozenCalibratedExplainer = FrozenCalibratedExplainer(
            calibrated_explainer
        )
        self.condition_source: str = condition_source
        self.x_test: np.ndarray = x
        self.y_threshold: Optional[Union[float, Tuple[float, float], List[Tuple[float, float]]]] = (
            y_threshold
        )
        self.low_high_percentiles: Optional[Tuple[float, float]] = None
        self.explanations: List[
            Union[FactualExplanation, AlternativeExplanation, FastExplanation]
        ] = []
        self.start_index: int = 0
        self.current_index: int = self.start_index
        self.end_index: int = len(x[:, 0])
        self.bins: Optional[Sequence[Any]] = bins
        self.total_explain_time: Optional[float] = None
        self.features_to_ignore: List[int] = (
            features_to_ignore if features_to_ignore is not None else []
        )
        # Optional per-instance feature ignore masks produced by the internal
        # FAST-based feature filter. When present, each entry corresponds to
        # the indices ignored for that instance on top of any global ignore.
        self.feature_filter_per_instance_ignore: Optional[Sequence[Sequence[int]]] = None
        # Optional telemetry from the internal FAST-based feature filter.
        self.filter_telemetry: Optional[Dict[str, Any]] = None
        # Derived caches (set during finalize of individual explanations)
        self._feature_names_cache: Optional[Sequence[str]] = None  # populated lazily
        self._predictions_cache: Optional[np.ndarray] = None
        self._probabilities_cache: Optional[np.ndarray] = None  # classification only
        self._lower_cache: Optional[np.ndarray] = None  # regression only
        self._upper_cache: Optional[np.ndarray] = None
        self._class_labels_cache: Optional[Sequence[str]] = None  # classification only

    def __iter__(self):
        """Return an iterator for the explanations."""
        self.current_index = self.start_index
        return self

    def __next__(self):
        """Return the next explanation."""
        if self.current_index >= self.end_index:
            raise StopIteration
        result = self[self.current_index]
        self.current_index += 1
        return result

    def __len__(self):
        """Return the number of explanations."""
        return len(self.x_test[:, 0])

    def build_rules_payload(self) -> List[Dict[str, Any]]:
        """Delegate payload materialisation to each stored explanation."""
        return [exp.build_rules_payload() for exp in self.explanations]

    def get_guarded_audit(self) -> Dict[str, Any]:
        """Return guarded interval audit for the collection and each instance.

        Raises
        ------
        ValidationError
            If called on a non-guarded explanation collection.
        """
        if not self.explanations:
            return {
                "summary": {
                    "n_instances": 0,
                    "intervals_tested": 0,
                    "intervals_conforming": 0,
                    "intervals_removed_guard": 0,
                    "intervals_emitted": 0,
                    "instances_with_any_removed_guard": 0,
                    "instances_all_intervals_removed_guard": 0,
                    "instances_with_zero_emitted": 0,
                },
                "instances": [],
            }

        if not all(hasattr(exp, "get_guarded_audit") for exp in self.explanations):
            raise ValidationError(
                "get_guarded_audit is only available for guarded explanation collections. "
                "Use explain_guarded_factual(...) or explore_guarded_alternatives(...).",
                details={"collection_type": type(self).__name__},
            )

        instances = [exp.get_guarded_audit() for exp in self.explanations]
        intervals_tested = int(sum(inst["summary"]["intervals_tested"] for inst in instances))
        intervals_conforming = int(
            sum(inst["summary"]["intervals_conforming"] for inst in instances)
        )
        intervals_removed_guard = int(
            sum(inst["summary"]["intervals_removed_guard"] for inst in instances)
        )
        intervals_emitted = int(sum(inst["summary"]["intervals_emitted"] for inst in instances))

        return {
            "summary": {
                "n_instances": int(len(instances)),
                "intervals_tested": intervals_tested,
                "intervals_conforming": intervals_conforming,
                "intervals_removed_guard": intervals_removed_guard,
                "intervals_emitted": intervals_emitted,
                "instances_with_any_removed_guard": int(
                    sum(1 for inst in instances if inst["summary"]["intervals_removed_guard"] > 0)
                ),
                "instances_all_intervals_removed_guard": int(
                    sum(
                        1
                        for inst in instances
                        if inst["summary"]["intervals_tested"] > 0
                        and inst["summary"]["intervals_removed_guard"]
                        == inst["summary"]["intervals_tested"]
                    )
                ),
                "instances_with_zero_emitted": int(
                    sum(1 for inst in instances if inst["summary"]["intervals_emitted"] == 0)
                ),
            },
            "instances": instances,
        }

    def copy(self, deep=False):
        """Return a copy of the collection.

        Parameters
        ----------
        deep : bool, default=False
            Determines whether to return a shallow or deep copy.

        Returns
        -------
        CalibratedExplanations
            A copy of the collection.
        """
        if deep:
            return deepcopy(self)
        return copy(self)

    def __getitem__(self, key: Union[int, slice, List[int], List[bool], np.ndarray]):
        """Return the explanation for the given key.

        In case the index key is an integer (or results in a single result), the function returns the explanation
        corresponding to the index. If the key is a slice or an integer or boolean list (or numpy array)
        resulting in more than one explanation, the function returns a new `CalibratedExplanations`
        object with the indexed explanations.
        """
        if isinstance(key, int):
            # Handle single item access
            return self.explanations[key]
        if isinstance(key, (slice, list, np.ndarray)):
            new_ = copy(self)
            if isinstance(key, slice):
                # Handle slicing
                new_.explanations = list(self.explanations[key])
            if isinstance(key, (list, np.ndarray)):
                if isinstance(key[0], (bool, np.bool_)):
                    # Handle boolean indexing
                    new_.explanations = [
                        exp for exp, include in zip(self.explanations, key, strict=False) if include
                    ]
                elif isinstance(key[0], int):
                    # Handle integer list indexing
                    new_.explanations = [self.explanations[i] for i in key]
            if len(new_.explanations) == 1:
                return new_.explanations[0]
            new_.start_index = 0
            new_.current_index = new_.start_index
            new_.end_index = len(new_.explanations)
            new_.bins = None if self.bins is None else [self.bins[e.index] for e in new_]
            new_.x_test = np.array([self.x_test[e.index, :] for e in new_])
            if self.y_threshold is None:
                new_.y_threshold = None
            elif isinstance(self.y_threshold, (int, float)):
                new_.y_threshold = float(self.y_threshold)
            elif isinstance(self.y_threshold, tuple):
                new_.y_threshold = self.y_threshold
            else:
                # assume list of tuples aligned with instances
                new_.y_threshold = [self.y_threshold[e.index] for e in new_]
            # Preserve per-instance feature ignore masks when present by slicing
            # them in the same way as bins/x_test/y_threshold.
            masks_value = getattr(self, "feature_filter_per_instance_ignore", None)
            if isinstance(masks_value, ABCSequence):
                try:
                    new_.feature_filter_per_instance_ignore = [masks_value[e.index] for e in new_]
                except IndexError:
                    new_.feature_filter_per_instance_ignore = None
            # Reset cached aggregates to avoid referencing stale state from the source
            new_._feature_names_cache = None
            new_._predictions_cache = None
            new_._probabilities_cache = None
            new_._lower_cache = None
            new_._upper_cache = None
            new_._class_labels_cache = None
            for i, e in enumerate(new_):
                e.index = i
            return new_
        raise ValidationError("Invalid argument type.", details={"argument": key})

    def __repr__(self) -> str:
        """Return the string representation of the CalibratedExplanations object."""
        explanations_str = "\n".join([str(e) for e in self.explanations])
        return f"CalibratedExplanations({len(self)} explanations):\n{explanations_str}"

    def __str__(self) -> str:
        """Return the string representation of the CalibratedExplanations object."""
        return self.__repr__()

    # ------------------------------------------------------------------
    # Plugin bridge helpers
    # ------------------------------------------------------------------

    def to_batch(self):
        """Serialise the collection into an :class:`ExplanationBatch`."""
        from ..plugins.builtins import collection_to_batch  # lazy import

        return collection_to_batch(self)

    @classmethod
    def from_batch(cls, batch):
        """Reconstruct a collection from an :class:`ExplanationBatch`."""
        from ..utils.exceptions import SerializationError, ValidationError

        # Check for required batch attributes (duck-typing for flexibility)
        if not hasattr(batch, "collection_metadata"):
            raise SerializationError(
                "ExplanationBatch payload has unexpected type",
                details={
                    "param": "batch",
                    "expected_type": "ExplanationBatch",
                    "actual_type": type(batch).__name__,
                },
            )

        # Get container_cls if available (may be None for duck-typed batches with template)
        container_cls = getattr(batch, "container_cls", None)

        metadata = dict(batch.collection_metadata)
        template = metadata.pop("container", None)

        if container_cls is None and template is not None:
            container_cls = type(template)

        # If neither container_cls nor template is present, raise error
        if container_cls is None:
            raise SerializationError(
                "ExplanationBatch payload missing container_cls and template",
                details={
                    "param": "batch",
                    "required": "container_cls or collection_metadata['container']",
                },
            )

        # Validate container_cls if present
        if not issubclass(container_cls, cls):
            raise ValidationError(
                "ExplanationBatch container metadata has unexpected type",
                details={
                    "param": "container_cls",
                    "expected_type": cls.__name__,
                    "actual_type": container_cls.__name__,
                },
            )

        # If template is a valid CalibratedExplanations instance, use it for metadata
        # but still reconstruct a new container from batch.instances to ensure
        # canonical reconstruction (ADR-015).
        if template is not None and not isinstance(template, cls):
            raise ValidationError(
                "ExplanationBatch container metadata has unexpected type",
                details={
                    "param": "container",
                    "expected_type": cls.__name__,
                    "actual_type": type(template).__name__,
                },
            )

        calibrated_explainer = metadata.get("calibrated_explainer")
        if calibrated_explainer is None:
            calibrated_explainer = metadata.get("explainer")
        if calibrated_explainer is None and template is not None:
            calibrated_explainer = template.calibrated_explainer

        x_test = metadata.get("x_test")
        if x_test is None:
            x_test = metadata.get("x")
        if x_test is None and template is not None:
            x_test = template.x_test

        y_threshold = metadata.get("y_threshold")
        if y_threshold is None and template is not None:
            y_threshold = template.y_threshold

        bins = metadata.get("bins")
        if bins is None and template is not None:
            bins = template.bins

        features_to_ignore = metadata.get("features_to_ignore")
        if features_to_ignore is None and template is not None:
            features_to_ignore = template.features_to_ignore

        condition_source = metadata.get("condition_source")
        if condition_source is None:
            if template is not None:
                condition_source = getattr(template, "condition_source", "prediction")
            else:
                condition_source = "prediction"

        if calibrated_explainer is None or x_test is None:
            raise SerializationError(
                "ExplanationBatch metadata missing explainer context",
                details={
                    "artifact": "ExplanationBatch",
                    "field": "calibrated_explainer",
                    "available_keys": tuple(sorted(metadata.keys())),
                },
            )

        container = container_cls(
            calibrated_explainer,
            x_test,
            y_threshold,
            bins,
            features_to_ignore,
            condition_source=condition_source,
        )
        container.low_high_percentiles = metadata.get(
            "low_high_percentiles", getattr(template, "low_high_percentiles", None)
        )
        container.total_explain_time = metadata.get(
            "total_explain_time", getattr(template, "total_explain_time", None)
        )
        container.feature_filter_per_instance_ignore = metadata.get(
            "feature_filter_per_instance_ignore",
            getattr(template, "feature_filter_per_instance_ignore", None),
        )
        container.batch_metadata = dict(metadata)

        # Propagate any full probability cube from instances into batch metadata
        # (keeps collection metadata aligned with telemetry exports). Also
        # populate a minimal `telemetry` attribute on the materialised
        # container so callers using `from_batch` directly receive the
        # same dependency hints and probability summaries as the orchestrator.
        container.telemetry = {"interval_dependencies": metadata.get("interval_dependencies")}
        full_probs = None
        for inst in getattr(batch, "instances", ()):
            pred = inst.get("prediction") if isinstance(inst, dict) else None
            if isinstance(pred, dict) and "__full_probabilities__" in pred:
                full_probs = pred.get("__full_probabilities__")
                break
        if full_probs is not None:
            container.batch_metadata.setdefault("__full_probabilities__", full_probs)
            with contextlib.suppress(Exception):  # adr002_allow
                arr = np.asarray(full_probs)
                container.telemetry = {
                    "full_probabilities_shape": tuple(arr.shape),
                    "full_probabilities_summary": {
                        "mean": float(np.mean(arr)),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                    },
                    "interval_dependencies": metadata.get("interval_dependencies"),
                }

        for index, instance in enumerate(batch.instances):
            explanation = instance.get("explanation")
            if explanation is None:
                raise SerializationError(
                    "ExplanationBatch instance missing explanation payload",
                    details={
                        "artifact": "ExplanationBatch",
                        "field": "explanation",
                        "instance_index": index,
                    },
                )
            if not isinstance(explanation, batch.explanation_cls):
                raise ValidationError(
                    "ExplanationBatch instance has unexpected explanation type",
                    details={
                        "param": "explanation",
                        "expected_type": batch.explanation_cls.__name__,
                        "actual_type": type(explanation).__name__,
                    },
                )
            explanation_copy = copy(explanation)
            explanation_copy.calibrated_explanations = container
            container.explanations.append(explanation_copy)

        return container

    # ------------------------------------------------------------------
    # JSON export helpers (schema v1 wrappers)
    # ------------------------------------------------------------------

    def to_json(self, *, include_version: bool = True) -> Mapping[str, Any]:
        """Return a JSON-friendly payload describing this collection.

        The payload wraps each explanation using the schema v1 helpers from
        :mod:`calibrated_explanations.serialization` and adds collection-level
        metadata (mode, thresholds, feature names, telemetry snapshot).

        Parameters
        ----------
        include_version:
            When ``True`` (default) the ``schema_version`` field is included on
            the top-level payload as well as on each explanation entry.
        """
        from ..serialization import to_json as _explanation_to_json

        instances = []
        for exp in self.explanations:
            domain = legacy_to_domain(exp.index, self._legacy_payload(exp))
            provenance = getattr(exp, "provenance", None)
            metadata = getattr(exp, "metadata", None)
            if provenance is not None:
                domain.provenance = cast(Optional[Mapping[str, Any]], _jsonify(provenance))
            if metadata is not None:
                domain.metadata = cast(Optional[Mapping[str, Any]], _jsonify(metadata))
            instances.append(_explanation_to_json(domain, include_version=include_version))

        payload: dict[str, Any] = {
            "collection": self._collection_metadata(),
            "explanations": instances,
        }
        if include_version:
            payload.setdefault("schema_version", "1.0.0")

        return payload

    def to_json_stream(self, *, chunk_size: int = 256, format: str = "jsonl"):
        """Stream the collection as JSON.

        This generator yields either a JSON Lines stream or chunked JSON arrays.

        Parameters
        ----------
        chunk_size:
            Number of explanations per yielded chunk (for "chunked") or
            used only for grouping when `format=="chunked"`.
        format:
            Either ``"jsonl"`` (default) for JSON Lines or ``"chunked"`` for
            chunked JSON arrays.

        Yields
        ------
        str
            UTF-8 JSON fragments (one per yield). The first yielded fragment is
            a small metadata object describing the collection and the export
            telemetry.
        """
        from ..serialization import to_json as _explanation_to_json

        if format not in {"jsonl", "chunked"}:
            raise ValidationError("Unsupported stream format", details={"format": format})

        start = time()
        tracemalloc.start()

        # Prepare collection metadata snapshot (without export telemetry yet)
        metadata = dict(self._collection_metadata())

        # Yield metadata first as a standalone JSON object line
        # Telemetry placeholders updated after the stream completes.
        meta_fragment = {"collection": metadata, "schema_version": "1.0.0"}
        yield json.dumps(meta_fragment, default=_jsonify)

        # Stream explanations
        chunk: List[str] = []
        n = 0
        for exp in self.explanations:
            domain = legacy_to_domain(exp.index, self._legacy_payload(exp))
            provenance = getattr(exp, "provenance", None)
            metadata_exp = getattr(exp, "metadata", None)
            if provenance is not None:
                domain.provenance = cast(Optional[Mapping[str, Any]], _jsonify(provenance))
            if metadata_exp is not None:
                domain.metadata = cast(Optional[Mapping[str, Any]], _jsonify(metadata_exp))
            item = _explanation_to_json(domain, include_version=True)
            line = json.dumps(item, default=_jsonify)
            n += 1
            if format == "jsonl":
                yield line
            else:  # chunked
                chunk.append(line)
                if len(chunk) >= chunk_size:
                    # yield a JSON array for this chunk
                    yield "[" + ",".join(chunk) + "]"
                    chunk = []

        # flush remaining chunk
        if format == "chunked" and chunk:
            yield "[" + ",".join(chunk) + "]"

        # stop tracemalloc and capture peak memory
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        elapsed = time() - start

        # Build telemetry
        telemetry = {
            "export_rows": n,
            "chunk_size": chunk_size,
            "mode": getattr(self.calibrated_explainer, "mode", None),
            "peak_memory_mb": round(float(peak) / (1024 * 1024), 3),
            "elapsed_seconds": round(float(elapsed), 3),
            "schema_version": "1.0.0",
            # feature_branch replaced by explicit fields
            "build_id": None,
            "feature_flags": None,
        }

        # Attach minimal telemetry to collection metadata and attempt to store
        # a more complete record on the underlying explainer if available.
        try:
            # minimal metadata
            metadata.setdefault("export_telemetry", {})
            metadata["export_telemetry"].update(telemetry)
            # attempt to update underlying explainer last telemetry
            underlying = getattr(self.calibrated_explainer, "_explainer", None)
            if underlying is not None:
                try:
                    last = getattr(underlying, "_last_telemetry", None) or {}
                    last.update({"export": telemetry})
                    underlying._last_telemetry = last
                except Exception:  # adr002_allow
                    # best-effort only: log for observability per fallback policy
                    _LOGGER.info(
                        "failed to attach export telemetry to underlying explainer",
                        exc_info=True,
                    )
        except Exception:  # adr002_allow
            _LOGGER.info("failed to attach export telemetry to collection", exc_info=True)

        # final telemetry fragment
        yield json.dumps({"export_telemetry": telemetry}, default=_jsonify)

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> ExportedExplanationCollection:
        """Materialise domain explanations from a :meth:`to_json` payload."""
        from ..serialization import from_json as _explanation_from_json

        explanations_blob = payload.get("explanations", []) or []
        domain: list[DomainExplanation] = []
        for item in explanations_blob:
            # Extract explicit multiclass annotations when present on the raw payload
            cls_idx = None
            cls_label = None
            if isinstance(item, Mapping):
                cls_idx = item.get("class_index")
                cls_label = item.get("class_label")
                # Also allow annotations under item['metadata'] when produced by other exporters
                meta = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else None
                if meta is not None:
                    if cls_idx is None:
                        cls_idx = meta.get("class_index")
                    if cls_label is None:
                        cls_label = meta.get("class_label")

            domain_exp = _explanation_from_json(item)

            # Ensure metadata is mutable dict and propagate class annotations
            m = dict(domain_exp.metadata) if isinstance(domain_exp.metadata, Mapping) else {}
            if cls_idx is not None:
                try:
                    m.setdefault("class_index", int(cls_idx))
                except (TypeError, ValueError, OverflowError):
                    m.setdefault("class_index", cls_idx)
            if cls_label is not None:
                m.setdefault("class_label", cls_label)
            # attach back
            domain_exp.metadata = m or None
            domain.append(domain_exp)

        metadata = payload.get("collection", {}) or {}
        return ExportedExplanationCollection(
            metadata=cast(Mapping[str, Any], _jsonify(metadata)), explanations=tuple(domain)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _legacy_payload(self, exp) -> Mapping[str, Any]:
        """Build a legacy-shaped payload from an explanation instance."""
        rules_blob = None
        # prefer conjunctive rules when present and populated
        if getattr(exp, "has_conjunctive_rules", False):
            rules_blob = getattr(exp, "conjunctive_rules", None)
        if not rules_blob:
            rules_blob = getattr(exp, "rules", None)
        if not rules_blob and hasattr(exp, "get_rules"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    rules_blob = exp.get_rules()  # type: ignore[attr-defined]
                except:  # noqa: E722
                    if not isinstance(sys.exc_info()[1], Exception):
                        raise
                    rules_blob = {}

        explanation_type = "factual"
        if isinstance(exp, AlternativeExplanation):
            explanation_type = "alternative"
        elif isinstance(exp, FastExplanation):
            explanation_type = "fast"

        payload: dict[str, Any] = {
            "task": getattr(
                exp, "get_mode", lambda: getattr(self.calibrated_explainer, "mode", None)
            )(),
            "rules": _jsonify(rules_blob or {}),
            "feature_weights": _jsonify(getattr(exp, "feature_weights", {})),
            "feature_predict": _jsonify(getattr(exp, "feature_predict", {})),
            "prediction": _jsonify(getattr(exp, "prediction", {})),
            "explanation_type": explanation_type,
        }
        return payload

    def _collection_metadata(self) -> Mapping[str, Any]:
        """Collect calibration metadata required to interpret the payload."""
        base = getattr(self, "calibrated_explainer", None)
        underlying = getattr(base, "_explainer", None)

        feature_names = None
        try:
            names = self.feature_names
            if names is not None:
                feature_names = list(names)
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            feature_names = None

        class_labels = None
        if hasattr(base, "class_labels"):
            try:
                class_labels = _jsonify(base.class_labels)  # type: ignore[attr-defined]
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                class_labels = None

        sample_percentiles = None
        if hasattr(base, "sample_percentiles"):
            try:
                sample_percentiles = _jsonify(base.sample_percentiles)  # type: ignore[attr-defined]
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                sample_percentiles = None

        runtime_telemetry = None
        if underlying is not None:
            try:
                runtime_telemetry = getattr(underlying, "runtime_telemetry", None)
                if callable(runtime_telemetry):
                    runtime_telemetry = runtime_telemetry()
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                runtime_telemetry = None

        metadata = {
            "size": len(self),
            "mode": getattr(base, "mode", None),
            "y_threshold": _jsonify(self.y_threshold),
            "low_high_percentiles": _jsonify(self.low_high_percentiles),
            "feature_names": _jsonify(feature_names),
            "class_labels": class_labels,
            "sample_percentiles": sample_percentiles,
            "runtime_telemetry": _jsonify(runtime_telemetry),
        }
        return {k: v for k, v in metadata.items() if v is not None}

    # Public wrappers for formerly-private helpers (temporary, Category A remediation)
    def collection_metadata(self) -> Mapping[str, Any]:
        """Public wrapper around internal collection metadata helper."""
        return self._collection_metadata()

    def legacy_payload(self, exp) -> Mapping[str, Any]:
        """Public wrapper to obtain the legacy payload for an explanation."""
        return self._legacy_payload(exp)

    @property
    def prediction_interval(self) -> List[Tuple[Optional[float], Optional[float]]]:
        """Return the prediction intervals for each explanation.

        Returns
        -------
        list of tuples
            A list of tuples containing (low, high) values of the prediction interval.
        """
        return [e.prediction_interval for e in self.explanations]

    @property
    def predict(self) -> List[Any]:
        """Return the scalar prediction for every explanation.

        Returns
        -------
        list
            A list of prediction value.
        """
        return [e.predict for e in self.explanations]

    # ---- Rich baseline exposure (Phase 1A golden snapshot enrichment) ----
    @property
    def feature_names(self):  # consistent naming with underlying explainer
        """Return cached feature names sourced from the underlying explainer."""
        if self._feature_names_cache is None:
            # Underlying FrozenCalibratedExplainer exposes feature_names via original explainer
            try:
                self._feature_names_cache = self.calibrated_explainer.feature_names
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                self._feature_names_cache = None
        return self._feature_names_cache

    @property
    def class_labels(self):
        """Return class labels for classification explanations if available."""
        if self._class_labels_cache is None:
            try:
                labels = getattr(self.calibrated_explainer, "class_labels", None)
                if labels is not None and isinstance(labels, dict):
                    # normalize to list ordered by class index if dict provided
                    # assume keys are numeric class indices
                    labels = [labels[k] for k in sorted(labels.keys())]
                self._class_labels_cache = labels
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                self._class_labels_cache = None
        return self._class_labels_cache

    @property
    def predictions(self):  # noqa: D401
        """Vector of scalar predictions for the explained instances (cached)."""
        if self._predictions_cache is None:
            try:
                self._predictions_cache = np.asarray([e.predict for e in self.explanations])
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                self._predictions_cache = None
        return self._predictions_cache

    @property
    def probabilities(self):  # classification only
        """Return cached probability matrices for classification explanations."""
        if self._probabilities_cache is None:
            try:
                # Each explanation may store:
                #  (a) its own probability vector (shape (n_classes,)) OR
                #  (b) the full matrix (n_instances, n_classes) due to earlier enrichment
                raw = [getattr(e, "prediction_probabilities", None) for e in self.explanations]
                if all(r is not None for r in raw):
                    # If first is a tuple (should not now), handle defensively
                    first = raw[0]
                    if isinstance(first, tuple):  # pragma: no cover - defensive
                        first = first[0]
                    first = np.asarray(first)
                    if first.ndim == 2 and first.shape[0] == len(self.explanations):
                        # Case (b): each explanation redundantly holds full matrix
                        self._probabilities_cache = first
                    else:
                        # Case (a): stack per-instance vectors
                        self._probabilities_cache = np.vstack(raw)
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                self._probabilities_cache = None
        return self._probabilities_cache

    @property
    def lower(self):  # regression only
        """Return cached lower bounds for regression prediction intervals."""
        if self._lower_cache is None:
            try:
                lows = [
                    getattr(e, "prediction_interval", (None, None))[0] for e in self.explanations
                ]
                if any(low is not None for low in lows):
                    self._lower_cache = np.asarray(lows)
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                self._lower_cache = None
        return self._lower_cache

    @property
    def upper(self):  # regression only
        """Return cached upper bounds for regression prediction intervals."""
        if self._upper_cache is None:
            try:
                highs = [
                    getattr(e, "prediction_interval", (None, None))[1] for e in self.explanations
                ]
                if any(h is not None for h in highs):
                    self._upper_cache = np.asarray(highs)
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                self._upper_cache = None
        return self._upper_cache

    @property
    def is_probabilistic_regression(self) -> bool:
        """Check if the explanations use probabilistic regression (thresholded).

        Probabilistic regression and thresholded regression are synonymous terms.
        See ADR-021 for terminology guidance.
        """
        return self.y_threshold is not None

    @property
    def is_one_sided(self) -> bool:
        """Check if the explanations are one-sided."""
        if self.low_high_percentiles is None:
            return False
        return np.isinf(self.get_low_percentile()) or np.isinf(self.get_high_percentile())

    def get_confidence(self) -> float:
        """Return the confidence level of the explanations.

        This method calculates the confidence interval for regression tasks by determining the distance between the lower and upper percentiles. By default, these percentiles are set to 5 and 95.

        Returns
        -------
        float
            The difference between the high and low percentiles, representing the confidence interval.

        Notes
        -----
        - This method is only applicable to regression tasks.
        - If the high percentile is infinite, the confidence is calculated as `100 - low_percentile`.
        - If the low percentile is infinite, the confidence is calculated as `high_percentile`.
        """
        if np.isinf(self.get_high_percentile()):
            return 100 - self.get_low_percentile()
        if np.isinf(self.get_low_percentile()):
            return self.get_high_percentile()
        return self.get_high_percentile() - self.get_low_percentile()

    def get_low_percentile(self) -> float:
        """Return the low percentile of the explanations.

        This method returns the first element of the `low_high_percentiles` attribute,
        which represents the lower bound of the percentile range for the explanation.

        Returns
        -------
        float
            The low percentile value of the explanation.
        """
        # mypy: low_high_percentiles is Optional; ensure it's set by callers before use
        assert self.low_high_percentiles is not None, "low_high_percentiles not set"
        return self.low_high_percentiles[0]  # pylint: disable=unsubscriptable-object

    def get_high_percentile(self) -> float:
        """Return the high percentile of the explanations.

        Returns
        -------
        float
            The high percentile value of the explanation.
        """
        assert self.low_high_percentiles is not None, "low_high_percentiles not set"
        return self.low_high_percentiles[1]  # pylint: disable=unsubscriptable-object

    # pylint: disable=too-many-arguments
    def finalize(
        self,
        binned,
        feature_weights,
        feature_predict,
        prediction,
        instance_time=None,
        total_time=None,
    ) -> "CalibratedExplanations":
        """
        Finalize the explanation by adding the binned data and the feature weights.

        Parameters
        ----------
        binned : array-like
            The binned data for the features.
        feature_weights : array-like
            The weights of the features.
        feature_predict : array-like
            The predicted values for the features.
        prediction : array-like
            The prediction values.
        instance_time : array-like, optional
            The time taken to explain each instance, by default None.
        total_time : float, optional
            The total time taken to explain all instances, by default None.

        Returns
        -------
        self : object
            Returns the instance of the class with explanations finalized.
        """
        for i, instance in enumerate(self.x_test):
            instance_bin = self.bins[i] if self.bins is not None else None
            if self.is_alternative():
                explanation: Union[FactualExplanation, AlternativeExplanation, FastExplanation]
                explanation = AlternativeExplanation(
                    self,
                    i,
                    instance,
                    binned,
                    feature_weights,
                    feature_predict,
                    prediction,
                    self.y_threshold,
                    instance_bin=instance_bin,
                    condition_source=self.condition_source,
                )
            else:
                explanation = FactualExplanation(
                    self,
                    i,
                    instance,
                    binned,
                    feature_weights,
                    feature_predict,
                    prediction,
                    self.y_threshold,
                    instance_bin=instance_bin,
                    condition_source=self.condition_source,
                )
            explanation.explain_time = instance_time[i] if instance_time is not None else None
            self.explanations.append(explanation)
        self.total_explain_time = time() - total_time if total_time is not None else None
        if self.is_alternative():
            return self.__convert_to_alternative_explanations()
        return self

    def __convert_to_alternative_explanations(self) -> "AlternativeExplanations":
        """Return an ``AlternativeExplanations`` view sharing this collection's backing data."""
        alternative_explanations = AlternativeExplanations.__new__(AlternativeExplanations)
        alternative_explanations.__dict__.update(self.__dict__)
        return alternative_explanations

    # pylint: disable=too-many-arguments
    def finalize_fast(
        self, feature_weights, feature_predict, prediction, instance_time=None, total_time=None
    ) -> None:
        """
        Finalize the explanation by adding the binned data and the feature weights.

        Parameters
        ----------
        binned : array-like
            The binned data for the features.
        feature_weights : array-like
            The weights of the features.
        feature_predict : array-like
            The predicted values for the features.
        prediction : array-like
            The prediction values.
        instance_time : array-like, optional
            The time taken to explain each instance, by default None.
        total_time : float, optional
            The total time taken to explain all instances, by default None.

        Notes
        -----
        - This method iterates over the test instances and creates a `FastExplanation` object for each instance.
        - The `FastExplanation` object is initialized with the provided feature weights, predictions, and other relevant data.
        - The explanation time for each instance is recorded if `instance_time` is provided.
        - The total explanation time is calculated if `total_time` is provided.
        """
        for i, instance in enumerate(self.x_test):
            instance_bin = self.bins[i] if self.bins is not None else None
            explanation = FastExplanation(
                self,
                i,
                instance,
                feature_weights,
                feature_predict,
                prediction,
                self.y_threshold,
                instance_bin=instance_bin,
                condition_source=self.condition_source,
            )
            explanation.explain_time = instance_time[i] if instance_time is not None else None
            self.explanations.append(explanation)
        self.total_explain_time = time() - total_time if total_time is not None else None

    def get_explainer(self):
        """Return the underlying :class:`~calibrated_explanations.core.calibrated_explainer.CalibratedExplainer` instance."""
        return self.calibrated_explainer

    def get_rules(self):
        """Return the materialised rule payload for each explanation in the collection."""
        return [
            # pylint: disable=protected-access
            explanation.get_rules()
            for explanation in self.explanations
        ]

    def add_conjunctions(self, n_top_features=5, max_rule_size=2, **kwargs):
        """
        Add conjunctive rules to the explanations.

        The conjunctive rules are added to the `conjunctive_rules` attribute of the `CalibratedExplanations`
        object.

        Parameters
        ----------
        n_top_features : int, optional
            The number of most important factual rules to try to combine into conjunctive rules. Defaults to 5.
        max_rule_size : int, optional
            The maximum size of the conjunctions. Defaults to 2 (meaning `rule_one and rule_two`).

        Returns
        -------
        CalibratedExplanations
            Returns a self reference, to allow for method chaining.
        """
        for explanation in self.explanations:
            explanation.add_conjunctions(n_top_features, max_rule_size, **kwargs)
        return self

    def reset(self):
        """Reset the explanations to their original state."""
        for explanation in self.explanations:
            explanation.reset()
        return self

    def remove_conjunctions(self):
        """Remove any conjunctive rules."""
        for explanation in self.explanations:
            explanation.remove_conjunctions()
        return self

    def filter_rule_sizes(
        self,
        *,
        rule_sizes: Optional[Any] = None,
        size_range: Optional[Tuple[int, int]] = None,
        copy: bool = True,
    ):
        """Filter rules by conjunctive rule size across the collection."""
        if copy:
            new_obj = self.copy()
            new_obj.explanations = [
                explanation.filter_rule_sizes(
                    rule_sizes=rule_sizes, size_range=size_range, copy=True
                )
                for explanation in self.explanations
            ]
            return new_obj

        for idx, explanation in enumerate(self.explanations):
            self.explanations[idx] = explanation.filter_rule_sizes(
                rule_sizes=rule_sizes, size_range=size_range, copy=False
            )
        return self

    def filter_features(
        self,
        *,
        exclude_features=None,
        include_features=None,
        copy: bool = True,
    ) -> "CalibratedExplanations":
        """Filter rules by feature inclusion or exclusion across all explanations.

        Parameters
        ----------
        exclude_features : str, int, or sequence of str/int, optional
            Feature names (str) or indices (int) to exclude. Rules containing any
            of these features will be removed.
        include_features : str, int, or sequence of str/int, optional
            Feature names (str) or indices (int) to include. Only rules containing
            these features will be kept.
        copy : bool, default=True
            If True, return a filtered copy without mutating the original.

        Returns
        -------
        CalibratedExplanations
            Filtered explanations object.
        """
        if copy:
            new_obj = self.copy()
            new_obj.explanations = [
                explanation.filter_features(
                    exclude_features=exclude_features, include_features=include_features, copy=True
                )
                for explanation in self.explanations
            ]
            return new_obj

        for idx, explanation in enumerate(self.explanations):
            self.explanations[idx] = explanation.filter_features(
                exclude_features=exclude_features, include_features=include_features, copy=False
            )
        return self

    def is_alternative(self):
        """Return True when the collection represents an alternative explanation workflow."""
        return isinstance(
            self.calibrated_explainer.discretizer, (RegressorDiscretizer, EntropyDiscretizer)
        )

    # pylint: disable=too-many-arguments, too-many-locals, unused-argument
    def plot(
        self,
        index=None,
        filter_top=10,
        show=True,
        filename="",
        uncertainty=False,
        style="regular",
        rnk_metric=None,
        rnk_weight=0.5,
        style_override=None,
        **kwargs,
    ):
        """Plot explanations for a given instance, with the option to show or save the plots.

        Parameters
        ----------
        index : int or None, default=None
            The index of the instance for which you want to plot the explanation. If None, the
            function will plot all the explanations.
        filter_top : int or None, default=10
            The number of top features to display in the plot. If set to `None`, all the
            features will be shown.
        show : bool, default=True
            Determines whether the plots should be displayed immediately after they are
            generated. Suitable to set to False when saving the plots to a file.
        filename : str, default=''
            The full path and filename of the plot image file that will be saved. If empty, the
            plot will not be saved.
        uncertainty : bool, default=False
            Determines whether to include uncertainty information in the plots.
        style : str, default='regular'
            The style of the plot. Supported styles are 'regular' and 'triangular'.
            Use ``style='ensured'`` as an alias for ``style='triangular'``.
        rnk_metric : str, default=None
            The metric used to rank the features. Supported metrics are 'ensured',
            'feature_weight', and 'uncertainty'. If None, the default from the explanation
            class is used.
        rnk_weight : float, default=0.5
            The weight of the uncertainty in the ranking. Used with the 'ensured' ranking
            metric.

        Returns
        -------
        None

        See Also
        --------
        FactualExplanation.plot
            Refer to the docstring for plot in FactualExplanation for details on default
            ranking ('feature_weight').
        AlternativeExplanation.plot
            Refer to the docstring for plot in AlternativeExplanation for details on default
            ranking ('ensured').
        FastExplanation.plot
            Refer to the docstring for plot in FastExplanation for details on default ranking
            ('feature_weight').
        """
        if style == "ensured":
            style = "triangular"

        if style == "narrative":
            from ..viz.narrative_plugin import NarrativePlotPlugin

            template_path = kwargs.pop("template_path", None)
            expertise_level = kwargs.pop(
                "expertise_level", ("beginner", "intermediate", "advanced")
            )
            output_format = kwargs.pop("output", "dataframe")

            plugin = NarrativePlotPlugin(template_path=template_path)

            if index is not None:
                # Delegate to single explanation helper when a specific index is requested
                return self[index].to_narrative(
                    template_path=template_path,
                    expertise_level=expertise_level,
                    output_format=output_format,
                    **kwargs,
                )

            return plugin.plot(
                self,
                template_path=template_path,
                expertise_level=expertise_level,
                output=output_format,
                **kwargs,
            )

        if len(filename) > 0:
            path, filename, title, ext = prepare_for_saving(filename)

        if index is not None:
            if len(filename) > 0:
                filename = path + title + str(index) + ext
            return self[index].plot(
                filter_top=filter_top,
                show=show,
                filename=filename,
                uncertainty=uncertainty,
                style=style,
                rnk_metric=rnk_metric,
                rnk_weight=rnk_weight,
                style_override=style_override,
                **kwargs,
            )
        else:
            results = []
            for i, explanation in enumerate(self.explanations):
                if len(filename) > 0:
                    filename = path + title + str(i) + ext
                results.append(
                    explanation.plot(
                        filter_top=filter_top,
                        show=show,
                        filename=filename,
                        uncertainty=uncertainty,
                        style=style,
                        rnk_metric=rnk_metric,
                        rnk_weight=rnk_weight,
                        style_override=style_override,
                        **kwargs,
                    )
                )
            if kwargs.get("return_plot_spec"):
                return results[0] if len(results) == 1 else results

    def to_narrative(
        self,
        template_path="exp.yaml",
        expertise_level=("beginner", "advanced"),
        output_format="dataframe",
        conjunction_separator=" AND ",
        align_weights=True,
        **kwargs,
    ):
        """
        Generate narrative explanations for the collection.

        This method provides a clean API for generating human-readable narratives
        from calibrated explanations using customizable templates.

        Parameters
        ----------
        template_path : str, default="exp.yaml"
            Path to the narrative template file (YAML or JSON).
            If the file doesn't exist, the default template will be used.
        expertise_level : str or tuple of str, default=("beginner", "advanced")
            The expertise level(s) for narrative generation. Can be a single
            level or a tuple of levels. Valid values: "beginner", "intermediate", "advanced".
        output_format : str, default="dataframe"
            Output format. Valid values: "dataframe", "text", "html", "dict", "markdown".
        conjunction_separator : str, default=" AND "
            Separator to use for conjunctive rules. Conjunctive rules combine
            multiple feature conditions (e.g., "Glucose > 120 AND BMI > 28").
        align_weights : bool, default=True
            If True, vertically align weight columns in the narrative output.
            If False, no alignment is applied.
        **kwargs : dict
            Additional keyword arguments passed to the narrative plugin.

        Returns
        -------
        pd.DataFrame or str or list of dict
            The generated narratives in the requested format:
            - "dataframe": pandas DataFrame with columns for each expertise level
            - "text": formatted text string with all narratives
            - "html": HTML table with all narratives
            - "dict": list of dictionaries, one per instance

        Raises
        ------
        FileNotFoundError
            If the template file is not found and no default is available.
        ValueError
            If an invalid expertise level or output format is specified.
        ImportError
            If pandas is not available and output_format="dataframe" is requested.

        Examples
        --------
        >>> from calibrated_explanations import CalibratedExplainer
        >>> explainer = CalibratedExplainer(model, x_train, y_train)
        >>> explanations = explainer.explain_factual(x_test)
        >>> narratives = explanations.to_narrative(
        ...     template_path="exp.yaml",
        ...     expertise_level=("beginner", "advanced"),
        ...     output_format="dataframe"
        ... )
        >>> print(narratives)

        See Also
        --------
        :meth:`.plot` : Plot explanations with various visual styles.
        """
        from ..viz.narrative_plugin import NarrativePlotPlugin

        # Create plugin instance
        plugin = NarrativePlotPlugin(template_path=template_path)

        # Generate narratives using the plugin
        return plugin.plot(
            self,
            template_path=template_path,
            expertise_level=expertise_level,
            output=output_format,
            conjunction_separator=conjunction_separator,
            align_weights=align_weights,
            **kwargs,
        )

    def to_dataframe(self, *args, **kwargs):
        """Return the narrative output as a pandas DataFrame.

        Call :meth:`to_narrative` with ``output_format='dataframe'`` and return
        the resulting DataFrame. Accepts the same arguments as
        :meth:`to_narrative`.
        """
        kwargs.setdefault("output_format", "dataframe")
        return self.to_narrative(*args, **kwargs)

    # pylint: disable=protected-access
    def as_lime(self, num_features_to_show=None):
        """Transform the explanations into LIME explanation objects.

        Returns
        -------
        list of lime.Explanation
            List of LIME explanation objects with the same values as the `CalibratedExplanations`.
        """
        _, lime_exp = self.calibrated_explainer.preload_lime()
        exp = []
        for explanation in self.explanations:  # range(len(self.x[:,0])):
            tmp = deepcopy(lime_exp)
            tmp.intercept[1] = 0
            tmp.local_pred = explanation.prediction["predict"]
            if "regression" in self.calibrated_explainer.mode:
                tmp.predicted_value = explanation.prediction["predict"]
                tmp.min_value = np.min(self.calibrated_explainer.y_cal)
                tmp.max_value = np.max(self.calibrated_explainer.y_cal)
            else:
                tmp.predict_proba[0], tmp.predict_proba[1] = (
                    1 - explanation.prediction["predict"],
                    explanation.prediction["predict"],
                )

            feature_weights = explanation.feature_weights["predict"]
            num_to_show = (
                num_features_to_show
                if num_features_to_show is not None
                else self.calibrated_explainer.num_features
            )
            features_to_plot = explanation.rank_features(feature_weights, num_to_show=num_to_show)
            define_conditions = getattr(explanation, "define_conditions", None)
            if define_conditions is None:
                define_conditions = getattr(explanation, "_define_conditions", None)
            rules = define_conditions() if define_conditions is not None else []
            for j, f in enumerate(features_to_plot[::-1]):  # pylint: disable=invalid-name
                tmp.local_exp[1][j] = (f, feature_weights[f])
            del tmp.local_exp[1][num_to_show:]
            tmp.domain_mapper.discretized_feature_names = rules
            tmp.domain_mapper.feature_values = explanation.x_test
            exp.append(tmp)
        return exp

    def as_shap(self):
        """Transform the explanations into a SHAP explanation object.

        Returns
        -------
        shap.Explanation
            SHAP explanation object with the same values as the explanation.
        """
        _, shap_exp = self.calibrated_explainer.preload_shap()
        shap_exp.base_values = np.resize(shap_exp.base_values, len(self))
        shap_exp.values = np.resize(shap_exp.values, (len(self), len(self.x_test[0, :])))
        shap_exp.data = self.x_test
        for i, explanation in enumerate(self.explanations):  # range(len(self.x[:,0])):
            # shap_exp.base_values[i] = explanation.prediction['predict']
            for f in range(len(self.x_test[0, :])):
                shap_exp.values[i][f] = -explanation.feature_weights["predict"][f]
        return shap_exp


class AlternativeExplanations(CalibratedExplanations):
    """A class for storing and visualizing alternative explanations.

    Inherits from :class:`.CalibratedExplanations` and provides methods specific to
    alternative explanations, such as filtering explanations by type.
    """

    def super_explanations(self, only_ensured=False, include_potential=True, copy=True):
        """
        Return a copy with only super-explanations.

        Super-explanations are individual rules with higher probability that support the predicted class.

        Parameters
        ----------
        only_ensured : bool, default=False
            Determines whether to return only ensured explanations.
        include_potential : bool, default=True
            Determines whether to include potential explanations in the super-explanations.
        copy : bool, default=True
            Determines whether to return a copy of the explanations or modify them in place.

        Returns
        -------
        AlternativeExplanations
            A new `AlternativeExplanations` object containing only super-factual or super-potential explanations.

        Notes
        -----
        Super-explanations are only available for `AlternativeExplanation` explanations.
        """
        if copy:
            new_obj = self.copy()
            new_obj.explanations = [
                explanation.super_explanations(
                    only_ensured=only_ensured, include_potential=include_potential, copy=True
                )
                for explanation in self.explanations
            ]
            return new_obj
        for explanation in self.explanations:
            explanation.super_explanations(
                only_ensured=only_ensured, include_potential=include_potential, copy=False
            )
        return self

    def super(self, only_ensured=False, include_potential=True, copy=True):
        """Shorthand delegator for :meth:`.super_explanations`."""
        return self.super_explanations(
            only_ensured=only_ensured, include_potential=include_potential, copy=copy
        )

    @classmethod
    def from_collection(cls, collection: "CalibratedExplanations"):
        """Create an AlternativeExplanations instance from an existing collection.

        This provides a safe public API for tests and callers that previously
        constructed an instance via low-level hacks like `__new__` and
        direct `__dict__` assignment.
        """
        inst = cls.__new__(cls)
        # Copy the public and necessary internal state conservatively.
        inst.calibrated_explainer = collection.calibrated_explainer
        inst.condition_source = getattr(collection, "condition_source", None)
        inst.x_test = getattr(collection, "x_test", None)
        inst.y_threshold = getattr(collection, "y_threshold", None)
        inst.low_high_percentiles = getattr(collection, "low_high_percentiles", None)
        inst.explanations = list(getattr(collection, "explanations", []))
        inst.start_index = getattr(collection, "start_index", 0)
        inst.current_index = getattr(collection, "current_index", inst.start_index)
        inst.end_index = getattr(
            collection, "end_index", len(inst.x_test[:, 0]) if inst.x_test is not None else 0
        )
        inst.bins = getattr(collection, "bins", None)
        inst.total_explain_time = getattr(collection, "total_explain_time", None)
        inst.features_to_ignore = list(getattr(collection, "features_to_ignore", []))
        inst.feature_filter_per_instance_ignore = getattr(
            collection, "feature_filter_per_instance_ignore", None
        )
        # Preserve caches if present
        inst._feature_names_cache = getattr(collection, "_feature_names_cache", None)
        inst._predictions_cache = getattr(collection, "_predictions_cache", None)
        inst._probabilities_cache = getattr(collection, "_probabilities_cache", None)
        inst._lower_cache = getattr(collection, "_lower_cache", None)
        inst._upper_cache = getattr(collection, "_upper_cache", None)
        inst._class_labels_cache = getattr(collection, "_class_labels_cache", None)
        return inst

    def semi_explanations(self, only_ensured=False, include_potential=True, copy=True):
        """
        Return a copy with only semi-explanations.

        Semi-explanations are individual rules with lower probability that support the predicted class.

        Parameters
        ----------
        only_ensured : bool, default=False
            Determines whether to return only ensured explanations.
        include_potential : bool, default=True
            Determines whether to include potential explanations in the semi-explanations.
        copy : bool, default=True
            Determines whether to return a copy of the explanations or modify them in place.

        Returns
        -------
        AlternativeExplanations
            A new `AlternativeExplanations` object containing only semi-factual or semi-potential explanations.

        Notes
        -----
        Semi-explanations are only available for `AlternativeExplanation` explanations.
        """
        if copy:
            new_obj = self.copy()
            new_obj.explanations = [
                explanation.semi_explanations(
                    only_ensured=only_ensured, include_potential=include_potential, copy=True
                )
                for explanation in self.explanations
            ]
            return new_obj
        for explanation in self.explanations:
            explanation.semi_explanations(
                only_ensured=only_ensured, include_potential=include_potential, copy=False
            )
        return self

    def semi(self, only_ensured=False, include_potential=True, copy=True):
        """Shorthand delegator for :meth:`.semi_explanations`."""
        return self.semi_explanations(
            only_ensured=only_ensured, include_potential=include_potential, copy=copy
        )

    def counter_explanations(self, only_ensured=False, include_potential=True, copy=True):
        """
        Return a copy with only counter-explanations.

        Counter-explanations are individual rules that do not support the predicted class.

        Parameters
        ----------
        only_ensured : bool, default=False
            Determines whether to return only ensured explanations.
        include_potential : bool, default=True
            Determines whether to include potential explanations in the counter-explanations.
        copy : bool, default=True
            Determines whether to return a copy of the explanations or modify them in place.

        Returns
        -------
        AlternativeExplanations
            A new `AlternativeExplanations` object containing only counter-factual or counter-potential explanations.

        Notes
        -----
        Counter-explanations are only available for `AlternativeExplanation` explanations.
        """
        if copy:
            new_obj = self.copy()
            new_obj.explanations = [
                explanation.counter_explanations(
                    only_ensured=only_ensured, include_potential=include_potential, copy=True
                )
                for explanation in self.explanations
            ]
            return new_obj
        for explanation in self.explanations:
            explanation.counter_explanations(
                only_ensured=only_ensured, include_potential=include_potential, copy=False
            )
        return self

    def counter(self, only_ensured=False, include_potential=True, copy=True):
        """Shorthand delegator for :meth:`.counter_explanations`."""
        return self.counter_explanations(
            only_ensured=only_ensured, include_potential=include_potential, copy=copy
        )

    def ensured_explanations(self, include_potential=True, copy=True):
        """
        Return a copy with only ensured explanations.

        Ensured explanations are individual rules that have a narrower uncertainty interval.

        Parameters
        ----------
        include_potential : bool, default=True
            Determines whether to include potential explanations in the ensured explanations.
        copy : bool, default=True
            Determines whether to return a copy of the explanations or modify them in place.

        Returns
        -------
        AlternativeExplanations
            A new `AlternativeExplanations` object containing only ensured explanations.
        """
        if copy:
            new_obj = self.copy()
            new_obj.explanations = [
                explanation.ensured_explanations(include_potential=include_potential, copy=True)
                for explanation in self.explanations
            ]
            return new_obj
        for explanation in self.explanations:
            explanation.ensured_explanations(include_potential=include_potential, copy=False)
        return self

    def ensured(self, include_potential=True, copy=True):
        """Shorthand delegator for :meth:`.ensured_explanations`."""
        return self.ensured_explanations(include_potential=include_potential, copy=copy)

    def pareto_explanations(self, include_potential=True, copy=True):
        """Return a copy with only output-envelope Pareto alternatives.

        Parameters
        ----------
        include_potential : bool, default=True
            Determines whether to include potential explanations before
            extracting the Pareto frontier.
        copy : bool, default=True
            Determines whether to return a copy of the explanations or modify
            them in place.

        Returns
        -------
        AlternativeExplanations
            A new ``AlternativeExplanations`` object containing Pareto-front
            alternatives.
        """
        if copy:
            new_obj = self.copy()
            new_obj.explanations = [
                explanation.pareto_explanations(include_potential=include_potential, copy=True)
                for explanation in self.explanations
            ]
            return new_obj
        for explanation in self.explanations:
            explanation.pareto_explanations(include_potential=include_potential, copy=False)
        return self

    def pareto(self, include_potential=True, copy=True):
        """Shorthand delegator for :meth:`.pareto_explanations`."""
        return self.pareto_explanations(include_potential=include_potential, copy=copy)


class FrozenCalibratedExplainer:
    """A class that wraps an explainer to provide a read-only interface.

    Prevents modification of the underlying explainer, ensuring its state remains unchanged.
    """

    def __init__(self, explainer):
        """Initialize a new instance of the FrozenCalibratedExplainer class.

        Parameters
        ----------
        explainer : CalibratedExplainer
            The explainer to be wrapped.
        """
        try:
            self._explainer = deepcopy(explainer)
        except (
            Exception
        ):  # adr002_allow  # pragma: no cover - defensive fallback for unpickleable state
            # Deepcopy of complex explainer objects can fail; log at DEBUG
            # instead of emitting a RuntimeWarning to avoid noisy test output.
            try:
                import logging

                logging.getLogger(__name__).debug(
                    "Deepcopy of explainer failed; using original instance for frozen wrapper"
                )
            except Exception:  # adr002_allow
                # If logging fails, fall back to warnings to preserve behavior
                warnings.warn(
                    "Deepcopy of explainer failed; using original instance for frozen wrapper",
                    UserWarning,
                    stacklevel=2,
                )
            self._explainer = explainer

    @property
    def explainer(self):
        """Return the wrapped explainer instance."""
        return self._explainer

    @property
    def x_cal(self):
        """
        Retrieves the calibrated feature matrix from the underlying explainer.

        This property provides access to the feature matrix used in the explainer, allowing users to understand the data being analyzed.

        Returns
        -------
            numpy.ndarray: The calibrated feature matrix.
        """
        return self._explainer.x_cal

    @property
    def y_cal(self):
        """
        Retrieves the calibrated target values from the underlying explainer.

        This property provides access to the target values used in the explainer, allowing users to understand the data being analyzed.

        Returns
        -------
            numpy.ndarray: The calibrated target values.
        """
        return self._explainer.y_cal

    @property
    def num_features(self):
        """
        Retrieves the number of features in the dataset.

        This property provides access to the count of features that the underlying explainer is using.
        It is useful for understanding the dimensionality of the data being analyzed.

        Returns
        -------
            int: The number of features in the dataset.
        """
        return self._explainer.num_features

    @property
    def categorical_features(self):
        """
        Retrieves the indices of categorical features from the underlying explainer.

        This property provides access to the indices of categorical features used in the explainer, allowing users to understand the data being analyzed.

        Returns
        -------
            list: The indices of categorical features.
        """
        return self._explainer.categorical_features

    @property
    def categorical_labels(self):
        """
        Retrieves the labels for categorical features from the underlying explainer.

        This property provides access to the labels for categorical features used in the explainer, allowing users to understand the data being analyzed.

        Returns
        -------
            list: The labels for categorical features.
        """
        return self._explainer.categorical_labels

    @property
    def feature_values(self):
        """
        Retrieves the unique values for each feature from the underlying explainer.

        This property provides access to the unique values for each feature used in the explainer, allowing users to understand the data being analyzed.

        Returns
        -------
            list: The unique values for each feature.
        """
        return self._explainer.feature_values

    @property
    def feature_names(self):
        """
        Retrieves the names of the features from the underlying explainer.

        This property provides access to the names of the features used in the explainer, allowing users to understand the data being analyzed.

        Returns
        -------
            list: The names of the features.
        """
        return self._explainer.feature_names

    @property
    def class_labels(self):
        """
        Retrieves the labels for the classes from the underlying explainer.

        This property provides access to the labels for the classes used in the explainer, allowing users to understand the data being analyzed.

        Returns
        -------
            list: The labels for the classes.
        """
        return self._explainer.class_labels

    @property
    def sample_percentiles(self):
        """
        Retrieves the sample percentiles from the underlying explainer.

        This property provides access to the percentiles of the samples used in the explainer,
        allowing users to understand the distribution of the data being analyzed.

        Returns
        -------
            list: The sample percentiles as a list.
        """
        return self._explainer.sample_percentiles

    @property
    def mode(self):
        """
        Retrieves the mode of the explainer from the underlying explainer.

        This property provides access to the mode of the explainer, allowing users to understand the type of problem being analyzed.

        Returns
        -------
            str: The mode of the explainer.
        """
        return self._explainer.mode

    @property
    def is_multiclass(self):
        """
        Retrieves a boolean indicating if the problem is multiclass from the underlying explainer.

        This property provides access to a boolean value indicating if the problem is multiclass, allowing users to understand the type of problem being analyzed.

        Returns
        -------
            bool: True if the problem is multiclass, False otherwise.
        """
        return self._explainer.is_multiclass

    @property
    def discretizer(self):
        """
        Retrieves the discretizer used by the explainer from the underlying explainer.

        This property provides access to the discretizer used by the explainer, allowing users to understand the discretization process.

        Returns
        -------
            Discretizer: The discretizer used by the explainer.
        """
        return self._explainer.discretizer

    @property
    def discretize(self):
        """Public accessor for the discretize function (testing helper)."""
        return self._explainer.discretize

    @property
    def rule_boundaries(self):
        """Expose the underlying rule boundaries helper."""
        return self._explainer.rule_boundaries

    @property
    def learner(self):
        """
        Retrieves the learner associated with the explainer from the underlying explainer.

        This property provides access to the learner associated with the explainer, allowing users to understand the learning process.

        Returns
        -------
            object: The learner associated with the explainer.
        """
        return self._explainer.learner

    @property
    def difficulty_estimator(self):
        """
        Retrieves the estimator for difficulty levels from the underlying explainer.

        This property provides access to the estimator for difficulty levels used in the explainer, allowing users to understand the learning process.

        Returns
        -------
            object: The estimator for difficulty levels.
        """
        return self._explainer.difficulty_estimator

    @property
    def prediction_orchestrator(self):
        """Expose the underlying prediction orchestrator (read-only)."""
        return self._explainer.prediction_orchestrator

    def predict(self, *args, **kwargs):
        """Forward the public prediction API to the underlying explainer."""
        return self._explainer.predict(*args, **kwargs)

    @property
    def _preload_lime(self):
        """
        Retrieves the preload_lime function from the underlying explainer.

        This property provides access to the preload_lime function used by the explainer, allowing users to understand the prediction process.

        Returns
        -------
            function: The preload_lime function used by the explainer.
        """
        return self._explainer.preload_lime

    @property
    def preload_lime(self):
        """Public accessor for the lime preload helper (testing helper)."""
        return self._explainer.preload_lime

    @property
    def preload_shap(self):
        """Public accessor for the shap preload helper (testing helper)."""
        return self._explainer.preload_shap

    def __setattr__(self, key, value):
        """Prevent modification of attributes except for '_explainer'."""
        if key == "_explainer":
            super().__setattr__(key, value)
        else:
            raise AttributeError("Cannot modify frozen instance")


class MultiClassCalibratedExplanations(CalibratedExplanations):
    """
    A class for storing and visualizing calibrated explanations for multi-class classification.

    This class extends `CalibratedExplanations` to support multi-class explanations,
    allowing storage and retrieval of explanations per instance using a dictionary.
    """

    def __init__(self, calibrated_explainer, x_test, bins, num_classes, explanations=None):
        """Initialize multiclass explanation storage for one or more instances."""
        x_test = validate_and_prepare_input(calibrated_explainer, x_test)
        super().__init__(calibrated_explainer, x_test, None, bins)
        self.num_classes = num_classes
        if explanations is None:
            self.explanations = [{} for _ in range(len(x_test))]
        else:
            self.explanations = deepcopy(explanations)

    def _first_explanation_for_instance(self, index):
        """Return the first explanation stored for an instance regardless of class key."""
        if index < 0 or index >= len(self.explanations):
            return None
        instance_explanations = self.explanations[index]
        if not instance_explanations:
            return None
        return next(iter(instance_explanations.values()))

    @property
    def X_test(self):  # noqa: N802
        """Backward-compatible alias for x_test."""
        return self.x_test

    def __repr__(self):
        """Return the string representation of the MultiClassCalibratedExplanations object."""
        explanations_str = (
            "\n" + f"MultiClassCalibratedExplanations({len(self.explanations)} explanations):\n"
        )
        first_explanation = self._first_explanation_for_instance(0)
        if first_explanation is None:
            return explanations_str
        labels = first_explanation.get_class_labels()
        for i in range(len(self.explanations)):
            explanations_str += f"explanation({i}):\n"
            for class_key, label in labels.items():
                label_explanation = self.__getitem__((i, class_key))
                explanations_str += f"explanation for label({label}):\n"
                explanations_str += str(label_explanation)
        return explanations_str

    def __getitem__(self, key):
        """
        Return the explanation for the given key.

        If key is an integer, return all class labels explanations at that index as MultiClassCalibratedExplanations.
        If key is a tuple (index, class_idx), return the explanation for a specific class label as FactualExplanation.
        """
        if isinstance(key, int):
            # Mirror CalibratedExplanations semantics: integer indexing returns
            # a single-instance view.
            x_single = np.atleast_2d(self.x_test[key])
            return MultiClassCalibratedExplanations(
                self.calibrated_explainer,
                x_single,
                self.bins,
                self.num_classes,
                [self.explanations[key]],
            )
        if isinstance(key, slice):
            return MultiClassCalibratedExplanations(
                self.calibrated_explainer,
                self.x_test[key],
                self.bins,
                self.num_classes,
                self.explanations[key],
            )
        if isinstance(key, (list, np.ndarray)):
            arr = np.asarray(key)
            if arr.dtype == bool:
                if len(arr) != len(self.explanations):
                    raise IndexError(
                        "Boolean index length must match number of explanations in collection."
                    )
                indices = np.where(arr)[0]
            else:
                indices = np.asarray(arr, dtype=int)
            selected_explanations = [self.explanations[int(i)] for i in indices]
            return MultiClassCalibratedExplanations(
                self.calibrated_explainer,
                self.x_test[indices],
                self.bins,
                self.num_classes,
                selected_explanations,
            )
        elif isinstance(key, tuple) and len(key) == 2:
            # Return Factual explanation of only one class label explanation

            index, class_idx = key
            # Accept both Python ints and numpy integer types
            if isinstance(class_idx, (int, np.integer)):
                return self.explanations[index].get(int(class_idx), None)
            elif isinstance(class_idx, str):
                first_explanation = self._first_explanation_for_instance(index)
                if first_explanation is None:
                    return None
                labels = first_explanation.get_class_labels()
                try:
                    class_idx = list(labels.keys())[list(labels.values()).index(class_idx)]
                except ValueError as exc:
                    raise KeyError(f"Unknown class label '{class_idx}' for index {index}.") from exc
                return self.explanations[index].get(int(class_idx), None)
        raise ValidationError("Invalid argument type. Use an index (int) or (index, class) tuple.")

    def get_explanation(self, index, class_idx=None):
        """Return explanation(s) at ``index``, optionally narrowed to ``class_idx``."""
        if class_idx is None:
            return self[index]
        return self[(index, class_idx)]

    # ------------------------------------------------------------------
    # Multiclass-specific overrides (dispatch into per-class dicts)
    # ------------------------------------------------------------------
    def __iter__(self):
        """Iterate yielding single-instance views (align with base semantics)."""
        for i in range(len(self.explanations)):
            yield self[i]

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> ExportedMultiClassExplanationCollection:
        """Materialise grouped multiclass explanations from exported JSON payload.

        Raises
        ------
        ValidationError
            If top-level or item-level schema versions are missing/unsupported,
            or required multiclass keys cannot be restored.
        """
        from ..serialization import from_json as _explanation_from_json

        expected_schema = "1.0.0"
        schema_version = payload.get("schema_version")
        if schema_version != expected_schema:
            raise ValidationError(
                "Unsupported multiclass payload schema version.",
                details={"expected": expected_schema, "received": schema_version},
            )

        explanations_blob = payload.get("explanations", [])
        if not isinstance(explanations_blob, list):
            raise ValidationError(
                "Multiclass payload explanations must be a list.",
                details={"type": type(explanations_blob).__name__},
            )

        grouped: dict[int, dict[int, DomainExplanation]] = {}
        for item in explanations_blob:
            if not isinstance(item, Mapping):
                raise ValidationError(
                    "Each multiclass explanation item must be a mapping.",
                    details={"type": type(item).__name__},
                )
            item_schema = item.get("schema_version")
            if item_schema != expected_schema:
                raise ValidationError(
                    "Unsupported multiclass explanation item schema version.",
                    details={"expected": expected_schema, "received": item_schema},
                )

            try:
                instance_index = int(item.get("index"))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValidationError(
                    "Multiclass explanation item is missing a valid instance index.",
                    details={"index": item.get("index")},
                ) from exc

            metadata_map = item.get("metadata")
            metadata_dict = metadata_map if isinstance(metadata_map, Mapping) else {}
            class_index_raw = item.get("class_index", metadata_dict.get("class_index"))
            class_label = item.get("class_label", metadata_dict.get("class_label"))
            if class_index_raw is None:
                raise ValidationError(
                    "Multiclass explanation item is missing class_index.",
                    details={"index": instance_index},
                )

            try:
                class_index = int(class_index_raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValidationError(
                    "Multiclass explanation item has invalid class_index.",
                    details={"index": instance_index, "class_index": class_index_raw},
                ) from exc

            domain_exp = _explanation_from_json(item)
            metadata_out = (
                dict(domain_exp.metadata) if isinstance(domain_exp.metadata, Mapping) else {}
            )
            metadata_out["class_index"] = class_index
            if class_label is not None:
                metadata_out["class_label"] = class_label
            domain_exp.metadata = metadata_out or None

            per_instance = grouped.setdefault(instance_index, {})
            if class_index in per_instance:
                raise ValidationError(
                    "Duplicate class_index for multiclass explanation item.",
                    details={"index": instance_index, "class_index": class_index},
                )
            per_instance[class_index] = domain_exp

        ordered = tuple(grouped[idx] for idx in sorted(grouped))
        metadata = payload.get("collection", {}) or {}
        return ExportedMultiClassExplanationCollection(
            metadata=cast(Mapping[str, Any], _jsonify(metadata)),
            explanations_by_instance=ordered,
        )

    def add_conjunctions(self, n_top_features=5, max_rule_size=2, **kwargs):
        """Apply add_conjunctions to every class-specific explanation."""
        for class_dict in self.explanations:
            for explanation in class_dict.values():
                explanation.add_conjunctions(n_top_features, max_rule_size, **kwargs)
        return self

    def remove_conjunctions(self):
        """Apply remove_conjunctions to every class-specific explanation."""
        for class_dict in self.explanations:
            for explanation in class_dict.values():
                explanation.remove_conjunctions()
        return self

    def reset(self):
        """Reset each class-specific explanation to original state."""
        for class_dict in self.explanations:
            for explanation in class_dict.values():
                explanation.reset()
        return self

    def filter_rule_sizes(
        self,
        *,
        rule_sizes: Optional[Any] = None,
        size_range: Optional[Tuple[int, int]] = None,
        copy: bool = True,
    ):
        """Filter rules by size across every class-specific explanation."""
        if copy:
            new_obj = self.copy()
            new_obj.explanations = [
                {
                    k: exp.filter_rule_sizes(
                        rule_sizes=rule_sizes, size_range=size_range, copy=True
                    )
                    for k, exp in class_dict.items()
                }
                for class_dict in self.explanations
            ]
            return new_obj

        for idx, class_dict in enumerate(self.explanations):
            for cls_key, explanation in class_dict.items():
                self.explanations[idx][cls_key] = explanation.filter_rule_sizes(
                    rule_sizes=rule_sizes, size_range=size_range, copy=False
                )
        return self

    def filter_features(self, *, exclude_features=None, include_features=None, copy: bool = True):
        """Filter features across every class-specific explanation."""
        if copy:
            new_obj = self.copy()
            new_obj.explanations = [
                {
                    k: exp.filter_features(
                        exclude_features=exclude_features,
                        include_features=include_features,
                        copy=True,
                    )
                    for k, exp in class_dict.items()
                }
                for class_dict in self.explanations
            ]
            return new_obj

        for idx, class_dict in enumerate(self.explanations):
            for cls_key, explanation in class_dict.items():
                self.explanations[idx][cls_key] = explanation.filter_features(
                    exclude_features=exclude_features, include_features=include_features, copy=False
                )
        return self

    def get_rules(self):
        """Return per-instance, per-class rule payloads.

        Returns
        -------
        list of dict
            Each item is a mapping {class_key: rules_payload} for that instance.
        """
        return [
            {cls_key: exp.get_rules() for cls_key, exp in class_dict.items()}
            for class_dict in self.explanations
        ]

    # Safe adapters / explicit not-implemented for adapters that assume flat lists
    def as_lime(self):
        """Raise for multiclass collections where a flat LIME export is undefined."""
        raise NotImplementedError(
            "as_lime() is not supported for multi-label collections. "
            "Call get_explanation(i, cls).as_lime() for a specific class, or iterate over the collection "
            "to build a per-class LIME mapping. If you need an aggregated LIME export, convert each per-class "
            "explanation via get_explanation(i, cls).as_lime() and combine the results in your caller."
        )

    def as_shap(self):
        """Raise for multiclass collections where a flat SHAP export is undefined."""
        raise NotImplementedError(
            "as_shap() is not supported for multi-label collections. "
            "Call get_explanation(i, cls).as_shap() for a specific class, or iterate and aggregate per-class SHAP outputs. "
            "Aggregating SHAP across classes is application-specific; prefer per-class SHAP objects for downstream use."
        )

    def to_narrative(self, *args, **kwargs):
        """
        Generate narratives for a multiclass (multi-label) collection.

        The method returns per-instance, per-class narratives. The behaviour depends
        on ``output_format`` (same semantics as single-instance :meth:`to_narrative`):

        - ``output_format='dict'``: returns ``List[Dict[class_key, narrative_dict]]``
          where each item corresponds to an instance and maps class keys to the
          narrative dict for that class.
        - ``output_format='text'``: returns a single combined text containing the
          narratives for every instance and class (human-readable).
        - ``output_format='dataframe'``: returns a pandas DataFrame with columns
          ``['instance', 'class', 'narrative']`` (requires pandas).

        For other formats (e.g., 'html', 'markdown') the implementation will
        attempt to coerce per-class outputs into the requested format where
        reasonable.
        """
        # Normalize kwargs used by the single-explanation API
        template_path = kwargs.pop("template_path", args[0] if len(args) > 0 else "exp.yaml")
        expertise_level = kwargs.pop(
            "expertise_level", kwargs.get("expertise_level", ("beginner", "advanced"))
        )
        output_format = kwargs.pop(
            "output_format", kwargs.get("output_format", kwargs.get("output", "dataframe"))
        )
        conjunction_separator = kwargs.pop(
            "conjunction_separator", kwargs.get("conjunction_separator", " AND ")
        )
        align_weights = kwargs.pop("align_weights", kwargs.get("align_weights", True))

        # Helper to convert a per-class explanation to the desired intermediate dict
        per_instance = []
        for _i, class_dict in enumerate(self.explanations):
            inst_map = {}
            for cls_key, explanation in class_dict.items():
                try:
                    narr = explanation.to_narrative(
                        template_path=template_path,
                        expertise_level=expertise_level,
                        output_format="dict",
                        conjunction_separator=conjunction_separator,
                        align_weights=align_weights,
                        **kwargs,
                    )
                except (AttributeError, TypeError, ValueError, KeyError):
                    # Fallback: try to obtain text output
                    narr = {
                        "text": explanation.to_narrative(
                            template_path=template_path,
                            expertise_level=expertise_level,
                            output_format="text",
                            conjunction_separator=conjunction_separator,
                            align_weights=align_weights,
                            **kwargs,
                        )
                    }
                inst_map[int(cls_key)] = narr
            per_instance.append(inst_map)

        # Return according to requested format
        if output_format == "dict":
            return per_instance

        if output_format == "text":
            parts = []
            for i, inst_map in enumerate(per_instance):
                parts.append(f"Instance {i}:")
                for cls_key, narr in inst_map.items():
                    label = None
                    first_exp = self._first_explanation_for_instance(i)
                    if first_exp is not None:
                        labels = first_exp.get_class_labels()
                        label = labels.get(cls_key, None)
                    hdr = f"  Class {cls_key}" + (f" ({label})" if label is not None else "")
                    parts.append(hdr)
                    if isinstance(narr, dict):
                        text = narr.get("text") or narr.get("short") or str(narr)
                    else:
                        text = str(narr)
                    parts.append(text)
                    parts.append("")
            return "\n".join(parts)

        if output_format == "dataframe":
            try:
                import pandas as pd
            except ImportError as exc:  # pragma: no cover - pandas import error path
                raise ImportError("pandas is required for output_format='dataframe'") from exc

            rows = []
            for i, inst_map in enumerate(per_instance):
                for cls_key, narr in inst_map.items():
                    # narr is a dict produced by single-explanation output_format='dict'
                    # Attempt to extract a compact textual narrative for a 'narrative' column
                    if isinstance(narr, dict):
                        text = narr.get("text") or narr.get("short") or str(narr)
                    else:
                        text = str(narr)
                    rows.append({"instance": i, "class": int(cls_key), "narrative": text})

            df = pd.DataFrame(rows)
            return df

        # Fall back to returning the dict structure for unknown formats
        return per_instance

    def to_json(self, *, include_version: bool = True) -> Mapping[str, Any]:
        """Return a JSON-friendly payload describing this multiclass collection.

        This mirrors :meth:`CalibratedExplanations.to_json` but emits one
        exported explanation per (instance, class) pair. Each legacy payload
        is augmented with ``class_index`` and, when available, ``class_label``.
        """
        from ..serialization import to_json as _explanation_to_json

        instances = []
        for idx, class_dict in enumerate(self.explanations):
            for cls_key, exp in class_dict.items():
                # Build legacy-shaped payload and annotate with class info
                payload = dict(self._legacy_payload(exp))
                payload["class_index"] = int(cls_key)
                try:
                    first = self._first_explanation_for_instance(idx)
                    if first is not None:
                        labels = first.get_class_labels()
                        payload.setdefault("class_label", labels.get(int(cls_key)))
                except (AttributeError, TypeError, ValueError, KeyError):
                    _LOGGER.debug(
                        "Failed to resolve class_label while exporting multiclass payload",
                        exc_info=True,
                    )

                domain = legacy_to_domain(int(idx), payload)
                provenance = getattr(exp, "provenance", None)
                metadata = getattr(exp, "metadata", None)
                if provenance is not None:
                    domain.provenance = cast(Optional[Mapping[str, Any]], _jsonify(provenance))
                if metadata is not None:
                    domain.metadata = cast(Optional[Mapping[str, Any]], _jsonify(metadata))
                instances.append(_explanation_to_json(domain, include_version=include_version))

        payload: dict[str, Any] = {
            "collection": self._collection_metadata(),
            "explanations": instances,
        }
        if include_version:
            payload.setdefault("schema_version", "1.0.0")

        return payload

    def to_json_stream(self, *, chunk_size: int = 256, format: str = "jsonl"):
        """Stream the multiclass collection as JSON.

        Yields the same fragments as :meth:`CalibratedExplanations.to_json_stream`
        but emits one item per (instance, class) pair.
        """
        from ..serialization import to_json as _explanation_to_json

        if format not in {"jsonl", "chunked"}:
            raise ValidationError("Unsupported stream format", details={"format": format})

        start = time()
        tracemalloc.start()

        metadata = dict(self._collection_metadata())
        meta_fragment = {"collection": metadata, "schema_version": "1.0.0"}
        yield json.dumps(meta_fragment, default=_jsonify)

        chunk: List[str] = []
        n = 0
        for idx, class_dict in enumerate(self.explanations):
            for cls_key, exp in class_dict.items():
                payload = dict(self._legacy_payload(exp))
                payload["class_index"] = int(cls_key)
                try:
                    first = self._first_explanation_for_instance(idx)
                    if first is not None:
                        labels = first.get_class_labels()
                        payload.setdefault("class_label", labels.get(int(cls_key)))
                except (AttributeError, TypeError, ValueError, KeyError):
                    _LOGGER.debug(
                        "Failed to resolve class_label while streaming multiclass payload",
                        exc_info=True,
                    )

                domain = legacy_to_domain(int(idx), payload)
                provenance = getattr(exp, "provenance", None)
                metadata_exp = getattr(exp, "metadata", None)
                if provenance is not None:
                    domain.provenance = cast(Optional[Mapping[str, Any]], _jsonify(provenance))
                if metadata_exp is not None:
                    domain.metadata = cast(Optional[Mapping[str, Any]], _jsonify(metadata_exp))
                item = _explanation_to_json(domain, include_version=True)
                line = json.dumps(item, default=_jsonify)
                n += 1
                if format == "jsonl":
                    yield line
                else:  # chunked
                    chunk.append(line)
                    if len(chunk) >= chunk_size:
                        yield "[" + ",".join(chunk) + "]"
                        chunk = []

        if format == "chunked" and chunk:
            yield "[" + ",".join(chunk) + "]"

        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        elapsed = time() - start

        telemetry = {
            "export_rows": n,
            "chunk_size": chunk_size,
            "mode": getattr(self.calibrated_explainer, "mode", None),
            "peak_memory_mb": round(float(peak) / (1024 * 1024), 3),
            "elapsed_seconds": round(float(elapsed), 3),
            "schema_version": "1.0.0",
            "build_id": None,
            "feature_flags": None,
        }

        try:
            metadata.setdefault("export_telemetry", {})
            metadata["export_telemetry"].update(telemetry)
            underlying = getattr(self.calibrated_explainer, "_explainer", None)
            if underlying is not None:
                try:
                    last = getattr(underlying, "_last_telemetry", None) or {}
                    last.update({"export": telemetry})
                    underlying._last_telemetry = last
                except Exception:  # adr002_allow
                    _LOGGER.info(
                        "failed to attach export telemetry to underlying explainer",
                        exc_info=True,
                    )
        except Exception:  # adr002_allow
            _LOGGER.info("failed to attach export telemetry to collection", exc_info=True)

        yield json.dumps({"export_telemetry": telemetry}, default=_jsonify)

    # Properties that aggregate per-class values into per-instance dicts
    @property
    def predictions(self):
        """Return per-instance per-class scalar predictions as a list of dicts."""
        return [
            {int(cls_key): getattr(exp, "predict", None) for cls_key, exp in class_dict.items()}
            for class_dict in self.explanations
        ]

    @property
    def prediction_interval(self):
        """Return per-instance per-class prediction intervals as a list of dicts."""
        return [
            {
                int(cls_key): getattr(exp, "prediction_interval", (None, None))
                for cls_key, exp in class_dict.items()
            }
            for class_dict in self.explanations
        ]

    @property
    def probabilities(self):
        """Return per-instance per-class probability vectors as a list of dicts.

        Each dict maps class_key -> the stored `prediction_probabilities` (if present) or
        None when unavailable.
        """
        return [
            {
                int(cls_key): getattr(exp, "prediction_probabilities", None)
                for cls_key, exp in class_dict.items()
            }
            for class_dict in self.explanations
        ]

    def plot(
        self,
        index=None,
        class_idx=None,
        filter_top=10,
        show=True,
        filename="",
        uncertainty=False,
        style="regular",
        **kwargs,
    ):
        """Plot multiclass explanations as factual or alternative views."""
        if len(self.explanations) > 0:
            first_explanation = self._first_explanation_for_instance(0)
            if isinstance(first_explanation, FactualExplanation):
                self.plot_factual(
                    index=index,
                    class_idx=class_idx,
                    filter_top=filter_top,
                    show=show,
                    filename=filename,
                    uncertainty=uncertainty,
                    style=style,
                    **kwargs,
                )
            elif isinstance(first_explanation, AlternativeExplanation):
                self.plot_alternative(
                    index=index,
                    class_idx=class_idx,
                    filter_top=filter_top,
                    show=show,
                    filename=filename,
                    uncertainty=uncertainty,
                    style=style,
                    **kwargs,
                )

        else:
            warnings.warn("No explanations found", stacklevel=2)

    def plot_alternative(
        self,
        index=None,
        class_idx=None,
        filter_top=10,
        show=True,
        filename="",
        uncertainty=False,
        style="regular",
        **kwargs,
    ):
        """
        Plot explanations for a given instance and class.

        If no class is specified, plots explanations for all classes at that index.
        """
        style_override = kwargs.get("style_override", get_multiclass_config())

        if index is not None:
            if class_idx is not None:
                explanation = self.get_explanation(index, class_idx)
                if explanation:
                    explanation.plot(
                        filter_top=filter_top,
                        show=show,
                        filename=filename,
                        uncertainty=uncertainty,
                        style=style,
                    )
                else:
                    warnings.warn(
                        f"No explanation found for instance {index}, class {class_idx}",
                        stacklevel=2,
                    )
            else:
                self.__getitem__(index).plot(
                    filter_top=filter_top,
                    show=show,
                    filename=filename,
                    uncertainty=uncertainty,
                    style=style,
                )
        else:
            import matplotlib.colors as mcolors

            rgb = np.array(list(permutations(range(0, 256, 11), 3))) / 255.0
            colors = [rgb.tolist()[i * 23] for i in range(25)]
            colors = list(mcolors.BASE_COLORS.values())
            for i, class_explanations in enumerate(self.explanations):
                # Ensure style_override gets passed through
                class_explanations_list = list(class_explanations.values())
                # Respect the explicit arguments passed to plot_alternative()
                # (do not override via kwargs in this multi-label/all-classes branch).
                iter_filename = filename
                iter_show = show

                rnk_metric = kwargs.get("rnk_metric", "ensured")
                if rnk_metric is None:
                    rnk_metric = "ensured"
                rnk_weight = kwargs.get("rnk_weight", 0.5)
                if rnk_metric == "uncertainty":
                    rnk_weight = 1.0
                    rnk_metric = "ensured"

                alternatives = []
                for ex in list(class_explanations.values()):
                    get_rules = getattr(ex, "get_rules", None)
                    if callable(get_rules):
                        alternatives.append(get_rules())
                    else:
                        alternatives.append(ex._get_rules())

                    # Ensure each explanation has a sensible `index` set before
                    # precondition checks or plotting. Some explanation objects
                    # may be frozen; use best-effort assignment.
                    for ex in class_explanations_list:
                        with contextlib.suppress(Exception):
                            ex.index = i
                        with contextlib.suppress(Exception):
                            ex._check_preconditions()
                    predicts = [getattr(ex, "prediction", None) for ex in class_explanations_list]

                filter_top = [len(alternative["rule"]) for alternative in alternatives]
                """if filter_top is None:
                    filter_top = num_features_to_show_list
                else:
                    filter_top = [filter_top for factual in factuals]
                filter_top = [np.min([num_features_to_show, filter_]) for num_features_to_show, filter_ in zip(num_features_to_show_list,filter_top)]"""

                if len(filter_top) <= 0:
                    warnings.warn(
                        f"The explanation has no rules to plot. The index of the instance is {i}",
                        stacklevel=2,
                    )
                    return

                if len(iter_filename) > 0:
                    path, iter_filename, title, ext = prepare_for_saving(iter_filename)
                    path = f"plots/{path}"
                    save_ext = [ext]
                else:
                    path = ""
                    title = ""
                    save_ext = []
                feature_predicts = [
                    {
                        "predict": alternative["predict"],
                        "low": alternative["predict_low"],
                        "high": alternative["predict_high"],
                        "classes": alternative["classes"],
                    }
                    for alternative in alternatives
                ]

                widths = [
                    np.reshape(
                        np.array(alternative["weight_high"]) - np.array(alternative["weight_low"]),
                        (len(alternative["weight"])),
                    )
                    for alternative in alternatives
                ]

                features_weights = [
                    np.reshape(alternative["weight"], (len(alternative["weight"])))
                    for alternative in alternatives
                ]

                def _rank_features_for_multiclass(explanation, *args, **rank_kwargs):
                    rank_fn = getattr(explanation, "rank_features", None)
                    if callable(rank_fn):
                        return rank_fn(*args, **rank_kwargs)
                    return explanation._rank_features(*args, **rank_kwargs)

                if rnk_metric == "feature_weight":
                    features_list_to_plot = [
                        _rank_features_for_multiclass(
                            ex, feature_weights, width=width, num_to_show=num_to_show
                        )
                        for ex, feature_weights, width, num_to_show in zip(
                            list(class_explanations.values()),
                            features_weights,
                            widths,
                            filter_top,
                            strict=False,
                        )
                    ]
                else:
                    predictions = [
                        alternative["predict"]
                        if predict["predict"] > 0.5
                        else [1 - p for p in alternative["predict"]]
                        for alternative, predict in zip(alternatives, predicts, strict=False)
                    ]
                    rankings = [
                        calculate_metrics(
                            uncertainty=[
                                alternative["predict_high"][i] - alternative["predict_low"][i]
                                for i in range(len(alternative["rule"]))
                            ],
                            prediction=prediction,
                            w=rnk_weight,
                            metric=rnk_metric,
                        )
                        for alternative, prediction in zip(alternatives, predictions, strict=False)
                    ]
                    features_list_to_plot = [
                        _rank_features_for_multiclass(ex, width=ranking, num_to_show=num_to_show)
                        for ex, ranking, num_to_show in zip(
                            list(class_explanations.values()), rankings, filter_top, strict=False
                        )
                    ]  ####################

                if "style" in kwargs and kwargs["style"] == "triangular":
                    raise ValidationError(
                        "triangular style does not support multi labels explanation, please set multi_explanation to None and try again!."
                    )
                    """probas = [predict["predict"] for predict in predicts]
                    uncertainties = [np.abs(predict["high"] - predict["low"]) for predict in predicts]
                    rule_probas = [alternative["predict"] for alternative in alternatives]
                    rule_uncertainties = [np.abs(
                        np.array(alternative["predict_high"]) - np.array(alternative["predict_low"])
                    ) for alternative in alternatives]
                    # Use list comprehension or NumPy array indexing to select elements
                    selected_rule_probas = [[rule_proba[i] for i in features_to_plot] \
                                            for rule_proba, features_to_plot in zip(rule_probas, features_list_to_plot)]
                    selected_rule_uncertainties = [[rule_uncertainty[i] for i in features_to_plot] \
                                            for rule_uncertainty, features_to_plot in zip(rule_uncertainties, features_list_to_plot)]

                    _plot_triangular(
                        self,
                        proba,
                        uncertainty,
                        selected_rule_proba,
                        selected_rule_uncertainty,
                        num_to_show_,
                        title=title,
                        path=path,
                        show=show,
                        save_ext=save_ext,
                        style_override=style_override,
                    )"""
                    return

                alternatives_values = [alternative["value"] for alternative in alternatives]

                column_names_list = [alternative["rule"] for alternative in alternatives]
                _plot_alternative_dict(
                    list(class_explanations.values()),
                    alternatives_values,
                    predicts,
                    feature_predicts,
                    features_list_to_plot,
                    num_to_show_list=filter_top,
                    colors=colors,
                    column_names_list=column_names_list,
                    title=title,
                    path=path,
                    show=iter_show,
                    save_ext=save_ext,
                    style_override=style_override,
                    idx=i,
                )

    def merge_rules(self, factuals):  # pragma: no cover  # dead code: zero callers
        """Merge rule dictionaries from multiple class-specific factual explanations."""
        merged_factuals = {
            "base_predict": [],
            "base_predict_low": [],
            "base_predict_high": [],
            "predict": [],
            "predict_low": [],
            "predict_high": [],
            "weight": [],
            "weight_low": [],
            "weight_high": [],
            "value": [],
            "rule": [],
            "feature": [],
            "feature_value": [],
            "is_conjunctive": [],
            "classes": [],
        }

        for _i, factual in enumerate(factuals):  # pylint: disable=invalid-name
            base_predicts = [factual["base_predict"][0] for _ in range(len(factual["rule"]))]
            base_predict_low = [factual["base_predict_low"][0] for _ in range(len(factual["rule"]))]
            base_predict_high = [
                factual["base_predict_high"][0] for _ in range(len(factual["rule"]))
            ]
            classes = [factual["classes"] for _ in range(len(factual["rule"]))]

            merged_factuals["base_predict"].extend(base_predicts)
            merged_factuals["base_predict_low"].extend(base_predict_low)
            merged_factuals["base_predict_high"].extend(base_predict_high)
            merged_factuals["classes"].extend(classes)

            merged_factuals["predict"].extend(factual["predict"])
            merged_factuals["predict_low"].extend(factual["predict_low"])
            merged_factuals["predict_high"].extend(factual["predict_high"])
            merged_factuals["weight"].extend(factual["weight"])
            merged_factuals["weight_low"].extend(factual["weight_low"])
            merged_factuals["weight_high"].extend(factual["weight_high"])
            merged_factuals["value"].extend(factual["value"])
            merged_factuals["rule"].extend(factual["rule"])
            merged_factuals["feature"].extend(factual["feature"])
            merged_factuals["feature_value"].extend(factual["feature_value"])
            merged_factuals["is_conjunctive"].extend(factual["is_conjunctive"])

        return merged_factuals

    def plot_factual(
        self,
        index=None,
        class_idx=None,
        filter_top=10,
        show=True,
        filename="",
        uncertainty=False,
        style="regular",
        **kwargs,
    ):
        """
        Plot explanations for a given instance and class.

        If no class is specified, plots explanations for all classes at that index.
        """
        style_override = kwargs.get("style_override", get_multiclass_config())

        if index is not None:
            if class_idx is not None:
                explanation = self.get_explanation(index, class_idx)
                if explanation:
                    explanation.plot(
                        filter_top=filter_top,
                        show=show,
                        filename=filename,
                        uncertainty=uncertainty,
                        style=style,
                    )
                else:
                    warnings.warn(
                        f"No explanation found for instance {index}, class {class_idx}",
                        stacklevel=2,
                    )
            else:
                self.__getitem__(index).plot(
                    filter_top=filter_top,
                    show=show,
                    filename=filename,
                    uncertainty=uncertainty,
                    style=style,
                )
        else:
            for i, class_explanations in enumerate(self.explanations):
                # Ensure style_override gets passed through
                # Delegate non-render payload construction to helper for testability
                payload = self._build_factual_plot_payload(
                    i=i,
                    class_explanations=class_explanations,
                    filename=filename,
                    show=show,
                    uncertainty=uncertainty,
                    style_override=style_override,
                    kwargs=kwargs,
                )

                if payload is None:
                    # No rules to plot for this instance
                    continue

                _plot_probabilistic_dict(
                    payload["class_explanations_list"],
                    payload["factual_values"],
                    payload["predicts"],
                    payload["feature_weights_list"],
                    payload["features_list_to_plot"],
                    payload["filter_top"],
                    payload["colors"],
                    payload["column_names_list"],
                    title=payload["title"],
                    path=payload["path"],
                    interval=payload["interval"],
                    show=payload["show"],
                    idx=payload["idx"],
                    save_ext=payload["save_ext"],
                    style_override=payload["style_override"],
                )

    def sort_factuals_by_rule(self, factuals):  # pragma: no cover  # ADR-023: multiclass viz
        """Group factual explanation entries by rule string across classes."""
        sorted_factuals = {}
        factual_rule = {
            "base_predict": [],
            "base_predict_low": [],
            "base_predict_high": [],
            "predict": [],
            "predict_low": [],
            "predict_high": [],
            "weight": [],
            "weight_low": [],
            "weight_high": [],
            "value": [],
            "rule": [],
            "feature": [],
            "feature_value": [],
            "is_conjunctive": [],
            "classes": [],
        }
        for _i, factual in enumerate(factuals):  # pylint: disable=invalid-name
            base_predict = factual["base_predict"][0]
            base_predict_low = factual["base_predict_low"][0]
            base_predict_high = factual["base_predict_high"][0]
            cls = factual["classes"]
            for j, rule in enumerate(factual["rule"]):
                if rule not in sorted_factuals:
                    sorted_factuals[rule] = deepcopy(factual_rule)

                sorted_factuals[rule]["base_predict"].append(base_predict)
                sorted_factuals[rule]["base_predict_low"].append(base_predict_low)
                sorted_factuals[rule]["base_predict_high"].append(base_predict_high)
                sorted_factuals[rule]["classes"].append(cls)

                sorted_factuals[rule]["predict"].append(factual["predict"][j])
                sorted_factuals[rule]["predict_low"].append(factual["predict_low"][j])
                sorted_factuals[rule]["predict_high"].append(factual["predict_high"][j])
                sorted_factuals[rule]["weight"].append(factual["weight"][j])
                sorted_factuals[rule]["weight_low"].append(factual["weight_low"][j])
                sorted_factuals[rule]["weight_high"].append(factual["weight_high"][j])
                sorted_factuals[rule]["value"].append(factual["value"][j])
                sorted_factuals[rule]["rule"].append(factual["rule"][j])
                sorted_factuals[rule]["feature"].append(factual["feature"][j])
                sorted_factuals[rule]["feature_value"].append(factual["feature_value"][j])
                sorted_factuals[rule]["is_conjunctive"].append(factual["is_conjunctive"][j])
        return sorted_factuals

    def _build_factual_plot_payload(
        self,
        *,
        i: int,
        class_explanations: Mapping[Any, Any],
        filename: str,
        show: bool,
        uncertainty: bool,
        style_override: Any,
        kwargs: Mapping[str, Any],
    ) -> dict | None:
        """Construct the non-render payload for plotting factual multiclass explanations.

        Returns a dict containing the exact arguments needed by `_plot_probabilistic_dict`.
        Returns ``None`` when there are no rules to plot for the given instance.
        """
        # Prepare colors similar to previous inline logic
        import matplotlib.colors as mcolors

        rgb = np.array(list(permutations(range(0, 256, 11), 3))) / 255.0
        colors = [rgb.tolist()[i * 23] for i in range(25)]
        colors = list(mcolors.BASE_COLORS.values())

        class_explanations_list = list(class_explanations.values())

        rnk_metric = kwargs.get("rnk_metric", "feature_weight")
        if rnk_metric is None:
            rnk_metric = "feature_weight"
        rnk_weight = kwargs.get("rnk_weight", 0.5)
        if rnk_metric == "uncertainty":
            rnk_weight = 1.0
            rnk_metric = "ensured"

        factuals = [ex.get_rules() for ex in class_explanations_list]
        factuals = self.sort_factuals_by_rule(factuals)

        # Ensure each explanation has a sensible `index` set before checks
        for ex in class_explanations_list:
            with contextlib.suppress(Exception):
                ex.index = i
            with contextlib.suppress(Exception):
                ex._check_preconditions()

        predicts = [getattr(ex, "prediction", None) for ex in class_explanations_list]

        filter_top = [len(factual["weight"]) for factual in list(factuals.values())]
        if len(filter_top) <= 0:
            return None

        if uncertainty:
            feature_weights_list = [
                {
                    "predict": factual["weight"],
                    "low": factual["weight_low"],
                    "high": factual["weight_high"],
                    "classes": factual["classes"],
                }
                for factual in list(factuals.values())
            ]
        else:
            feature_weights_list = [
                {"predict": factual["weight"], "classes": factual["classes"]}
                for factual in list(factuals.values())
            ]

        widths = [
            np.reshape(
                np.array(factual["weight_high"]) - np.array(factual["weight_low"]),
                (len(factual["weight"])),
            )
            for factual in list(factuals.values())
        ]

        first_explanation = next(iter(class_explanations.values()))
        rank_features = getattr(first_explanation, "rank_features", None)
        if not callable(rank_features):
            rank_features = first_explanation._rank_features

        if rnk_metric == "feature_weight":
            features_list_to_plot = [
                rank_features(factual["weight"], width=width, num_to_show=num_to_show)
                for factual, width, num_to_show in zip(
                    list(factuals.values()), widths, filter_top, strict=False
                )
            ]
        else:
            rankings = [
                calculate_metrics(
                    uncertainty=[
                        factual["predict_high"][j] - factual["predict_low"][j]
                        for j in range(len(factual["weight"]))
                    ],
                    prediction=factual["predict"],
                    w=rnk_weight,
                    metric=rnk_metric,
                )
                for factual in list(factuals.values())
            ]
            features_list_to_plot = [
                rank_features(width=ranking, num_to_show=num_to_show)
                for ranking, num_to_show in zip(rankings, filter_top, strict=False)
            ]

        column_names_list = list(factuals)
        factual_values = [factual["value"] for factual in list(factuals.values())]

        # Prepare filename/path/title/save_ext
        if len(filename) > 0:
            path, _, title, ext = prepare_for_saving(str(i) + "_" + filename)
            path = f"plots/{path}"
            save_ext = [ext]
        else:
            path = ""
            title = ""
            save_ext = []

        return {
            "class_explanations_list": list(class_explanations.values()),
            "factual_values": factual_values,
            "predicts": predicts,
            "feature_weights_list": feature_weights_list,
            "features_list_to_plot": features_list_to_plot,
            "filter_top": filter_top,
            "colors": colors,
            "column_names_list": column_names_list,
            "title": title,
            "path": path,
            "interval": uncertainty,
            "show": show,
            "idx": i,
            "save_ext": save_ext,
            "style_override": style_override,
        }
