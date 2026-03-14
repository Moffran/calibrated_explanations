"""Types for reject-aware explanation envelopes."""

from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from math import isclose
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np

from ..utils.exceptions import DataShapeError, ValidationError
from .explanations import AlternativeExplanations, CalibratedExplanations


class RejectPolicy(Enum):
    """Describe how rejection should affect prediction/explanation invocation.

    Policies
    --------
    - NONE: Preserve legacy behaviour (no reject orchestration).
    - FLAG: Process all instances and tag rejection status in the envelope.
    - ONLY_REJECTED: Process only rejected (uncertain) instances.
    - ONLY_ACCEPTED: Process only non-rejected (confident) instances.

    Deprecated Aliases (emit DeprecationWarning, removed in v1.0.0)
    ---------------------------------------------------------------
    - PREDICT_AND_FLAG -> FLAG
    - EXPLAIN_ALL -> FLAG
    - EXPLAIN_REJECTS -> ONLY_REJECTED
    - EXPLAIN_NON_REJECTS -> ONLY_ACCEPTED
    - SKIP_ON_REJECT -> ONLY_ACCEPTED
    """

    NONE = "none"
    FLAG = "flag"
    ONLY_REJECTED = "only_rejected"
    ONLY_ACCEPTED = "only_accepted"

    @classmethod
    def _missing_(cls, value: object) -> RejectPolicy | None:
        """Handle deprecated policy names with warnings."""
        if not isinstance(value, str):
            return None

        deprecation_map = {
            "predict_and_flag": ("FLAG", cls.FLAG),
            "explain_all": ("FLAG", cls.FLAG),
            "explain_rejects": ("ONLY_REJECTED", cls.ONLY_REJECTED),
            "explain_non_rejects": ("ONLY_ACCEPTED", cls.ONLY_ACCEPTED),
            "skip_on_reject": ("ONLY_ACCEPTED", cls.ONLY_ACCEPTED),
        }

        lower_value = value.lower()
        if lower_value in deprecation_map:
            new_name, new_policy = deprecation_map[lower_value]
            from ..utils.deprecations import deprecate_alias

            # Emit a standardized alias deprecation message
            deprecate_alias(lower_value, new_name, stacklevel=3)
            return new_policy
        return None


_VALID_NCF = frozenset({"default", "ensured"})
_LEGACY_NCF_SILENT_MAP = {"entropy": "default"}
_REMOVED_EXPLICIT_NCF = frozenset({"hinge", "margin"})


class RejectContractWarning(UserWarning):
    """Visible warning for reject contract fallback/coercion paths."""


def normalize_reject_ncf_choice(ncf: str) -> str:
    """Normalize a user-facing reject NCF choice.

    Accepted public values are ``default`` and ``ensured``.
    Legacy ``entropy`` is silently mapped to ``default``.
    Explicit ``hinge``/``margin`` inputs are rejected.
    """
    lowered = str(ncf).strip().lower()
    if lowered in _LEGACY_NCF_SILENT_MAP:
        return _LEGACY_NCF_SILENT_MAP[lowered]
    if lowered in _REMOVED_EXPLICIT_NCF:
        raise ValueError(  # adr002_allow - public dataclass validation contract uses ValueError
            "Explicit ncf values 'hinge' and 'margin' are no longer supported; "
            "use ncf='default' instead."
        )
    if lowered not in _VALID_NCF:
        raise ValueError(  # adr002_allow - public dataclass validation contract uses ValueError
            f"ncf must be one of {sorted(_VALID_NCF)!r}; got {ncf!r}"
        )
    return lowered


def canonical_reject_ncf_w(ncf: str, w: float) -> float:
    """Return the effective canonical ``w`` for the given NCF.

    ``w`` is operational only for ``ensured``. For other NCFs the value is
    accepted for API compatibility but ignored and normalized.
    """
    if ncf == "ensured":
        return float(w)
    return 0.0


@dataclass(eq=False)
class RejectPolicySpec:
    """Bundle a RejectPolicy with a non-conformity function (NCF) configuration.

    Parameters
    ----------
    policy : RejectPolicy
        The reject policy (FLAG, ONLY_REJECTED, ONLY_ACCEPTED).
    ncf : str, default 'default'
        Reject NCF mode: ``default`` (task-dependent internal default score)
        or ``ensured`` (interval-width blended with default score).
        Legacy ``entropy`` is accepted and silently mapped to ``default``.
    w : float, default 0.5
        Blending weight in [0, 1] used only when ``ncf='ensured'``.
        For ``ncf='default'``, the value is accepted but ignored and
        normalized to a canonical effective value.

    Class Methods
    -------------
    flag(ncf, w) / only_rejected(ncf, w) / only_accepted(ncf, w)
        Convenience constructors for common policy values.

    Examples
    --------
    >>> spec = RejectPolicySpec(RejectPolicy.FLAG, ncf='ensured', w=0.5)
    >>> spec = RejectPolicySpec.flag(ncf='default')
    """

    policy: RejectPolicy = RejectPolicy.NONE
    ncf: str = "default"
    w: float = 0.5  # pylint: disable=invalid-name

    def __post_init__(self) -> None:
        """Validate and canonicalize the NCF configuration after initialization."""
        self.ncf = normalize_reject_ncf_choice(self.ncf)
        if not 0.0 <= self.w <= 1.0:
            raise ValueError(  # adr002_allow - public dataclass validation contract uses ValueError
                f"w must be in [0, 1]; got {self.w}"
            )
        self.w = canonical_reject_ncf_w(self.ncf, float(self.w))

    @classmethod
    def flag(cls, ncf: str = "default", w: float = 0.5) -> "RejectPolicySpec":  # pylint: disable=invalid-name
        """Return a FLAG RejectPolicySpec with the given NCF configuration."""
        return cls(RejectPolicy.FLAG, ncf=ncf, w=w)

    @classmethod
    def only_rejected(cls, ncf: str = "default", w: float = 0.5) -> "RejectPolicySpec":  # pylint: disable=invalid-name
        """Return an ONLY_REJECTED RejectPolicySpec with the given NCF configuration."""
        return cls(RejectPolicy.ONLY_REJECTED, ncf=ncf, w=w)

    @classmethod
    def only_accepted(cls, ncf: str = "default", w: float = 0.5) -> "RejectPolicySpec":  # pylint: disable=invalid-name
        """Return an ONLY_ACCEPTED RejectPolicySpec with the given NCF configuration."""
        return cls(RejectPolicy.ONLY_ACCEPTED, ncf=ncf, w=w)

    def __eq__(self, other: object) -> bool:
        """Support equality comparison with RejectPolicy enum members and other specs.

        ``spec == RejectPolicy.ONLY_REJECTED`` returns True when
        ``spec.policy is RejectPolicy.ONLY_REJECTED``, enabling mixed policy
        lists to be compared with ``==`` regardless of type.
        """
        if isinstance(other, RejectPolicy):
            return self.policy == other
        if isinstance(other, RejectPolicySpec):
            return (
                self.policy == other.policy
                and self.ncf == other.ncf
                and isclose(self.w, other.w, rel_tol=1e-9, abs_tol=0.0)
            )
        return NotImplemented

    def __hash__(self) -> int:
        """Return a stable hash aligned with normalized equality semantics."""
        return hash((self.policy, self.ncf, round(self.w, 12)))

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical serializable representation of this policy spec.

        Serialization is intentionally strict and deterministic.
        """
        return {"policy": self.policy.value, "ncf": self.ncf, "w": float(self.w)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RejectPolicySpec":
        """Create a policy spec from :meth:`to_dict` payload."""
        try:
            policy = payload["policy"]
            ncf = payload["ncf"]
            w = payload["w"]
        except KeyError as exc:
            raise ValidationError(
                "Missing key in RejectPolicySpec.from_dict",
                details={"missing": str(exc)},
            ) from exc
        if not isinstance(ncf, str):
            raise ValidationError(
                "RejectPolicySpec.from_dict expects a string ncf.",
                details={"ncf_type": type(ncf).__name__},
            )
        return cls(RejectPolicy(policy), ncf=ncf, w=float(w))

    @property
    def value(self) -> str:
        """String value compatible with RejectPolicy.value for use in mixed policy lists."""
        return f"{self.policy.value}[ncf={self.ncf},w={self.w:.12g}]"


@dataclass
class RejectResult:
    """Envelope returned when a reject policy is active.

    Fields are intentionally optional to allow gradual rollout and
    compatibility with existing consumers.
    """

    prediction: Optional[Any] = None
    explanation: Optional[Any] = None
    rejected: Optional[Any] = None
    policy: RejectPolicy = RejectPolicy.NONE
    metadata: Dict[str, Any] | None = None


@dataclass
class RejectContext:
    """Expertise-adaptable reject information attached to explanations.

    Fields mirror the recommended integration: minimal structured metadata
    plus optional rendered strings for each expertise level.
    """

    rejected: bool
    reject_type: str | None = None  # "ambiguity" | "novelty" | None
    prediction_set_size: int = 1
    confidence: float | None = None
    prediction_set_ref: dict[str, Any] | None = None
    # Rendered strings (optional) - templates preferred instead of hardcoding
    beginner_text: str | None = None
    intermediate_text: str | None = None
    advanced_text: str | None = None

    def materialize_prediction_set(self, explainer: Any) -> Any | None:
        """Materialize prediction-set labels/indices from a lightweight reference."""
        if self.prediction_set_ref is None:
            return None
        indices = self.prediction_set_ref.get("indices")
        if not isinstance(indices, list):
            return self.prediction_set_ref.get("summary")
        labels = getattr(explainer, "class_labels", None)
        if labels:
            return {labels.get(i, i) for i in indices}
        return set(indices)


def _to_json_safe(value: Any) -> Any:
    """Recursively normalize nested metadata values to JSON-safe Python types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]
    return value


def as_packed_bits(arr: np.ndarray) -> np.ndarray:
    """Pack a boolean array into uint8 bits."""
    return np.packbits(np.asarray(arr, dtype=np.bool_), bitorder="little")


def as_unpacked_bits(packed: np.ndarray, length: int) -> np.ndarray:
    """Unpack a uint8 bit-array into a boolean array of ``length``."""
    unpacked = np.unpackbits(np.asarray(packed, dtype=np.uint8), bitorder="little")
    return unpacked[:length].astype(np.bool_)


def _normalize_raw_count_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy raw-count metadata aliases to the canonical key.

    Preserve any existing ``raw_reject_counts`` mapping and merge legacy
    ``_raw_reject_counts`` values when present.
    """
    if not meta:
        return {"raw_reject_counts": {}}

    normalized = dict(meta)
    legacy_counts = normalized.pop("_raw_reject_counts", None)

    existing = normalized.get("raw_reject_counts")
    if existing is None or not isinstance(existing, dict):
        normalized["raw_reject_counts"] = {}
    else:
        normalized["raw_reject_counts"] = dict(existing)

    if isinstance(legacy_counts, dict):
        normalized["raw_reject_counts"].update(legacy_counts)

    return normalized


def _canonicalize_degraded_mode(value: Any) -> tuple[str, ...]:
    """Return deterministic degraded-mode tuple from arbitrary metadata payloads."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set)):
        return tuple(str(v) for v in value if v)
    return (str(value),)


def _normalize_contract_metadata(
    *,
    metadata: dict[str, Any] | None,
    policy: RejectPolicy,
    rejected: Any | None,
    source_indices: Any | None,
    original_count: int | None,
) -> dict[str, Any]:
    """Ensure required non-NONE reject metadata keys are present and coherent."""
    normalized = _normalize_raw_count_metadata(dict(metadata or {}))

    if original_count is None:
        if normalized.get("original_count") is not None:
            original_count_int = int(normalized["original_count"])
        elif rejected is not None:
            original_count_int = int(len(np.asarray(rejected)))
        else:
            original_count_int = 0
    else:
        original_count_int = int(original_count)

    if source_indices is None:
        if normalized.get("source_indices") is not None:
            idxs = [int(v) for v in np.asarray(normalized.get("source_indices")).tolist()]
        else:
            idxs = list(range(original_count_int))
    else:
        idxs = [int(v) for v in np.asarray(source_indices).tolist()]

    rejected_arr = None if rejected is None else np.asarray(rejected, dtype=bool)
    rejected_count = normalized.get("rejected_count")
    if rejected_count is None:
        if rejected_arr is not None and len(rejected_arr) == original_count_int:
            rejected_count = int(np.sum(rejected_arr))
        else:
            rejected_count = int(normalized.get("raw_reject_counts", {}).get("rejected", 0))
    rejected_count = int(rejected_count)
    accepted_count = int(
        normalized.get("accepted_count", max(original_count_int - rejected_count, 0))
    )
    reject_rate = normalized.get("reject_rate")
    if reject_rate is None:
        reject_rate = float(rejected_count / original_count_int) if original_count_int > 0 else 0.0
    degraded_mode = _canonicalize_degraded_mode(normalized.get("degraded_mode"))
    init_error = bool(normalized.get("init_error", False))
    init_ok = bool(normalized.get("init_ok", not init_error))
    fallback_used = bool(normalized.get("fallback_used", bool(degraded_mode) or init_error))

    normalized.update(
        {
            "policy": str(normalized.get("policy", policy.value)),
            "reject_rate": float(reject_rate),
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "effective_confidence": normalized.get("effective_confidence"),
            "effective_threshold": normalized.get("effective_threshold"),
            "source_indices": idxs,
            "original_count": original_count_int,
            "init_ok": init_ok,
            "fallback_used": fallback_used,
            "init_error": init_error,
            "degraded_mode": degraded_mode,
        }
    )
    return normalized


class RejectMixin:
    """Mixin to hold global reject metadata on a CalibratedExplanations collection."""

    policy: RejectPolicy
    rejected: np.ndarray | None
    _metadata: Dict[str, Any] | None
    ambiguity_mask: np.ndarray | None
    novelty_mask: np.ndarray | None
    prediction_set_size: np.ndarray | None
    prediction_set: np.ndarray | None
    epsilon: float | None

    def initialize_reject_metadata(self) -> None:
        """Initialize per-instance reject metadata fields on wrapper objects."""
        self.policy = RejectPolicy.NONE
        self.rejected = None
        self._metadata = {}
        self.ambiguity_mask = None
        self.novelty_mask = None
        self.prediction_set_size = None
        self.prediction_set = None
        self.epsilon = None

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return a lightweight, serialisable view of reject metadata."""
        meta = dict(self._metadata) if self._metadata else {}
        meta = _normalize_raw_count_metadata(meta)
        for heavy_key in (
            "ambiguity_mask",
            "novelty_mask",
            "prediction_set_size",
            "rejected",
            "prediction_set",
        ):
            meta.pop(heavy_key, None)

        if self.ambiguity_mask is not None:
            ambiguity = np.asarray(self.ambiguity_mask)
            meta["ambiguity_rate"] = float(np.mean(ambiguity)) if len(ambiguity) else 0.0
            meta["ambiguity_count"] = int(np.sum(ambiguity))
            meta["ambiguity_shape"] = tuple(ambiguity.shape)
        if self.novelty_mask is not None:
            novelty = np.asarray(self.novelty_mask)
            meta["novelty_rate"] = float(np.mean(novelty)) if len(novelty) else 0.0
            meta["novelty_count"] = int(np.sum(novelty))
            meta["novelty_shape"] = tuple(novelty.shape)
        if self.prediction_set_size is not None:
            sizes = np.asarray(self.prediction_set_size)
            meta["prediction_set_size_summary"] = {
                "min": int(np.min(sizes)) if len(sizes) else 0,
                "max": int(np.max(sizes)) if len(sizes) else 0,
                "mean": float(np.mean(sizes)) if len(sizes) else 0.0,
            }
        if self.rejected is not None:
            rejected = np.asarray(self.rejected)
            payload_rejected_count = int(np.sum(rejected))
            payload_count = int(len(rejected))
            payload_reject_rate = float(np.mean(rejected)) if payload_count else 0.0
            meta.setdefault("rejected_count", payload_rejected_count)
            meta.setdefault("accepted_count", max(payload_count - payload_rejected_count, 0))
            meta.setdefault("reject_rate", payload_reject_rate)
            meta["payload_rejected_count"] = payload_rejected_count
            meta["payload_accepted_count"] = max(payload_count - payload_rejected_count, 0)
            meta["payload_reject_rate"] = payload_reject_rate
            meta["payload_count"] = payload_count
        meta = _normalize_contract_metadata(
            metadata=meta,
            policy=getattr(self, "policy", RejectPolicy.NONE),
            rejected=None,
            source_indices=meta.get("source_indices"),
            original_count=meta.get("original_count"),
        )
        return meta

    def metadata_summary(self) -> Dict[str, Any]:
        """Return lightweight metadata summary (alias of ``metadata``)."""
        return self.metadata

    def metadata_full(self) -> Dict[str, Any]:
        """Return full metadata including per-instance arrays as JSON-safe values."""
        meta = dict(self._metadata) if self._metadata else {}
        meta = _normalize_raw_count_metadata(meta)
        if self.ambiguity_mask is not None:
            meta["ambiguity_mask"] = np.asarray(self.ambiguity_mask)
        if self.novelty_mask is not None:
            meta["novelty_mask"] = np.asarray(self.novelty_mask)
        if self.prediction_set_size is not None:
            meta["prediction_set_size"] = np.asarray(self.prediction_set_size)
        if self.rejected is not None:
            meta["rejected"] = np.asarray(self.rejected)
        if self.prediction_set is not None:
            meta["prediction_set"] = np.asarray(self.prediction_set)
        meta = _normalize_contract_metadata(
            metadata=meta,
            policy=getattr(self, "policy", RejectPolicy.NONE),
            rejected=None,
            source_indices=meta.get("source_indices"),
            original_count=meta.get("original_count"),
        )
        return _to_json_safe(meta)

    def clear_reject_arrays(self, keep_summary: bool = True) -> None:
        """Drop per-instance reject arrays while optionally preserving summary metadata."""
        if not keep_summary:
            self._metadata = {}
        self.rejected = None
        self.ambiguity_mask = None
        self.novelty_mask = None
        self.prediction_set_size = None
        self.prediction_set = None

    def memory_profile(self) -> dict[str, int]:
        """Return byte estimates for heavy reject arrays and the total bytes."""
        profile: dict[str, int] = {}
        total = 0
        for name in (
            "rejected",
            "ambiguity_mask",
            "novelty_mask",
            "prediction_set_size",
            "prediction_set",
        ):
            value = getattr(self, name, None)
            if value is None:
                profile[name] = 0
                continue
            arr = np.asarray(value)
            size = int(arr.nbytes)
            profile[name] = size
            total += size
        profile["total_bytes"] = total
        return profile

    def to_packed_masks(self) -> None:
        """Pack boolean reject masks to compact uint8 bit-arrays in place."""
        if self.rejected is not None:
            self.rejected = as_packed_bits(np.asarray(self.rejected, dtype=np.bool_))
        if self.ambiguity_mask is not None:
            self.ambiguity_mask = as_packed_bits(np.asarray(self.ambiguity_mask, dtype=np.bool_))
        if self.novelty_mask is not None:
            self.novelty_mask = as_packed_bits(np.asarray(self.novelty_mask, dtype=np.bool_))

    @staticmethod
    def _validate_key_indexing(key: Any, src_len: int, field: str) -> None:
        """Validate indexing keys for reject-field slicing.

        Supported indexers are: integers, slices, boolean masks, and integer
        index lists/arrays. Mixed or object indexers are rejected.
        """
        if isinstance(key, int):
            if key >= src_len or key < -src_len:
                raise DataShapeError(
                    f"Reject field '{field}' integer index out of bounds",
                    details={
                        "field": field,
                        "source_length": src_len,
                        "key_kind": "int",
                        "key": key,
                    },
                )
            return
        if isinstance(key, slice):
            return
        if not isinstance(key, (list, np.ndarray)):
            raise DataShapeError(
                f"Unsupported key type for reject slicing: {type(key)}",
                details={"field": field, "source_length": src_len, "key_kind": type(key).__name__},
            )

        indexer = np.asarray(key)
        if np.issubdtype(indexer.dtype, np.bool_):
            if len(indexer) != src_len:
                raise DataShapeError(
                    f"Reject field '{field}' boolean mask length mismatch",
                    details={
                        "field": field,
                        "source_length": src_len,
                        "key_kind": "bool_mask",
                        "mask_length": len(indexer),
                    },
                )
            return

        if np.issubdtype(indexer.dtype, np.integer):
            if indexer.size == 0:
                return
            if np.any(indexer >= src_len) or np.any(indexer < -src_len):
                raise DataShapeError(
                    f"Reject field '{field}' integer list index out of bounds",
                    details={
                        "field": field,
                        "source_length": src_len,
                        "key_kind": "index_list",
                        "key": key,
                    },
                )
            return

        raise DataShapeError(
            f"Reject field '{field}' indexer must be boolean mask or integer list/ndarray",
            details={"field": field, "source_length": src_len, "key_kind": type(key).__name__},
        )

    def _slice_reject_fields(self, key, source: RejectMixin):
        """Slice reject metadata in sync with explanation slicing with strict validation.

        Notes
        -----
        ``raw_reject_counts`` stores canonical *sums* for sliced arrays:
        boolean masks store the count of ``True`` values and
        ``prediction_set_size`` stores the numeric total of set sizes.
        """
        fields = [
            "rejected",
            "ambiguity_mask",
            "novelty_mask",
            "prediction_set_size",
            "prediction_set",
        ]
        raw_counts: dict[str, int] = {}

        for field in fields:
            val = getattr(source, field, None)
            if val is None:
                setattr(self, field, None)
                raw_counts[field] = 0
                continue
            arr = np.asarray(val)
            if arr.ndim == 0:
                raise DataShapeError(
                    f"Reject field '{field}' is scalar and cannot be sliced.",
                    details={"field": field, "key_kind": type(key).__name__},
                )
            src_len = arr.shape[0]
            self._validate_key_indexing(key, src_len, field)
            try:
                sliced = arr[key]
            except (IndexError, TypeError, ValueError) as exc:
                raise DataShapeError(
                    f"Slicing reject field '{field}' failed: {exc}",
                    details={"field": field, "src_len": src_len, "key": repr(key)},
                ) from exc
            sliced_arr = np.array(sliced, copy=True)
            setattr(self, field, sliced_arr)
            # Canonical raw counts are sums (True-count for masks, numeric sum
            # for prediction_set_size), aligned with RejectOrchestrator output.
            if field in ("rejected", "ambiguity_mask", "novelty_mask", "prediction_set_size"):
                raw_counts[field] = int(np.sum(sliced_arr))
            elif field == "prediction_set":
                raw_counts.setdefault("prediction_set_size", int(np.sum(sliced_arr)))

        # Constant fields just copy
        self.policy = source.policy
        self.epsilon = source.epsilon
        # Copy base metadata and preserve original-batch contract semantics.
        self._metadata = deepcopy(source._metadata) if source._metadata is not None else {}
        if self._metadata is not None:
            self._metadata = _normalize_raw_count_metadata(self._metadata)
            source_meta = (
                source._metadata if isinstance(getattr(source, "_metadata", None), dict) else {}
            )
            source_indices_meta = source_meta.get("source_indices")
            try:
                source_indices_arr = (
                    np.asarray(source_indices_meta, dtype=int)
                    if source_indices_meta is not None
                    else np.arange(
                        len(source.rejected)
                        if source.rejected is not None
                        else len(source.explanations)
                    )
                )
                self._validate_key_indexing(key, len(source_indices_arr), "source_indices")
                sliced_source_indices = np.asarray(source_indices_arr[key], dtype=int).reshape(-1)
            except Exception:  # adr002_allow - best-effort metadata continuity
                sliced_source_indices = np.arange(
                    len(self.rejected) if self.rejected is not None else 0, dtype=int
                )

            if source_meta.get("raw_total_examples") is not None:
                self._metadata["raw_total_examples"] = int(source_meta["raw_total_examples"])
            else:
                self._metadata["raw_total_examples"] = (
                    int(len(source.rejected)) if source.rejected is not None else 0
                )
            self._metadata["source_indices"] = [int(v) for v in sliced_source_indices.tolist()]
            if source_meta.get("original_count") is not None:
                self._metadata["original_count"] = int(source_meta["original_count"])
            elif source.rejected is not None:
                self._metadata["original_count"] = int(len(source.rejected))
            else:
                self._metadata["original_count"] = int(len(source.explanations))
            self._metadata.setdefault("raw_reject_counts", {})
            self._metadata["raw_reject_counts"].update(raw_counts)
            rejected_arr = np.asarray(self.rejected) if self.rejected is not None else None
            n = len(rejected_arr) if rejected_arr is not None else 0  # pylint: disable=invalid-name
            self._metadata["sliced_total_examples"] = int(n)
            recomputed: dict = {
                "payload_reject_rate": 0.0,
                "payload_rejected_count": 0,
                "payload_accepted_count": 0,
                "payload_count": int(n),
                "ambiguity_rate": 0.0,
                "novelty_rate": 0.0,
                "error_rate_defined": False,
                "error_rate": 0.0,
            }
            if n > 0:
                if self.rejected is not None:
                    payload_rejected_count = int(np.sum(self.rejected))
                    recomputed["payload_reject_rate"] = float(np.mean(self.rejected))
                    recomputed["payload_rejected_count"] = payload_rejected_count
                    recomputed["payload_accepted_count"] = max(int(n) - payload_rejected_count, 0)
                if self.ambiguity_mask is not None:
                    recomputed["ambiguity_rate"] = float(np.mean(self.ambiguity_mask))
                if self.novelty_mask is not None:
                    recomputed["novelty_rate"] = float(np.mean(self.novelty_mask))
                sizes = (
                    np.asarray(self.prediction_set_size)
                    if self.prediction_set_size is not None
                    else None
                )
                epsilon = (
                    self.epsilon if self.epsilon is not None else self._metadata.get("epsilon")
                )
                if sizes is not None and epsilon is not None:
                    singleton = int(np.sum(sizes == 1))
                    empty = int(np.sum(sizes == 0))
                    if singleton > 0:
                        recomputed["error_rate"] = float(
                            max(0.0, min(1.0, (n * float(epsilon) - empty) / singleton))
                        )
                        recomputed["error_rate_defined"] = True
            self._metadata = {**self._metadata, **recomputed}
            self._metadata = _normalize_contract_metadata(
                metadata=self._metadata,
                policy=self.policy,
                rejected=None,
                source_indices=self._metadata.get("source_indices"),
                original_count=self._metadata.get("original_count"),
            )


def _ensure_runtime_available(instance: Any) -> None:
    """Raise a clear error when operating on read-only unpickled wrappers."""
    if getattr(instance, "_pickled_readonly", False) is True:
        raise RuntimeError(  # adr002_allow - API contract requires RuntimeError for pickled read-only access
            "read-only pickled RejectCalibratedExplanations; call .reconstruct_runtime(...) to restore runtime handles"
        )


def _copy_collection_attributes(obj: Any, base: Any) -> None:
    """Copy minimal collection attributes without duplicating calibration-heavy state."""
    import logging
    from types import ModuleType

    copy_by_reference = {
        "plugin_manager",
        "prediction_orchestrator",
        "perf_cache",
        "calibrator_cache",
        "calibrated_explainer",
        "frozen_calibrated_explainer",
        "learner",
        "predict_function",
        "rng",
        "reject_orchestrator",
        "_predict_bridge",
        "explanations",
        "x_test",
    }
    # Any attribute name containing one of these markers is considered
    # calibration/heavy state and should not be duplicated on reject wrappers.
    calibration_markers = (
        "x_cal",
        "_X_cal",
        "scaled_x_cal",
        "scaled_y_cal",
        "fast_x_cal",
        "calibrator",
        "calibrat",
        "interval_learner",
        "calibrator_cache",
        "feature_frequencies",
        "feature_values",
    )

    for name, value in vars(base).items():
        if name.startswith("__") and name.endswith("__"):
            continue
        descriptor = getattr(type(base), name, None)
        if isinstance(descriptor, property):
            continue
        if callable(value) or isinstance(value, (logging.Logger, ModuleType)):
            continue

        if name in copy_by_reference:
            setattr(obj, name, value)
            continue
        if any(marker in name for marker in calibration_markers):
            continue
        if isinstance(value, np.ndarray):
            setattr(obj, name, np.copy(value))
        elif isinstance(value, (list, dict, set)):
            setattr(obj, name, value.copy())
        else:
            setattr(obj, name, value)


def _copy_array_or_none(value: Any, *, ensure_dtype: Any | None = None) -> np.ndarray | None:
    """Return a copied numpy array for per-instance reject fields, preserving dtype."""
    if value is None:
        return None
    arr = np.asarray(value)
    if ensure_dtype is not None:
        arr = arr.astype(ensure_dtype, copy=False)
    return np.array(arr, copy=True)


def _lightweight_reject_metadata(metadata: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return canonical lightweight reject metadata without per-instance arrays."""
    if not metadata:
        return {"raw_reject_counts": {}}
    normalized = _normalize_raw_count_metadata(dict(metadata))
    for heavy_key in (
        "ambiguity_mask",
        "novelty_mask",
        "prediction_set_size",
        "rejected",
        "prediction_set",
    ):
        normalized.pop(heavy_key, None)
    return normalized


def _resolve_source_indices_for_wrapper(
    *,
    policy: RejectPolicy,
    metadata: Dict[str, Any] | None,
    rejected: np.ndarray | None,
    payload_count: int,
) -> np.ndarray:
    """Resolve a strict source-index mapping for wrapper alignment.

    Raises
    ------
    DataShapeError
        If a safe, deterministic source mapping cannot be established.
    """
    logger = logging.getLogger(__name__)
    source_raw = (metadata or {}).get("source_indices")
    if source_raw is not None:
        idx_arr = np.asarray(source_raw)
        if idx_arr.ndim != 1 or not np.issubdtype(idx_arr.dtype, np.integer):
            raise DataShapeError(
                "source_indices must be a one-dimensional integer sequence",
                details={"source_indices_type": type(source_raw).__name__},
            )
        idxs = idx_arr.astype(int, copy=False)
        if len(idxs) != payload_count:
            raise DataShapeError(
                "source_indices length mismatch for filtered reject payload",
                details={
                    "source_indices_len": int(len(idxs)),
                    "payload_count": int(payload_count),
                },
            )
        if np.any(idxs < 0):
            raise DataShapeError(
                "source_indices must be non-negative",
                details={"source_indices": idxs.tolist()},
            )
        if len(set(idxs.tolist())) != len(idxs):
            raise DataShapeError(
                "source_indices must be unique",
                details={"source_indices": idxs.tolist()},
            )
        if np.any(np.diff(idxs) <= 0):
            raise DataShapeError(
                "source_indices must preserve source ordering",
                details={"source_indices": idxs.tolist()},
            )
        original_count = (metadata or {}).get("original_count")
        if original_count is not None:
            original_count_int = int(original_count)
            if np.any(idxs >= original_count_int):
                raise DataShapeError(
                    "source_indices must be < original_count",
                    details={
                        "source_indices": idxs.tolist(),
                        "original_count": original_count_int,
                    },
                )
        return np.array(idxs, copy=True)

    if rejected is None:
        if payload_count == 0:
            return np.array([], dtype=int)
        raise DataShapeError(
            "Cannot align filtered reject payload without source_indices metadata or rejected mask.",
            details={"payload_count": int(payload_count), "policy": policy.value},
        )

    rejected_arr = np.asarray(rejected, dtype=bool)
    if policy is RejectPolicy.FLAG:
        idxs = np.arange(len(rejected_arr), dtype=int)
    elif policy is RejectPolicy.ONLY_REJECTED:
        idxs = np.flatnonzero(rejected_arr)
    elif policy is RejectPolicy.ONLY_ACCEPTED:
        idxs = np.flatnonzero(~rejected_arr)
    else:
        idxs = np.arange(len(rejected_arr), dtype=int)

    if len(idxs) != payload_count:
        raise DataShapeError(
            "Cannot derive deterministic source_indices from policy/rejected mask.",
            details={
                "derived_len": int(len(idxs)),
                "payload_count": int(payload_count),
                "policy": policy.value,
            },
        )
    warnings.warn(
        "Reject result is missing source_indices metadata; derived mapping from policy/rejected mask.",
        RejectContractWarning,
        stacklevel=3,
    )
    logger.info("Derived source_indices fallback for reject wrapper alignment.")
    return np.array(idxs, dtype=int)


def _align_reject_field_to_payload(
    *,
    name: str,
    value: Any,
    payload_count: int,
    source_indices: np.ndarray,
    ensure_dtype: Any | None = None,
) -> np.ndarray | None:
    """Align a reject per-instance field to payload length using source indices."""
    arr = _copy_array_or_none(value, ensure_dtype=ensure_dtype)
    if arr is None:
        return None
    if arr.ndim == 0:
        raise DataShapeError(
            f"Reject field '{name}' is scalar and cannot be aligned.",
            details={"field": name},
        )
    if len(arr) == payload_count:
        return arr
    if np.any(source_indices >= len(arr)):
        raise DataShapeError(
            f"Reject field '{name}' is shorter than source_indices require.",
            details={
                "field": name,
                "field_length": int(len(arr)),
                "max_source_index": int(np.max(source_indices)) if len(source_indices) else -1,
            },
        )
    return np.array(arr[source_indices], copy=True)


def _prune_unpickleable_state(state: dict[str, Any]) -> dict[str, Any]:
    """Prune runtime-only, heavy, or unpicklable attributes before pickling."""
    prune_keys = {
        "plugin_manager",
        "prediction_orchestrator",
        "reject_orchestrator",
        "_predict_bridge",
        "perf_cache",
        "rng",
        "latest_explanation",
    }
    st = dict(state)
    for key in prune_keys:
        st.pop(key, None)
    return st


class RejectCalibratedExplanations(CalibratedExplanations, RejectMixin):
    """A CalibratedExplanations collection that carries rejection metadata."""

    @classmethod
    def from_collection(
        cls,
        base: CalibratedExplanations,
        metadata: Dict[str, Any],
        policy: RejectPolicy,
        rejected: Any = None,
    ) -> RejectCalibratedExplanations:
        """Create reject-aware collection without mutating ``__class__`` of a copied object."""
        obj = object.__new__(cls)
        obj.initialize_reject_metadata()
        _copy_collection_attributes(obj, base)
        obj.calibrated_explainer = getattr(base, "calibrated_explainer", None)
        obj.explanations = base.explanations
        obj.policy = policy
        rejected_arr = _copy_array_or_none(rejected, ensure_dtype=bool)
        payload_count = int(len(base.explanations))
        source_indices = _resolve_source_indices_for_wrapper(
            policy=policy,
            metadata=metadata,
            rejected=rejected_arr,
            payload_count=payload_count,
        )
        obj.rejected = _align_reject_field_to_payload(
            name="rejected",
            value=rejected_arr,
            payload_count=payload_count,
            source_indices=source_indices,
            ensure_dtype=bool,
        )
        if metadata and metadata.get("original_count") is not None:
            original_count = int(metadata["original_count"])
        elif rejected_arr is not None:
            original_count = int(len(rejected_arr))
        else:
            original_count = int(payload_count)
        obj._metadata = _normalize_contract_metadata(
            metadata=_lightweight_reject_metadata(metadata),
            policy=policy,
            rejected=rejected_arr,
            source_indices=source_indices,
            original_count=original_count,
        )
        # Unpack masks to fields for slicing
        obj.ambiguity_mask = _align_reject_field_to_payload(
            name="ambiguity_mask",
            value=metadata.get("ambiguity_mask") if metadata else None,
            payload_count=payload_count,
            source_indices=source_indices,
            ensure_dtype=bool,
        )
        obj.novelty_mask = _align_reject_field_to_payload(
            name="novelty_mask",
            value=metadata.get("novelty_mask") if metadata else None,
            payload_count=payload_count,
            source_indices=source_indices,
            ensure_dtype=bool,
        )
        obj.prediction_set_size = _align_reject_field_to_payload(
            name="prediction_set_size",
            value=metadata.get("prediction_set_size") if metadata else None,
            payload_count=payload_count,
            source_indices=source_indices,
            ensure_dtype=int,
        )
        obj.prediction_set = _align_reject_field_to_payload(
            name="prediction_set",
            value=metadata.get("prediction_set") if metadata else None,
            payload_count=payload_count,
            source_indices=source_indices,
        )
        obj.epsilon = metadata.get("epsilon") if metadata else None
        for name in ("x_cal", "_X_cal", "scaled_x_cal", "fast_x_cal", "scaled_y_cal"):
            if hasattr(obj, name):
                raise RuntimeError(  # adr002_allow - defensive invariant breach guard
                    f"reject wrapper should not copy calibration array {name}"
                )
        return cast(RejectCalibratedExplanations, obj)

    def __getstate__(self):
        """Return pickle state with lightweight metadata and no runtime handles."""
        state = dict(self.__dict__)
        state["_metadata"] = _lightweight_reject_metadata(state.get("_metadata") or {})
        state["_ce_version"] = "reject_v0.11.1"
        return _prune_unpickleable_state(state)

    def __setstate__(self, state):
        """Restore pickle state in read-only mode until runtime is reconstructed."""
        migrated = dict(state)
        migrated["_metadata"] = _lightweight_reject_metadata(migrated.get("_metadata") or {})
        for runtime_key in (
            "plugin_manager",
            "prediction_orchestrator",
            "reject_orchestrator",
            "_predict_bridge",
            "perf_cache",
            "rng",
        ):
            migrated.setdefault(runtime_key, None)
        migrated["_pickled_readonly"] = True
        self.__dict__.update(migrated)

    def is_readonly_pickled(self) -> bool:
        """Return True when reconstructed from a pickled, read-only state."""
        return bool(getattr(self, "_pickled_readonly", False))

    def __copy__(self):
        """Return a shallow copy preserving references to runtime-coupled fields."""
        new_obj = object.__new__(type(self))
        new_obj.__dict__ = self.__dict__.copy()
        return new_obj

    def reconstruct_runtime(
        self,
        plugin_manager=None,
        prediction_orchestrator=None,
        reject_orchestrator=None,
    ):
        """Restore runtime handles after unpickling and clear read-only mode."""
        if plugin_manager is not None:
            self.plugin_manager = plugin_manager
        if prediction_orchestrator is not None:
            self.prediction_orchestrator = prediction_orchestrator
        if reject_orchestrator is not None:
            self.reject_orchestrator = reject_orchestrator
        self._pickled_readonly = False
        return self

    def __getitem__(self, key: Union[int, slice, List[int], List[bool], np.ndarray]):
        """Return the explanation(s) for the given key, preserving reject metadata."""
        _ensure_runtime_available(self)
        src_len = len(self.rejected) if self.rejected is not None else len(self.explanations)
        self._validate_key_indexing(key, src_len, "rejected")
        # Call base implementation. If it returns a single Explanation, we return it directly.
        # If it returns a new CalibratedExplanations (collection), we assume it's currently
        # a RejectCalibratedExplanations (because base implementation calls copy(self)).
        new_inst = super().__getitem__(key)
        if not isinstance(new_inst, CalibratedExplanations):
            return new_inst
        if isinstance(new_inst, RejectCalibratedExplanations):
            new_inst._slice_reject_fields(key, source=self)
            return new_inst
        rejected_for_wrap = None
        if self.rejected is not None:
            rejected_for_wrap = np.asarray(self.rejected)[key]
        wrapped = RejectCalibratedExplanations.from_collection(
            new_inst,
            {},
            self.policy,
            rejected=rejected_for_wrap,
        )
        wrapped._slice_reject_fields(key, source=self)
        return wrapped


class RejectAlternativeExplanations(AlternativeExplanations, RejectMixin):
    """An AlternativeExplanations collection that carries rejection metadata."""

    @classmethod
    def from_collection(
        cls,
        base: AlternativeExplanations,
        metadata: Dict[str, Any],
        policy: RejectPolicy,
        rejected: Any = None,
    ) -> "RejectAlternativeExplanations":
        """Create a reject-aware alternative collection from an existing collection."""
        obj = object.__new__(cls)
        obj.initialize_reject_metadata()
        _copy_collection_attributes(obj, base)
        obj.policy = policy
        rejected_arr = _copy_array_or_none(rejected, ensure_dtype=bool)
        payload_count = int(len(base.explanations))
        source_indices = _resolve_source_indices_for_wrapper(
            policy=policy,
            metadata=metadata,
            rejected=rejected_arr,
            payload_count=payload_count,
        )
        obj.rejected = _align_reject_field_to_payload(
            name="rejected",
            value=rejected_arr,
            payload_count=payload_count,
            source_indices=source_indices,
            ensure_dtype=bool,
        )
        if metadata and metadata.get("original_count") is not None:
            original_count = int(metadata["original_count"])
        elif rejected_arr is not None:
            original_count = int(len(rejected_arr))
        else:
            original_count = int(payload_count)
        obj._metadata = _normalize_contract_metadata(
            metadata=_lightweight_reject_metadata(metadata),
            policy=policy,
            rejected=rejected_arr,
            source_indices=source_indices,
            original_count=original_count,
        )
        obj.ambiguity_mask = _align_reject_field_to_payload(
            name="ambiguity_mask",
            value=metadata.get("ambiguity_mask") if metadata else None,
            payload_count=payload_count,
            source_indices=source_indices,
            ensure_dtype=bool,
        )
        obj.novelty_mask = _align_reject_field_to_payload(
            name="novelty_mask",
            value=metadata.get("novelty_mask") if metadata else None,
            payload_count=payload_count,
            source_indices=source_indices,
            ensure_dtype=bool,
        )
        obj.prediction_set_size = _align_reject_field_to_payload(
            name="prediction_set_size",
            value=metadata.get("prediction_set_size") if metadata else None,
            payload_count=payload_count,
            source_indices=source_indices,
            ensure_dtype=int,
        )
        obj.prediction_set = _align_reject_field_to_payload(
            name="prediction_set",
            value=metadata.get("prediction_set") if metadata else None,
            payload_count=payload_count,
            source_indices=source_indices,
        )
        obj.epsilon = metadata.get("epsilon") if metadata else None
        return cast(RejectAlternativeExplanations, obj)

    def __getstate__(self):
        """Return pickle state with lightweight metadata and no runtime handles."""
        state = dict(self.__dict__)
        state["_metadata"] = _lightweight_reject_metadata(state.get("_metadata") or {})
        state["_ce_version"] = "reject_v0.11.1"
        return _prune_unpickleable_state(state)

    def __setstate__(self, state):
        """Restore pickle state in read-only mode until runtime is reconstructed."""
        migrated = dict(state)
        migrated["_metadata"] = _lightweight_reject_metadata(migrated.get("_metadata") or {})
        for runtime_key in (
            "plugin_manager",
            "prediction_orchestrator",
            "reject_orchestrator",
            "_predict_bridge",
            "perf_cache",
            "rng",
        ):
            migrated.setdefault(runtime_key, None)
        migrated["_pickled_readonly"] = True
        self.__dict__.update(migrated)

    def is_readonly_pickled(self) -> bool:
        """Return True when reconstructed from a pickled, read-only state."""
        return bool(getattr(self, "_pickled_readonly", False))

    def __copy__(self):
        """Return a shallow copy preserving references to runtime-coupled fields."""
        new_obj = object.__new__(type(self))
        new_obj.__dict__ = self.__dict__.copy()
        return new_obj

    def reconstruct_runtime(
        self,
        plugin_manager=None,
        prediction_orchestrator=None,
        reject_orchestrator=None,
    ):
        """Restore runtime handles after unpickling and clear read-only mode."""
        if plugin_manager is not None:
            self.plugin_manager = plugin_manager
        if prediction_orchestrator is not None:
            self.prediction_orchestrator = prediction_orchestrator
        if reject_orchestrator is not None:
            self.reject_orchestrator = reject_orchestrator
        self._pickled_readonly = False
        return self

    def __getitem__(self, key: Union[int, slice, List[int], List[bool], np.ndarray]):
        """Return sliced alternatives while preserving reject metadata alignment."""
        _ensure_runtime_available(self)
        src_len = len(self.rejected) if self.rejected is not None else len(self.explanations)
        self._validate_key_indexing(key, src_len, "rejected")
        new_inst = super().__getitem__(key)
        if not isinstance(new_inst, AlternativeExplanations):
            return new_inst
        if isinstance(new_inst, RejectAlternativeExplanations):
            new_inst._slice_reject_fields(key, source=self)
            return new_inst
        rejected_for_wrap = None
        if self.rejected is not None:
            rejected_for_wrap = np.asarray(self.rejected)[key]
        wrapped = RejectAlternativeExplanations.from_collection(
            new_inst,
            {},
            self.policy,
            rejected=rejected_for_wrap,
        )
        wrapped._slice_reject_fields(key, source=self)
        return wrapped
