"""Types for reject-aware explanation envelopes."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np

from .explanations import CalibratedExplanations


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


_VALID_NCF = frozenset({"hinge", "ensured", "entropy", "margin"})


@dataclass(eq=False)
class RejectPolicySpec:
    """Bundle a RejectPolicy with a non-conformity function (NCF) configuration.

    Parameters
    ----------
    policy : RejectPolicy
        The reject policy (FLAG, ONLY_REJECTED, ONLY_ACCEPTED).
    ncf : str, default 'hinge'
        Non-conformity function: 'hinge' (default/existing), 'ensured'
        (Venn-Abers interval width), 'entropy' (Shannon entropy), or
        'margin' (top-two probability gap).
    w : float, default 0.5
        Hinge weight in [0, 1]. ``w=1.0`` reduces to pure hinge (identical
        to the default behaviour). ``w=0.5`` gives equal weight to the
        uncertainty measure and the hinge term.

    Class Methods
    -------------
    flag(ncf, w) / only_rejected(ncf, w) / only_accepted(ncf, w)
        Convenience constructors for common policy values.

    Examples
    --------
    >>> spec = RejectPolicySpec(RejectPolicy.FLAG, ncf='ensured', w=0.5)
    >>> spec = RejectPolicySpec.flag(ncf='ensured', w=0.5)
    """

    policy: RejectPolicy
    ncf: str = "hinge"
    w: float = 0.5  # pylint: disable=invalid-name

    def __post_init__(self) -> None:
        if self.ncf not in _VALID_NCF:
            raise ValueError(
                f"ncf must be one of {sorted(_VALID_NCF)!r}; got {self.ncf!r}"
            )
        if not 0.0 <= self.w <= 1.0:
            raise ValueError(f"w must be in [0, 1]; got {self.w}")

    @classmethod
    def flag(cls, ncf: str = "hinge", w: float = 0.5) -> "RejectPolicySpec":  # pylint: disable=invalid-name
        """Return a FLAG RejectPolicySpec with the given NCF configuration."""
        return cls(RejectPolicy.FLAG, ncf=ncf, w=w)

    @classmethod
    def only_rejected(cls, ncf: str = "hinge", w: float = 0.5) -> "RejectPolicySpec":  # pylint: disable=invalid-name
        """Return an ONLY_REJECTED RejectPolicySpec with the given NCF configuration."""
        return cls(RejectPolicy.ONLY_REJECTED, ncf=ncf, w=w)

    @classmethod
    def only_accepted(cls, ncf: str = "hinge", w: float = 0.5) -> "RejectPolicySpec":  # pylint: disable=invalid-name
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
            return self.policy == other.policy and self.ncf == other.ncf and self.w == other.w
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.policy, self.ncf, self.w))

    @property
    def value(self) -> str:
        """String value compatible with RejectPolicy.value for use in mixed policy lists."""
        return f"{self.policy.value}[ncf={self.ncf},w={self.w}]"


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
    prediction_set: Any | None = None
    # Rendered strings (optional) - templates preferred instead of hardcoding
    beginner_text: str | None = None
    intermediate_text: str | None = None
    advanced_text: str | None = None


class RejectMixin:
    """Mixin to hold global reject metadata on a CalibratedExplanations collection."""

    policy: RejectPolicy = RejectPolicy.NONE
    rejected: np.ndarray | None = None
    _metadata: Dict[str, Any] | None = None
    ambiguity_mask: np.ndarray | None = None
    novelty_mask: np.ndarray | None = None
    prediction_set_size: np.ndarray | None = None
    epsilon: float | None = None

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return a dynamic view of the metadata to mimic RejectResult.metadata."""
        # Start with stashed metadata (rates etc)
        meta = dict(self._metadata) if self._metadata else {}
        # Overlay the current (possibly sliced) masks
        if self.ambiguity_mask is not None:
            meta["ambiguity_mask"] = self.ambiguity_mask
        if self.novelty_mask is not None:
            meta["novelty_mask"] = self.novelty_mask
        if self.prediction_set_size is not None:
            meta["prediction_set_size"] = self.prediction_set_size
        return meta

    def _slice_reject_fields(self, key, source: RejectMixin):
        """Slice the reject metadata arrays in sync with the explanations."""
        # Fields to slice using the same key as the explanations
        fields = ["rejected", "ambiguity_mask", "novelty_mask", "prediction_set_size"]

        for field in fields:
            val = getattr(source, field, None)
            if val is not None and isinstance(val, (np.ndarray, list)):
                try:
                    if isinstance(key, (slice, int)):
                        new_val = val[key]
                    elif isinstance(key, (list, np.ndarray)):
                        # Handle boolean or integer indexing
                        if isinstance(val, np.ndarray):
                            new_val = val[key]
                        else:
                            # list fallback
                            new_val = (
                                [val[i] for i in key] if hasattr(key, "__iter__") else val[key]
                            )
                    else:
                        new_val = val

                    # Note: we don't have to handle single-item return here because
                    # __getitem__ returns the wrapper only if it's a collection.
                    setattr(self, field, new_val)
                except (IndexError, TypeError):
                    # Fallback or mismatched dimensions
                    setattr(self, field, None)

        # Constant fields just copy
        self.policy = source.policy
        self.epsilon = source.epsilon
        # Metadata (rates) are invariant to slicing for now? Or should they be recomputed?
        # For simple parity, we copy the original metadata dict reference or values.
        # Since rates won't match the slice, copying static metadata is safer than nothing.
        self._metadata = copy(source._metadata)


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
        """Upgrade a standard collection to a rejected collection."""
        obj = copy(base)
        obj.__class__ = cls
        obj.policy = policy
        obj.rejected = rejected
        # Stash the raw metadata
        obj._metadata = metadata
        # Unpack masks to fields for slicing
        obj.ambiguity_mask = metadata.get("ambiguity_mask")
        obj.novelty_mask = metadata.get("novelty_mask")
        obj.prediction_set_size = metadata.get("prediction_set_size")
        obj.epsilon = metadata.get("epsilon")
        return cast(RejectCalibratedExplanations, obj)

    def __getitem__(self, key: Union[int, slice, List[int], List[bool], np.ndarray]):
        """Return the explanation(s) for the given key, preserving reject metadata."""
        # Call base implementation. If it returns a single Explanation, we return it directly.
        # If it returns a new CalibratedExplanations (collection), we assume it's currently
        # a RejectCalibratedExplanations (because base implementation calls copy(self)).
        new_inst = super().__getitem__(key)

        if isinstance(new_inst, RejectCalibratedExplanations):
            # It's a new collection (sliced). We need to slice our metadata.
            new_inst._slice_reject_fields(key, source=self)

        return new_inst
