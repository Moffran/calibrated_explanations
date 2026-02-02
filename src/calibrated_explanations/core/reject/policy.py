"""Reject policy enum for the `core.reject` package."""

from __future__ import annotations

import warnings
from enum import Enum
from typing import Any


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
            warnings.warn(
                f"RejectPolicy('{value}') is deprecated and will be removed in v1.0.0. "
                f"Use RejectPolicy.{new_name} instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            return new_policy
        return None


# Deprecation map for attribute-style access (e.g., RejectPolicy.PREDICT_AND_FLAG)
_DEPRECATED_ATTRS = {
    "PREDICT_AND_FLAG": ("FLAG", RejectPolicy.FLAG),
    "EXPLAIN_ALL": ("FLAG", RejectPolicy.FLAG),
    "EXPLAIN_REJECTS": ("ONLY_REJECTED", RejectPolicy.ONLY_REJECTED),
    "EXPLAIN_NON_REJECTS": ("ONLY_ACCEPTED", RejectPolicy.ONLY_ACCEPTED),
    "SKIP_ON_REJECT": ("ONLY_ACCEPTED", RejectPolicy.ONLY_ACCEPTED),
}


def __getattr__(name: str) -> Any:
    """Handle deprecated attribute access on the module (e.g., policy.PREDICT_AND_FLAG)."""
    if name in _DEPRECATED_ATTRS:
        new_name, new_policy = _DEPRECATED_ATTRS[name]
        warnings.warn(
            f"RejectPolicy.{name} is deprecated and will be removed in v1.0.0. "
            f"Use RejectPolicy.{new_name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return new_policy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def is_policy_enabled(policy: Any) -> bool:
    """Return True if the provided policy requires reject orchestration."""
    try:
        resolved = RejectPolicy(policy)
        return resolved.value != RejectPolicy.NONE.value
    except Exception:  # adr002_allow
        return False
