"""Reject policy enum for the `core.reject` package."""

from __future__ import annotations

import warnings
from typing import Any

from ...explanations.reject import RejectPolicy, RejectPolicySpec

__all__ = ["RejectPolicy", "RejectPolicySpec", "is_policy_enabled"]

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
    if isinstance(policy, RejectPolicySpec):
        return policy.policy != RejectPolicy.NONE
    try:
        resolved = RejectPolicy(policy)
        return resolved.value != RejectPolicy.NONE.value
    except Exception:  # adr002_allow
        return False
