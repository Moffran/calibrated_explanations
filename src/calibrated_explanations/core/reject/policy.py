"""Reject policy exports for the ``core.reject`` package.

This module re-exports :class:`RejectPolicy` and :class:`RejectPolicySpec`.
For stable serialization of policy specs, prefer ``RejectPolicySpec.to_dict()``
and ``RejectPolicySpec.from_dict()`` over parsing ``RejectPolicySpec.value``.
Serialization supports the canonical string NCF values.
"""

from __future__ import annotations

from typing import Any

from ...explanations.reject import RejectPolicy, RejectPolicySpec

__all__ = ["RejectPolicy", "RejectPolicySpec", "is_policy_enabled"]


def is_policy_enabled(policy: Any) -> bool:
    """Return True if the provided policy requires reject orchestration."""
    if isinstance(policy, RejectPolicySpec):
        return policy.policy != RejectPolicy.NONE
    try:
        resolved = RejectPolicy(policy)
        return resolved.value != RejectPolicy.NONE.value
    except Exception:  # adr002_allow
        return False
