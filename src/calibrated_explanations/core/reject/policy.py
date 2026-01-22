"""Reject policy enum placed under the `core.reject` package to avoid module/package name collisions."""

from __future__ import annotations

from enum import Enum
from typing import Any


class RejectPolicy(Enum):
    """Describe how rejection should affect prediction/explanation invocation.

    - NONE: preserve legacy behaviour (no reject orchestration).
    - PREDICT_AND_FLAG: always predict and attach a rejected flag in the envelope.
    - EXPLAIN_ALL: predict and explain all inputs, tagging rejected status.
    - EXPLAIN_REJECTS: predict all, explain only rejected instances.
    - EXPLAIN_NON_REJECTS: predict all, explain only non-rejected instances.
    - SKIP_ON_REJECT: short-circuit prediction/explanation when rejected.
    """

    NONE = "none"
    PREDICT_AND_FLAG = "predict_and_flag"
    EXPLAIN_ALL = "explain_all"
    EXPLAIN_REJECTS = "explain_rejects"
    EXPLAIN_NON_REJECTS = "explain_non_rejects"
    SKIP_ON_REJECT = "skip_on_reject"


def is_policy_enabled(policy: Any) -> bool:
    """Return True if the provided policy requires reject orchestration."""
    try:
        return RejectPolicy(policy) is not RejectPolicy.NONE
    except Exception:  # adr002_allow
        return False
