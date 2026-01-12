"""Types for reject-aware explanation envelopes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..core.reject.policy import RejectPolicy


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
