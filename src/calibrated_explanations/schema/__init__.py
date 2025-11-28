"""Explanation schema package.

This package provides schema validation and loading helpers for ADR-005
explanation envelopes.

Part of ADR-001: Core Decomposition Boundaries (Stage 1c).
"""

from .validation import validate_payload

__all__ = ["validate_payload"]
