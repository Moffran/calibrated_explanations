"""Deterministic categorical encoder with JSON-safe mapping snapshots."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..utils.exceptions import NotFittedError, ValidationError


class BuiltinEncoder:
    """A tiny deterministic column-wise categorical encoder.

    - Learns ordered categories per column and maps to integers starting at 0.
    - Exposes JSON-safe mapping via ``get_mapping_snapshot`` and
      accepts snapshots via ``set_mapping``.
    - Deterministic: sorts observed categories to ensure stable ordering.
    - Supports unseen category policy when transforming.
    """

    def __init__(self, unseen_policy: str = "error") -> None:
        self.unseen_policy = unseen_policy
        self.mapping_: Dict[str, List[Any]] | None = None

    def fit(self, x: Any) -> "BuiltinEncoder":
        """Learn per-column category mappings from input data."""
        arr = self._as_2d(x)
        mapping: Dict[str, List[Any]] = {}
        for i, col in enumerate(arr.T):
            # numpy will produce object dtype for mixed types; use Python set
            cats = sorted({self._safe_val(v) for v in col})
            mapping[f"col_{i}"] = cats
        self.mapping_ = mapping
        return self

    def fit_transform(self, x: Any) -> Any:
        """Fit the encoder and return transformed values."""
        self.fit(x)
        return self.transform(x)

    def transform(self, x: Any) -> Any:
        """Map input categories to learned integer indices."""
        arr = self._as_2d(x)
        if self.mapping_ is None:
            raise NotFittedError("Encoder not fitted")
        out = np.zeros_like(arr, dtype=float)
        for i, col in enumerate(arr.T):
            cats = self.mapping_.get(f"col_{i}", [])
            for j, v in enumerate(col):
                v_safe = self._safe_val(v)
                if v_safe in cats:
                    out[j, i] = float(cats.index(v_safe))
                else:
                    if self.unseen_policy == "ignore":
                        out[j, i] = -1.0
                    else:
                        raise ValidationError(f"Unseen category {v_safe} in column {i}")
        return out

    def get_mapping_snapshot(self) -> Dict[str, Any] | None:
        """Return a shallow copy of the learned mapping."""
        return dict(self.mapping_) if self.mapping_ is not None else None

    def set_mapping(self, mapping: Dict[str, Any]) -> None:
        """Load a mapping produced by ``get_mapping_snapshot``."""
        # Accepts plain dicts produced by get_mapping_snapshot
        self.mapping_ = (
            {str(k): list(v) for k, v in mapping.items()} if mapping is not None else None
        )

    @staticmethod
    def _as_2d(x: Any) -> np.ndarray:
        arr = np.asarray(x)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    @staticmethod
    def _safe_val(v: Any) -> Any:
        # Ensure JSON-safe primitive ordering for snapshot determinism
        if v is None:
            return "__none__"
        if isinstance(v, (int, float, str, bool)):
            return v
        try:
            return str(v)
        except (TypeError, ValueError):
            return repr(v)
