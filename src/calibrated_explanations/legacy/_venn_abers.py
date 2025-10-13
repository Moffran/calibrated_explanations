"""Deprecated shim for :mod:`calibrated_explanations._venn_abers`."""

from __future__ import annotations

from warnings import warn

from ..core.venn_abers import VennAbers

warn(
    "'calibrated_explanations._venn_abers' is deprecated; import from "
    "'calibrated_explanations.core.venn_abers' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["VennAbers"]
