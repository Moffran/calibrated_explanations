"""PlotSpec MVP (ADR-007): Backend-agnostic plotting specification.

Minimal structures to represent 1–2 existing plots (regression/probabilistic
feature bars with an optional header interval). Default plotting behavior
remains unchanged; use an adapter to render a PlotSpec via matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple


@dataclass
class IntervalHeaderSpec:
    """Header panel expressing a scalar prediction with an interval."""

    pred: float
    low: float
    high: float
    xlim: Tuple[float, float] | None = None
    xlabel: str | None = None
    ylabel: str | None = None


@dataclass
class BarItem:
    """One horizontal bar for a feature contribution, with optional interval."""

    label: str
    value: float
    interval_low: float | None = None
    interval_high: float | None = None
    color_role: str | None = None  # e.g., "positive" | "negative" | "regression"
    instance_value: Any | None = None


@dataclass
class BarHPanelSpec:
    """Panel with horizontal bars representing feature contributions."""

    bars: Sequence[BarItem]
    xlabel: str | None = None
    ylabel: str | None = None


@dataclass
class PlotSpec:
    """A simple multi-panel plot specification."""

    title: str | None = None
    figure_size: Tuple[float, float] | None = None
    header: IntervalHeaderSpec | None = None
    body: BarHPanelSpec | None = None


__all__ = [
    "PlotSpec",
    "IntervalHeaderSpec",
    "BarHPanelSpec",
    "BarItem",
]
