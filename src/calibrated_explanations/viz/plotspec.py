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
    # When True, render two bands (negative/positive) stacked as in legacy probabilistic plot
    dual: bool = True
    # Optional explicit class labels for dual probabilistic header (neg_label, pos_label)
    neg_label: str | None = None
    pos_label: str | None = None
    # Optional fully resolved captions for the dual header bands. When provided these
    # strings are rendered verbatim instead of applying ``P(y=...)`` formatting.
    neg_caption: str | None = None
    pos_caption: str | None = None
    # Flag indicating whether uncertainty/interval shading should be rendered. Legacy
    # probabilistic plots always drew the translucent bands when an interval was
    # requested; setting this flag lets the adapter mirror that behaviour precisely.
    show_intervals: bool = False
    # Optional override for the grey uncertainty overlay color and alpha (per-PlotSpec)
    uncertainty_color: str | None = None
    uncertainty_alpha: float | None = None


@dataclass
class BarItem:
    """One horizontal bar for a feature contribution, with optional interval."""

    label: str
    value: float
    interval_low: float | None = None
    interval_high: float | None = None
    color_role: str | None = None  # e.g., "positive" | "negative" | "regression"
    instance_value: Any | None = None
    # When True, follow legacy behavior: suppress drawing solid when interval crosses zero
    solid_on_interval_crosses_zero: bool = True


@dataclass
class BarHPanelSpec:
    """Panel with horizontal bars representing feature contributions."""

    bars: Sequence[BarItem]
    xlabel: str | None = None
    ylabel: str | None = None
    # When True, follow legacy behavior: suppress drawing solids when intervals cross zero
    solid_on_interval_crosses_zero: bool = True
    # Legacy alternative-plot specific metadata
    is_alternative: bool = False
    base_segments: Sequence["IntervalSegment"] | None = None
    base_lines: Sequence[tuple[float, str, float | None]] | None = None
    pivot: float | None = None
    xlim: Tuple[float, float] | None = None
    xticks: Sequence[float] | None = None
    bar_span: float = 0.2
    # Legacy probabilistic plots only shaded the prediction interval backdrop when
    # uncertainty information was provided. This flag lets the adapter decide whether
    # to draw that grey band.
    show_base_interval: bool = False


@dataclass
class SaveBehavior:
    """Hints describing how and where a plot should be saved or exported.

    - path: Optional filesystem path to write files to. When None, a headless
      export may be requested and adapters can return bytes instead of writing.
    - title: Suggested filename or title to use for saved artifacts.
    - default_exts: Preferred output extensions (e.g., ["png", "svg"]).
    """

    path: str | None = None
    title: str | None = None
    default_exts: Sequence[str] | None = None


@dataclass
class PlotSpec:
    """A simple multi-panel plot specification."""

    title: str | None = None
    figure_size: Tuple[float, float] | None = None
    header: IntervalHeaderSpec | None = None
    body: BarHPanelSpec | None = None

    # Metadata fields added in v0.10.1
    kind: str | None = None
    mode: str | None = None
    feature_order: Sequence[str] | None = None
    plotspec_version: str = "1.0.0"
    save_behavior: SaveBehavior | None = None

    # Optional provenance fields for reproducibility/audit
    data_slice_id: str | None = None
    rendering_seed: int | None = None


@dataclass
class TriangularSpec:
    """Specification for triangular plot data (quiver/scatter)."""

    proba: Any | None = None
    uncertainty: Any | None = None
    rule_proba: Sequence[float] | None = None
    rule_uncertainty: Sequence[float] | None = None
    num_to_show: int = 50
    is_probabilistic: bool = True


@dataclass
class GlobalSpec:
    """Specification for global plot data (scatter of uncertainty vs proba/predict)."""

    proba: Sequence[float] | None = None
    predict: Sequence[float] | None = None
    low: Sequence[float] | None = None
    high: Sequence[float] | None = None
    uncertainty: Sequence[float] | None = None
    y_test: Sequence[Any] | None = None


@dataclass
class TriangularPlotSpec:
    """PlotSpec for triangular plots (non-panel)."""

    title: str | None = None
    figure_size: Tuple[float, float] | None = None
    triangular: TriangularSpec | None = None

    # Metadata fields
    kind: str | None = None
    mode: str | None = None
    plotspec_version: str = "1.0.0"
    save_behavior: SaveBehavior | None = None

    # Optional provenance fields
    data_slice_id: str | None = None
    rendering_seed: int | None = None


@dataclass
class GlobalPlotSpec:
    """PlotSpec for global plots (non-panel)."""

    title: str | None = None
    figure_size: Tuple[float, float] | None = None
    global_entries: GlobalSpec | None = None

    # Metadata fields
    kind: str | None = None
    mode: str | None = None
    plotspec_version: str = "1.0.0"
    save_behavior: SaveBehavior | None = None

    # Optional provenance fields
    data_slice_id: str | None = None
    rendering_seed: int | None = None


@dataclass(frozen=True)
class IntervalSegment:
    """Segment describing an interval fill for alternative probability plots."""

    low: float
    high: float
    color: str
    alpha: float | None = None


__all__ = [
    "PlotSpec",
    "IntervalHeaderSpec",
    "BarHPanelSpec",
    "BarItem",
]
