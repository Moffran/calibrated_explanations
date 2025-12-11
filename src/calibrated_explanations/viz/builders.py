"""Builders to convert existing plotting inputs to PlotSpec structures.

This keeps default plotting intact while offering an internal pathway to
render via the PlotSpec + matplotlib adapter for selected plots.
"""

from __future__ import annotations

import math
import sys
import contextlib
from typing import Any, Sequence

import numpy as np

from .plotspec import (
    BarHPanelSpec,
    BarItem,
    IntervalHeaderSpec,
    IntervalSegment,
    PlotSpec,
)

_PROBABILITY_TOL = 1e-9


def is_valid_probability_values(*values: float) -> bool:
    """Check if all provided values are valid probabilities in [0, 1] (with tolerance).

    This function validates probability values used in visualization, checking that:
    - All values can be converted to floats
    - All values are finite
    - All values lie within [0 - tolerance, 1 + tolerance]

    Used to auto-detect when predictions should be rendered as probabilities
    vs generic scalar values (e.g., in build_probabilistic_bars_spec).

    Parameters
    ----------
    *values : float
        One or more values to validate as probabilities.

    Returns
    -------
    bool
        True if all values are valid probabilities, False otherwise.
    """
    finite: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except:
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            return False
        if not math.isfinite(numeric):
            return False
        finite.append(numeric)
    if not finite:
        return False
    tol = _PROBABILITY_TOL
    return all(-tol <= v <= 1.0 + tol for v in finite)


# Backward compatibility alias for private naming
_looks_like_probability_values = is_valid_probability_values


def _ensure_indexable_length(name: str, seq: Sequence[Any] | None, *, max_index: int) -> None:
    """Ensure ``seq`` can satisfy ``max_index`` when provided.

    The ADR contract requires that feature-oriented arrays (feature weights,
    column names, rule labels, instance vectors) all cover the same indices
    requested via ``features_to_plot``. Raise ``ValueError`` with a descriptive
    message when a sequence is too short so tests can detect drift early.
    """
    if seq is None or max_index < 0:
        return
    with contextlib.suppress(TypeError):
        length = len(seq)
        if length <= max_index:
            from ..core.exceptions import ValidationError

            raise ValidationError(
                f"{name} length {length} does not cover feature index {max_index}",
                details={
                    "param": name,
                    "length": length,
                    "required_to_cover": max_index,
                    "shortfall": max_index - length + 1,
                },
            )


def _normalize_interval_bounds(
    low: float,
    high: float,
    *,
    y_minmax: tuple[float, float] | None,
) -> tuple[float, float, tuple[float, float] | None]:
    """Clamp non-finite interval bounds to ``y_minmax`` when available."""
    if y_minmax is None:
        return low, high, None

    floor = float(y_minmax[0])
    ceil = float(y_minmax[1])
    if not math.isfinite(low):
        low = floor
    if not math.isfinite(high):
        high = ceil
    xlim = (float(min(low, floor)), float(max(high, ceil)))
    return low, high, xlim


def _legacy_color_brew(n: int) -> list[tuple[int, int, int]]:
    """Reproduce the legacy colour palette for probability fills."""
    color_list: list[tuple[int, int, int]] = []
    s, v = 0.75, 0.9
    c = s * v
    m = v - c
    for h in np.arange(5, 385, 490.0 / n).astype(int):
        h_bar = h / 60.0
        x = c * (1 - abs((h_bar % 2) - 1))
        rgb = [
            (c, x, 0),
            (x, c, 0),
            (0, c, x),
            (0, x, c),
            (x, 0, c),
            (c, 0, x),
            (c, x, 0),
        ]
        r, g, b = rgb[int(h_bar)]
        rgb = (
            int(255 * (r + m)),
            int(255 * (g + m)),
            int(255 * (b + m)),
        )
        color_list.append(rgb)
    color_list.reverse()
    return color_list


def _legacy_get_fill_color(probability: float, reduction: float = 1.0) -> str:
    """Mirror legacy ``__get_fill_color`` for probability intensities."""
    colors = _legacy_color_brew(2)
    winner_class = int(probability >= 0.5)
    color = colors[winner_class]
    alpha = probability if winner_class == 1 else 1 - probability
    alpha = ((alpha - 0.5) / (1 - 0.5)) * (1 - 0.25) + 0.25
    if reduction != 1:
        alpha = reduction
    alpha = float(alpha)
    blended = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
    hex_color = "#%02x%02x%02x" % tuple(blended)
    if (
        reduction == 1.0
        and math.isfinite(probability)
        and math.isclose(probability, 1.0, rel_tol=1e-9, abs_tol=1e-12)
    ):
        return "#ff0000"
    return hex_color


REGRESSION_BAR_COLOR = _legacy_get_fill_color(1.0, 1.0)
REGRESSION_BASE_COLOR = _legacy_get_fill_color(1.0, 0.15)


def _build_probability_segments(
    *,
    low: float,
    high: float,
    center: float,
    reduction: float,
    pivot: float | None,
) -> tuple[IntervalSegment, ...]:
    """Construct coloured segments for a probability interval."""
    lo = float(low)
    hi = float(high)
    if lo > hi:
        lo, hi = hi, lo
    if pivot is not None and lo < pivot < hi:
        left_color = _legacy_get_fill_color(lo, reduction)
        right_color = _legacy_get_fill_color(hi, reduction)
        return (
            IntervalSegment(low=lo, high=pivot, color=left_color),
            IntervalSegment(low=pivot, high=hi, color=right_color),
        )
    return (IntervalSegment(low=lo, high=hi, color=_legacy_get_fill_color(center, reduction)),)


def build_regression_bars_spec(
    *,
    title: str | None,
    predict: dict[str, float],
    feature_weights: dict[str, Sequence[float]] | Sequence[float],
    features_to_plot: Sequence[int],
    column_names: Sequence[str] | None,
    rule_labels: Sequence[str] | None = None,
    instance: Sequence[Any] | None,
    y_minmax: tuple[float, float] | None,
    interval: bool,
    confidence: float | None = None,
    uncertainty_color: str | None = None,
    uncertainty_alpha: float | None = None,
    sort_by: str | None = None,
    ascending: bool = False,
    legacy_solid_behavior: bool = True,
) -> PlotSpec:
    """Create a PlotSpec for the regression bar plot variant.

    Mirrors a subset of _plot_regression behavior sufficient for MVP.

    Parameters
    ----------
    sort_by: Optional[str]
        How to sort features. One of:
                - None or "none": keep provided order
                - "value": by signed value
                - "abs": by absolute value (|value - 0|)
                - "width" or "interval": by interval width (|high - low|) when available, else 0.
                    Width ties are broken by absolute value to keep ordering deterministic.
                - "label": lexicographically by label
    ascending: bool
        Sort ascending (default False = descending)
    """
    max_index = max(features_to_plot) if features_to_plot else -1
    if isinstance(feature_weights, dict):
        for key in ("predict", "low", "high"):
            _ensure_indexable_length(
                f"feature_weights['{key}']",
                feature_weights.get(key),
                max_index=max_index,
            )
    else:
        _ensure_indexable_length("feature_weights", feature_weights, max_index=max_index)
    _ensure_indexable_length("column_names", column_names, max_index=max_index)
    _ensure_indexable_length("rule_labels", rule_labels, max_index=max_index)
    _ensure_indexable_length("instance", instance, max_index=max_index)

    # Header (interval around prediction)
    pred = float(predict["predict"]) if "predict" in predict else 0.0
    low = float(predict.get("low", pred))
    high = float(predict.get("high", pred))
    low, high, xlim = _normalize_interval_bounds(low, high, y_minmax=y_minmax)
    has_interval = not math.isclose(low, high, rel_tol=1e-12, abs_tol=1e-12)
    if confidence is not None:
        try:
            confidence_label = f"Prediction interval with {confidence}% confidence"
        except:
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            confidence_label = "Prediction interval"
    else:
        confidence_label = (
            "Prediction interval with unknown confidence"
            if y_minmax is None
            else "Prediction interval"
        )

    header = IntervalHeaderSpec(
        pred=pred,
        low=low,
        high=high,
        xlim=xlim,
        xlabel=confidence_label,
        ylabel="Median prediction",
        dual=False,
        show_intervals=has_interval,
        uncertainty_color=uncertainty_color,
        uncertainty_alpha=uncertainty_alpha,
    )

    # Body (bars)
    bars: list[BarItem] = []
    if isinstance(feature_weights, dict) and interval:
        pred_w = feature_weights["predict"]
        wl = feature_weights["low"]
        wh = feature_weights["high"]
        for j in features_to_plot:
            val = float(pred_w[j])
            bars.append(
                BarItem(
                    label=(
                        rule_labels[j]
                        if rule_labels is not None
                        else (column_names[j] if column_names is not None else str(j))
                    ),
                    value=val,
                    interval_low=float(wl[j]),
                    interval_high=float(wh[j]),
                    instance_value=(instance[j] if instance is not None else None),
                    color_role="regression",
                    solid_on_interval_crosses_zero=legacy_solid_behavior,
                )
            )
    else:
        # Simple numeric sequence
        arr = feature_weights  # type: ignore[assignment]
        for j in features_to_plot:
            val = float(arr[j])  # type: ignore[index]
            bars.append(
                BarItem(
                    label=(
                        rule_labels[j]
                        if rule_labels is not None
                        else (column_names[j] if column_names is not None else str(j))
                    ),
                    value=val,
                    instance_value=(instance[j] if instance is not None else None),
                    color_role="regression",
                    solid_on_interval_crosses_zero=legacy_solid_behavior,
                )
            )

    # Optional sorting of bars
    def _key(item: BarItem):  # return type varies by mode (float, str, or tuple for width)
        s = (sort_by or "none").lower()
        if s in ("none", ""):
            # identity sort key: preserve order
            return 0.0
        if s == "value":
            return item.value
        if s == "abs":
            return abs(item.value)
        if s in ("interval", "width"):
            if item.interval_low is not None and item.interval_high is not None:
                w = abs(float(item.interval_high) - float(item.interval_low))
                # Tie-break on |value| to make ordering deterministic when widths are equal
                return (w, abs(item.value))
            return 0.0
        if s == "label":
            return item.label
        # Fallback to no sort if unknown
        return 0.0

    if sort_by and sort_by.lower() not in ("none", ""):
        # Stable sort by key; reverse for descending when ascending=False
        bars = sorted(bars, key=_key, reverse=not ascending)

    body = BarHPanelSpec(
        bars=bars,
        xlabel="Feature weights",
        ylabel="Rules",
        solid_on_interval_crosses_zero=legacy_solid_behavior,
    )
    return PlotSpec(title=title, header=header, body=body)


def build_factual_probabilistic_spec(**kwargs) -> PlotSpec:
    """Create a wrapper for the factual probabilistic plot kind (ADR-016).

    Currently delegates to `build_probabilistic_bars_spec`. This wrapper
    provides a stable name for future refactors and makes plot kinds explicit.
    """
    return build_probabilistic_bars_spec(**kwargs)


def build_factual_regression_spec(**kwargs) -> PlotSpec:
    """Create a wrapper for the factual regression plot kind (ADR-016)."""
    return build_regression_bars_spec(**kwargs)


def build_alternative_probabilistic_spec(
    *,
    title: str | None,
    predict: dict[str, float],
    feature_weights: dict[str, Sequence[float]] | Sequence[float],
    features_to_plot: Sequence[int],
    column_names: Sequence[str] | None,
    rule_labels: Sequence[str] | None = None,
    instance: Sequence[Any] | None,
    y_minmax: tuple[float, float] | None,
    interval: bool,
    sort_by: str | None = None,
    ascending: bool = False,
    legacy_solid_behavior: bool = True,
    neg_label: str | None = None,
    pos_label: str | None = None,
    uncertainty_color: str | None = None,
    uncertainty_alpha: float | None = None,
    xlabel: str | None = None,
    xlim: tuple[float, float] | None = None,
    xticks: Sequence[float] | None = None,
) -> PlotSpec:
    """Build an alternative probabilistic PlotSpec mirroring legacy visuals."""
    max_index = max(features_to_plot) if features_to_plot else -1
    if isinstance(feature_weights, dict):
        _ensure_indexable_length(
            "feature_weights['predict']", feature_weights.get("predict"), max_index=max_index
        )
        if interval:
            _ensure_indexable_length(
                "feature_weights['low']", feature_weights.get("low"), max_index=max_index
            )
            _ensure_indexable_length(
                "feature_weights['high']", feature_weights.get("high"), max_index=max_index
            )
    else:
        _ensure_indexable_length("feature_weights", feature_weights, max_index=max_index)
    _ensure_indexable_length("column_names", column_names, max_index=max_index)
    _ensure_indexable_length("rule_labels", rule_labels, max_index=max_index)
    _ensure_indexable_length("instance", instance, max_index=max_index)

    base_pred = float(predict.get("predict", 0.0))
    base_low = float(predict.get("low", base_pred))
    base_high = float(predict.get("high", base_pred))

    if y_minmax is not None:
        floor = float(y_minmax[0])
        ceil = float(y_minmax[1])
        base_low = float(min(max(base_low, floor), ceil))
        base_high = float(min(max(base_high, floor), ceil))
        default_xlim = (floor, ceil)
    else:
        default_xlim = (0.0, 1.0)

    if xlim is None:
        xlim = default_xlim
    xlim = (float(xlim[0]), float(xlim[1])) if xlim is not None else default_xlim

    if xlabel is None:
        body_xlabel = (
            f"Probability for class '{pos_label}'" if pos_label is not None else "Probability"
        )
    else:
        body_xlabel = xlabel

    if xticks is None and xlim == (0.0, 1.0):
        xticks = [float(x) for x in np.linspace(0.0, 1.0, 11)]

    pivot = 0.5 if xlim[0] <= 0.5 <= xlim[1] else None
    base_segments = _build_probability_segments(
        low=base_low,
        high=base_high,
        center=base_pred,
        reduction=0.15,
        pivot=pivot,
    )

    bars: list[BarItem] = []

    def _label_for(index: int) -> str:
        if rule_labels is not None:
            return str(rule_labels[index])
        if column_names is not None:
            return str(column_names[index])
        return str(index)

    if isinstance(feature_weights, dict):
        preds = feature_weights.get("predict", ())
        lows = feature_weights.get("low", ()) if interval else None
        highs = feature_weights.get("high", ()) if interval else None
        for j in features_to_plot:
            val = float(preds[j])
            color_role = (
                "positive"
                if (pivot is not None and val >= pivot)
                else ("positive" if pivot is None and val >= base_pred else "negative")
            )
            if interval and lows is not None and highs is not None:
                lo = float(lows[j])
                hi = float(highs[j])
            else:
                lo = hi = val
            segments = _build_probability_segments(
                low=lo,
                high=hi,
                center=val,
                reduction=0.99,
                pivot=pivot,
            )
            item = BarItem(
                label=_label_for(j),
                value=val,
                interval_low=lo,
                interval_high=hi,
                instance_value=(instance[j] if instance is not None else None),
                solid_on_interval_crosses_zero=legacy_solid_behavior,
                color_role=color_role,
            )
            item.segments = segments
            bars.append(item)
    else:
        arr = feature_weights
        for j in features_to_plot:
            val = float(arr[j])  # type: ignore[index]
            color_role = (
                "positive"
                if (pivot is not None and val >= pivot)
                else ("positive" if pivot is None and val >= base_pred else "negative")
            )
            segments = _build_probability_segments(
                low=val,
                high=val,
                center=val,
                reduction=0.99,
                pivot=pivot,
            )
            item = BarItem(
                label=_label_for(j),
                value=val,
                interval_low=val,
                interval_high=val,
                instance_value=(instance[j] if instance is not None else None),
                solid_on_interval_crosses_zero=legacy_solid_behavior,
                color_role=color_role,
            )
            item.segments = segments
            bars.append(item)

    def _key(item: BarItem):
        s = (sort_by or "none").lower()
        if s in ("none", ""):
            return 0.0
        if s == "value":
            return item.value
        if s == "abs":
            return abs(item.value)
        if s in ("interval", "width"):
            if item.interval_low is not None and item.interval_high is not None:
                w = abs(float(item.interval_high) - float(item.interval_low))
                return (w, abs(item.value))
            return 0.0
        if s == "label":
            return item.label
        return 0.0

    if sort_by and sort_by.lower() not in ("none", ""):
        bars = sorted(bars, key=_key, reverse=not ascending)

    xtick_values = tuple(float(x) for x in xticks) if xticks is not None else None

    body = BarHPanelSpec(
        bars=bars,
        xlabel=body_xlabel,
        ylabel="Alternative rules",
        solid_on_interval_crosses_zero=legacy_solid_behavior,
        is_alternative=True,
        base_segments=base_segments,
        base_lines=None,
        pivot=pivot,
        xlim=xlim,
        xticks=xtick_values,
        bar_span=0.2,
    )

    height = max(len(bars), 1) * 0.5
    return PlotSpec(title=title, figure_size=(10.0, height), header=None, body=body)


def build_alternative_regression_spec(
    *,
    title: str | None,
    predict: dict[str, float],
    feature_weights: dict[str, Sequence[float]] | Sequence[float],
    features_to_plot: Sequence[int],
    column_names: Sequence[str] | None,
    rule_labels: Sequence[str] | None = None,
    instance: Sequence[Any] | None,
    y_minmax: tuple[float, float] | None,
    interval: bool,
    sort_by: str | None = None,
    ascending: bool = False,
    legacy_solid_behavior: bool = True,
    neg_label: str | None = None,
    pos_label: str | None = None,
    uncertainty_color: str | None = None,
    uncertainty_alpha: float | None = None,
    xlabel: str | None = None,
    xlim: tuple[float, float] | None = None,
    xticks: Sequence[float] | None = None,
) -> PlotSpec:
    """Build an alternative regression PlotSpec mirroring legacy visuals."""
    max_index = max(features_to_plot) if features_to_plot else -1
    if isinstance(feature_weights, dict):
        _ensure_indexable_length(
            "feature_weights['predict']", feature_weights.get("predict"), max_index=max_index
        )
        if interval:
            _ensure_indexable_length(
                "feature_weights['low']", feature_weights.get("low"), max_index=max_index
            )
            _ensure_indexable_length(
                "feature_weights['high']", feature_weights.get("high"), max_index=max_index
            )
    else:
        _ensure_indexable_length("feature_weights", feature_weights, max_index=max_index)
    _ensure_indexable_length("column_names", column_names, max_index=max_index)
    _ensure_indexable_length("rule_labels", rule_labels, max_index=max_index)
    _ensure_indexable_length("instance", instance, max_index=max_index)

    base_pred = float(predict.get("predict", 0.0))
    base_low = float(predict.get("low", base_pred))
    base_high = float(predict.get("high", base_pred))

    if y_minmax is not None:
        floor = float(y_minmax[0])
        ceil = float(y_minmax[1])
        base_low = float(min(max(base_low, floor), ceil))
        base_high = float(min(max(base_high, floor), ceil))
        default_xlim = (floor, ceil)
    else:
        lo, hi = (min(base_low, base_high), max(base_low, base_high))
        default_xlim = (lo, hi if not math.isclose(lo, hi, rel_tol=1e-9) else lo + 1.0)

    if xlim is None:
        xlim = default_xlim
    xlim = (float(xlim[0]), float(xlim[1])) if xlim is not None else default_xlim

    body_xlabel = "Prediction interval" if xlabel is None else xlabel

    xtick_values = tuple(float(x) for x in xticks) if xticks is not None else None

    lo, hi = (min(base_low, base_high), max(base_low, base_high))
    base_segments = (IntervalSegment(low=lo, high=hi, color=REGRESSION_BASE_COLOR),)
    base_lines = ((base_pred, REGRESSION_BAR_COLOR, 0.3),)

    bars: list[BarItem] = []

    def _label_for(index: int) -> str:
        if rule_labels is not None:
            return str(rule_labels[index])
        if column_names is not None:
            return str(column_names[index])
        return str(index)

    if isinstance(feature_weights, dict):
        preds = feature_weights.get("predict", ())
        lows = feature_weights.get("low", ()) if interval else None
        highs = feature_weights.get("high", ()) if interval else None
        for j in features_to_plot:
            val = float(preds[j])
            has_interval = interval and lows is not None and highs is not None
            if has_interval:
                lo = float(lows[j])
                hi = float(highs[j])
            else:
                lo = hi = val
            lo_draw, hi_draw = (min(lo, hi), max(lo, hi))
            segments: tuple[IntervalSegment, ...] = ()
            if has_interval and not math.isclose(lo_draw, hi_draw, rel_tol=1e-9, abs_tol=1e-9):
                segments = (
                    IntervalSegment(
                        low=lo_draw, high=hi_draw, color=REGRESSION_BAR_COLOR, alpha=0.4
                    ),
                )
            if not segments:
                # Fall back to drawing contribution span between base prediction and rule value.
                base_span_low, base_span_high = (min(val, base_pred), max(val, base_pred))
                lo_draw, hi_draw = base_span_low, base_span_high
            interval_low = lo_draw
            interval_high = hi_draw
            item = BarItem(
                label=_label_for(j),
                value=val,
                interval_low=interval_low,
                interval_high=interval_high,
                instance_value=(instance[j] if instance is not None else None),
                solid_on_interval_crosses_zero=legacy_solid_behavior,
                color_role=REGRESSION_BAR_COLOR,
            )
            if segments:
                item.segments = segments
            item.line = val
            item.line_color = REGRESSION_BAR_COLOR
            item.line_alpha = 1.0
            bars.append(item)
    else:
        arr = feature_weights
        for j in features_to_plot:
            val = float(arr[j])  # type: ignore[index]
            segments: tuple[IntervalSegment, ...] = ()
            interval_low = min(val, base_pred)
            interval_high = max(val, base_pred)
            item = BarItem(
                label=_label_for(j),
                value=val,
                interval_low=interval_low,
                interval_high=interval_high,
                instance_value=(instance[j] if instance is not None else None),
                solid_on_interval_crosses_zero=legacy_solid_behavior,
                color_role=REGRESSION_BAR_COLOR,
            )
            if not math.isclose(val, base_pred, rel_tol=1e-9, abs_tol=1e-9):
                lo_draw, hi_draw = (min(val, base_pred), max(val, base_pred))
                item.segments = (
                    IntervalSegment(
                        low=lo_draw, high=hi_draw, color=REGRESSION_BAR_COLOR, alpha=0.4
                    ),
                )
            item.line = val
            item.line_color = REGRESSION_BAR_COLOR
            item.line_alpha = 1.0
            bars.append(item)

    def _key(item: BarItem):
        s = (sort_by or "none").lower()
        if s in ("none", ""):
            return 0.0
        if s == "value":
            return item.value
        if s == "abs":
            return abs(item.value)
        if s in ("interval", "width"):
            if item.interval_low is not None and item.interval_high is not None:
                w = abs(float(item.interval_high) - float(item.interval_low))
                return (w, abs(item.value))
            return 0.0
        if s == "label":
            return item.label
        return 0.0

    if sort_by and sort_by.lower() not in ("none", ""):
        bars = sorted(bars, key=_key, reverse=not ascending)

    body = BarHPanelSpec(
        bars=bars,
        xlabel=body_xlabel,
        ylabel="Alternative rules",
        solid_on_interval_crosses_zero=legacy_solid_behavior,
        is_alternative=True,
        base_segments=base_segments,
        base_lines=base_lines,
        pivot=None,
        xlim=xlim,
        xticks=xtick_values,
        bar_span=0.2,
    )

    height = max(len(bars), 1) * 0.5
    return PlotSpec(title=title, figure_size=(10.0, height), header=None, body=body)


__all__ = [
    "build_regression_bars_spec",
    "build_probabilistic_bars_spec",
    "build_factual_probabilistic_spec",
    "build_alternative_probabilistic_spec",
    "build_factual_regression_spec",
    "build_alternative_regression_spec",
]


def build_probabilistic_bars_spec(
    *,
    title: str | None,
    predict: dict[str, float],
    feature_weights: dict[str, Sequence[float]] | Sequence[float],
    features_to_plot: Sequence[int],
    column_names: Sequence[str] | None,
    rule_labels: Sequence[str] | None = None,
    instance: Sequence[Any] | None,
    y_minmax: tuple[float, float] | None,
    interval: bool,
    sort_by: str | None = None,
    ascending: bool = False,
    legacy_solid_behavior: bool = True,
    neg_label: str | None = None,
    pos_label: str | None = None,
    uncertainty_color: str | None = None,
    uncertainty_alpha: float | None = None,
    neg_caption: str | None = None,
    pos_caption: str | None = None,
    header_xlabel: str | None = None,
    header_ylabel: str | None = None,
    body_ylabel: str | None = None,
) -> PlotSpec:
    """Create a PlotSpec for the probabilistic bar plot variant.

    This mirrors the body behavior of the legacy probabilistic bar plot. The header
    is an IntervalHeaderSpec approximating the prediction interval (when available)
    and the body is a horizontal bar panel with rule labels on the left and
    instance values on the right.
    """
    max_index = max(features_to_plot) if features_to_plot else -1
    if isinstance(feature_weights, dict):
        for key in ("predict", "low", "high"):
            _ensure_indexable_length(
                f"feature_weights['{key}']",
                feature_weights.get(key),
                max_index=max_index,
            )
    else:
        _ensure_indexable_length("feature_weights", feature_weights, max_index=max_index)
    _ensure_indexable_length("column_names", column_names, max_index=max_index)
    _ensure_indexable_length("rule_labels", rule_labels, max_index=max_index)
    _ensure_indexable_length("instance", instance, max_index=max_index)

    # Header: use prediction interval when available
    pred = float(predict.get("predict", 0.0))
    low = float(predict.get("low", pred))
    high = float(predict.get("high", pred))
    if y_minmax is not None:
        low, high, xlim = _normalize_interval_bounds(low, high, y_minmax=y_minmax)
    else:
        xlim = (0.0, 1.0)
    if is_valid_probability_values(pred, low, high):
        pred = float(min(max(pred, 0.0), 1.0))
        low = float(min(max(low, 0.0), 1.0))
        high = float(min(max(high, 0.0), 1.0))
        xlim = (0.0, 1.0)
    has_interval = not math.isclose(low, high, rel_tol=1e-12, abs_tol=1e-12)
    header = IntervalHeaderSpec(
        pred=pred,
        low=low,
        high=high,
        xlim=xlim,
        xlabel=(header_xlabel if header_xlabel is not None else "Probability"),
        ylabel=(
            header_ylabel
            if header_ylabel is not None
            else ("Median prediction" if y_minmax is not None else "Probability")
        ),
        dual=True,
        neg_label=neg_label,
        pos_label=pos_label,
        neg_caption=neg_caption,
        pos_caption=pos_caption,
        show_intervals=has_interval,
        uncertainty_color=uncertainty_color,
        uncertainty_alpha=uncertainty_alpha,
    )

    bars: list[BarItem] = []
    if isinstance(feature_weights, dict) and interval:
        pred_w = feature_weights["predict"]
        wl = feature_weights["low"]
        wh = feature_weights["high"]
        for j in features_to_plot:
            val = float(pred_w[j])
            bars.append(
                BarItem(
                    label=(
                        rule_labels[j]
                        if rule_labels is not None
                        else (column_names[j] if column_names is not None else str(j))
                    ),
                    value=val,
                    interval_low=float(wl[j]),
                    interval_high=float(wh[j]),
                    instance_value=(instance[j] if instance is not None else None),
                    color_role=("positive" if val > 0 else "negative"),
                    solid_on_interval_crosses_zero=legacy_solid_behavior,
                )
            )
    else:
        arr = feature_weights  # type: ignore[assignment]
        for j in features_to_plot:
            val = float(arr[j])  # type: ignore[index]
            bars.append(
                BarItem(
                    label=(
                        rule_labels[j]
                        if rule_labels is not None
                        else (column_names[j] if column_names is not None else str(j))
                    ),
                    value=val,
                    instance_value=(instance[j] if instance is not None else None),
                    color_role=("positive" if val > 0 else "negative"),
                    solid_on_interval_crosses_zero=legacy_solid_behavior,
                )
            )

    # Sorting similar to regression builder
    def _key(item: BarItem):
        s = (sort_by or "none").lower()
        if s in ("none", ""):
            return 0.0
        if s == "value":
            return item.value
        if s == "abs":
            return abs(item.value)
        if s in ("interval", "width"):
            if item.interval_low is not None and item.interval_high is not None:
                w = abs(float(item.interval_high) - float(item.interval_low))
                return (w, abs(item.value))
            return 0.0
        if s == "label":
            return item.label
        return 0.0

    if sort_by and sort_by.lower() not in ("none", ""):
        bars = sorted(bars, key=_key, reverse=not ascending)

    # For the body we display feature contributions (weights) centered at zero
    body = BarHPanelSpec(
        bars=bars,
        xlabel="Feature weights",
        ylabel=(body_ylabel if body_ylabel is not None else "Rules"),
        solid_on_interval_crosses_zero=legacy_solid_behavior,
        show_base_interval=interval,
    )
    return PlotSpec(title=title, header=header, body=body)


# --- Triangular and global builders (return JSON-serializable dicts) ---
def build_triangular_plotspec_dict(
    *,
    title: str | None,
    proba: list[float] | float,
    uncertainty: list[float] | float,
    rule_proba: list[float],
    rule_uncertainty: list[float],
    num_to_show: int = 50,
    is_probabilistic: bool = True,
) -> dict:
    """Create a triangular PlotSpec dict describing quiver+scatter data.

    This does not return a PlotSpec dataclass because triangular plots are
    non-panel (no header/body) and are conveyed as a payload in the dict
    form consumed by adapters via the PlotSpec JSON contract.
    """
    return {
        "plot_spec": {
            "kind": "triangular",
            "mode": "classification" if is_probabilistic else "regression",
            "header": None,
            "body": None,
            "style": "triangular",
            "uncertainty": True,
            "feature_order": [],
            "triangular": {
                "proba": proba,
                "uncertainty": uncertainty,
                "rule_proba": rule_proba,
                "rule_uncertainty": rule_uncertainty,
                "num_to_show": int(num_to_show),
                "is_probabilistic": bool(is_probabilistic),
            },
        }
    }


def build_global_plotspec_dict(
    *,
    title: str | None,
    proba: list[float] | None,
    predict: list[float] | None,
    low: list[float],
    high: list[float],
    uncertainty: list[float],
    y_test: list | None = None,
    is_regularized: bool = True,
) -> dict:
    """Create a global PlotSpec dict for scatter of uncertainty vs proba/predict.

    Contains arrays and axis hints; adapters should render scatter accordingly.
    """
    # basic axis hints
    min_x = None
    max_x = None
    min_y = None
    max_y = None
    try:
        if proba is not None:
            arr = list(proba)
        elif predict is not None:
            arr = list(predict)
        else:
            arr = []
        if arr:
            min_x = float(min(arr))
            max_x = float(max(arr))
    except:
        if not isinstance(sys.exc_info()[1], Exception):
            raise
        min_x = 0.0
        max_x = 1.0
    try:
        if uncertainty is not None:
            min_y = float(min(uncertainty))
            max_y = float(max(uncertainty))
    except:
        if not isinstance(sys.exc_info()[1], Exception):
            raise
        min_y = 0.0
        max_y = 1.0

    axis_hints = {"xlim": [min_x or 0.0, max_x or 1.0], "ylim": [min_y or 0.0, max_y or 1.0]}
    return {
        "plot_spec": {
            "kind": "global_probabilistic" if is_regularized else "global_regression",
            "mode": "classification" if is_regularized else "regression",
            "header": None,
            "body": None,
            "style": "regular",
            "uncertainty": True,
            "feature_order": [],
            "global_entries": {
                "proba": list(proba) if proba is not None else None,
                "predict": list(predict) if predict is not None else None,
                "low": list(low),
                "high": list(high),
                "uncertainty": list(uncertainty),
                "y_test": list(y_test) if y_test is not None else None,
            },
            "axis_hints": axis_hints,
        }
    }


def plotspec_to_dict(spec: PlotSpec) -> dict:
    """Convert a PlotSpec dataclass to a JSON-serializable dict matching plotspec_schema.json."""
    out = {
        "plot_spec": {
            "kind": (
                "factual_probabilistic"
                if (spec.header is not None and spec.header.dual)
                else "factual_regression"
            ),
            "mode": (
                "classification" if spec.header is not None and spec.header.dual else "regression"
            ),
            "header": {
                "pred": float(spec.header.pred) if spec.header is not None else None,
                "low": float(spec.header.low) if spec.header is not None else None,
                "high": float(spec.header.high) if spec.header is not None else None,
                "xlim": (
                    list(spec.header.xlim)
                    if (spec.header is not None and spec.header.xlim is not None)
                    else None
                ),
                "xlabel": spec.header.xlabel if spec.header is not None else None,
                "ylabel": spec.header.ylabel if spec.header is not None else None,
                "dual": bool(spec.header.dual) if spec.header is not None else False,
            }
            if spec.header is not None
            else None,
            "body": None,
            "style": "regular",
            "uncertainty": False,
            "feature_order": [],
        }
    }
    if spec.body is not None:
        entries = []
        for i, b in enumerate(spec.body.bars):
            e = {
                "index": i,
                "name": b.label,
                "weight": float(b.value),
                "low": (float(b.interval_low) if b.interval_low is not None else None),
                "high": (float(b.interval_high) if b.interval_high is not None else None),
                "instance_value": b.instance_value,
            }
            entries.append(e)
        out["plot_spec"]["body"] = {
            "bars_count": len(entries),
            "xlabel": spec.body.xlabel,
            "ylabel": spec.body.ylabel,
        }
        out["plot_spec"]["feature_entries"] = entries
        out["plot_spec"]["feature_order"] = list(range(len(entries)))
        # uncertainty flag if any bar has interval
        out["plot_spec"]["uncertainty"] = any(
            (b.interval_low is not None and b.interval_high is not None) for b in spec.body.bars
        )
    return out


def build_factual_probabilistic_plotspec_dict(**kwargs) -> dict:
    """Build a PlotSpec for factual probabilistic and return as serializable dict.

    This calls the existing `build_probabilistic_bars_spec` and converts the
    dataclass to the agreed PlotSpec dict shape for schema validation and
    downstream adapter consumption.
    """
    spec = build_probabilistic_bars_spec(**kwargs)
    return plotspec_to_dict(spec)
