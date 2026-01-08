"""Builders to convert existing plotting inputs to PlotSpec structures.

This keeps default plotting intact while offering an internal pathway to
render via the PlotSpec + matplotlib adapter for selected plots.
"""

from __future__ import annotations

import contextlib
import math
import sys
from typing import Any, Sequence

import numpy as np

from .plotspec import (
    BarHPanelSpec,
    BarItem,
    GlobalPlotSpec,
    GlobalSpec,
    IntervalHeaderSpec,
    IntervalSegment,
    PlotSpec,
    TriangularPlotSpec,
    TriangularSpec,
)
from .serializers import (
    global_plotspec_to_dict,
    plotspec_to_dict,
    triangular_plotspec_to_dict,
    validate_plotspec,
)
from dataclasses import asdict, is_dataclass


class _PlotSpecDictWrapper(dict):
    """A dict-like wrapper that also exposes underlying dataclass attributes.

    This lets builders return a JSON-serializable envelope (dict) while
    preserving attribute access used by tests that treat the return value as
    a dataclass (`spec.save_behavior = ...`).
    """

    def __init__(self, payload: dict, spec_obj: object):
        super().__init__(payload)
        object.__setattr__(self, "_spec_obj", spec_obj)
        object.__setattr__(self, "_payload", payload)

    def __getattr__(self, name: str):
        spec = object.__getattribute__(self, "_spec_obj")
        if hasattr(spec, name):
            return getattr(spec, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value):
        # write through to underlying dataclass when possible
        spec = object.__getattribute__(self, "_spec_obj")
        if hasattr(spec, name):
            try:
                setattr(spec, name, value)
            except Exception:
                serialized = _serialize_plot_attr(value)
            else:
                serialized = _serialize_plot_attr(value)
            payload = object.__getattribute__(self, "_payload")
            inner = payload.setdefault("plot_spec", {})
            inner[name] = serialized
            super().__setitem__(name, serialized)
        else:
            super().__setitem__(name, value)

_PROBABILITY_TOL = 1e-9


def _serialize_plot_attr(value: Any) -> Any:
    """Serialize dataclass-like attributes for the wrapper dict."""
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (list, tuple, dict)):
        return value
    return value


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
        except BaseException:
            exc_info = sys.exc_info()[1]
            if not isinstance(exc_info, Exception):
                raise
            return False
        if not math.isfinite(numeric):
            return False
        finite.append(numeric)
    if not finite:
        return False
    tol = _PROBABILITY_TOL
    return all(-tol <= v <= 1.0 + tol for v in finite)


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
            from ..utils.exceptions import ValidationError

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
        except BaseException:
            exc_info = sys.exc_info()[1]
            if not isinstance(exc_info, Exception):
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
    spec = PlotSpec(title=title, header=header, body=body)
    # metadata
    spec.kind = "factual_regression"
    spec.mode = "regression"
    if spec.body is not None:
        spec.feature_order = tuple(range(len(spec.body.bars)))
    validate_plotspec(plotspec_to_dict(spec))
    return spec


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
    explicit_header_labels: bool = False,
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

    # Remember whether the caller provided `xticks`. If not provided we may
    # default them for plotting, but header-creation logic should only
    # consider explicitly-provided hints.
    provided_xticks = xticks is not None
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
    # Optionally include a dual header for alternative probabilistic plots
    # when the x-axis is a probability scale and legacy solid behavior is
    # enabled. Some callers/tests expect headerless specs when parity-mode
    # (legacy solid behavior) is disabled, so only create the header when
    # `legacy_solid_behavior` is True and header-related hints are present.
    is_prob_scale = xlim == (0.0, 1.0)
    has_base_interval = bool(base_segments)
    header_hints = explicit_header_labels or provided_xticks
    header_needed = is_prob_scale and header_hints
    if header_needed and legacy_solid_behavior:
        header = IntervalHeaderSpec(
            pred=base_pred,
            low=float(min(base_low, base_high)),
            high=float(max(base_low, base_high)),
            xlim=xlim,
            xlabel=body_xlabel,
            ylabel=("Median prediction" if y_minmax is not None else "Probability"),
            dual=True,
            neg_label=neg_label,
            pos_label=pos_label,
            show_intervals=has_base_interval,
            uncertainty_color=uncertainty_color,
            uncertainty_alpha=uncertainty_alpha,
        )
    else:
        header = None

    # When a header is present, expose a base line (prediction marker)
    # within the body so the adapter can render a marker line for the
    # prediction baseline. This mirrors the regression alternative builder
    # behaviour and satisfies parity tests that expect `lines` primitives.
    if header is not None:
        try:
            body.base_lines = ((base_pred, REGRESSION_BAR_COLOR, 0.3),)
        except Exception:
            body.base_lines = None

    spec = PlotSpec(title=title, figure_size=(10.0, height), header=header, body=body)
    spec.kind = "alternative_probabilistic"
    spec.mode = "classification"
    spec.feature_order = tuple(range(len(spec.body.bars)))
    validate_plotspec(plotspec_to_dict(spec))
    return spec


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
    spec = PlotSpec(title=title, figure_size=(10.0, height), header=None, body=body)
    spec.kind = "alternative_regression"
    spec.mode = "regression"
    spec.feature_order = tuple(range(len(spec.body.bars)))
    validate_plotspec(plotspec_to_dict(spec))
    return spec


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
    spec = PlotSpec(title=title, header=header, body=body)
    spec.kind = "factual_probabilistic"
    spec.mode = "classification"
    spec.feature_order = tuple(range(len(spec.body.bars)))
    validate_plotspec(plotspec_to_dict(spec))
    return spec


# --- Triangular and global builders (return JSON-serializable dicts) ---
def build_triangular_plotspec(
    *,
    title: str | None,
    proba: list[float] | float,
    uncertainty: list[float] | float,
    rule_proba: list[float],
    rule_uncertainty: list[float],
    num_to_show: int = 50,
    is_probabilistic: bool = True,
) -> TriangularPlotSpec:
    """Create a TriangularPlotSpec dataclass for triangular plots."""
    triangular = TriangularSpec(
        proba=proba,
        uncertainty=uncertainty,
        rule_proba=rule_proba,
        rule_uncertainty=rule_uncertainty,
        num_to_show=num_to_show,
        is_probabilistic=is_probabilistic,
    )
    spec = TriangularPlotSpec(
        title=title,
        triangular=triangular,
        kind="triangular",
        mode="classification" if is_probabilistic else "regression",
    )
    payload = triangular_plotspec_to_dict(spec)
    validate_plotspec(payload)
    # Return a wrapper that is dict-like for parity tests but preserves
    # attribute access to the underlying dataclass for roundtrip tests.
    return _PlotSpecDictWrapper(payload, spec)


def build_global_plotspec(
    *,
    title: str | None,
    proba: list[float] | None,
    predict: list[float] | None,
    low: list[float],
    high: list[float],
    uncertainty: list[float],
    y_test: list | None = None,
    is_regularized: bool = True,
) -> GlobalPlotSpec:
    """Create a GlobalPlotSpec dataclass for global plots."""
    global_entries = GlobalSpec(
        proba=proba,
        predict=predict,
        low=low,
        high=high,
        uncertainty=uncertainty,
        y_test=y_test,
    )
    spec = GlobalPlotSpec(
        title=title,
        global_entries=global_entries,
        kind="global_probabilistic" if is_regularized else "global_regression",
        mode="classification" if is_regularized else "regression",
    )
    payload = global_plotspec_to_dict(spec)
    validate_plotspec(payload)
    return _PlotSpecDictWrapper(payload, spec)


def build_factual_probabilistic_plotspec_dict(**kwargs) -> dict:
    """Build a PlotSpec for factual probabilistic and return as serializable dict.

    This calls the existing `build_probabilistic_bars_spec` and converts the
    dataclass to the agreed PlotSpec dict shape for schema validation and
    downstream adapter consumption.
    """
    spec = build_probabilistic_bars_spec(**kwargs)
    return plotspec_to_dict(spec)


def build_triangular_plotspec_dict(**kwargs) -> dict:
    """Build a TriangularPlotSpec and return a JSON-serializable dict.

    Wrapper for `build_triangular_plotspec` that converts the dataclass
    into the envelope expected by adapters and `plotting.py`.
    """
    spec = build_triangular_plotspec(**kwargs)
    # If the builder already returned a dict envelope, pass it through.
    if isinstance(spec, dict):
        return spec
    return triangular_plotspec_to_dict(spec)


def build_global_plotspec_dict(**kwargs) -> dict:
    """Build a GlobalPlotSpec and return a JSON-serializable dict.

    Wrapper for `build_global_plotspec` that converts the dataclass
    into the envelope expected by adapters and plugin builders.
    """
    spec = build_global_plotspec(**kwargs)
    # If the builder already returned a dict envelope, pass it through.
    if isinstance(spec, dict):
        return spec
    return global_plotspec_to_dict(spec)
