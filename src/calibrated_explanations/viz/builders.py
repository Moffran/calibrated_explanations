"""Builders to convert existing plotting inputs to PlotSpec structures.

This keeps default plotting intact while offering an internal pathway to
render via the PlotSpec + matplotlib adapter for selected plots.
"""

from __future__ import annotations

from typing import Any, Sequence

from .plotspec import BarHPanelSpec, BarItem, IntervalHeaderSpec, PlotSpec


def build_regression_bars_spec(
    *,
    title: str | None,
    predict: dict[str, float],
    feature_weights: dict[str, Sequence[float]] | Sequence[float],
    features_to_plot: Sequence[int],
    column_names: Sequence[str] | None,
    instance: Sequence[Any] | None,
    y_minmax: tuple[float, float] | None,
    interval: bool,
    uncertainty_color: str | None = None,
    uncertainty_alpha: float | None = None,
    sort_by: str | None = None,
    ascending: bool = False,
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
    # Header (interval around prediction)
    pred = float(predict["predict"]) if "predict" in predict else 0.0
    low = float(predict.get("low", pred))
    high = float(predict.get("high", pred))
    xlim = None
    if y_minmax is not None:
        xlim = (float(min(low, y_minmax[0])), float(max(high, y_minmax[1])))
    header = IntervalHeaderSpec(
        pred=pred,
        low=low,
        high=high,
        xlim=xlim,
        xlabel=(
            "Prediction interval with unknown confidence"
            if y_minmax is None
            else "Prediction interval"
        ),
        ylabel="Median prediction",
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
                    label=(column_names[j] if column_names is not None else str(j)),
                    value=val,
                    interval_low=float(wl[j]),
                    interval_high=float(wh[j]),
                    instance_value=(instance[j] if instance is not None else None),
                    color_role="regression",
                )
            )
    else:
        # Simple numeric sequence
        arr = feature_weights  # type: ignore[assignment]
        for j in features_to_plot:
            val = float(arr[j])  # type: ignore[index]
            bars.append(
                BarItem(
                    label=(column_names[j] if column_names is not None else str(j)),
                    value=val,
                    instance_value=(instance[j] if instance is not None else None),
                    color_role="regression",
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

    body = BarHPanelSpec(bars=bars, xlabel="Feature weights", ylabel="Rules")
    return PlotSpec(title=title, header=header, body=body)


__all__ = ["build_regression_bars_spec"]


def build_probabilistic_bars_spec(
    *,
    title: str | None,
    predict: dict[str, float],
    feature_weights: dict[str, Sequence[float]] | Sequence[float],
    features_to_plot: Sequence[int],
    column_names: Sequence[str] | None,
    instance: Sequence[Any] | None,
    y_minmax: tuple[float, float] | None,
    interval: bool,
    sort_by: str | None = None,
    ascending: bool = False,
    neg_label: str | None = None,
    pos_label: str | None = None,
    uncertainty_color: str | None = None,
    uncertainty_alpha: float | None = None,
) -> PlotSpec:
    """Create a PlotSpec for the probabilistic bar plot variant.

    This mirrors the body behavior of the legacy probabilistic bar plot. The header
    is an IntervalHeaderSpec approximating the prediction interval (when available)
    and the body is a horizontal bar panel with rule labels on the left and
    instance values on the right.
    """
    # Header: use prediction interval when available
    pred = float(predict.get("predict", 0.0))
    low = float(predict.get("low", pred))
    high = float(predict.get("high", pred))
    xlim = None
    if y_minmax is not None:
        xlim = (float(min(low, y_minmax[0])), float(max(high, y_minmax[1])))
    header = IntervalHeaderSpec(
        pred=pred,
        low=low,
        high=high,
        xlim=xlim,
        xlabel=("Probability" if y_minmax is None else "Probability"),
        ylabel=("Median prediction" if y_minmax is not None else "Probability"),
        dual=True,
        neg_label=neg_label,
        pos_label=pos_label,
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
                    label=(column_names[j] if column_names is not None else str(j)),
                    value=val,
                    interval_low=float(wl[j]),
                    interval_high=float(wh[j]),
                    instance_value=(instance[j] if instance is not None else None),
                    color_role=("positive" if val > 0 else "negative"),
                )
            )
    else:
        arr = feature_weights  # type: ignore[assignment]
        for j in features_to_plot:
            val = float(arr[j])  # type: ignore[index]
            bars.append(
                BarItem(
                    label=(column_names[j] if column_names is not None else str(j)),
                    value=val,
                    instance_value=(instance[j] if instance is not None else None),
                    color_role=("positive" if val > 0 else "negative"),
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

    body = BarHPanelSpec(bars=bars, xlabel="Probability", ylabel="Rules")
    return PlotSpec(title=title, header=header, body=body)


__all__ = ["build_regression_bars_spec", "build_probabilistic_bars_spec"]
