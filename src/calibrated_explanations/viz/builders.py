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
    rule_labels: Sequence[str] | None = None,
    instance: Sequence[Any] | None,
    y_minmax: tuple[float, float] | None,
    interval: bool,
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
        dual=False,
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


__all__ = ["build_regression_bars_spec"]


def build_factual_probabilistic_spec(**kwargs) -> PlotSpec:
    """Wrapper for factual probabilistic plot kind (ADR-016).

    Currently delegates to `build_probabilistic_bars_spec`. This wrapper
    provides a stable name for future refactors and makes plot kinds explicit.
    """
    return build_probabilistic_bars_spec(**kwargs)


def build_alternative_probabilistic_spec(**kwargs) -> PlotSpec:
    """Wrapper for alternative probabilistic plot kind (ADR-016)."""
    return build_probabilistic_bars_spec(**kwargs)


def build_factual_regression_spec(**kwargs) -> PlotSpec:
    """Wrapper for factual regression plot kind (ADR-016)."""
    return build_regression_bars_spec(**kwargs)


def build_alternative_regression_spec(**kwargs) -> PlotSpec:
    """Wrapper for alternative regression plot kind (ADR-016)."""
    return build_regression_bars_spec(**kwargs)


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
    # For probabilistic plots, default x-axis range is [0,1] unless caller supplies y_minmax
    if y_minmax is not None:
        xlim = (float(min(low, y_minmax[0])), float(max(high, y_minmax[1])))
    else:
        xlim = (0.0, 1.0)
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
        ylabel="Rules",
        solid_on_interval_crosses_zero=legacy_solid_behavior,
    )
    return PlotSpec(title=title, header=header, body=body)


__all__ = ["build_regression_bars_spec", "build_probabilistic_bars_spec"]


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
    except Exception:
        min_x = 0.0
        max_x = 1.0
    try:
        if uncertainty is not None:
            min_y = float(min(uncertainty))
            max_y = float(max(uncertainty))
    except Exception:
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
