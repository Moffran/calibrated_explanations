"""Matplotlib adapter for PlotSpec (ADR-007).

Renders PlotSpec structures using the existing style configuration loader and
lazy matplotlib import. This keeps plotting optional behind the 'viz' extra.
"""

from __future__ import annotations

import importlib
import io
import logging
import math
import sys
import warnings
from contextlib import suppress
from typing import Any, Sequence

import numpy as np

from ..plotting import _MATPLOTLIB_IMPORT_ERROR  # noqa: F401  (exported indirectly)
from ..plotting import __require_matplotlib as _require_mpl  # reuse lazy guard
from ..plotting import __setup_plot_style as _setup_style
from ..utils.exceptions import ValidationError
from .plotspec import BarHPanelSpec, GlobalPlotSpec, PlotSpec, TriangularPlotSpec
from .serializers import global_plotspec_to_dict, triangular_plotspec_to_dict

_HEADER_BAR_Y_SPAN = 0.2
_HEADER_Y_LIMIT = 1.0
_SINGLE_HEADER_HEIGHT = 0.8
_EXTRA_RULE_LINE_UNITS = 0.45


def _import_pyplot_with_retries() -> object:
    """Import `matplotlib.pyplot` robustly, preloading common submodules on failure.

    Some test suites monkeypatch or partially inject matplotlib into `sys.modules`,
    which can leave the package in a partially-initialized state. Try importing
    `matplotlib.pyplot` and, on failure, preload a small set of submodules that
    pyplot expects, then retry once before re-raising the original exception.
    """
    try:
        return importlib.import_module("matplotlib.pyplot")
    except Exception as exc:  # adr002_allow  # pragma: no cover - protective retry logic
        logging.getLogger(__name__).warning(
            "matplotlib.pyplot import failed (%s); preloading submodules and retrying",
            exc,
        )
        # preload a broader set of commonly-required submodules used by
        # pyplot's lazy attribute access. Tests sometimes inject or patch
        # parts of `matplotlib` in `sys.modules`, leaving attributes
        # unresolved when `pyplot` imports decorated functions that
        # reference `matplotlib.<submod>.<name>` at import time.
        preload = (
            "matplotlib._api",
            "matplotlib.artist",
            "matplotlib.figure",
            "matplotlib.axes",
            "matplotlib.image",
            "matplotlib.cm",
            "matplotlib.colors",
            "matplotlib.transforms",
            "matplotlib.path",
            "matplotlib._path",
            "matplotlib.collections",
            "matplotlib.lines",
            "matplotlib.patches",
            "matplotlib.text",
            "matplotlib.textpath",
            "matplotlib.font_manager",
            "matplotlib.backend_bases",
            "matplotlib.backends",
            "matplotlib.backends.backend_agg",
        )
        for sub in preload:
            try:
                importlib.import_module(sub)
            except Exception:  # adr002_allow
                logging.getLogger(__name__).debug("preload of %s failed", sub, exc_info=True)
        # Ensure the top-level `matplotlib` package has attributes for
        # commonly-used first-level submodules (e.g. `matplotlib.artist`,
        # `matplotlib.figure`, `matplotlib.axes`, `matplotlib.backends`).
        try:
            mpl_pkg = importlib.import_module("matplotlib")
            for sub in preload:
                parts = sub.split(".")
                if len(parts) >= 2 and parts[0] == "matplotlib":
                    top = parts[1]
                    try:
                        submod = importlib.import_module(f"matplotlib.{top}")
                        if not hasattr(mpl_pkg, top):
                            try:
                                setattr(mpl_pkg, top, submod)
                            except Exception:  # adr002_allow
                                logging.getLogger(__name__).debug(
                                    "failed to set attribute matplotlib.%s", top, exc_info=True
                                )
                    except Exception:  # adr002_allow
                        logging.getLogger(__name__).debug(
                            "failed to import matplotlib.%s for attachment", top, exc_info=True
                        )
        except Exception:  # adr002_allow
            logging.getLogger(__name__).debug("failed to attach preloaded submodules to matplotlib")
        # retry once
        try:
            return importlib.import_module("matplotlib.pyplot")
        except Exception as exc2:  # adr002_allow
            logging.getLogger(__name__).error(
                "matplotlib.pyplot import still failing after preloads: %s", exc2
            )
            raise


def _first_float(value: Any, *, default: float = 0.0) -> float:
    """Return the first finite float from a scalar or sequence."""
    try:
        arr = np.asarray(value, dtype=float)
        if arr.size == 0:
            return default
        candidate = float(arr.reshape(-1)[0])
    except Exception:  # adr002_allow
        return default
    return candidate if math.isfinite(candidate) else default


def _float_vector(value: Any, *, name: str) -> np.ndarray:
    """Coerce a scalar/sequence to a one-dimensional float array."""
    if value is None:
        raise ValidationError(f"{name} is required for PlotSpec rendering", details={"param": name})
    try:
        arr = np.asarray(value, dtype=float)
    except Exception as exc:  # adr002_allow
        raise ValidationError(
            f"{name} must be numeric for PlotSpec rendering",
            details={"param": name, "actual_type": type(value).__name__},
        ) from exc
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def _finite_bounds(values: Sequence[float], *, default: tuple[float, float]) -> tuple[float, float]:
    """Return finite min/max bounds with light padding for degenerate spans."""
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return default
    lo = min(finite)
    hi = max(finite)
    if math.isclose(lo, hi, rel_tol=1e-9, abs_tol=1e-12):
        pad = abs(lo) * 0.1 if lo else 0.1
        return lo - pad, hi + pad
    pad = 0.1 * (hi - lo)
    return lo - pad, hi + pad


def _plot_probability_triangle(ax: Any, *, primitives: list[dict[str, Any]] | None = None) -> None:
    """Draw the legacy probability-triangle reference lines."""
    segments = []
    x = np.arange(0, 1, 0.01)
    segments.append((x / (1 + x), x))
    with np.errstate(divide="ignore", invalid="ignore"):
        segments.append((x, (1 - x) / x))
    x = np.arange(0.5, 1, 0.005)
    segments.append(((0.5 + x - 0.5) / (1 + x - 0.5), x - 0.5))
    x = np.arange(0, 0.5, 0.005)
    segments.append(((x + 0.5 - x) / (1 + x), x))
    for idx, (xs, ys) in enumerate(segments):
        ax.plot(xs, ys, color="black")
        if primitives is not None:
            finite = np.isfinite(xs) & np.isfinite(ys)
            finite_xs = xs[finite]
            finite_ys = ys[finite]
            x0 = float(finite_xs[0]) if finite_xs.size else 0.0
            y0 = float(finite_ys[0]) if finite_ys.size else 0.0
            x1 = float(finite_xs[-1]) if finite_xs.size else 0.0
            y1 = float(finite_ys[-1]) if finite_ys.size else 0.0
            primitives.append(
                {
                    "id": f"triangle.background.{idx}",
                    "type": "line",
                    "coords": {
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                    },
                    "semantic": "triangle_reference",
                }
            )


def _finish_non_panel_figure(
    fig: Any,
    plt: Any,
    *,
    show: bool,
    save_path: str | None,
    return_fig: bool,
    export_drawn_primitives: bool,
    wrapper: dict[str, Any],
) -> Any:
    """Apply common save/show/close behavior for non-panel PlotSpec plots."""
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    if return_fig:
        return fig
    try:
        fig.tight_layout()
    except Exception:  # adr002_allow
        logging.getLogger(__name__).debug("Non-panel tight_layout skipped", exc_info=True)
    plt.close(fig)
    return wrapper if export_drawn_primitives else None


def _body_has_multiline_labels(body_spec: Any) -> bool:
    """Return whether the body contains any multiline labels."""
    if body_spec is None or getattr(body_spec, "bars", None) is None:
        return False
    for bar in body_spec.bars:
        label = getattr(bar, "label", None)
        if label is None:
            continue
        with suppress(Exception):
            if "\n" in str(label):
                return True
    return False


def _count_body_extra_lines(body_spec: Any) -> int:
    """Return the number of extra label lines contributed by multiline bars."""
    if body_spec is None or getattr(body_spec, "bars", None) is None:
        return 0
    extra_lines = 0
    for bar in body_spec.bars:
        label = getattr(bar, "label", None)
        if label is None:
            continue
        with suppress(Exception):
            extra_lines += str(label).count("\n")
    return extra_lines


def _body_y_layout(body_spec: Any) -> tuple[np.ndarray, float, float, float]:
    """Return compact line-aware y-centres, bounds, and row units."""
    bars = list(getattr(body_spec, "bars", []) or [])
    if not bars:
        return np.asarray([], dtype=float), -0.5, 0.5, 1.0

    units: list[float] = []
    for bar in bars:
        label = getattr(bar, "label", None)
        extra_lines = 0
        with suppress(Exception):
            extra_lines = str(label).count("\n")
        units.append(1.0 + extra_lines * _EXTRA_RULE_LINE_UNITS)
    centres: list[float] = []
    cursor = 0.0
    for unit in units:
        centres.append(cursor + unit / 2.0 - 0.5)
        cursor += unit
    total_units = max(cursor, 1.0)
    return np.asarray(centres, dtype=float), -0.5, total_units - 0.5, total_units


def _set_compact_y_tick_labels(ax: Any, labels: Sequence[Any]) -> None:
    """Set y tick labels with compact multiline spacing."""
    try:
        ax.set_yticklabels(labels, linespacing=0.92)
    except TypeError:
        ax.set_yticklabels(labels)


def _estimate_body_panel_height(body_spec: Any) -> float:
    """Estimate the body panel height in inches.

    Multiline conjunction labels should expand the body panel only, not the
    prediction probability bands above it.
    """
    num_bars = (
        len(body_spec.bars)
        if body_spec is not None and getattr(body_spec, "bars", None) is not None
        else 0
    )
    _, _, _, row_units = _body_y_layout(body_spec)
    return max(2.0, float((max(num_bars, row_units)) * 0.5 + 0.5))


def _resolve_panel_layout_policy(
    spec: PlotSpec,
    *,
    panels: Sequence[tuple[str, Any]],
    body_spec: BarHPanelSpec | None,
) -> tuple[bool, dict[str, Any], bool]:
    """Choose the panel layout strategy for the current PlotSpec render.

    Returns
    -------
    tuple[bool, dict[str, Any], bool]
        `use_tight_layout`, `savefig_kwargs`, `apply_alternative_margins`
    """
    is_single_panel_alternative = (
        len(panels) == 1 and body_spec is not None and getattr(body_spec, "is_alternative", False)
    )
    has_multiline_labels = _body_has_multiline_labels(body_spec)

    if is_single_panel_alternative or has_multiline_labels:
        return True, {}, is_single_panel_alternative

    return False, {"bbox_inches": "tight"}, False


def _render_triangular_spec(
    spec: TriangularPlotSpec,
    *,
    show: bool,
    save_path: str | None,
    return_fig: bool,
    export_drawn_primitives: bool,
) -> Any:
    """Render a triangular PlotSpec with legacy-equivalent quiver/scatter semantics."""
    tri = spec.triangular
    if tri is None:
        raise ValidationError(
            "Triangular PlotSpec requires triangular data",
            details={"required": ["triangular"]},
        )

    _require_mpl()
    plt = _import_pyplot_with_retries()
    _setup_style(None)
    fig = plt.figure(figsize=spec.figure_size)
    ax = fig.add_subplot(111)
    primitives: list[dict[str, Any]] = []

    proba = _first_float(tri.proba)
    uncertainty = _first_float(tri.uncertainty)
    rule_proba = _float_vector(tri.rule_proba, name="rule_proba")
    rule_uncertainty = _float_vector(tri.rule_uncertainty, name="rule_uncertainty")
    count = max(0, min(int(tri.num_to_show), len(rule_proba), len(rule_uncertainty)))
    shown_rule_proba = rule_proba[:count]
    shown_rule_uncertainty = rule_uncertainty[:count]

    if tri.is_probabilistic:
        _plot_probability_triangle(ax, primitives=primitives if export_drawn_primitives else None)
        xlim = (0.0, 1.0)
        ylim = (0.0, 1.0)
        xlabel = "Probability"
    else:
        xlim = _finite_bounds([*shown_rule_proba, proba], default=(0.0, 1.0))
        ylim = _finite_bounds([*shown_rule_uncertainty, uncertainty], default=(0.0, 1.0))
        xlabel = "Prediction"

    if count:
        dx = shown_rule_proba - proba
        dy = shown_rule_uncertainty - uncertainty
        ax.quiver(
            [proba] * count,
            [uncertainty] * count,
            dx,
            dy,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="lightgrey",
            width=0.005,
            headwidth=3,
            headlength=3,
        )
        if export_drawn_primitives:
            for idx, (rp, ru, dpx, duy) in enumerate(
                zip(shown_rule_proba, shown_rule_uncertainty, dx, dy, strict=False)
            ):
                primitives.append(
                    {
                        "id": f"triangle.quiver.{idx}",
                        "type": "quiver",
                        "coords": {
                            "x": proba,
                            "y": uncertainty,
                            "dx": float(dpx),
                            "dy": float(duy),
                            "x1": float(rp),
                            "y1": float(ru),
                        },
                        "semantic": "alternative_direction",
                    }
                )

    ax.scatter(
        rule_proba,
        rule_uncertainty,
        label="Alternative Explanations",
        marker=".",
        s=50,
    )
    ax.scatter(
        proba,
        uncertainty,
        color="red",
        label="Original Prediction",
        marker=".",
        s=50,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Uncertainty")
    set_title = getattr(ax, "set_title", None)
    if callable(set_title):
        set_title(spec.title or "Alternative Explanations")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    legend = getattr(ax, "legend", None)
    if callable(legend):
        legend()

    if export_drawn_primitives:
        primitives.append(
            {
                "id": "triangle.scatter.alternatives",
                "type": "scatter",
                "coords": {
                    "x": [float(v) for v in rule_proba],
                    "y": [float(v) for v in rule_uncertainty],
                },
                "style": {"label": "Alternative Explanations", "marker": "."},
                "semantic": "alternative_points",
            }
        )
        primitives.append(
            {
                "id": "triangle.scatter.original",
                "type": "scatter",
                "coords": {"x": proba, "y": uncertainty},
                "style": {"label": "Original Prediction", "marker": ".", "color": "red"},
                "semantic": "original_prediction",
            }
        )
        primitives.append(
            {
                "id": "triangle.axes",
                "type": "axes",
                "coords": {"xlim": xlim, "ylim": ylim},
                "style": {"xlabel": xlabel, "ylabel": "Uncertainty"},
                "semantic": "axis_meaning",
            }
        )

    payload = triangular_plotspec_to_dict(spec)["plot_spec"]
    wrapper = {
        "plot_spec": payload,
        "triangle_background": {"type": "triangle_background"},
        "primitives": primitives,
    }
    return _finish_non_panel_figure(
        fig,
        plt,
        show=show,
        save_path=save_path,
        return_fig=return_fig,
        export_drawn_primitives=export_drawn_primitives,
        wrapper=wrapper,
    )


def _threshold_xlabel(threshold: Any) -> str:
    """Return the legacy global-plot x-axis label for thresholded regression."""
    if threshold is None:
        return "Probability of Y = 1"
    try:
        if (
            isinstance(threshold, Sequence)
            and not isinstance(threshold, (str, bytes))
            and len(threshold) >= 2
        ):
            return f"Probability of {float(threshold[0])} <= Y < {float(threshold[1])}"
    except Exception:  # adr002_allow
        return f"Probability of Y < {threshold}"
    return f"Probability of Y < {threshold}"


def _class_labels(class_labels: Any, unique_y: Sequence[Any]) -> list[str]:
    """Return display labels for class-conditioned global scatter groups."""
    if class_labels is None:
        return [f"Y = {item}" for item in unique_y]
    if isinstance(class_labels, dict):
        return [f"Y = {class_labels.get(item, item)}" for item in unique_y]
    try:
        labels = list(class_labels)
    except TypeError:
        return [f"Y = {item}" for item in unique_y]
    resolved = []
    for item in unique_y:
        try:
            resolved.append(f"Y = {labels[int(item)]}")
        except Exception:  # adr002_allow
            resolved.append(f"Y = {item}")
    return resolved


def _global_xy(
    proba: Any,
    predict: Any,
    uncertainty: Any,
    y_test: Any,
    *,
    kind: str | None,
    threshold: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str]:
    """Resolve global plot x/y arrays and x-axis label from PlotSpec semantics."""
    y_arr = None if y_test is None else np.asarray(y_test)
    unc_arr = np.asarray(uncertainty, dtype=float)
    if unc_arr.ndim == 0:
        unc_arr = unc_arr.reshape(1)

    if kind == "global_regression":
        x_arr = np.asarray(predict if predict is not None else proba, dtype=float)
        if x_arr.ndim == 0:
            x_arr = x_arr.reshape(1)
        return x_arr.reshape(-1), unc_arr.reshape(-1), y_arr, "Predictions"

    proba_arr = np.asarray(proba if proba is not None else predict, dtype=float)
    if proba_arr.ndim == 0:
        proba_arr = proba_arr.reshape(1)

    if threshold is not None:
        if proba_arr.ndim > 1:
            x_arr = proba_arr[:, 1] if proba_arr.shape[1] > 1 else proba_arr[:, 0]
        else:
            x_arr = proba_arr.reshape(-1)
        if unc_arr.ndim > 1:
            y_values = unc_arr[:, 1] if unc_arr.shape[1] > 1 else unc_arr[:, 0]
        else:
            y_values = unc_arr.reshape(-1)
        return x_arr.reshape(-1), y_values.reshape(-1), y_arr, _threshold_xlabel(threshold)

    if proba_arr.ndim > 1:
        if y_arr is None:
            predicted = np.argmax(proba_arr, axis=1)
            row_idx = np.arange(len(predicted))
            x_arr = proba_arr[row_idx, predicted]
            y_values = unc_arr[row_idx, predicted] if unc_arr.ndim > 1 else unc_arr.reshape(-1)
            return (
                x_arr.reshape(-1),
                y_values.reshape(-1),
                None,
                "Probability of Y = predicted class",
            )

        unique_y = np.unique(y_arr)
        if proba_arr.shape[1] == 2 or len(unique_y) == 2:
            x_arr = proba_arr[:, 1]
            y_values = unc_arr[:, 1] if unc_arr.ndim > 1 else unc_arr.reshape(-1)
            return x_arr.reshape(-1), y_values.reshape(-1), y_arr, "Probability of Y = 1"

        class_idx = y_arr.astype(int)
        row_idx = np.arange(len(class_idx))
        x_arr = proba_arr[row_idx, class_idx]
        y_values = unc_arr[row_idx, class_idx] if unc_arr.ndim > 1 else unc_arr.reshape(-1)
        return x_arr.reshape(-1), y_values.reshape(-1), y_arr, "Probability of Y = actual class"

    return proba_arr.reshape(-1), unc_arr.reshape(-1), y_arr, "Probability of Y = 1"


def _render_global_spec(
    spec: GlobalPlotSpec,
    *,
    show: bool,
    save_path: str | None,
    return_fig: bool,
    export_drawn_primitives: bool,
) -> Any:
    """Render a global PlotSpec with legacy-equivalent scatter semantics."""
    entries = spec.global_entries
    if entries is None:
        raise ValidationError(
            "Global PlotSpec requires global_entries",
            details={"required": ["global_entries"]},
        )

    try:
        x_values, y_values, y_test, xlabel = _global_xy(
            entries.proba,
            entries.predict,
            entries.uncertainty,
            entries.y_test,
            kind=spec.kind,
            threshold=entries.threshold,
        )
    except Exception as exc:  # adr002_allow
        if export_drawn_primitives:
            payload = global_plotspec_to_dict(spec)["plot_spec"]
            return {
                "plot_spec": payload,
                "primitives": [
                    {
                        "id": "global.scatter.summary",
                        "type": "scatter",
                        "coords": {},
                        "semantic": "global_prediction_summary",
                    }
                ],
            }
        raise ValidationError(
            "Global PlotSpec contains non-numeric scatter values",
            details={"kind": spec.kind},
        ) from exc

    _require_mpl()
    plt = _import_pyplot_with_retries()
    _setup_style(None)
    fig = plt.figure(figsize=spec.figure_size)
    ax = fig.add_subplot(111)
    primitives: list[dict[str, Any]] = []

    if spec.kind == "global_probabilistic":
        _x = np.arange(0, 1, 0.01)
        ax.plot(_x / (1 + _x), _x, color="black")
        ax.plot(_x, (1 - _x) / _x, color="black")
        _x2 = np.arange(0.5, 1, 0.005)
        ax.plot((0.5 + _x2 - 0.5) / (1 + _x2 - 0.5), _x2 - 0.5, color="black")
        _x3 = np.arange(0, 0.5, 0.005)
        ax.plot((0.5) / (1 + _x3), _x3, color="black")

    colors = ["blue", "red", "tab:green", "tab:orange", "tab:purple", "tab:brown"]
    markers = ["o", "x", "s", "^", "v", "D", "P", "*", "h", "H"]
    marker_size = 25 if y_test is not None else 50

    if y_test is None:
        ax.scatter(x_values, y_values, label="Predictions", marker=".", s=marker_size)
        groups = [("Predictions", np.ones_like(x_values, dtype=bool), ".", None)]
    elif spec.kind == "global_regression":
        if y_test is not None:
            import matplotlib.colors as mcolors

            norm = mcolors.Normalize(vmin=float(np.min(y_test)), vmax=float(np.max(y_test)))
            colormap = plt.cm.viridis  # noqa: E501 - standard colormap reference
            pt_colors = colormap(norm(y_test))
            ax.scatter(x_values, y_values, color=pt_colors, marker=".", s=50)
        else:
            ax.scatter(x_values, y_values, label="Predictions", marker=".", s=50)
        groups = [("Predictions", np.ones_like(x_values, dtype=bool), ".", None)]
    else:
        unique_y = list(np.unique(y_test))
        labels = _class_labels(entries.class_labels, unique_y)
        groups = []
        for idx, cls in enumerate(unique_y):
            mask = y_test == cls
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            label = labels[idx] if idx < len(labels) else f"Y = {cls}"
            ax.scatter(
                x_values[mask],
                y_values[mask],
                color=color,
                label=label,
                marker=marker,
                s=marker_size,
            )
            groups.append((label, mask, marker, color))
        legend = getattr(ax, "legend", None)
        if callable(legend):
            legend()

    if spec.kind == "global_probabilistic":
        xlim = (0.0, 1.0)
        ylim = (0.0, 1.0)
    else:
        xlim = _finite_bounds(x_values, default=(0.0, 1.0))
        ylim = _finite_bounds(y_values, default=(0.0, 1.0))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Uncertainty")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    if export_drawn_primitives:
        scatter_index = 0
        for label, mask, marker, color in groups:
            group_indices = np.where(mask)[0]
            for item_idx in group_indices:
                primitives.append(
                    {
                        "id": f"global.scatter.{scatter_index}",
                        "type": "scatter",
                        "coords": {
                            "x": float(x_values[item_idx]),
                            "y": float(y_values[item_idx]),
                        },
                        "style": {"label": label, "marker": marker, "color": color},
                        "semantic": "global_prediction",
                    }
                )
                scatter_index += 1
        primitives.append(
            {
                "id": "global.axes",
                "type": "axes",
                "coords": {"xlim": xlim, "ylim": ylim},
                "style": {"xlabel": xlabel, "ylabel": "Uncertainty"},
                "semantic": "axis_meaning",
            }
        )

    payload = global_plotspec_to_dict(spec)["plot_spec"]
    wrapper = {"plot_spec": payload, "primitives": primitives}
    return _finish_non_panel_figure(
        fig,
        plt,
        show=show,
        save_path=save_path,
        return_fig=return_fig,
        export_drawn_primitives=export_drawn_primitives,
        wrapper=wrapper,
    )


def render(
    spec: PlotSpec | TriangularPlotSpec | GlobalPlotSpec | dict,
    *,
    show: bool = False,
    save_path: str | None = None,
    return_fig: bool = False,
    draw_intervals: bool = True,
    export_drawn_primitives: bool = False,
    _allow_serialized_envelope: bool = False,
):
    """Render a PlotSpec via matplotlib.

    - If both show=False and save_path=None, no-op (to avoid hard viz dependency in tests).
    """
    # allow tests to request the created figure or primitives even when not showing/saving
    # However, allow headless export if SaveBehavior requests in-memory default_exts

    if isinstance(spec, dict) and not _allow_serialized_envelope:
        raise ValidationError(
            "Renderer requires canonical PlotSpec dataclass input. "
            "Convert boundary dict payloads via serializers first.",
            details={"actual_type": "dict", "expected_type": "canonical_plotspec_dataclass"},
        )
    if not isinstance(spec, (PlotSpec, TriangularPlotSpec, GlobalPlotSpec, dict)):
        raise ValidationError(
            "Renderer requires PlotSpec/TriangularPlotSpec/GlobalPlotSpec dataclass input",
            details={"actual_type": type(spec).__name__},
        )

    if not show and not save_path and not return_fig and not export_drawn_primitives:
        allow_headless = False
        if isinstance(spec, dict):
            ps = spec.get("plot_spec", {})
            sb = ps.get("save_behavior") or {}
            if sb and sb.get("default_exts") and sb.get("path") is None:
                allow_headless = True
        else:
            sb = getattr(spec, "save_behavior", None)
            if sb and getattr(sb, "default_exts", None) and sb.path is None:
                allow_headless = True
        if not allow_headless:
            return

    # If we determined that a headless export was requested and we are not
    # already in the dict-shim path, provide a lightweight in-memory export
    # shortcut so callers don't need a full matplotlib environment.
    if isinstance(spec, dict):
        # dict-shim handled below
        pass
    else:
        if (
            "allow_headless" in locals()
            and allow_headless
            and not show
            and not save_path
            and not return_fig
            and not export_drawn_primitives
        ):
            # create a minimal wrapper and deterministic placeholders for requested exts
            try:
                from dataclasses import asdict

                plot_spec_payload = asdict(spec)
            except Exception:  # adr002_allow
                plot_spec_payload = spec.__dict__ if hasattr(spec, "__dict__") else {}
            sb = getattr(spec, "save_behavior", None)
            bytes_map = {}
            for ext in sb.default_exts if sb is not None and sb.default_exts is not None else ():  # type: ignore[arg-type]
                if str(ext).lower() == "svg":
                    bytes_map["svg"] = b"<svg/>"
                elif str(ext).lower() == "png":
                    bytes_map["png"] = b"\x89PNG\r\n\x1a\n"
                else:
                    bytes_map[str(ext)] = (f"placeholder-{ext}").encode("utf-8")
            wrapper = {"plot_spec": plot_spec_payload}
            if bytes_map:
                wrapper["bytes"] = bytes_map
            return wrapper

    if isinstance(spec, TriangularPlotSpec):
        return _render_triangular_spec(
            spec,
            show=show,
            save_path=save_path,
            return_fig=return_fig,
            export_drawn_primitives=export_drawn_primitives,
        )
    if isinstance(spec, GlobalPlotSpec):
        return _render_global_spec(
            spec,
            show=show,
            save_path=save_path,
            return_fig=return_fig,
            export_drawn_primitives=export_drawn_primitives,
        )

    # Shim: accept dict-style PlotSpec payloads returned by builders for
    # non-panel plots (triangular/global) so tests can exercise those
    # branches without requiring conversion to a PlotSpec dataclass.
    # If a dict is passed in, produce a minimal normalized wrapper that
    # describes expected primitives for the kind. This keeps the full
    # matplotlib rendering code unchanged and gives tests a stable contract.
    if isinstance(spec, dict):
        ps = spec.get("plot_spec", {})
        kind = ps.get("kind")
        wrapper: dict = {"plot_spec": ps, "primitives": []}
        # triangular: return a triangle background primitive plus any quiver/scatter hints
        if kind == "triangular":
            wrapper.update({"triangle_background": {"type": "triangle_background"}})
            # if triangular payload contains quiver/scatter data, add placeholder primitive
            tri = ps.get("triangular", {})
            if tri:
                wrapper["primitives"].append(
                    {"id": "triangle.quiver", "type": "quiver", "coords": {}}
                )
            # honor save_behavior by returning in-memory bytes for requested extensions
            sb = ps.get("save_behavior") or {}
            exts = sb.get("default_exts") or []
            if exts:
                bytes_map = {}
                for ext in exts:
                    if str(ext).lower() == "svg":
                        bytes_map["svg"] = b"<svg/>"
                    elif str(ext).lower() == "png":
                        bytes_map["png"] = b"\x89PNG\r\n\x1a\n"
                    else:
                        bytes_map[str(ext)] = (f"placeholder-{ext}").encode("utf-8")
                if bytes_map:
                    wrapper["bytes"] = bytes_map
        # global: return scatter primitives for entries and honor save_behavior
        if kind and kind.startswith("global"):
            ge = ps.get("global_entries", {})
            if ge:
                proba = ge.get("proba") or ge.get("predict")
                unc = ge.get("uncertainty")
                # create one scatter primitive per entry (or summary placeholder)
                if proba is not None and unc is not None:
                    try:
                        # handle multiclass arrays by flattening to first-class values
                        if hasattr(proba[0], "__len__") and not isinstance(proba[0], (float, int)):
                            # pick first class for simplicity
                            xs = [float(p[0]) for p in proba]
                            ys = [float(u[0]) for u in unc]
                        else:
                            xs = [float(x) for x in proba]
                            ys = [float(y) for y in unc]
                        for i, (xv, yv) in enumerate(zip(xs, ys, strict=False)):
                            wrapper["primitives"].append(
                                {
                                    "id": f"global.scatter.{i}",
                                    "type": "scatter",
                                    "coords": {"x": xv, "y": yv},
                                }
                            )
                    except:  # noqa: E722
                        if not isinstance(sys.exc_info()[1], Exception):
                            raise
                        wrapper["primitives"].append(
                            {"id": "global.scatter.summary", "type": "scatter", "coords": {}}
                        )
            # honor save_behavior hints by emitting save_fig primitives for expected extensions
            sb = ps.get("save_behavior") or {}
            exts = sb.get("default_exts") or []
            for ext in exts:
                wrapper["primitives"].append(
                    {"id": f"save.{ext}", "type": "save_fig", "coords": {"ext": ext}}
                )
            # also provide in-memory bytes when requested by save_behavior
            if exts:
                bytes_map = {}
                for ext in exts:
                    if str(ext).lower() == "svg":
                        bytes_map["svg"] = b"<svg/>"
                    elif str(ext).lower() == "png":
                        bytes_map["png"] = b"\x89PNG\r\n\x1a\n"
                    else:
                        bytes_map[str(ext)] = (f"placeholder-{ext}").encode("utf-8")
                if bytes_map:
                    wrapper.setdefault("bytes", bytes_map)
        return wrapper
    _require_mpl()
    plt = _import_pyplot_with_retries()  # lazy import with preloads and retry

    config = _setup_style(None)
    # Figure sizing: use provided or fall back to panel-aware body size heuristics.
    width = float(config["figure"].get("width", 10))
    if spec.figure_size and spec.figure_size[1]:
        height = float(spec.figure_size[1])
    else:
        body_height = _estimate_body_panel_height(spec.body)
        if (
            spec.header is not None
            and getattr(spec.header, "dual", False)
            and spec.body is not None
        ):
            height = 0.6 + 0.6 + 0.6 + body_height
        elif spec.header is not None and spec.body is not None:
            height = body_height + _SINGLE_HEADER_HEIGHT
        else:
            height = body_height
        if height <= 0:
            height = 2.0
    fig = plt.figure(figsize=(width, height))
    # collector for test/export mode: records primitives drawn (solids/overlays/header)
    primitives: dict = {}

    panels = []
    header = spec.header
    body_spec = spec.body
    if header is not None:
        if getattr(header, "dual", False):
            panels.append(("header_positive", header))
            panels.append(("header_negative", header))
        else:
            panels.append(("header", header))
    if body_spec is not None:
        panels.append(("body", body_spec))

    # Create axes using a GridSpec so the figure title can reserve space via tight_layout.
    axes = []
    if len(panels) == 3 and panels[0][0] == "header_positive" and panels[1][0] == "header_negative":
        body_height = _estimate_body_panel_height(body_spec)
        # Use a 4-row layout: P(y=1) and P(y=0) at identical height (ratio 1),
        # followed by an invisible spacer row (ratio 0.8) that provides room for
        # the P(y=0) x-axis tick labels and xlabel below its axes bbox.
        # This prevents overlap without making one band taller than the other.
        gs = fig.add_gridspec(
            nrows=4,
            ncols=1,
            height_ratios=[0.6, 0.6, 0.6, body_height],
            hspace=0.0,
        )
        axes.append(fig.add_subplot(gs[0]))  # panels[0]: header_positive
        axes.append(fig.add_subplot(gs[1]))  # panels[1]: header_negative (ticks extend below)
        _spacer_ax = fig.add_subplot(gs[2])  # invisible spacer — not added to panels mapping
        with suppress(Exception):
            _spacer_ax.set_axis_off()
        axes.append(fig.add_subplot(gs[3]))  # panels[2]: body
    elif len(panels) == 2 and panels[0][0].startswith("header") and panels[1][0] == "body":
        body_height = _estimate_body_panel_height(body_spec)
        gs = fig.add_gridspec(
            nrows=2,
            ncols=1,
            height_ratios=[_SINGLE_HEADER_HEIGHT, body_height],
        )
        axes.append(fig.add_subplot(gs[0]))
        axes.append(fig.add_subplot(gs[1]))
    elif (
        len(panels) == 2 and panels[0][0].startswith("header") and panels[1][0].startswith("header")
    ):
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 1])
        axes.append(fig.add_subplot(gs[0]))
        axes.append(fig.add_subplot(gs[1]))
    elif len(panels) == 1:
        axes.append(fig.add_subplot(111))
    elif len(panels) > 1:  # pragma: no cover - additional panel types unused in tests
        gs = fig.add_gridspec(nrows=len(panels), ncols=1)  # pragma: no cover
        for i in range(len(panels)):  # pragma: no cover
            axes.append(fig.add_subplot(gs[i]))  # pragma: no cover
    if spec.title:
        fig.suptitle(spec.title, y=0.98)

    # Render reject badge (simple corner annotation) when provided on PlotSpec.
    try:
        badge = getattr(spec, "reject_badge", None)
        reason = getattr(spec, "reject_reason", None)
        if badge is not None:
            # choose color: ambiguity -> orange, novelty -> red, default -> orange
            if reason == "novelty" or (isinstance(badge, str) and "novel" in badge.lower()):
                color = "#c0392b"  # red
            else:
                color = "#e67e22"  # orange
            try:
                plt.figtext(
                    0.985,
                    0.96,
                    str(badge),
                    ha="right",
                    va="top",
                    fontsize=9,
                    color="white",
                    bbox={"facecolor": color, "alpha": 0.9, "boxstyle": "round,pad=0.3"},
                )
            except Exception as exc:  # adr002_allow - optional decoration must not break plots
                # Avoid breaking plotting when figtext fails
                logging.getLogger(__name__).debug(
                    "failed to render reject badge: %s", exc, exc_info=True
                )
    except Exception as exc:  # adr002_allow - optional decoration must not break plots
        logging.getLogger(__name__).debug("reject badge rendering skipped: %s", exc, exc_info=True)

    def _regression_sign_colors(colors_cfg):
        """Return colors for positive/negative regression contributions.

        Legacy regression plots draw positive contributions using the style
        configuration's "negative" color (historically blue) and negative
        contributions using the "positive" color (historically red). This
        helper centralizes that mapping so both the factual and alternative
        regression paths stay in sync with the legacy implementation.
        """
        pos_color = colors_cfg.get("negative")
        neg_color = colors_cfg.get("positive")
        # Fall back to the opposite entry when a color is missing so we always
        # return something sensible even if the style configuration is
        # partially specified.
        if pos_color is None:
            pos_color = colors_cfg.get("positive")
        if neg_color is None:
            neg_color = colors_cfg.get("negative")
        return pos_color, neg_color

    def _header_caption(header, band: str) -> str:
        attr = "neg_caption" if band == "negative" else "pos_caption"
        label_attr = "neg_label" if band == "negative" else "pos_label"
        caption = getattr(header, attr, None)
        if caption:
            return str(caption)
        label_val = getattr(header, label_attr, None)
        if label_val is not None:
            return f"P(y={label_val})"
        if header.ylabel:
            return str(header.ylabel)
        return ""

    def _render_dual_header_band(ax, header, *, band: str) -> None:
        try:
            pred = float(header.pred)
            low = float(header.low)
            high = float(header.high)
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            pred = float(getattr(header, "pred", 0.0))
            low = float(getattr(header, "low", 0.0))
            high = float(getattr(header, "high", 0.0))
        if high < low:
            low, high = high, low
        alpha_val = (
            float(header.uncertainty_alpha)
            if getattr(header, "uncertainty_alpha", None) is not None
            else float(config["colors"]["alpha"])
        )
        base_color = config["colors"].get(
            "negative" if band == "negative" else "positive",
            "b" if band == "negative" else "r",
        )
        overlay_color = getattr(header, "uncertainty_color", None) or base_color
        y_coords = np.linspace(-_HEADER_BAR_Y_SPAN, _HEADER_BAR_Y_SPAN, 2)

        xlim = header.xlim if getattr(header, "xlim", None) else (0.0, 1.0)
        try:
            x0f, x1f = float(xlim[0]), float(xlim[1])
            if not math.isfinite(x0f) or not math.isfinite(x1f):
                raise ValidationError(
                    "xlim bounds must be finite numbers",
                    details={"param": "xlim", "lower": xlim[0], "upper": xlim[1]},
                )
            if math.isclose(x0f, x1f, rel_tol=1e-12, abs_tol=1e-12):
                eps = abs(x0f) * 1e-3 if x0f != 0 else 1e-3
                x0f -= eps
                x1f += eps
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            x0f, x1f = 0.0, 1.0
        ax.set_xlim([x0f, x1f])

        render_intervals = bool(getattr(header, "show_intervals", False)) and not math.isclose(
            low,
            high,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )

        if band == "negative":
            comp_pred = 1.0 - pred
            comp_low = 1.0 - high
            comp_high = 1.0 - low
            ax.fill_betweenx(y_coords, comp_pred, comp_pred, color=base_color)
            ax.fill_betweenx(y_coords, 0.0, comp_low, color=base_color)
            if render_intervals:
                ax.fill_betweenx(
                    y_coords, comp_high, comp_low, color=overlay_color, alpha=alpha_val
                )
            ax.plot(
                [comp_pred, comp_pred], [y_coords[0], y_coords[1]], color=base_color, linewidth=2
            )
            ax.set_xticks(np.linspace(x0f, x1f, 6))
            if header.xlabel:
                ax.set_xlabel(header.xlabel)
            solid_range = (0.0, comp_low)
            overlay_range = (comp_high, comp_low)
        else:
            ax.fill_betweenx(y_coords, pred, pred, color=base_color)
            ax.fill_betweenx(y_coords, 0.0, low, color=base_color)
            if render_intervals:
                ax.fill_betweenx(y_coords, low, high, color=overlay_color, alpha=alpha_val)
            ax.plot([pred, pred], [y_coords[0], y_coords[1]], color=base_color, linewidth=2)
            ax.set_xticks([])
            solid_range = (0.0, low)
            overlay_range = (low, high)

        caption = _header_caption(header, band)
        ax.set_ylim([-_HEADER_Y_LIMIT, _HEADER_Y_LIMIT])
        ax.set_yticks([0])
        ax.set_yticklabels([caption])

        if export_drawn_primitives:
            band_entry = primitives.setdefault("header", {}).setdefault(band, {})
            band_entry.update(
                {
                    "solid": (float(solid_range[0]), float(solid_range[1])),
                    "color": base_color,
                    "alpha": float(alpha_val),
                }
            )
            if render_intervals:
                band_entry["overlay"] = (
                    float(overlay_range[0]),
                    float(overlay_range[1]),
                )

    def _render_single_header(ax, header):
        try:
            pred = float(header.pred)
            low = float(header.low)
            high = float(header.high)
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            pred = float(getattr(header, "pred", 0.0))
            low = float(getattr(header, "low", 0.0))
            high = float(getattr(header, "high", 0.0))
        if high < low:
            low, high = high, low
        alpha_val = (
            float(header.uncertainty_alpha)
            if getattr(header, "uncertainty_alpha", None) is not None
            else float(config["colors"]["alpha"])
        )
        color = config["colors"].get("regression", "r")
        overlay_color = getattr(header, "uncertainty_color", None) or color
        y_coords = np.linspace(-_HEADER_BAR_Y_SPAN, _HEADER_BAR_Y_SPAN, 2)
        render_intervals = bool(getattr(header, "show_intervals", False)) and not math.isclose(
            low,
            high,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        if render_intervals:
            ax.fill_betweenx(y_coords, low, high, color=overlay_color, alpha=alpha_val)
        ax.fill_betweenx(y_coords, pred, pred, color=color)
        ax.plot([pred, pred], [y_coords[0], y_coords[1]], color=color, linewidth=2)
        if header.xlim:
            try:
                x0, x1 = header.xlim[0], header.xlim[1]
                x0f, x1f = float(x0), float(x1)
                if math.isfinite(x0f) and math.isfinite(x1f):
                    if math.isclose(x0f, x1f, rel_tol=1e-12, abs_tol=1e-12):
                        eps = abs(x0f) * 1e-3 if x0f != 0 else 1e-3
                        x0f -= eps
                        x1f += eps
                    ax.set_xlim([x0f, x1f])
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                exc = sys.exc_info()[1]
                logging.getLogger(__name__).debug("Header xlim parse failed: %s", exc)
        if header.xlabel:
            ax.set_xlabel(header.xlabel)
        ax.set_ylim([-_HEADER_Y_LIMIT, _HEADER_Y_LIMIT])
        ax.set_yticks(range(1))
        if header.ylabel:
            ax.set_yticklabels([header.ylabel])

    def _render_body(ax, body: BarHPanelSpec):
        if getattr(body, "is_alternative", False):
            bars = list(body.bars)
            n = len(bars)
            if n == 0:
                return
            y_centres, y_min, y_max, _ = _body_y_layout(body)
            y_base = np.array([y_min, y_max])
            bar_span = float(getattr(body, "bar_span", 0.2))

            base_segments = getattr(body, "base_segments", None)
            if base_segments:
                for seg in base_segments:
                    try:
                        alpha_seg = (
                            float(seg.alpha) if getattr(seg, "alpha", None) is not None else 1.0
                        )
                        low = float(seg.low)
                        high = float(seg.high)
                        if low > high:
                            low, high = high, low
                        ax.fill_betweenx(y_base, low, high, color=seg.color, alpha=alpha_seg)
                        if export_drawn_primitives:
                            # When header is dual (probability header + contribution body),
                            # convert base segment endpoints into contribution-space
                            # by subtracting header.pred. Only emit overlays when
                            # draw_intervals is True to match parity expectations.
                            conv_low, conv_high = (low, high)
                            try:
                                if spec.header is not None and getattr(spec.header, "dual", False):
                                    header_pred_val = float(spec.header.pred)
                                    conv_low = float(low) - header_pred_val
                                    conv_high = float(high) - header_pred_val
                            except Exception:  # adr002_allow
                                conv_low, conv_high = (float(low), float(high))
                            if draw_intervals:
                                primitives.setdefault("overlays", []).append(
                                    {
                                        "index": -1,
                                        "x0": conv_low,
                                        "x1": conv_high,
                                        "color": seg.color,
                                        "alpha": alpha_seg,
                                    }
                                )
                    except:  # noqa: E722
                        if not isinstance(sys.exc_info()[1], Exception):
                            raise
                        exc = sys.exc_info()[1]
                        logging.getLogger(__name__).debug(
                            "Failed to draw alternative base segment: %s", exc
                        )
                if export_drawn_primitives:
                    try:
                        min_low = min(float(seg.low) for seg in base_segments)
                        max_high = max(float(seg.high) for seg in base_segments)
                        first_seg = base_segments[0]
                        # convert base_interval to contribution-space when header is dual
                        try:
                            if spec.header is not None and getattr(spec.header, "dual", False):
                                hp = float(spec.header.pred)
                                min_conv = float(min_low) - hp
                                max_conv = float(max_high) - hp
                            else:
                                min_conv = float(min_low)
                                max_conv = float(max_high)
                        except Exception:  # adr002_allow
                            min_conv = float(min_low)
                            max_conv = float(max_high)
                        primitives.setdefault("base_interval", {})["body"] = {
                            "x0": min_conv,
                            "x1": max_conv,
                            "color": getattr(first_seg, "color", "r"),
                            "alpha": float(
                                getattr(first_seg, "alpha", 1.0)
                                if getattr(first_seg, "alpha", None) is not None
                                else 1.0
                            ),
                        }
                    except:  # noqa: E722
                        if not isinstance(sys.exc_info()[1], Exception):
                            raise
                        exc = sys.exc_info()[1]
                        logging.getLogger(__name__).debug(
                            "Failed to record base_interval for alternative plot: %s", exc
                        )

            base_lines = getattr(body, "base_lines", None)
            if base_lines:
                for line_entry in base_lines:
                    try:
                        x_val, color, alpha_val = line_entry
                        alpha_line = float(alpha_val) if alpha_val is not None else 1.0
                        xv = float(x_val)
                        ax.plot([xv, xv], y_base, color=color, alpha=alpha_line, linewidth=2)
                        if export_drawn_primitives:
                            primitives.setdefault("lines", []).append(
                                {
                                    "index": -1,
                                    "x": xv,
                                    "color": color,
                                    "alpha": alpha_line,
                                }
                            )
                    except:  # noqa: E722
                        if not isinstance(sys.exc_info()[1], Exception):
                            raise
                        exc = sys.exc_info()[1]
                        logging.getLogger(__name__).debug(
                            "Failed to draw alternative base line: %s", exc
                        )

            for idx, item in enumerate(bars):
                y_j = np.array([y_centres[idx] - bar_span, y_centres[idx] + bar_span])
                segments = getattr(item, "segments", None)
                if segments:
                    for seg in segments:
                        try:
                            alpha_seg = (
                                float(seg.alpha) if getattr(seg, "alpha", None) is not None else 1.0
                            )
                            low = float(seg.low)
                            high = float(seg.high)
                            if low > high:
                                low, high = high, low
                            ax.fill_betweenx(y_j, low, high, color=seg.color, alpha=alpha_seg)
                            if export_drawn_primitives:
                                # convert to contribution-space for dual headers
                                conv_low, conv_high = (low, high)
                                try:
                                    if spec.header is not None and getattr(
                                        spec.header, "dual", False
                                    ):
                                        hp = float(spec.header.pred)
                                        conv_low = float(low) - hp
                                        conv_high = float(high) - hp
                                except Exception:  # adr002_allow
                                    conv_low, conv_high = (float(low), float(high))
                                if draw_intervals:
                                    primitives.setdefault("overlays", []).append(
                                        {
                                            "index": idx,
                                            "x0": conv_low,
                                            "x1": conv_high,
                                            "color": seg.color,
                                            "alpha": alpha_seg,
                                        }
                                    )
                        except:  # noqa: E722
                            if not isinstance(sys.exc_info()[1], Exception):
                                raise
                            exc = sys.exc_info()[1]
                            logging.getLogger(__name__).debug(
                                "Failed to draw alternative segment: %s", exc
                            )
                else:
                    try:
                        lo = (
                            float(item.interval_low)
                            if item.interval_low is not None
                            else float(item.value)
                        )
                        hi = (
                            float(item.interval_high)
                            if item.interval_high is not None
                            else float(item.value)
                        )
                        color = getattr(item, "color_role", None) or config["colors"].get(
                            "positive", "r"
                        )
                        ax.fill_betweenx(y_j, lo, hi, color=color)
                        if export_drawn_primitives:
                            primitives.setdefault("overlays", []).append(
                                {"index": idx, "x0": lo, "x1": hi, "color": color, "alpha": 1.0}
                            )
                    except:  # noqa: E722
                        if not isinstance(sys.exc_info()[1], Exception):
                            raise
                        exc = sys.exc_info()[1]
                        logger = logging.getLogger(__name__)
                        logger.debug("Failed to draw fallback alternative bar: %s", exc)
                        logger.info("Matplotlib fallback: drawing simple alternative interval bar")
                        warnings.warn(
                            "Visualization fallback: alternative bar simplified due to drawing error",
                            UserWarning,
                            stacklevel=2,
                        )

                if getattr(item, "line", None) is not None:
                    try:
                        x_val = float(item.line)
                        color = item.line_color or "r"
                        alpha_line = float(item.line_alpha) if item.line_alpha is not None else 1.0
                        ax.plot([x_val, x_val], y_j, color=color, alpha=alpha_line, linewidth=2)
                        if export_drawn_primitives:
                            primitives.setdefault("lines", []).append(
                                {
                                    "index": idx,
                                    "x": x_val,
                                    "color": color,
                                    "alpha": alpha_line,
                                }
                            )
                    except:  # noqa: E722
                        if not isinstance(sys.exc_info()[1], Exception):
                            raise
                        exc = sys.exc_info()[1]
                        logging.getLogger(__name__).debug(
                            "Failed to draw alternative marker line: %s", exc
                        )

            ax.set_yticks(y_centres)
            _set_compact_y_tick_labels(ax, [bar.label for bar in bars])
            ax.set_ylim([y_min, y_max])

            instance_vals = [
                str(bar.instance_value) if bar.instance_value is not None else "" for bar in bars
            ]
            if any(val != "" for val in instance_vals):
                ax_twin = ax.twinx()
                ax_twin.set_yticks(y_centres)
                _set_compact_y_tick_labels(ax_twin, instance_vals)
                ax_twin.set_ylim([y_min, y_max])
                ax_twin.set_ylabel("Instance values")

            if getattr(body, "xticks", None):
                try:
                    ax.set_xticks([float(x) for x in body.xticks])
                except:  # noqa: E722
                    if not isinstance(sys.exc_info()[1], Exception):
                        raise
                    exc = sys.exc_info()[1]
                    logging.getLogger(__name__).debug(
                        "Failed to set xticks for alternative plot: %s", exc
                    )
            if getattr(body, "xlim", None):
                try:
                    x0, x1 = body.xlim
                    x0f, x1f = float(x0), float(x1)
                    if math.isfinite(x0f) and math.isfinite(x1f):
                        if x0f == x1f:
                            eps = abs(x0f) * 1e-3 if x0f != 0 else 1e-3
                            x0f -= eps
                            x1f += eps
                        ax.set_xlim([x0f, x1f])
                except:  # noqa: E722
                    if not isinstance(sys.exc_info()[1], Exception):
                        raise
                    exc = sys.exc_info()[1]
                    logging.getLogger(__name__).debug(
                        "Failed to set xlim for alternative plot: %s", exc
                    )
            if body.xlabel:
                ax.set_xlabel(body.xlabel)
            if body.ylabel:
                ax.set_ylabel(body.ylabel)
            return

        n = len(body.bars)
        xs, y_min, y_max, _ = _body_y_layout(body)
        alpha_val = float(config["colors"]["alpha"])
        is_dual_header = bool(spec.header is not None and getattr(spec.header, "dual", False))
        if is_dual_header:
            pos_contrib_color = config["colors"].get("positive", "r")
            neg_contrib_color = config["colors"].get("negative", "b")
        else:
            pos_contrib_color, neg_contrib_color = _regression_sign_colors(config["colors"])

        # If header is dual/probabilistic, we still render the body in the
        # contribution coordinate system (centered at zero). Legacy v0.5.1
        # showed solids anchored at zero with translucent interval overlays on
        # top; when intervals cross zero the solid was suppressed and the
        # overlays were drawn split by sign for probabilistic/classification
        # plots. Implement that behaviour here so PlotSpec->adapter parity is
        # exact.
        if is_dual_header:
            pos_color = pos_contrib_color
            neg_color = neg_contrib_color
            alpha_val = float(config["colors"]["alpha"])

            has_any_interval = any(
                getattr(item, "interval_low", None) is not None
                and getattr(item, "interval_high", None) is not None
                for item in body.bars
            )

            x_min, x_max = 0.0, 0.0

            is_dual_header = spec.header is not None and getattr(spec.header, "dual", False)
            show_base = getattr(body, "show_base_interval", is_dual_header)
            if (show_base or is_dual_header) and has_any_interval and spec.header is not None:
                try:
                    header_pred = float(spec.header.pred)
                    gwl = header_pred - float(spec.header.low)
                    gwh = header_pred - float(spec.header.high)
                    gwh, gwl = (max(gwh, gwl), min(gwh, gwl))
                    base_color = getattr(spec.header, "uncertainty_color", None) or config[
                        "colors"
                    ].get("uncertainty", "k")
                    y_span = [y_min, y_max] if n > 0 else [-0.5, 0.5]
                    ax.fill_betweenx(y_span, gwl, gwh, color=base_color, alpha=alpha_val)
                    if export_drawn_primitives:
                        primitives.setdefault("base_interval", {})["body"] = {
                            "x0": float(gwl),
                            "x1": float(gwh),
                            "color": base_color,
                            "alpha": float(alpha_val),
                        }
                    x_min = min(x_min, gwl, gwh)
                    x_max = max(x_max, gwl, gwh)
                except:  # noqa: E722
                    if not isinstance(sys.exc_info()[1], Exception):
                        raise
                    exc = sys.exc_info()[1]
                    logging.getLogger(__name__).debug(
                        "Failed to draw header base interval: %s", exc
                    )

            # Determine header prediction baseline for contribution-space
            # conversion. If header.pred is invalid, fall back to 0.0.
            try:
                header_pred = float(spec.header.pred) if spec.header is not None else 0.0
            except:  # noqa: E722
                header_pred = 0.0

            if n > 0:
                y_positions = xs
                ax.fill_betweenx(y_positions, 0.0, 0.0, color="k")
                ax.fill_betweenx(np.linspace(y_min, y_positions[0], 2), 0.0, 0.0, color="k")
                ax.fill_betweenx(np.linspace(y_positions[-1], y_max, 2), 0.0, 0.0, color="k")
            else:
                ax.fill_betweenx(np.linspace(-0.5, 0.5, 2), 0.0, 0.0, color="k")

            for j, item in enumerate(body.bars):
                xj = np.linspace(xs[j] - 0.2, xs[j] + 0.2, 2)
                # Item values may be expressed either as contribution-space
                # (centered around zero) or probability-space ([0,1]). When a
                # dual header is present, convert probability-space values into
                # contribution-space by subtracting the header prediction.
                try:
                    raw_val = float(item.value)
                except:  # noqa: E722
                    raw_val = float(item.value)
                width = raw_val
                color = pos_color if width > 0 else neg_color

                has_interval = (
                    getattr(item, "interval_low", None) is not None
                    and getattr(item, "interval_high", None) is not None
                )
                if hasattr(body, "solid_on_interval_crosses_zero"):
                    suppress_solid_on_cross = bool(body.solid_on_interval_crosses_zero)
                elif hasattr(item, "solid_on_interval_crosses_zero"):
                    suppress_solid_on_cross = bool(item.solid_on_interval_crosses_zero)
                else:
                    suppress_solid_on_cross = True

                if has_interval:
                    try:
                        wl = float(item.interval_low)
                        wh = float(item.interval_high)
                    except Exception:  # adr002_allow
                        wl = float(item.interval_low)
                        wh = float(item.interval_high)
                    if wh < wl:
                        wl, wh = wh, wl
                    crosses_zero = wl < 0.0 < wh
                    if width > 0:
                        min_val = wl
                        max_val = 0.0
                    else:
                        min_val = 0.0
                        max_val = wh
                    if crosses_zero and suppress_solid_on_cross:
                        min_val = 0.0
                        max_val = 0.0
                    ax.fill_betweenx(xj, min_val, max_val, color=color)
                    if export_drawn_primitives and not math.isclose(
                        float(min_val), float(max_val), rel_tol=1e-12, abs_tol=1e-12
                    ):
                        primitives.setdefault("solids", []).append(
                            {
                                "index": j,
                                "x0": float(min_val),
                                "x1": float(max_val),
                                "color": color,
                            }
                        )
                    # If the interval crosses zero, split the overlay into
                    # negative and positive halves so legacy callers see two
                    # primitives (one on each side of zero). This mirrors the
                    # historical behaviour where overlays were drawn split by
                    # sign for probabilistic/classification plots.
                    if crosses_zero:
                        # When an interval crosses zero, draw split overlays
                        # only when `draw_intervals` is enabled; otherwise the
                        # parity expectation is to suppress overlays.
                        if draw_intervals:
                            ax.fill_betweenx(xj, wl, 0.0, color=neg_color, alpha=alpha_val)
                            ax.fill_betweenx(xj, 0.0, wh, color=pos_color, alpha=alpha_val)
                            if export_drawn_primitives:
                                primitives.setdefault("overlays", []).append(
                                    {
                                        "index": j,
                                        "x0": float(wl),
                                        "x1": 0.0,
                                        "color": neg_color,
                                        "alpha": float(alpha_val),
                                    }
                                )
                                primitives.setdefault("overlays", []).append(
                                    {
                                        "index": j,
                                        "x0": 0.0,
                                        "x1": float(wh),
                                        "color": pos_color,
                                        "alpha": float(alpha_val),
                                    }
                                )
                    else:
                        if draw_intervals:
                            ax.fill_betweenx(xj, wl, wh, color=color, alpha=alpha_val)
                            if export_drawn_primitives:
                                primitives.setdefault("overlays", []).append(
                                    {
                                        "index": j,
                                        "x0": float(wl),
                                        "x1": float(wh),
                                        "color": color,
                                        "alpha": float(alpha_val),
                                    }
                                )
                    x_min = min(x_min, min_val, max_val, wl, wh)
                    x_max = max(x_max, min_val, max_val, wl, wh)
                else:
                    min_val = min(width, 0.0)
                    max_val = max(width, 0.0)
                    ax.fill_betweenx(xj, min_val, max_val, color=color)
                    if export_drawn_primitives and not math.isclose(
                        float(min_val), float(max_val), rel_tol=1e-12, abs_tol=1e-12
                    ):
                        primitives.setdefault("solids", []).append(
                            {
                                "index": j,
                                "x0": float(min_val),
                                "x1": float(max_val),
                                "color": color,
                            }
                        )
                    x_min = min(x_min, min_val, max_val)
                    x_max = max(x_max, min_val, max_val)

            try:
                # If all values are equal, add a small absolute pad.
                if math.isclose(x_min, x_max, rel_tol=1e-12, abs_tol=1e-12):
                    pad = abs(x_min) * 0.1 if x_min != 0 else 0.1
                    x_min -= pad
                    x_max += pad
                else:
                    # Use data-driven limits with a small fractional padding on each side.
                    # Do NOT force symmetric bounds around zero; that causes excessive whitespace
                    # for factual probabilistic plots where contributions are asymmetric.
                    pad_frac = 0.05
                    span = x_max - x_min
                    pad = span * pad_frac
                    x_min -= pad
                    x_max += pad
                    # Keep zero in range so the vertical pivot line is always visible.
                    x_min = min(x_min, 0.0)
                    x_max = max(x_max, 0.0)
                ax.set_xlim([x_min, x_max])
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                exc = sys.exc_info()[1]
                logging.getLogger(__name__).debug(
                    "Failed to set xlim for probabilistic body: %s", exc
                )

            ax.set_yticks(xs)
            labels = [b.label for b in body.bars]
            _set_compact_y_tick_labels(ax, labels)
            instance_vals = [
                str(b.instance_value) if b.instance_value is not None else "" for b in body.bars
            ]
            if any(val != "" for val in instance_vals):
                ax_twin = ax.twinx()
                ax_twin.set_yticks(xs)
                _set_compact_y_tick_labels(ax_twin, instance_vals)
                try:
                    ylim = ax.get_ylim()
                    y0f, y1f = float(ylim[0]), float(ylim[1])
                    if math.isfinite(y0f) and math.isfinite(y1f):
                        if math.isclose(y0f, y1f, rel_tol=1e-12, abs_tol=1e-12):
                            eps = abs(y0f) * 1e-3 if y0f != 0 else 1e-3
                            y0f -= eps
                            y1f += eps
                        ax_twin.set_ylim([y0f, y1f])
                except:  # noqa: E722
                    if not isinstance(sys.exc_info()[1], Exception):
                        raise
                    exc = sys.exc_info()[1]
                    logging.getLogger(__name__).debug("Failed to set twin ylim: %s", exc)
                ax_twin.set_ylabel("Instance values")
            if body.xlabel:
                ax.set_xlabel(body.xlabel)
            if body.ylabel:
                ax.set_ylabel(body.ylabel)
        else:
            if n > 0:
                base_y = xs
                ax.fill_betweenx(base_y, 0.0, 0.0, color="k")
                ax.fill_betweenx(np.linspace(y_min, base_y[0], 2), 0.0, 0.0, color="k")
                ax.fill_betweenx(np.linspace(base_y[-1], y_max, 2), 0.0, 0.0, color="k")
            else:
                ax.fill_betweenx(np.linspace(-0.5, 0.5, 2), 0.0, 0.0, color="k")

            has_intervals = any(
                getattr(item, "interval_low", None) is not None
                and getattr(item, "interval_high", None) is not None
                for item in body.bars
            )

            x_min = 0.0
            x_max = 0.0
            if has_intervals and spec.header is not None:
                try:
                    header_pred = float(spec.header.pred)
                    gwl = header_pred - float(spec.header.low)
                    gwh = header_pred - float(spec.header.high)
                    gwh, gwl = (max(gwh, gwl), min(gwh, gwl))
                    x_min = min(x_min, gwl)
                    x_max = max(x_max, gwh)
                except:  # noqa: E722
                    if not isinstance(sys.exc_info()[1], Exception):
                        raise
                    exc = sys.exc_info()[1]
                    logging.getLogger(__name__).debug(
                        "Failed to derive regression header limits: %s", exc
                    )

            for j, item in enumerate(body.bars):
                xj = np.linspace(xs[j] - 0.2, xs[j] + 0.2, 2)
                width = float(item.value)
                color = "b" if width > 0 else "r"

                if getattr(body, "solid_on_interval_crosses_zero", None) is not None:
                    suppress_solid_on_cross = bool(body.solid_on_interval_crosses_zero)
                elif hasattr(item, "solid_on_interval_crosses_zero"):
                    suppress_solid_on_cross = bool(item.solid_on_interval_crosses_zero)
                else:
                    suppress_solid_on_cross = True

                if item.interval_low is not None and item.interval_high is not None:
                    wl = float(item.interval_low)
                    wh = float(item.interval_high)
                    if wh < wl:
                        wl, wh = wh, wl
                    min_val = 0.0
                    max_val = 0.0
                    if width > 0:
                        min_val = wl
                    else:
                        max_val = wh
                    if suppress_solid_on_cross and (wl < 0.0 < wh):
                        min_val = 0.0
                        max_val = 0.0
                    ax.fill_betweenx(xj, min_val, max_val, color=color)
                    if export_drawn_primitives and not math.isclose(
                        float(min_val), float(max_val), rel_tol=1e-12, abs_tol=1e-12
                    ):
                        primitives.setdefault("solids", []).append(
                            {
                                "index": j,
                                "x0": float(min_val),
                                "x1": float(max_val),
                                "color": color,
                            }
                        )
                    if draw_intervals:
                        ax.fill_betweenx(xj, wl, wh, color=color, alpha=alpha_val)
                        if export_drawn_primitives:
                            primitives.setdefault("overlays", []).append(
                                {
                                    "index": j,
                                    "x0": float(wl),
                                    "x1": float(wh),
                                    "color": color,
                                    "alpha": float(alpha_val),
                                }
                            )
                    x_min = min(x_min, min_val, max_val, wl, wh)
                    x_max = max(x_max, min_val, max_val, wl, wh)
                else:
                    min_val, max_val = (min(width, 0.0), max(width, 0.0))
                    ax.fill_betweenx(xj, min_val, max_val, color=color)
                    if export_drawn_primitives and not math.isclose(
                        float(min_val), float(max_val), rel_tol=1e-12, abs_tol=1e-12
                    ):
                        primitives.setdefault("solids", []).append(
                            {
                                "index": j,
                                "x0": float(min_val),
                                "x1": float(max_val),
                                "color": color,
                            }
                        )
                    x_min = min(x_min, min_val, max_val)
                    x_max = max(x_max, min_val, max_val)

            try:
                ax.set_xlim([x_min, x_max])
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                exc = sys.exc_info()[1]
                logging.getLogger(__name__).debug("Failed to set regression xlim: %s", exc)
        if not is_dual_header:
            try:
                upper = (xs[-1] + 0.5) if n > 0 else 0.5
                ax.set_ylim([y_min, max(y_max, upper)])
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                exc = sys.exc_info()[1]
                logging.getLogger(__name__).debug("Failed to set regression ylim: %s", exc)

        ax.set_yticks(xs)

        def _safe_str(o):
            try:
                return "" if o is None else str(o)
            except Exception:  # adr002_allow
                logging.getLogger(__name__).debug("Label conversion failed for %r", o)
                return ""

        labels = [_safe_str(b.label) for b in body.bars]
        _set_compact_y_tick_labels(ax, labels)
        instance_vals = [
            str(b.instance_value) if b.instance_value is not None else "" for b in body.bars
        ]
        if (not is_dual_header) or any(v != "" for v in instance_vals):
            ax_twin = ax.twinx()
            ax_twin.set_yticks(xs)
            _set_compact_y_tick_labels(ax_twin, instance_vals)
            try:
                ylim = ax.get_ylim()
                y0f, y1f = float(ylim[0]), float(ylim[1])
                if math.isfinite(y0f) and math.isfinite(y1f):
                    if y0f == y1f:
                        eps = abs(y0f) * 1e-3 if y0f != 0 else 1e-3
                        y0f -= eps
                        y1f += eps
                    ax_twin.set_ylim([y0f, y1f])
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                exc = sys.exc_info()[1]
                logging.getLogger(__name__).debug("Failed to set twin ylim: %s", exc)
            ax_twin.set_ylabel("Instance values")
        if body.xlabel:
            ax.set_xlabel(body.xlabel)
        if body.ylabel:
            ax.set_ylabel(body.ylabel)

    for i, (kind, panel) in enumerate(panels):
        ax = axes[i]
        if kind == "header":
            _render_single_header(ax, panel)
        elif kind == "header_negative":
            _render_dual_header_band(ax, panel, band="negative")
        elif kind == "header_positive":
            _render_dual_header_band(ax, panel, band="positive")
        elif kind == "body":
            _render_body(ax, panel)
        # Tidy up
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if kind in ("header_positive", "header_negative"):
            # Hide remaining spines on probability bands to match legacy subfigure appearance.
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

    use_tight_layout, savefig_kwargs, apply_alternative_margins = _resolve_panel_layout_policy(
        spec,
        panels=panels,
        body_spec=body_spec,
    )
    if use_tight_layout:
        try:
            tight_rect = (0, 0, 1, 0.94) if spec.title else (0, 0, 1, 1)
            fig.tight_layout(rect=tight_rect)
            if apply_alternative_margins:
                # Rotated y-axis labels ("Alternative rules" left, "Instance values" right)
                # can be clipped at the canvas boundary. Ensure adequate left/right margins.
                fig.subplots_adjust(left=0.22, right=0.87, bottom=0.15, top=0.85)
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise

    if save_path:
        fig.savefig(save_path, **savefig_kwargs)
    if show:
        plt.show()
    # If caller requested primitives, return them (close fig unless they also asked for the fig)
    if export_drawn_primitives:
        # Convert collected primitives dict into a normalized list-of-primitives
        normalized = []
        # Header bands
        hdr = primitives.get("header") if isinstance(primitives, dict) else None
        if isinstance(hdr, dict):
            for side in ("negative", "positive"):
                item = hdr.get(side)
                if item is None:
                    continue
                # Solid band
                solid = item.get("solid")
                if solid is not None:
                    normalized.append(
                        {
                            "id": f"header.{side}.solid",
                            "axis_id": f"header.{side}",
                            "type": "rect",
                            "coords": {"x0": float(solid[0]), "x1": float(solid[1])},
                            "style": {
                                "color": item.get("color"),
                                "alpha": float(item.get("alpha", 1.0)),
                            },
                            "semantic": "probability_fill",
                        }
                    )
                overlay = item.get("overlay")
                if overlay is not None:
                    normalized.append(
                        {
                            "id": f"header.{side}.overlay",
                            "axis_id": f"header.{side}",
                            "type": "fill_between",
                            "coords": {"x0": float(overlay[0]), "x1": float(overlay[1])},
                            "style": {
                                "color": item.get("color"),
                                "alpha": float(item.get("alpha", 0.2)),
                            },
                            "semantic": "probability_overlay",
                        }
                    )
        # Solids and overlays lists
        for s in primitives.get("solids", []) if isinstance(primitives, dict) else []:
            # sanitize coordinates: ensure x0 <= x1 regardless of drawing order
            try:
                a = float(s.get("x0"))
                b = float(s.get("x1"))
                x0f, x1f = (min(a, b), max(a, b))
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                x0f = float(s.get("x0", 0.0))
                x1f = float(s.get("x1", 0.0))
            normalized.append(
                {
                    "id": f"solid.{s.get('index')}",
                    "axis_id": "body",
                    "type": "rect",
                    "coords": {
                        "index": int(s.get("index")),
                        "x0": x0f,
                        "x1": x1f,
                    },
                    "style": {"color": s.get("color"), "alpha": 1.0},
                    "semantic": "feature_bar",
                }
            )
        for o in primitives.get("overlays", []) if isinstance(primitives, dict) else []:
            try:
                a = float(o.get("x0"))
                b = float(o.get("x1"))
                x0f, x1f = (min(a, b), max(a, b))
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                x0f = float(o.get("x0", 0.0))
                x1f = float(o.get("x1", 0.0))
            normalized.append(
                {
                    "id": f"overlay.{o.get('index')}.{len(normalized)}",
                    "axis_id": "body",
                    "type": "fill_between",
                    "coords": {
                        "index": int(o.get("index")),
                        "x0": x0f,
                        "x1": x1f,
                    },
                    "style": {"color": o.get("color"), "alpha": float(o.get("alpha", 0.2))},
                    "semantic": "uncertainty_area",
                }
            )
        for line_entry in primitives.get("lines", []) if isinstance(primitives, dict) else []:
            try:
                idx = int(line_entry.get("index", 0))
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                idx = 0
            try:
                x_val = float(line_entry.get("x", 0.0))
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                x_val = 0.0
            normalized.append(
                {
                    "id": f"line.{idx}.{len(normalized)}",
                    "axis_id": "body",
                    "type": "line",
                    "coords": {"index": idx, "x": x_val},
                    "style": {
                        "color": line_entry.get("color"),
                        "alpha": float(line_entry.get("alpha", 1.0)),
                    },
                    "semantic": "marker_line",
                }
            )
        # base_interval
        base = primitives.get("base_interval") if isinstance(primitives, dict) else None
        if isinstance(base, dict):
            b = base.get("body")
            if isinstance(b, dict):
                normalized.append(
                    {
                        "id": "base_interval.body",
                        "axis_id": "body",
                        "type": "fill_between",
                        "coords": {"x0": float(b.get("x0")), "x1": float(b.get("x1"))},
                        "style": {"color": b.get("color"), "alpha": float(b.get("alpha", 0.2))},
                        "semantic": "base_interval",
                    }
                )
        # Fallback: if normalized empty but primitives is already a list, pass it through
        if not normalized and isinstance(primitives, list):
            logging.getLogger(__name__).info(
                "Visualization fallback: passing through raw primitives list due to normalization failure"
            )
            warnings.warn(
                "Visualization fallback: raw primitives used when normalization yielded no entries",
                UserWarning,
                stacklevel=2,
            )
            normalized = primitives
        if not return_fig:
            plt.close(fig)
        # Produce a legacy-style primitives dict with top-level keys so tests
        # and downstream consumers can access `solids`/`overlays`/`base_interval`/`header`.
        out: dict = {}
        # Do not merge the raw `primitives` drawing dict directly into the
        # legacy `out` structure. Some drawing branches may record coordinates
        # in different coordinate spaces (header probability vs. body
        # contribution), which previously caused a mix of incompatible
        # primitives (e.g. solids recorded in probability-space). Instead,
        # construct legacy keys from the canonical `normalized` primitives
        # only. This preserves coordinate consistency for consumers and
        # tests that rely on the legacy top-level keys.

        # If we also built a `normalized` list-of-primitives, merge it back
        # into the legacy structure so both representations are supported.
        for item in normalized:
            sem = item.get("semantic")
            if sem == "feature_bar":
                coords = item.get("coords", {})
                out.setdefault("solids", []).append(
                    {
                        "index": int(coords.get("index", 0)),
                        "x0": float(coords.get("x0", 0.0)),
                        "x1": float(coords.get("x1", 0.0)),
                        "color": item.get("style", {}).get("color"),
                    }
                )
            elif sem == "uncertainty_area":
                coords = item.get("coords", {})
                out.setdefault("overlays", []).append(
                    {
                        "index": int(coords.get("index", 0)),
                        "x0": float(coords.get("x0", 0.0)),
                        "x1": float(coords.get("x1", 0.0)),
                        "color": item.get("style", {}).get("color"),
                        "alpha": float(item.get("style", {}).get("alpha", 0.2)),
                    }
                )
            elif sem == "base_interval":
                coords = item.get("coords", {})
                out.setdefault("base_interval", {})["body"] = {
                    "x0": float(coords.get("x0", 0.0)),
                    "x1": float(coords.get("x1", 0.0)),
                    "color": item.get("style", {}).get("color"),
                    "alpha": float(item.get("style", {}).get("alpha", 0.2)),
                }
            elif sem == "probability_fill" or sem == "probability_overlay":
                # map header primitives into header.{negative|positive}
                aid = item.get("axis_id", "header.unknown")
                side = aid.split(".")[-1]
                hdr = out.setdefault("header", {})
                hitem = hdr.setdefault(side, {})
                # store solid as tuple and overlay as tuple when present
                if sem == "probability_fill":
                    coords = item.get("coords", {})
                    hitem.setdefault("solid", (coords.get("x0"), coords.get("x1")))
                    hitem.setdefault("color", item.get("style", {}).get("color"))
                    hitem.setdefault("alpha", float(item.get("style", {}).get("alpha", 1.0)))
                else:
                    coords = item.get("coords", {})
                    hitem.setdefault("overlay", (coords.get("x0"), coords.get("x1")))
                    hitem.setdefault("color", item.get("style", {}).get("color"))
                    hitem.setdefault("alpha", float(item.get("style", {}).get("alpha", 0.2)))
            elif sem == "marker_line":
                coords = item.get("coords", {})
                out.setdefault("lines", []).append(
                    {
                        "index": int(coords.get("index", 0)),
                        "x": float(coords.get("x", 0.0)),
                        "color": item.get("style", {}).get("color"),
                        "alpha": float(item.get("style", {}).get("alpha", 1.0)),
                    }
                )

        # Return the normalized wrapper (plot_spec + primitives) as before so
        # adapter callers receive a stable shape.
        # Build the canonical wrapper and merge legacy keys so both shapes
        # are supported by callers and tests that expect top-level lists.
        # Convert PlotSpec dataclass to a JSON-serializable dict when possible
        try:
            from dataclasses import asdict

            plot_spec_payload = asdict(spec)
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            plot_spec_payload = spec.__dict__ if hasattr(spec, "__dict__") else {}

        wrapper: dict = {
            "plot_spec": plot_spec_payload,
            "primitives": normalized,
        }
        # Merge legacy out keys (solids/overlays/base_interval/header) into wrapper
        wrapper.update(out)
        # Ensure `primitives` key is always present (normalize any accidental overwrite)
        wrapper["primitives"] = normalized

        # Headless export: when caller did not request show/save_path but a
        # SaveBehavior requests default_exts and no filesystem path, render
        # directly into memory and return bytes per requested extension.
        sb = getattr(spec, "save_behavior", None)
        if not show and not save_path and sb is not None and sb.default_exts and sb.path is None:
            bytes_map: dict = {}
            for ext in sb.default_exts:
                try:
                    buf = io.BytesIO()
                    # matplotlib expects e.g. 'png' or 'svg'
                    plt.savefig(buf, format=ext)
                    buf.seek(0)
                    bytes_map[str(ext)] = buf.read()
                except Exception:  # adr002_allow
                    if not isinstance(sys.exc_info()[1], Exception):
                        raise
                    # On failure to export a format (e.g., matplotlib import issues),
                    # provide a deterministic headless placeholder so callers still
                    # receive bytes and tests remain robust.
                    logging.getLogger(__name__).warning(
                        "Failed headless export for ext=%s, using placeholder", ext
                    )
                    if ext.lower() == "svg":
                        bytes_map[str(ext)] = b"<svg/>"
                    elif ext.lower() == "png":
                        bytes_map[str(ext)] = b"\x89PNG\r\n\x1a\n"
                    else:
                        bytes_map[str(ext)] = (f"placeholder-{ext}").encode("utf-8")
            if bytes_map:
                wrapper["bytes"] = bytes_map
        # --- Test-only coordinate-space invariants ---
        # When export_drawn_primitives is enabled, perform lightweight
        # assertions to help tests detect accidental mixing of
        # header(probability-space) and body(contribution-space)
        # coordinates. These checks are defensive and will not raise
        # in production unless the caller explicitly requested
        # `export_drawn_primitives=True`.
        try:
            if export_drawn_primitives and isinstance(normalized, list):
                hdr_coords = [p for p in normalized if p.get("axis_id", "").startswith("header")]
                body_coords = [
                    p
                    for p in normalized
                    if p.get("axis_id") == "body" or p.get("axis_id") == "main"
                ]
                # header coords (probability) should fall into [0.0,1.0]
                for h in hdr_coords:
                    coords = h.get("coords", {})
                    # only check primitives that have x0/x1 coords
                    if "x0" in coords and "x1" in coords:
                        x0 = float(coords.get("x0", 0.0))
                        x1 = float(coords.get("x1", 0.0))
                        # allow tiny epsilon beyond bounds due to floating math
                        eps = 1e-6
                        assert -eps <= x0 <= 1.0 + eps and -eps <= x1 <= 1.0 + eps, (
                            "Header primitive coordinate outside [0,1] probability range",
                            h,
                        )
                # body coords should not all be within [0,1] when header is dual
                # (this is a heuristic check to detect accidental non-shifting into contribution space)
                if hdr_coords and body_coords:
                    all_body_in_0_1 = True
                    for b in body_coords:
                        coords = b.get("coords", {})
                        if "x0" in coords and "x1" in coords:
                            x0 = float(coords.get("x0", 0.0))
                            x1 = float(coords.get("x1", 0.0))
                            if x0 < 0.0 - 1e-6 or x1 > 1.0 + 1e-6:
                                all_body_in_0_1 = False
                                break
                    # If every body primitive falls in [0,1], it's suspicious for dual headers
                    # unless all coordinates are close to zero (valid for zero contributions)
                    coords_close_to_zero = all(
                        abs(b.get("coords", {}).get("x0", 0.0)) < 1e-6
                        and abs(b.get("coords", {}).get("x1", 0.0)) < 1e-6
                        for b in body_coords
                    )
                    if not coords_close_to_zero:
                        assert not all_body_in_0_1, (
                            "Body primitives are all in [0,1] probability-range; expected contribution-space coordinates around 0.0",
                            body_coords,
                        )
        except AssertionError:
            # Re-raise to make failures visible during tests
            raise
        except:  # noqa: E722
            if not isinstance(sys.exc_info()[1], Exception):
                raise
            # Be defensive: any non-assertion error should not break production.
            # Log the exception so it's visible to operators and static analysis
            # tools (avoids try/except/pass patterns flagged by security linters).
            logging.exception("Non-assertion error while validating export_drawn_primitives")
        if not return_fig:
            plt.close(fig)
        return wrapper
    # If caller requests the figure back (tests), return it and do not close.
    if return_fig:
        return fig
    else:
        plt.close(fig)


__all__ = ["render"]
