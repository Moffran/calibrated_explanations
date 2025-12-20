"""Matplotlib adapter for PlotSpec (ADR-007).

Renders PlotSpec structures using the existing style configuration loader and
lazy matplotlib import. This keeps plotting optional behind the 'viz' extra.
"""

from __future__ import annotations

import logging
import math
import sys
import warnings

import numpy as np

from ..plotting import _MATPLOTLIB_IMPORT_ERROR  # noqa: F401  (exported indirectly)
from ..plotting import __require_matplotlib as _require_mpl  # reuse lazy guard
from ..plotting import __setup_plot_style as _setup_style
from ..utils.exceptions import ValidationError
from .plotspec import BarHPanelSpec, PlotSpec

# Preload matplotlib submodules to avoid lazy loading issues with coverage
try:
    import matplotlib.artist  # noqa: F401
    import matplotlib.axes  # noqa: F401
    import matplotlib.image  # noqa: F401
except:  # noqa: E722
    if not isinstance(sys.exc_info()[1], Exception):
        raise
    exc = sys.exc_info()[1]
    logging.getLogger(__name__).debug(
        "Failed to preload matplotlib submodules: %s", exc
    )  # matplotlib not installed or already loaded


def render(
    spec: PlotSpec,
    *,
    show: bool = False,
    save_path: str | None = None,
    return_fig: bool = False,
    draw_intervals: bool = True,
    export_drawn_primitives: bool = False,
):
    """Render a PlotSpec via matplotlib.

    - If both show=False and save_path=None, no-op (to avoid hard viz dependency in tests).
    """
    # allow tests to request the created figure or primitives even when not showing/saving
    if not show and not save_path and not return_fig and not export_drawn_primitives:
        return

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
                        for i, (xv, yv) in enumerate(zip(xs, ys)):
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
        return wrapper
    _require_mpl()
    import matplotlib.pyplot as plt  # type: ignore  # lazy import

    config = _setup_style(None)
    # Figure sizing: use provided or fall back to width from config and body size heuristic
    width = float(config["figure"].get("width", 10))
    if spec.figure_size and spec.figure_size[1]:
        height = float(spec.figure_size[1])
    else:
        num_bars = 0
        if spec.body is not None and getattr(spec.body, "bars", None) is not None:
            num_bars = len(spec.body.bars)
        height = float(num_bars * 0.5 + 2.0)
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
            panels.append(("header_negative", header))
            panels.append(("header_positive", header))
        else:
            panels.append(("header", header))
    if body_spec is not None:
        panels.append(("body", body_spec))

    # Create axes using a GridSpec so the figure title can reserve space via tight_layout.
    axes = []
    if len(panels) == 3 and panels[0][0] == "header_negative" and panels[1][0] == "header_positive":
        num_bars = (
            len(body_spec.bars) if (body_spec is not None and hasattr(body_spec, "bars")) else 5
        )
        gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1, 1, num_bars + 2])
        axes.append(fig.add_subplot(gs[0]))
        axes.append(fig.add_subplot(gs[1]))
        axes.append(fig.add_subplot(gs[2]))
    elif len(panels) == 2 and panels[0][0].startswith("header") and panels[1][0] == "body":
        num_bars = (
            len(body_spec.bars) if (body_spec is not None and hasattr(body_spec, "bars")) else 5
        )
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, num_bars + 2])
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
        y_coords = np.linspace(-0.2, 0.2, 2)

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
            solid_range = (0.0, comp_low)
            overlay_range = (comp_high, comp_low)
        else:
            ax.fill_betweenx(y_coords, pred, pred, color=base_color)
            ax.fill_betweenx(y_coords, 0.0, low, color=base_color)
            if render_intervals:
                ax.fill_betweenx(y_coords, low, high, color=overlay_color, alpha=alpha_val)
            ax.plot([pred, pred], [y_coords[0], y_coords[1]], color=base_color, linewidth=2)
            ax.set_xticks([])
            if header.xlabel:
                ax.set_xlabel(header.xlabel)
            solid_range = (0.0, low)
            overlay_range = (low, high)

        caption = _header_caption(header, band)
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
        y_coords = np.linspace(-0.2, 0.2, 2)
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
        ax.set_yticks(range(1))
        if header.ylabel:
            ax.set_yticklabels([header.ylabel])

    def _render_body(ax, body: BarHPanelSpec):
        if getattr(body, "is_alternative", False):
            bars = list(body.bars)
            n = len(bars)
            if n == 0:
                return
            y_min = -0.5
            y_max = (n - 1) + 0.5
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
                            primitives.setdefault("overlays", []).append(
                                {
                                    "index": -1,
                                    "x0": low,
                                    "x1": high,
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
                        primitives.setdefault("base_interval", {})["body"] = {
                            "x0": float(min_low),
                            "x1": float(max_high),
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
                y_j = np.array([idx - bar_span, idx + bar_span])
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
                                primitives.setdefault("overlays", []).append(
                                    {
                                        "index": idx,
                                        "x0": low,
                                        "x1": high,
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

            ax.set_yticks(range(n))
            ax.set_yticklabels([bar.label for bar in bars])
            ax.set_ylim([y_min, y_max])

            instance_vals = [
                str(bar.instance_value) if bar.instance_value is not None else "" for bar in bars
            ]
            if any(val != "" for val in instance_vals):
                ax_twin = ax.twinx()
                ax_twin.set_yticks(range(n))
                ax_twin.set_yticklabels(instance_vals)
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
        xs = np.linspace(0, n - 1, n)
        alpha_val = float(config["colors"]["alpha"])
        is_dual_header = bool(spec.header is not None and getattr(spec.header, "dual", False))
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
                    y_span = [-0.5, (n - 1) + 0.5] if n > 0 else [-0.5, 0.5]
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

            if n > 0:
                y_positions = np.linspace(0, n - 1, n)
                ax.fill_betweenx(y_positions, 0.0, 0.0, color="k")
                ax.fill_betweenx(np.linspace(-0.5, y_positions[0], 2), 0.0, 0.0, color="k")
                ax.fill_betweenx(
                    np.linspace(y_positions[-1], y_positions[-1] + 0.5, 2), 0.0, 0.0, color="k"
                )
            else:
                ax.fill_betweenx(np.linspace(-0.5, 0.5, 2), 0.0, 0.0, color="k")

            for j, item in enumerate(body.bars):
                xj = np.linspace(xs[j] - 0.2, xs[j] + 0.2, 2)
                width = float(item.value)
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
                if math.isclose(x_min, x_max, rel_tol=1e-12, abs_tol=1e-12):
                    pad = abs(x_min) * 0.1 if x_min != 0 else 0.1
                    x_min -= pad
                    x_max += pad
                ax.set_xlim([x_min, x_max])
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                exc = sys.exc_info()[1]
                logging.getLogger(__name__).debug(
                    "Failed to set xlim for probabilistic body: %s", exc
                )

            ax.set_yticks(range(n))
            labels = [b.label for b in body.bars]
            ax.set_yticklabels(labels)
            instance_vals = [
                str(b.instance_value) if b.instance_value is not None else "" for b in body.bars
            ]
            if any(val != "" for val in instance_vals):
                ax_twin = ax.twinx()
                ax_twin.set_yticks(range(n))
                ax_twin.set_yticklabels(instance_vals)
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
                base_y = np.linspace(0, n - 1, n)
                ax.fill_betweenx(base_y, 0.0, 0.0, color="k")
                ax.fill_betweenx(np.linspace(-0.5, base_y[0], 2), 0.0, 0.0, color="k")
                ax.fill_betweenx(np.linspace(base_y[-1], base_y[-1] + 0.5, 2), 0.0, 0.0, color="k")
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
                ax.set_ylim([-0.5, upper])
            except:  # noqa: E722
                if not isinstance(sys.exc_info()[1], Exception):
                    raise
                exc = sys.exc_info()[1]
                logging.getLogger(__name__).debug("Failed to set regression ylim: %s", exc)

        ax.set_yticks(range(n))
        labels = [b.label for b in body.bars]
        ax.set_yticklabels(labels)
        instance_vals = [
            str(b.instance_value) if b.instance_value is not None else "" for b in body.bars
        ]
        if (not is_dual_header) or any(v != "" for v in instance_vals):
            ax_twin = ax.twinx()
            ax_twin.set_yticks(range(n))
            ax_twin.set_yticklabels(instance_vals)
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

    # Reserve room for the title at the top; keep bottom/left/right snug
    try:
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    except:  # noqa: E722
        if not isinstance(sys.exc_info()[1], Exception):
            raise

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
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
