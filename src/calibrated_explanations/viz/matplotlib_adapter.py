"""Matplotlib adapter for PlotSpec (ADR-007).

Renders PlotSpec structures using the existing style configuration loader and
lazy matplotlib import. This keeps plotting optional behind the 'viz' extra.
"""

from __future__ import annotations

import contextlib
import math

import numpy as np

from .._plots import _MATPLOTLIB_IMPORT_ERROR  # noqa: F401  (exported indirectly)
from .._plots import __require_matplotlib as _require_mpl  # reuse lazy guard
from .._plots import __setup_plot_style as _setup_style
from .plotspec import BarHPanelSpec, PlotSpec


def render(spec: PlotSpec, *, show: bool = False, save_path: str | None = None) -> None:
    """Render a PlotSpec via matplotlib.

    - If both show=False and save_path=None, no-op (to avoid hard viz dependency in tests).
    """
    if not show and not save_path:
        return
    _require_mpl()
    import matplotlib.pyplot as plt  # type: ignore  # lazy import

    config = _setup_style(None)
    # Figure sizing: use provided or fall back to width from config and body size heuristic
    width = float(config["figure"].get("width", 10))
    # Auto-adjust height when not explicitly provided: base + per-bar + multiline adjustment
    if spec.figure_size and spec.figure_size[1]:
        height = spec.figure_size[1]
    else:
        # estimate number of bars (body may be None)
        num_bars = 0
        max_label_lines = 1
        if spec.body is not None and getattr(spec.body, "bars", None) is not None:
            bars = spec.body.bars
            num_bars = len(bars)
            # compute maximum number of wrapped lines across labels (labels may include newlines)
            try:
                max_label_lines = max((str(b.label).count("\n") + 1) for b in bars) if bars else 1
            except Exception:
                max_label_lines = 1
        # heuristics (tunable): base height for header + body; give extra per bar and per extra label line
        base = 1.0
        per_bar = 0.35
        per_extra_line = 0.1
        height = base + per_bar * max(1, num_bars) + per_extra_line * max(0, (max_label_lines - 1))
        # clamp to sensible range
        height = float(max(3.0, min(height, 22.0)))
    fig = plt.figure(figsize=(width, height))

    panels = []
    if spec.header is not None:
        panels.append(("header", spec.header))
    if spec.body is not None:
        panels.append(("body", spec.body))

    # Create axes using a GridSpec so the figure title can reserve space via tight_layout.
    axes = []
    if len(panels) == 2 and panels[0][0] == "header" and panels[1][0] == "body":
        body_spec = panels[1][1]
        num_bars = len(body_spec.bars) if hasattr(body_spec, "bars") else 5
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, num_bars + 2])
        axes.append(fig.add_subplot(gs[0]))
        axes.append(fig.add_subplot(gs[1]))
    elif len(panels) == 1:
        axes.append(fig.add_subplot(111))
    elif len(panels) > 1:
        gs = fig.add_gridspec(nrows=len(panels), ncols=1)
        for i in range(len(panels)):
            axes.append(fig.add_subplot(gs[i]))
    if spec.title:
        fig.suptitle(spec.title, y=0.98)

    def _render_header(ax, header):
        pred, low, high = header.pred, header.low, header.high
        # layout: explicit band centers for top (negative) and bottom (positive)
        alpha_val = float(config["colors"]["alpha"])
        # tune these values to adjust vertical spacing/height of bands
        top_center = 0.75
        bot_center = 0.25
        band_thickness = 0.18
        y_top = np.linspace(top_center - band_thickness / 2.0, top_center + band_thickness / 2.0, 2)
        y_bot = np.linspace(bot_center - band_thickness / 2.0, bot_center + band_thickness / 2.0, 2)
        # small fallback x sample used for single-band header
        x = np.linspace(0, 1, 2)
        xj = np.linspace(x[0] - 0.2, x[0] + 0.2, 2)
        # Dual-mode: render two stacked bands (negative top, positive bottom) to mimic legacy probabilistic plot
        if getattr(header, "dual", False):
            # Map colors: keep config mapping stable (positive vs negative)
            pos_color = config["colors"]["positive"]
            neg_color = config["colors"]["negative"]
            # Negative band (top) - complement endpoints and conservative solid area
            comp_lo = 1.0 - high
            comp_hi = 1.0 - low
            solid_end = min(comp_lo, comp_hi)
            # solid area (conservative) and translucent uncertainty overlay
            ax.fill_betweenx(y_top, 0.0, solid_end, color=neg_color)
            ax.fill_betweenx(y_top, comp_lo, comp_hi, color=neg_color, alpha=alpha_val)
            # Negative pred marker line (drawn across band center)
            ax.plot([1.0 - pred, 1.0 - pred], [y_top[0], y_top[1]], color=neg_color, linewidth=2)
            # Positive band (bottom): solid area to low and translucent overlay low->high
            ax.fill_betweenx(y_bot, 0.0, low, color=pos_color)
            ax.fill_betweenx(y_bot, low, high, color=pos_color, alpha=alpha_val)
            ax.plot([pred, pred], [y_bot[0], y_bot[1]], color=pos_color, linewidth=2)
            # set sensible xlim/labels
            if header.xlim:
                try:
                    x0, x1 = header.xlim[0], header.xlim[1]
                    x0f, x1f = float(x0), float(x1)
                    if math.isfinite(x0f) and math.isfinite(x1f):
                        # avoid identical limits which matplotlib dislikes
                        if x0f == x1f:
                            eps = abs(x0f) * 1e-3 if x0f != 0 else 1e-3
                            x0f -= eps
                            x1f += eps
                        ax.set_xlim([x0f, x1f])
                except Exception as exc:
                    # defensive: skip setting xlim on invalid input but log the error
                    import logging

                    logging.getLogger(__name__).debug("Header xlim parse failed: %s", exc)
            if header.xlabel:
                ax.set_xlabel(header.xlabel)
            # Two y tick positions for top (neg) and bottom (pos) bands (match band centers)
            ax.set_yticks([top_center, bot_center])
            neg_lab = getattr(header, "neg_label", None)
            pos_lab = getattr(header, "pos_label", None)
            if neg_lab or pos_lab:
                lab_top = f"P(y={neg_lab})" if neg_lab else ""
                lab_bot = f"P(y={pos_lab})" if pos_lab else ""
                ax.set_yticklabels([lab_top, lab_bot])
            else:
                if header.ylabel:
                    ax.set_yticklabels([header.ylabel, ""])  # fallback
        else:
            # Single-band header (regression-like)
            color = config["colors"]["regression"]
            ax.fill_betweenx(xj, low, high, color=color, alpha=alpha_val)
            ax.fill_betweenx(xj, pred, pred, color=color)
            if header.xlim:
                try:
                    x0, x1 = header.xlim[0], header.xlim[1]
                    x0f, x1f = float(x0), float(x1)
                    if math.isfinite(x0f) and math.isfinite(x1f):
                        if x0f == x1f:
                            eps = abs(x0f) * 1e-3 if x0f != 0 else 1e-3
                            x0f -= eps
                            x1f += eps
                        ax.set_xlim([x0f, x1f])
                except Exception as exc:
                    # defensive: skip setting xlim on invalid input but log the error
                    logging.getLogger(__name__).debug("Header xlim parse failed: %s", exc)
            if header.xlabel:
                ax.set_xlabel(header.xlabel)
            ax.set_yticks(range(1))
            if header.ylabel:
                ax.set_yticklabels([header.ylabel])

    def _render_body(ax, body: BarHPanelSpec):
        n = len(body.bars)
        xs = np.linspace(0, n - 1, n)
        alpha_val = float(config["colors"]["alpha"])
        # If header carries uncertainty intervals (around a prediction), draw a grey background
        # Map header probability interval to the feature-weight axis by subtracting the prediction
        if (
            spec.header is not None
            and getattr(spec.header, "low", None) is not None
            and getattr(spec.header, "high", None) is not None
            and getattr(spec.header, "pred", None) is not None
        ):
            try:
                # allow per-PlotSpec override, otherwise use config default
                band_color = getattr(spec.header, "uncertainty_color", None) or config[
                    "colors"
                ].get("uncertainty", "#bbbbbb")
                # If header is dual/probabilistic, keep the overlay mapped to the positive-band direction
                if getattr(spec.header, "dual", False):
                    lo = float(spec.header.pred) - float(spec.header.high)
                    hi = float(spec.header.pred) - float(spec.header.low)
                else:
                    # Regression: map overlay relative to median prediction so it points in the same direction
                    lo = float(spec.header.low) - float(spec.header.pred)
                    hi = float(spec.header.high) - float(spec.header.pred)
                lo, hi = (lo, hi) if lo <= hi else (hi, lo)
                header_alpha = getattr(spec.header, "uncertainty_alpha", None)
                use_alpha = header_alpha if header_alpha is not None else min(alpha_val * 2.0, 0.7)
                ax.fill_betweenx([-0.5, n - 0.5], lo, hi, color=band_color, alpha=use_alpha)
            except Exception as exc:  # log and continue; plotting overlay is optional
                import logging

                logging.getLogger(__name__).debug("Header overlay failed: %s", exc)
        for j, item in enumerate(body.bars):
            xj = np.linspace(xs[j] - 0.2, xs[j] + 0.2, 2)
            val = float(item.value)
            # Match legacy factual regression: positive -> 'positive' color, negative -> 'negative' color
            pos = config["colors"]["positive"]
            neg = config["colors"]["negative"]
            color = pos if val > 0 else neg
            if item.interval_low is not None and item.interval_high is not None:
                wl, wh = float(item.interval_low), float(item.interval_high)
                # ensure proper ordering
                lo, hi = (wl, wh) if wl <= wh else (wh, wl)
                # Solid contribution around zero follows legacy logic:
                # - If interval crosses zero, avoid solid bar entirely.
                # - Otherwise, draw opaque bar from wl->0 (for positive val) or 0->wh (for negative val).
                if lo < 0 < hi:
                    min_val, max_val = 0.0, 0.0
                else:
                    if val > 0:
                        min_val, max_val = lo, 0.0
                    else:
                        min_val, max_val = 0.0, hi
                ax.fill_betweenx(xj, min_val, max_val, color=color)
                # Transparent interval overlay on top
                ax.fill_betweenx(xj, lo, hi, color=color, alpha=alpha_val)
            else:
                # Simple bar without interval
                min_val, max_val = (min(val, 0.0), max(val, 0.0))
                ax.fill_betweenx(xj, min_val, max_val, color=color)
        ax.set_yticks(range(n))
        labels = [b.label for b in body.bars]
        ax.set_yticklabels(labels)
        # Add right-side twin axis with instance values if present on BarItem
        instance_vals = [
            str(b.instance_value) if b.instance_value is not None else "" for b in body.bars
        ]
        if any(v != "" for v in instance_vals):
            ax_twin = ax.twinx()
            ax_twin.set_yticks(range(n))
            ax_twin.set_yticklabels(instance_vals)
            try:
                ylim = ax.get_ylim()
                y0f, y1f = float(ylim[0]), float(ylim[1])
                if math.isfinite(y0f) and math.isfinite(y1f):
                    # avoid identical limits
                    if y0f == y1f:
                        eps = abs(y0f) * 1e-3 if y0f != 0 else 1e-3
                        y0f -= eps
                        y1f += eps
                    ax_twin.set_ylim([y0f, y1f])
            except Exception as exc:
                # if ylim is invalid, skip setting twin ylim but log the error
                import logging

                logging.getLogger(__name__).debug("Failed to set twin ylim: %s", exc)
            ax_twin.set_ylabel("Instance values")
        if body.xlabel:
            ax.set_xlabel(body.xlabel)
        if body.ylabel:
            ax.set_ylabel(body.ylabel)

    for i, (kind, panel) in enumerate(panels):
        ax = axes[i]
        if kind == "header":
            _render_header(ax, panel)
        elif kind == "body":
            _render_body(ax, panel)
        # Tidy up
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Reserve room for the title at the top; keep bottom/left/right snug
    with contextlib.suppress(Exception):
        fig.tight_layout(rect=(0, 0, 1, 0.94))

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


__all__ = ["render"]
