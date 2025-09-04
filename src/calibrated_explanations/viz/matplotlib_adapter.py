"""Matplotlib adapter for PlotSpec (ADR-007).

Renders PlotSpec structures using the existing style configuration loader and
lazy matplotlib import. This keeps plotting optional behind the 'viz' extra.
"""

from __future__ import annotations

import contextlib

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
    height = spec.figure_size[1] if (spec.figure_size and spec.figure_size[1]) else 4.0
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
        x = np.linspace(0, 1, 2)
        xj = np.linspace(x[0] - 0.2, x[0] + 0.2, 2)
        # Match legacy: header uses the dedicated 'regression' color with alpha overlay
        color = config["colors"]["regression"]
        alpha_val = float(config["colors"]["alpha"])
        ax.fill_betweenx(xj, low, high, color=color, alpha=alpha_val)
        ax.fill_betweenx(xj, pred, pred, color=color)
        if header.xlim:
            ax.set_xlim(list(header.xlim))
        if header.xlabel:
            ax.set_xlabel(header.xlabel)
        ax.set_yticks(range(1))
        if header.ylabel:
            ax.set_yticklabels([header.ylabel])

    def _render_body(ax, body: BarHPanelSpec):
        n = len(body.bars)
        xs = np.linspace(0, n - 1, n)
        alpha_val = float(config["colors"]["alpha"])
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
        ax.set_yticklabels([b.label for b in body.bars])
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
