"""Color helpers for plotting.

Extracted from legacy `_plots.py` to provide a shared helper for both the
legacy plotting functions and the MVP matplotlib adapter.
"""

from __future__ import annotations

import sys

import numpy as np


def color_brew(n: int):
    """Return a list of RGB color tuples (0-255) for n colors.

    This mirrors the small hue-generation routine used previously in
    `_plots.__color_brew` and is intentionally lightweight.
    """
    color_list = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in np.arange(5, 385, 490.0 / n).astype(int):
        h_bar = h / 60.0
        x = c * (1 - abs((h_bar % 2) - 1))
        rgb = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x), (c, x, 0)]
        r, g, b = rgb[int(h_bar)]
        rgb = [(int(255 * (r + m))), (int(255 * (g + m))), (int(255 * (b + m)))]
        color_list.append(rgb)
    color_list.reverse()
    return color_list


def get_fill_color(venn_abers: dict, reduction: float = 1.0) -> str:
    """Compute a hex color string based on a Venn-Abers-style probability dict.

    Parameters
    ----------
    venn_abers : dict
        Dictionary with key ``'predict'`` (float in [0, 1]) and optionally
        ``'low_high'`` (sequence of two floats). The function uses ``'predict'``
        to pick the winning class and to set the blend alpha.
    reduction : float, optional
        Override for the alpha used when mixing with white.

    Returns
    -------
    str
        A color string ``'#RRGGBB'``.
    """
    colors = color_brew(2)
    # determine winning class: 1 if predict >= 0.5 else 0
    try:
        pred = float(venn_abers.get("predict", 0.0))
    except:
        if not isinstance(sys.exc_info()[1], Exception):
            raise
        pred = 0.0
    winner_class = int(pred >= 0.5)
    color = colors[winner_class]

    alpha = pred if winner_class == 1 else 1.0 - pred
    # Normalize alpha into [.25, 1] range like legacy code
    try:
        alpha = ((alpha - 0.5) / (1 - 0.5)) * (1 - 0.25) + 0.25
    except:
        if not isinstance(sys.exc_info()[1], Exception):
            raise
        alpha = 0.25
    if reduction != 1:
        alpha = reduction

    alpha = float(alpha)
    color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
    return "#{:02x}{:02x}{:02x}".format(*color)
