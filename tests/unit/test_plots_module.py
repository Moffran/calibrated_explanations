"""Tests for lightweight utilities in :mod:`calibrated_explanations.plotting`."""

from __future__ import annotations

import warnings

# Suppress internal deprecation warning from viz.plots importing legacy plotting
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
