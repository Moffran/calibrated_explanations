"""Canonical plotting module alias for :mod:`calibrated_explanations.plotting`."""

from __future__ import annotations

import sys as _sys

import calibrated_explanations.plotting as _plotting

_sys.modules[__name__] = _plotting
