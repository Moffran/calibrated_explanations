"""Test configuration for matplotlib-dependent visualization tests."""

from __future__ import annotations

import contextlib
import importlib

import pytest

matplotlib = pytest.importorskip(
    "matplotlib", reason="matplotlib is required for viz unit tests"
)

_missing_reasons: list[str] = []

if not hasattr(matplotlib, "artist"):
    _missing_reasons.append("matplotlib.artist attribute missing")
else:
    try:
        importlib.import_module("matplotlib.artist")
    except Exception as exc:  # pragma: no cover - defensive guard
        _missing_reasons.append(f"matplotlib.artist import failed: {exc}")

try:
    _axes_module = importlib.import_module("matplotlib.axes")
except Exception as exc:  # pragma: no cover - defensive guard
    _missing_reasons.append(f"matplotlib.axes import failed: {exc}")
else:
    if not hasattr(_axes_module, "Axes"):
        _missing_reasons.append("matplotlib.axes.Axes missing")

if _missing_reasons:
    pytest.skip(
        "matplotlib installation lacks required primitives: "
        + ", ".join(_missing_reasons),
        allow_module_level=True,
    )

with contextlib.suppress(Exception):
    matplotlib.use("Agg", force=True)
