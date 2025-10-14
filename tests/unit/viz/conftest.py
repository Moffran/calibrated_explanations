"""Test configuration for matplotlib-dependent visualization tests."""

from __future__ import annotations

# CRITICAL: Preload matplotlib submodules BEFORE pytest-cov instruments code.
# This is placed immediately after __future__ imports (which must be first).
# matplotlib 3.8+ uses lazy loading that breaks when coverage instruments __getattr__.
try:
    import matplotlib
    import matplotlib.image  # noqa: F401
    import matplotlib.axes  # noqa: F401
    import matplotlib.artist  # noqa: F401
    import matplotlib.pyplot  # noqa: F401 - Force full pyplot initialization
except Exception:  # pragma: no cover
    pass  # matplotlib not installed

import contextlib
import importlib

import pytest

matplotlib = pytest.importorskip(
    "matplotlib", reason="matplotlib is required for viz unit tests"
)

_missing_reasons: list[str] = []

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
