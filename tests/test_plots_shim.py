"""Tests for the deprecated ``calibrated_explanations._plots`` shim."""

from __future__ import annotations

import importlib
import sys
import warnings


def test_plots_shim_warns_and_reexports():
    """Importing the shim emits a deprecation warning and mirrors ``plotting``."""

    canonical = importlib.import_module("calibrated_explanations.plotting")
    module_name = "calibrated_explanations._plots"

    # Ensure module body executes so the warning and re-export logic run.
    sys.modules.pop(module_name, None)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        shim = importlib.import_module(module_name)

    assert any(issubclass(item.category, DeprecationWarning) for item in captured)

    expected_exports = {name for name in dir(canonical) if not name.startswith("__")}
    assert set(shim.__all__) == expected_exports

    for name in expected_exports:
        assert getattr(shim, name) is getattr(canonical, name)
