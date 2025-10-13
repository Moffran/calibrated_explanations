"""Tests for the deprecated legacy plotting shim."""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_plots_legacy_shim_emits_warning_and_reexports():
    """Importing the legacy shim should warn and surface legacy plotting helpers."""

    module_name = "calibrated_explanations._plots_legacy"
    legacy_name = "calibrated_explanations.legacy.plotting"

    # Ensure we import a fresh copy of the shim so the warning is triggered.
    sys.modules.pop(module_name, None)
    parent = sys.modules.get("calibrated_explanations")
    if parent is not None and hasattr(parent, "_plots_legacy"):
        delattr(parent, "_plots_legacy")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        module = importlib.import_module(module_name)

    assert any(isinstance(item.message, DeprecationWarning) for item in caught)

    legacy = importlib.import_module(legacy_name)
    expected_exports = tuple(sorted(name for name in dir(legacy) if not name.startswith("__")))

    assert tuple(sorted(module.__all__)) == expected_exports

    sample_exports = expected_exports[:5] if len(expected_exports) >= 5 else expected_exports
    for name in sample_exports:
        assert getattr(module, name) is getattr(legacy, name)

    # The shim should be reattached to the package namespace for future imports.
    assert getattr(sys.modules["calibrated_explanations"], "_plots_legacy") is module
