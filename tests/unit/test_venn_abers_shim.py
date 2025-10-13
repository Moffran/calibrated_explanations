"""Tests for the deprecated ``calibrated_explanations._venn_abers`` shim."""

from __future__ import annotations

import importlib
import sys
import warnings


def test_venn_abers_shim_emits_warning_and_reexports():
    """Importing the shim should warn and expose the modern implementation."""

    module_name = "calibrated_explanations._venn_abers"

    # Force a fresh import to make sure the warning is triggered.
    sys.modules.pop(module_name, None)
    parent = sys.modules.get("calibrated_explanations")
    if parent is not None and hasattr(parent, "_venn_abers"):
        delattr(parent, "_venn_abers")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        module = importlib.import_module(module_name)

    assert any(isinstance(item.message, DeprecationWarning) for item in caught)

    from calibrated_explanations.core.venn_abers import VennAbers

    # The shim should surface the canonical implementation.
    assert module.VennAbers is VennAbers

    # It should also be reattached to the package namespace for subsequent imports.
    assert getattr(sys.modules["calibrated_explanations"], "_venn_abers") is module
