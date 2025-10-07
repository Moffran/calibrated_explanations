"""Tests for the deprecated ``calibrated_explanations.core`` shim."""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path


def test_deprecated_core_module_emits_warning():
    """Executing the legacy shim should emit a ``DeprecationWarning``."""

    module_name = "calibrated_explanations.core"
    shim_path = Path(__file__).resolve().parents[2] / "src" / "calibrated_explanations" / "core.py"

    sys.modules.pop(module_name, None)

    parent = sys.modules.get("calibrated_explanations")
    if parent is not None and hasattr(parent, "core"):
        delattr(parent, "core")

    spec = importlib.util.spec_from_file_location(module_name, shim_path)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            spec.loader.exec_module(module)
        assert any(isinstance(item.message, DeprecationWarning) for item in caught)
    finally:
        sys.modules.pop(module_name, None)
