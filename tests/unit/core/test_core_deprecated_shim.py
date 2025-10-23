"""Tests for the legacy ``calibrated_explanations.core`` shim module."""

from __future__ import annotations

import importlib
import warnings
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_core_shim():
    """Load the deprecated shim module and return it with captured warnings."""

    module_path = (
        Path(__file__).resolve().parents[3] / "src" / "calibrated_explanations" / "core.py"
    )

    spec = spec_from_file_location("_ce_core_shim_test", module_path)
    assert spec and spec.loader, "failed to create module spec for shim"
    module = module_from_spec(spec)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        spec.loader.exec_module(module)  # type: ignore[union-attr]

    return module, caught


def test_core_shim_emits_deprecation_warning():
    module, caught = _load_core_shim()
    del module  # module used implicitly to ensure execution

    deprecation_messages = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert deprecation_messages, "Expected DeprecationWarning from shim import"
    assert "legacy module 'calibrated_explanations.core' is deprecated" in deprecation_messages[0]


def test_core_shim_reexports_package_symbols():
    module, _ = _load_core_shim()

    # The shim should proxy exports from the real package module to maintain backward compatibility.
    package_module = importlib.import_module("calibrated_explanations.core")
    assert module.CalibratedExplainer is package_module.CalibratedExplainer
