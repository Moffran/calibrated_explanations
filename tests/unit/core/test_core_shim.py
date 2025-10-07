"""Tests for the deprecated ``calibrated_explanations.core`` shim module."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import calibrated_explanations

def _load_core_shim(module_name: str):
    """Load ``core.py`` under a temporary module name and return the module."""

    repo_root = Path(__file__).resolve().parents[3]
    shim_path = repo_root / "src" / "calibrated_explanations" / "core.py"
    spec = importlib.util.spec_from_file_location(module_name, shim_path)
    assert spec is not None and spec.loader is not None  # sanity check for loader resolution
    module = importlib.util.module_from_spec(spec)
    with pytest.deprecated_call():
        spec.loader.exec_module(module)
    return module


def test_core_shim_reexports_calibrated_explainer():
    """The shim should warn about deprecation while re-exporting the explainer."""

    module = _load_core_shim("tests.calibrated_explanations.core_shim")
    from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer

    assert module.CalibratedExplainer is CalibratedExplainer
"""Tests for the legacy ``calibrated_explanations.core`` shim module."""


def _load_legacy_core_module(module_name: str):
    """Load the legacy ``core.py`` module under a dedicated name."""

    module_path = Path(calibrated_explanations.__file__).with_name("core.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)

    try:
        sys.modules[module_name] = module
        with pytest.warns(DeprecationWarning):
            spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)

    return module


def test_legacy_core_shim_emits_warning_and_reexports():
    module_name = "calibrated_explanations._legacy_core_for_tests"
    legacy_module = _load_legacy_core_module(module_name)

    from calibrated_explanations.core import CalibratedExplainer

    assert legacy_module.CalibratedExplainer is CalibratedExplainer
