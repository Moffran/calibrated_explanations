from __future__ import annotations

import importlib
import sys

import pytest


def test_core_calibration_import_emits_deprecation(monkeypatch):
    module_name = "calibrated_explanations.core.calibration"
    if module_name in sys.modules:
        monkeypatch.delitem(sys.modules, module_name)

    with pytest.warns(DeprecationWarning, match="deprecated"):
        importlib.import_module(module_name)
