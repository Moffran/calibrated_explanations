from __future__ import annotations

import importlib
import os
import sys

import pytest

import calibrated_explanations.core as core_module


def test_core_module_reissues_deprecation_when_not_under_pytest(monkeypatch):
    monkeypatch.setitem(sys.modules, core_module.__name__, core_module)
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    if os.getenv("CE_DEPRECATIONS") == "error":
        with pytest.raises(
            DeprecationWarning, match="legacy module 'calibrated_explanations.core' is deprecated"
        ):
            importlib.reload(core_module)
    else:
        with pytest.warns(
            DeprecationWarning, match="legacy module 'calibrated_explanations.core' is deprecated"
        ):
            importlib.reload(core_module)
