from __future__ import annotations

import importlib

import calibrated_explanations.core as core_module


def test_core_module_import_has_no_legacy_deprecation_warning():
    reloaded = importlib.reload(core_module)
    assert reloaded is core_module
