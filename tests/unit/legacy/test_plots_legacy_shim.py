import importlib
import sys

import pytest


@pytest.fixture
def reload_legacy_module():
    """Import the deprecated legacy plotting shim fresh for each test."""
    module_name = "calibrated_explanations.legacy._plots_legacy"
    sys.modules.pop(module_name, None)
    with pytest.warns(DeprecationWarning):
        module = importlib.import_module(module_name)
    return module


def test_shim_exports_match_plotting_module(reload_legacy_module):
    legacy_module = reload_legacy_module
    plotting = importlib.import_module("calibrated_explanations.legacy.plotting")

    exported = set(legacy_module.__all__)
    expected = {name for name in dir(plotting) if not name.startswith("__")}

    assert exported == expected


def test_reexported_objects_are_identical(reload_legacy_module):
    legacy_module = reload_legacy_module
    plotting = importlib.import_module("calibrated_explanations.legacy.plotting")

    assert legacy_module._plot_regression is plotting._plot_regression
    assert legacy_module._plot_probabilistic is plotting._plot_probabilistic
