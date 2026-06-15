"""Ensure removed calibration compatibility shims fail closed."""

from __future__ import annotations

import importlib
import sys

import pytest

from calibrated_explanations.calibration import IntervalRegressor, VennAbers


@pytest.mark.parametrize(
    "module_path",
    [
        "calibrated_explanations.core.calibration",
        "calibrated_explanations.core.calibration.interval_learner",
        "calibrated_explanations.core.calibration.interval_regressor",
        "calibrated_explanations.core.calibration.state",
        "calibrated_explanations.core.calibration.summaries",
        "calibrated_explanations.core.calibration.venn_abers",
    ],
)
def test_should_raise_import_error_when_core_calibration_shim_imported(module_path: str):
    """Removed core calibration shims should no longer import."""
    for name in list(sys.modules):
        if name == module_path or name.startswith(module_path + "."):
            sys.modules.pop(name)
    with pytest.raises((ImportError, ModuleNotFoundError)):
        importlib.import_module(module_path)


def test_canonical_calibration_imports_remain_available():
    assert IntervalRegressor.__name__ == "IntervalRegressor"
    assert VennAbers.__name__ == "VennAbers"
