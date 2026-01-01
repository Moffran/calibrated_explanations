"""Ensure calibration compatibility shims emit deprecation warnings."""

from __future__ import annotations

import importlib
import warnings

import pytest


@pytest.mark.parametrize(
    "module_path",
    [
        "calibrated_explanations.core.calibration.interval_learner",
        "calibrated_explanations.core.calibration.interval_regressor",
        "calibrated_explanations.core.calibration.state",
        "calibrated_explanations.core.calibration.summaries",
        "calibrated_explanations.core.calibration.venn_abers",
    ],
)
def test_calibration_shims_forward_imports(module_path: str):
    """Import shims twice to ensure warnings fire and objects are exported."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        module = importlib.reload(importlib.import_module(module_path))
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert hasattr(module, "__all__")
    for name in module.__all__:
        assert hasattr(module, name)
