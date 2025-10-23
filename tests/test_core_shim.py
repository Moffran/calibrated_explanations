"""Tests for the legacy :mod:`calibrated_explanations.core` shim module."""

import importlib.util
import sys
from pathlib import Path
from unittest import mock


def test_core_module_issues_deprecation_warning():
    """Importing the shim should emit a :class:`DeprecationWarning`."""

    module_name = "calibrated_explanations.core"

    # Ensure we execute the module body within the test to record coverage.
    sys.modules.pop(module_name, None)

    module_path = (
        Path(__file__).resolve().parents[1] / "src" / "calibrated_explanations" / "core.py"
    )
    spec = importlib.util.spec_from_file_location("tests.core_legacy_shim", module_path)
    module = importlib.util.module_from_spec(spec)

    with mock.patch("warnings.warn") as mock_warn:
        assert spec.loader is not None  # for type-checkers
        spec.loader.exec_module(module)

    mock_warn.assert_called_once()
    message, category = mock_warn.call_args.args[:2]
    assert "legacy module" in str(message)
    assert category is DeprecationWarning
    assert module.__file__ == str(module_path)

    # The shim re-exports names from the package. Spot check a key attribute.
    from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer

    assert module.CalibratedExplainer is CalibratedExplainer
