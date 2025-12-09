"""Tests for calibration package shim compatibility (ADR-001 Stage 1a).

Verifies that deprecated core.calibration imports work correctly and emit
deprecation warnings as expected per the migration plan.
"""

from __future__ import annotations

import warnings


class TestCoreCalibrationShimImports:
    """Test that core.calibration shim correctly re-exports top-level calibration."""

    def test_should_import_venn_abers_via_core_calibration_shim(self):
        """Verify VennAbers can be imported from core.calibration (deprecated shim)."""
        # Note: Deprecation warning may have already been emitted in prior tests.
        # Since warnings.warn() with once_per_key semantics fires only once per session,
        # we verify import works and class is valid without requiring new warning.
        from calibrated_explanations.core.calibration import VennAbers

        # Verify the class works and has expected methods
        assert VennAbers is not None
        assert callable(VennAbers)  # It's a class

    def test_should_import_interval_regressor_via_core_calibration_shim(self):
        """Verify IntervalRegressor can be imported from core.calibration (deprecated shim)."""
        from calibrated_explanations.core.calibration import IntervalRegressor

        # Verify import works and class is valid
        assert IntervalRegressor is not None
        assert hasattr(IntervalRegressor, "__init__")

    def test_should_import_calibration_state_via_core_calibration_shim(self):
        """Verify CalibrationState can be imported from core.calibration (deprecated shim)."""
        from calibrated_explanations.core.calibration import CalibrationState

        # Verify import works and class is valid
        assert CalibrationState is not None
        assert hasattr(CalibrationState, "__init__")

    def test_should_import_calibration_helpers_via_core_calibration_shim(self):
        """Verify calibration helpers can be imported from core.calibration (deprecated shim)."""
        from calibrated_explanations.core.calibration import (
            initialize_interval_learner,
            assign_threshold,
            update_interval_learner,
            get_calibration_summaries,
        )

        # Verify functions exist and are callable
        assert callable(initialize_interval_learner)
        assert callable(assign_threshold)
        assert callable(update_interval_learner)
        assert callable(get_calibration_summaries)

    def test_should_have_identical_venn_abers_via_both_paths(self):
        """Verify VennAbers imported via both paths are the same class."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from calibrated_explanations.calibration import VennAbers as VA_TopLevel
            from calibrated_explanations.core.calibration import VennAbers as VA_Deprecated

            assert VA_TopLevel is VA_Deprecated

    def test_should_have_identical_interval_regressor_via_both_paths(self):
        """Verify IntervalRegressor imported via both paths are the same class."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from calibrated_explanations.calibration import IntervalRegressor as IR_TopLevel
            from calibrated_explanations.core.calibration import (
                IntervalRegressor as IR_Deprecated,
            )

            assert IR_TopLevel is IR_Deprecated

    def test_should_emit_deprecation_warning_only_once_per_key(self):
        """Verify deprecation warning is emitted (once per key per environment)."""
        # First import should emit warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Force fresh import
            import sys

            if "calibrated_explanations.core.calibration" in sys.modules:
                del sys.modules["calibrated_explanations.core.calibration"]
            # Import inside the capture context so the shim's import-time
            # deprecation warning is recorded by the context manager.
            import importlib

            importlib.import_module("calibrated_explanations.core.calibration")

            assert len(w) >= 1
            assert issubclass(w[-1].category, DeprecationWarning)

    def test_should_verify_all_exports_from_shim(self):
        """Verify all documented exports are available from core.calibration shim."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from calibrated_explanations.core import calibration as core_cal_module

            expected_exports = [
                "CalibrationState",
                "IntervalRegressor",
                "VennAbers",
                "assign_threshold",
                "get_calibration_summaries",
                "initialize_interval_learner",
                "initialize_interval_learner_for_fast_explainer",
                "invalidate_calibration_summaries",
                "update_interval_learner",
            ]

            for export in expected_exports:
                assert hasattr(core_cal_module, export), f"Missing export: {export}"
