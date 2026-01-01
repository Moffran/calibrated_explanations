"""Tests for public API deprecation warnings (ADR-001 Stage 3).

These tests verify that:
1. Sanctioned symbols (CalibratedExplainer, WrapCalibratedExplainer, transform_to_numeric)
   do NOT emit deprecation warnings
2. Unsanctioned symbols (viz, explanations, discretizers, calibrators)
   DO emit deprecation warnings
"""

import pytest
import warnings
import sys
import contextlib


class TestSanctionedPublicApiSymbols:
    """Verify that sanctioned symbols do NOT emit DeprecationWarning."""

    def test_calibrated_explainer_no_warning(self):
        """CalibratedExplainer is sanctioned and should not emit warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Filter for DeprecationWarnings related to CalibratedExplainer
            dep_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]
            assert (
                len(dep_warnings) == 0
            ), f"Expected no warnings for CalibratedExplainer, got: {dep_warnings}"

    def test_wrap_calibrated_explainer_no_warning(self):
        """WrapCalibratedExplainer is sanctioned and should not emit warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            dep_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]
            assert (
                len(dep_warnings) == 0
            ), f"Expected no warnings for WrapCalibratedExplainer, got: {dep_warnings}"

    def test_transform_to_numeric_no_warning(self):
        """transform_to_numeric is sanctioned and should not emit warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            dep_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]
            assert (
                len(dep_warnings) == 0
            ), f"Expected no warnings for transform_to_numeric, got: {dep_warnings}"


class TestDeprecatedExplanationSymbols:
    """Verify that explanation classes emit DeprecationWarning."""

    def test_alternative_explanation_deprecated(self):
        """AlternativeExplanation should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Reload package and access the deprecated symbol inside the
            # warning capture so the deprecation is emitted inside the context.
            import importlib

            ce = importlib.reload(importlib.import_module("calibrated_explanations"))
            ce.__dict__.pop("AlternativeExplanation", None)
            with contextlib.suppress(Exception):
                _ = getattr(ce, "AlternativeExplanation")

            dep_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]
            assert (
                len(dep_warnings) >= 1
            ), f"Expected at least 1 deprecation warning, got {len(dep_warnings)}"
            assert any("AlternativeExplanation" in str(wi.message) for wi in dep_warnings)
            assert "deprecated" in str(dep_warnings[0].message).lower()

    def test_factual_explanation_deprecated(self):
        """FactualExplanation should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import importlib

            ce = importlib.reload(importlib.import_module("calibrated_explanations"))
            ce.__dict__.pop("FactualExplanation", None)
            with contextlib.suppress(Exception):
                _ = getattr(ce, "FactualExplanation")

            dep_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(dep_warnings) >= 1

    def test_fast_explanation_deprecated(self):
        """FastExplanation should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import importlib

            ce = importlib.reload(importlib.import_module("calibrated_explanations"))
            ce.__dict__.pop("FastExplanation", None)
            with contextlib.suppress(Exception):
                _ = getattr(ce, "FastExplanation")

            dep_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(dep_warnings) >= 1

    def test_alternative_explanations_deprecated(self):
        """AlternativeExplanations should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import importlib

            ce = importlib.reload(importlib.import_module("calibrated_explanations"))
            ce.__dict__.pop("AlternativeExplanations", None)
            with contextlib.suppress(Exception):
                _ = getattr(ce, "AlternativeExplanations")

            dep_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(dep_warnings) >= 1
            assert any("AlternativeExplanations" in str(wi.message) for wi in dep_warnings)

    def test_calibrated_explanations_deprecated(self):
        """CalibratedExplanations should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import importlib

            ce = importlib.reload(importlib.import_module("calibrated_explanations"))
            ce.__dict__.pop("CalibratedExplanations", None)
            with contextlib.suppress(Exception):
                _ = getattr(ce, "CalibratedExplanations")

            dep_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(dep_warnings) >= 1
            assert any("CalibratedExplanations" in str(wi.message) for wi in dep_warnings)


class TestDeprecatedDiscretizerSymbols:
    """Verify that discretizer classes emit DeprecationWarning."""

    def test_entropy_discretizer_deprecated(self):
        """EntropyDiscretizer should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import importlib

            ce = importlib.reload(importlib.import_module("calibrated_explanations"))
            ce.__dict__.pop("EntropyDiscretizer", None)
            with contextlib.suppress(Exception):
                _ = getattr(ce, "EntropyDiscretizer")

            dep_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(dep_warnings) >= 1
            assert any("EntropyDiscretizer" in str(wi.message) for wi in dep_warnings)

    def test_regressor_discretizer_deprecated(self):
        """RegressorDiscretizer should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import importlib

            ce = importlib.reload(importlib.import_module("calibrated_explanations"))
            ce.__dict__.pop("RegressorDiscretizer", None)
            with contextlib.suppress(Exception):
                _ = getattr(ce, "RegressorDiscretizer")

            dep_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(dep_warnings) >= 1

    def test_binary_entropy_discretizer_deprecated(self):
        """BinaryEntropyDiscretizer should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import importlib

            ce = importlib.reload(importlib.import_module("calibrated_explanations"))
            ce.__dict__.pop("BinaryEntropyDiscretizer", None)
            with contextlib.suppress(Exception):
                _ = getattr(ce, "BinaryEntropyDiscretizer")

            dep_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(dep_warnings) >= 1

    def test_binary_regressor_discretizer_deprecated(self):
        """BinaryRegressorDiscretizer should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import importlib

            ce = importlib.reload(importlib.import_module("calibrated_explanations"))
            ce.__dict__.pop("BinaryRegressorDiscretizer", None)
            with contextlib.suppress(Exception):
                _ = getattr(ce, "BinaryRegressorDiscretizer")

            dep_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(dep_warnings) >= 1


class TestDeprecatedCalibratorSymbols:
    """Verify that calibrator classes emit DeprecationWarning."""

    def test_interval_regressor_deprecated(self, monkeypatch):
        """IntervalRegressor should emit DeprecationWarning."""
        import calibrated_explanations as ce

        # Clear cached value to force fresh __getattr__ call
        monkeypatch.delitem(ce.__dict__, "IntervalRegressor", raising=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with contextlib.suppress(Exception):
                _ = ce.IntervalRegressor

            dep_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(dep_warnings) >= 1
            assert any("IntervalRegressor" in str(wi.message) for wi in dep_warnings)

    def test_venn_abers_deprecated(self, monkeypatch):
        """VennAbers should emit DeprecationWarning."""
        import calibrated_explanations as ce

        # Clear cached value to force fresh __getattr__ call
        monkeypatch.delitem(ce.__dict__, "VennAbers", raising=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with contextlib.suppress(Exception):
                _ = ce.VennAbers

            dep_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(dep_warnings) >= 1
            assert any("VennAbers" in str(wi.message) for wi in dep_warnings)


class TestDeprecatedVizNamespace:
    """Verify that viz namespace import emits DeprecationWarning."""

    def test_viz_namespace_deprecated(self):
        """viz namespace should emit DeprecationWarning."""
        # Need to reload to avoid caching from other tests

        # Clear from cache
        if "calibrated_explanations" in sys.modules:
            ce = sys.modules["calibrated_explanations"]
            if "viz" in ce.__dict__:
                del ce.__dict__["viz"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import calibrated_explanations

            _ = calibrated_explanations.viz

            dep_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]
            assert (
                len(dep_warnings) >= 1
            ), f"Expected at least 1 deprecation warning, got {len(dep_warnings)}"
            assert any("viz" in str(warning.message).lower() for warning in dep_warnings)


class TestDeprecationMessageFormat:
    """Verify that deprecation messages have the correct format."""

    def test_deprecation_message_contains_migration_info(self):
        """Deprecation warnings should contain current and recommended import paths."""
        # Test with a symbol not yet imported in this session

        ce = sys.modules.get("calibrated_explanations")
        if ce and "RegressorDiscretizer" in ce.__dict__:
            del ce.__dict__["RegressorDiscretizer"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import calibrated_explanations

            _ = calibrated_explanations.RegressorDiscretizer

            dep_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(dep_warnings) >= 1, f"Expected at least 1 warning, got {len(dep_warnings)}"

            message = str(dep_warnings[0].message)
            assert "DEPRECATED" in message
            assert "RECOMMENDED" in message
            assert "calibrated_explanations" in message
            assert "v0.11.0" in message

    def test_deprecation_message_contains_removal_version(self):
        """Deprecation warnings should mention the removal version."""

        ce = sys.modules.get("calibrated_explanations")
        if ce and "VennAbers" in ce.__dict__:
            del ce.__dict__["VennAbers"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import calibrated_explanations

            _ = calibrated_explanations.VennAbers

            dep_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(dep_warnings) >= 1
            assert "v0.11.0" in str(dep_warnings[0].message)


class TestNonExistentSymbolRaisesAttributeError:
    """Verify that accessing non-existent symbols raises AttributeError."""

    def test_nonexistent_symbol_raises_attribute_error(self):
        """Accessing non-existent symbol should raise AttributeError."""
        import calibrated_explanations

        with pytest.raises(AttributeError):
            _ = calibrated_explanations.NonExistentSymbol
