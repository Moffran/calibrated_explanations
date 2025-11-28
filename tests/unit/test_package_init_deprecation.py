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
from importlib import reload


class TestSanctionedPublicApiSymbols:
    """Verify that sanctioned symbols do NOT emit DeprecationWarning."""

    def test_calibrated_explainer_no_warning(self):
        """CalibratedExplainer is sanctioned and should not emit warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from calibrated_explanations import CalibratedExplainer
            
            # Filter for DeprecationWarnings related to CalibratedExplainer
            dep_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(dep_warnings) == 0, f"Expected no warnings for CalibratedExplainer, got: {dep_warnings}"

    def test_wrap_calibrated_explainer_no_warning(self):
        """WrapCalibratedExplainer is sanctioned and should not emit warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from calibrated_explanations import WrapCalibratedExplainer
            
            dep_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(dep_warnings) == 0, f"Expected no warnings for WrapCalibratedExplainer, got: {dep_warnings}"

    def test_transform_to_numeric_no_warning(self):
        """transform_to_numeric is sanctioned and should not emit warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from calibrated_explanations import transform_to_numeric
            
            dep_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(dep_warnings) == 0, f"Expected no warnings for transform_to_numeric, got: {dep_warnings}"


class TestDeprecatedExplanationSymbols:
    """Verify that explanation classes emit DeprecationWarning."""

    def test_alternative_explanation_deprecated(self):
        """AlternativeExplanation should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from calibrated_explanations import AlternativeExplanation
            
            dep_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(dep_warnings) == 1, f"Expected 1 deprecation warning, got {len(dep_warnings)}"
            assert "AlternativeExplanation" in str(dep_warnings[0].message)
            assert "deprecated" in str(dep_warnings[0].message).lower()

    def test_factual_explanation_deprecated(self):
        """FactualExplanation should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from calibrated_explanations import FactualExplanation
            
            dep_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(dep_warnings) == 1

    def test_fast_explanation_deprecated(self):
        """FastExplanation should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from calibrated_explanations import FastExplanation
            
            dep_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(dep_warnings) == 1

    def test_alternative_explanations_deprecated(self):
        """AlternativeExplanations should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from calibrated_explanations import AlternativeExplanations
            
            dep_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
            assert "AlternativeExplanations" in str(dep_warnings[0].message)

    def test_calibrated_explanations_deprecated(self):
        """CalibratedExplanations should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from calibrated_explanations import CalibratedExplanations
            
            dep_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
            assert "CalibratedExplanations" in str(dep_warnings[0].message)


class TestDeprecatedDiscretizerSymbols:
    """Verify that discretizer classes emit DeprecationWarning."""

    def test_entropy_discretizer_deprecated(self):
        """EntropyDiscretizer should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from calibrated_explanations import EntropyDiscretizer
            
            dep_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
            assert "EntropyDiscretizer" in str(dep_warnings[0].message)

    def test_regressor_discretizer_deprecated(self):
        """RegressorDiscretizer should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from calibrated_explanations import RegressorDiscretizer
            
            dep_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(dep_warnings) == 1

    def test_binary_entropy_discretizer_deprecated(self):
        """BinaryEntropyDiscretizer should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from calibrated_explanations import BinaryEntropyDiscretizer
            
            dep_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(dep_warnings) == 1

    def test_binary_regressor_discretizer_deprecated(self):
        """BinaryRegressorDiscretizer should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from calibrated_explanations import BinaryRegressorDiscretizer
            
            dep_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(dep_warnings) == 1


class TestDeprecatedCalibratorSymbols:
    """Verify that calibrator classes emit DeprecationWarning."""

    def test_interval_regressor_deprecated(self):
        """IntervalRegressor should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from calibrated_explanations import IntervalRegressor
            
            dep_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
            assert "IntervalRegressor" in str(dep_warnings[0].message)

    def test_venn_abers_deprecated(self):
        """VennAbers should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from calibrated_explanations import VennAbers
            
            dep_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
            assert "VennAbers" in str(dep_warnings[0].message)


class TestDeprecatedVizNamespace:
    """Verify that viz namespace import emits DeprecationWarning."""

    def test_viz_namespace_deprecated(self):
        """viz namespace should emit DeprecationWarning."""
        # Need to reload to avoid caching from other tests
        import sys
        import importlib
        
        # Clear from cache
        if 'calibrated_explanations' in sys.modules:
            ce = sys.modules['calibrated_explanations']
            if 'viz' in ce.__dict__:
                del ce.__dict__['viz']
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import calibrated_explanations
            viz_module = calibrated_explanations.viz
            
            dep_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1, f"Expected at least 1 deprecation warning, got {len(dep_warnings)}"
            assert any("viz" in str(warning.message).lower() for warning in dep_warnings)


class TestDeprecationMessageFormat:
    """Verify that deprecation messages have the correct format."""

    def test_deprecation_message_contains_migration_info(self):
        """Deprecation warnings should contain current and recommended import paths."""
        # Test with a symbol not yet imported in this session
        import sys
        import importlib
        
        ce = sys.modules.get('calibrated_explanations')
        if ce and 'RegressorDiscretizer' in ce.__dict__:
            del ce.__dict__['RegressorDiscretizer']
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import calibrated_explanations
            discretizer = calibrated_explanations.RegressorDiscretizer
            
            dep_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1, f"Expected at least 1 warning, got {len(dep_warnings)}"
            
            message = str(dep_warnings[0].message)
            assert "DEPRECATED" in message
            assert "RECOMMENDED" in message
            assert "calibrated_explanations" in message
            assert "v0.11.0" in message

    def test_deprecation_message_contains_removal_version(self):
        """Deprecation warnings should mention the removal version."""
        import sys
        
        ce = sys.modules.get('calibrated_explanations')
        if ce and 'VennAbers' in ce.__dict__:
            del ce.__dict__['VennAbers']
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import calibrated_explanations
            va = calibrated_explanations.VennAbers
            
            dep_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "v0.11.0" in str(dep_warnings[0].message)


class TestNonExistentSymbolRaisesAttributeError:
    """Verify that accessing non-existent symbols raises AttributeError."""

    def test_nonexistent_symbol_raises_attribute_error(self):
        """Accessing non-existent symbol should raise AttributeError."""
        import calibrated_explanations
        with pytest.raises(AttributeError):
            _ = calibrated_explanations.NonExistentSymbol
