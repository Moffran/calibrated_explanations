"""Tests for LIME and SHAP integration helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from calibrated_explanations.integrations.lime import LimeHelper
from calibrated_explanations.integrations.shap import ShapHelper


class TestLimeHelper:
    """Tests for LIME integration helper."""

    def test_should_initialize_disabled(self):
        """LimeHelper should initialize with _enabled=False."""
        mock_explainer = MagicMock()
        helper = LimeHelper(explainer=mock_explainer)

        assert helper.is_enabled() is False
        assert helper.explainer_instance is None
        assert helper.reference_explanation is None

    def test_should_set_enabled_flag(self):
        """set_enabled() should update the _enabled flag."""
        mock_explainer = MagicMock()
        helper = LimeHelper(explainer=mock_explainer)

        assert helper.is_enabled() is False
        helper.set_enabled(True)
        assert helper.is_enabled() is True

    def test_should_reset_on_disable(self):
        """set_enabled(False) should call reset()."""
        mock_explainer = MagicMock()
        helper = LimeHelper(explainer=mock_explainer)

        # Manually set some state
        helper.enabled = True
        helper.explainer_instance = MagicMock()
        helper.reference_explanation = MagicMock()

        # Disable
        helper.set_enabled(False)

        # State should be cleared
        assert helper.is_enabled() is False
        assert helper.explainer_instance is None
        assert helper.reference_explanation is None

    def test_should_reset_clears_state(self):
        """reset() should clear all cached state."""
        mock_explainer = MagicMock()
        helper = LimeHelper(explainer=mock_explainer)

        # Set some state
        helper.enabled = True
        helper.explainer_instance = MagicMock()
        helper.reference_explanation = MagicMock()

        # Call reset
        helper.reset()

        # Verify state cleared
        assert helper.enabled is False
        assert helper.explainer_instance is None
        assert helper.reference_explanation is None

    def test_should_return_none_when_lime_unavailable(self):
        """preload() should return (None, None) when lime is not installed."""
        mock_explainer = MagicMock()
        helper = LimeHelper(explainer=mock_explainer)

        with patch("calibrated_explanations.integrations.lime.safe_import", return_value=None):
            instance, explanation = helper.preload()

            assert instance is None
            assert explanation is None

    def test_should_expose_cached_explainer_instance(self):
        """explainer_instance property should return cached instance."""
        mock_explainer = MagicMock()
        helper = LimeHelper(explainer=mock_explainer)

        # Preload with unavailable LIME (returns None)
        with patch("calibrated_explanations.integrations.lime.safe_import", return_value=None):
            instance = helper.explainer_instance

            assert instance is None

    def test_should_expose_cached_reference_explanation(self):
        """reference_explanation property should return cached explanation."""
        mock_explainer = MagicMock()
        helper = LimeHelper(explainer=mock_explainer)

        # Preload with unavailable LIME (returns None)
        with patch("calibrated_explanations.integrations.lime.safe_import", return_value=None):
            explanation = helper.reference_explanation

            assert explanation is None

    def test_should_not_preload_when_already_enabled(self):
        """preload() should skip initialization if already enabled."""
        mock_explainer = MagicMock()
        helper = LimeHelper(explainer=mock_explainer)

        # Mark as already enabled with a cached instance
        helper.enabled = True
        mock_instance = MagicMock()
        helper.explainer_instance = mock_instance
        mock_explanation = MagicMock()
        helper.reference_explanation = mock_explanation

        # Call preload - should not reinitialize
        instance, explanation = helper.preload()

        assert instance is mock_instance
        assert explanation is mock_explanation


class TestShapHelper:
    """Tests for SHAP integration helper."""

    def test_should_initialize_disabled(self):
        """ShapHelper should initialize with _enabled=False."""
        mock_explainer = MagicMock()
        helper = ShapHelper(explainer=mock_explainer)

        assert helper.is_enabled() is False
        assert helper.explainer_instance is None
        assert helper.reference_explanation is None

    def test_should_set_enabled_flag(self):
        """set_enabled() should update the _enabled flag."""
        mock_explainer = MagicMock()
        helper = ShapHelper(explainer=mock_explainer)

        assert helper.is_enabled() is False
        helper.set_enabled(True)
        assert helper.is_enabled() is True

    def test_should_reset_on_disable(self):
        """set_enabled(False) should call reset()."""
        mock_explainer = MagicMock()
        helper = ShapHelper(explainer=mock_explainer)

        # Manually set some state
        helper.enabled = True
        helper.explainer_instance = MagicMock()
        helper.reference_explanation = MagicMock()

        # Disable
        helper.set_enabled(False)

        # State should be cleared
        assert helper.is_enabled() is False
        assert helper.explainer_instance is None
        assert helper.reference_explanation is None

    def test_should_reset_clears_state(self):
        """reset() should clear all cached state."""
        mock_explainer = MagicMock()
        helper = ShapHelper(explainer=mock_explainer)

        # Set some state
        helper.enabled = True
        helper.explainer_instance = MagicMock()
        helper.reference_explanation = MagicMock()

        # Call reset
        helper.reset()

        # Verify state cleared
        assert helper.enabled is False
        assert helper.explainer_instance is None
        assert helper.reference_explanation is None

    def test_should_return_none_when_shap_unavailable(self):
        """preload() should return (None, None) when shap is not installed."""
        mock_explainer = MagicMock()
        helper = ShapHelper(explainer=mock_explainer)

        with patch("calibrated_explanations.integrations.shap.safe_import", return_value=None):
            instance, explanation = helper.preload()

            assert instance is None
            assert explanation is None

    def test_should_expose_cached_explainer_instance(self):
        """explainer_instance property should return cached instance."""
        mock_explainer = MagicMock()
        helper = ShapHelper(explainer=mock_explainer)

        # Preload with unavailable SHAP (returns None)
        with patch("calibrated_explanations.integrations.shap.safe_import", return_value=None):
            instance = helper.explainer_instance

            assert instance is None

    def test_should_expose_cached_reference_explanation(self):
        """reference_explanation property should return cached explanation."""
        mock_explainer = MagicMock()
        helper = ShapHelper(explainer=mock_explainer)

        # Preload with unavailable SHAP (returns None)
        with patch("calibrated_explanations.integrations.shap.safe_import", return_value=None):
            explanation = helper.reference_explanation

            assert explanation is None

    def test_should_not_preload_when_already_enabled(self):
        """preload() should skip initialization if already enabled."""
        mock_explainer = MagicMock()
        helper = ShapHelper(explainer=mock_explainer)

        # Mark as already enabled with a cached instance
        helper.enabled = True
        mock_instance = MagicMock()
        helper.explainer_instance = mock_instance
        mock_explanation = MagicMock()
        helper.reference_explanation = mock_explanation

        # Call preload - should not reinitialize
        instance, explanation = helper.preload()

        assert instance is mock_instance
        assert explanation is mock_explanation
