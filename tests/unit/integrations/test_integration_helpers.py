"""Tests for LIME and SHAP integration helpers."""

from __future__ import annotations

import numpy as np
from unittest.mock import MagicMock, patch


from calibrated_explanations.integrations.lime import LimeHelper
from calibrated_explanations.integrations.shap import ShapHelper


class TestLimeHelper:
    """Tests for LIME integration helper."""

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

    def test_should_disable_when_lime_import_missing(self):
        mock_explainer = MagicMock()
        helper = LimeHelper(explainer=mock_explainer)

        with patch(
            "calibrated_explanations.integrations.lime.safe_import",
            side_effect=ImportError("missing"),
        ):
            instance, explanation = helper.preload()

        assert instance is None
        assert explanation is None
        assert helper.enabled is False

    def test_should_preload_regression_mode(self):
        class FakeLimeTabularExplainer:
            def __init__(self, _x_cal, **_kwargs):
                self.calls = []

            def explain_instance(self, row, predict_fn, num_features):
                self.calls.append((row, num_features))
                return {"ok": True, "pred": predict_fn(np.array([row]))}

        mock_explainer = MagicMock()
        mock_explainer.feature_names = ["f1", "f2"]
        mock_explainer.x_cal = np.array([[0.1, 0.2], [0.2, 0.3]])
        mock_explainer.mode = "regression"
        mock_explainer.num_features = 2
        mock_explainer.learner.predict = lambda x: np.sum(np.asarray(x), axis=1)
        helper = LimeHelper(explainer=mock_explainer)

        with patch(
            "calibrated_explanations.integrations.lime.safe_import",
            return_value=FakeLimeTabularExplainer,
        ):
            instance, explanation = helper.preload()

        assert instance is not None
        assert explanation["ok"] is True
        assert helper.enabled is True


class TestShapHelper:
    """Tests for SHAP integration helper."""

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
