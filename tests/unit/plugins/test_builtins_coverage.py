"""Unit tests for coverage of builtins.py."""

import pytest
from unittest.mock import Mock
import numpy as np
from calibrated_explanations.plugins.builtins import (
    derive_threshold_labels,
    LegacyPredictBridge,
    ValidationError,
    collection_to_batch,
    LegacyIntervalCalibratorPlugin,
)
from calibrated_explanations.plugins.intervals import IntervalCalibratorContext
from calibrated_explanations.core.exceptions import NotFittedError


def make_interval_context(task, metadata=None):
    meta = {"task": task}
    if metadata:
        meta.update(metadata)
    calibration_splits = ((np.asarray([[1.0]]), np.asarray([1.0])),)
    return IntervalCalibratorContext(
        learner=Mock(),
        calibration_splits=calibration_splits,
        bins={"calibration": "bins"},
        residuals={},
        difficulty={"estimator": Mock()},
        metadata=meta,
        fast_flags={},
    )


class TestBuiltinsCoverage:
    def test_derive_threshold_labels(self):
        """Test derive_threshold_labels heuristic."""
        # Case 1: scalar
        labels = derive_threshold_labels(0.5)
        assert labels == ("Y < 0.50", "Y ≥ 0.50")

        # Case 2: interval (sequence)
        labels = derive_threshold_labels([0.4, 0.6])
        assert labels == ("0.40 <= Y < 0.60", "Outside interval")

        # Case 3: invalid/string?
        labels = derive_threshold_labels("foo")
        assert labels == ("Target within threshold", "Outside threshold")


    def test_legacy_predict_bridge_regression_invariants(self):
        """Test invariant checks in LegacyPredictBridge."""
        mock_explainer = Mock()

        # 1. Low > High
        probs = np.array([0.5])
        low = np.array([0.6])  # > high
        high = np.array([0.4])

        mock_explainer.predict.return_value = (probs, (low, high))
        bridge = LegacyPredictBridge(mock_explainer)

        with pytest.raises(ValidationError, match="Interval invariant violated"):
            bridge.predict("X", mode="regression", task="regression")

        # 2. Prediction outside interval
        probs = np.array([0.9])  # > high
        low = np.array([0.4])
        high = np.array([0.6])

        mock_explainer.predict.return_value = (probs, (low, high))

        with pytest.raises(ValidationError, match="Prediction invariant violated"):
            bridge.predict("X", mode="regression", task="regression")

    def test_collection_to_batch(self):
        """Test collection_to_batch helper."""
        pass

    def test_legacy_interval_calibrator_plugin_create_missing_explainer(self):
        """Test LegacyIntervalCalibratorPlugin create method errors."""
        plugin = LegacyIntervalCalibratorPlugin()
        # Mock context
        context = Mock()
        context.metadata = {"task": "regression"}
        context.bins = {"calibration": None}
        context.difficulty = {"estimator": None}
        context.calibration_splits = [(None, None)]
        context.learner = None

        # When explainer is None in context.metadata
        with pytest.raises(NotFittedError):
            plugin.create(context)


