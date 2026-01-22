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

    def test_legacy_predict_bridge_classification(self):
        """Test LegacyPredictBridge in classification mode."""
        mock_explainer = Mock()
        # Mock predict return: (probs, (low_probs, high_probs))
        probs = np.array([[0.2, 0.8]])
        low = np.array([[0.1, 0.7]])
        high = np.array([[0.3, 0.9]])

        def predict_side_effect(*args, **kwargs):
            if kwargs.get("uq_interval"):
                return (probs, (low, high))
            if kwargs.get("calibrated"):
                return probs
            return probs  # Default

        mock_explainer.predict.side_effect = predict_side_effect
        mock_explainer.predict_proba.return_value = probs

        bridge = LegacyPredictBridge(mock_explainer)

        result = bridge.predict("X", mode="classification", task="classification")

        assert "classes" in result
        np.testing.assert_array_equal(result["classes"], probs)
        np.testing.assert_array_equal(result["predict"], probs)
        np.testing.assert_array_equal(result["low"], low)
        np.testing.assert_array_equal(result["high"], high)
        assert result["mode"] == "classification"
        assert result["task"] == "classification"

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

    def test_collection_to_batch_includes_metadata(self):
        """Ensure expanded metadata travels through the batch helper."""

        class DummyCollection:
            def __init__(self):
                self.explanations = [Mock()]
                self.mode = "classification"
                self.calibrated_explainer = "legacy"
                self.x_test = "X"
                self.y_threshold = 0.5
                self.bins = "bucket"
                self.features_to_ignore = (1,)
                self.low_high_percentiles = (1, 99)
                self.feature_filter_per_instance_ignore = ((0,),)
                self.filter_telemetry = {"enabled": True}

        collection = DummyCollection()
        batch = collection_to_batch(collection)
        assert batch.collection_metadata["mode"] == "classification"
        assert batch.collection_metadata["calibrated_explainer"] == "legacy"
        assert batch.collection_metadata["x_test"] == "X"
        assert batch.collection_metadata["filter_telemetry"] == {"enabled": True}
        assert batch.instances[0]["explanation"] == collection.explanations[0]

    def test_legacy_interval_calibrator_plugin_returns_cached(self):
        plugin = LegacyIntervalCalibratorPlugin()
        sentinel = object()
        context = make_interval_context("classification", metadata={"calibrator": sentinel})
        assert plugin.create(context) is sentinel

    def test_legacy_interval_calibrator_plugin_regression_path(self, monkeypatch):
        plugin = LegacyIntervalCalibratorPlugin()
        explainer = Mock()
        context = make_interval_context("regression", metadata={"explainer": explainer})
        context.metadata["explainer"].interval_learner = None

        called = {}

        class DummyRegressor:
            def __init__(self, explainer):
                called["explainer"] = explainer

        monkeypatch.setattr(
            "calibrated_explanations.calibration.interval_regressor.IntervalRegressor",
            DummyRegressor,
            raising=False,
        )
        calibrator = plugin.create(context)
        assert isinstance(calibrator, DummyRegressor)
        assert called["explainer"] == context.metadata["explainer"]

    def test_legacy_interval_calibrator_plugin_classification_path(self, monkeypatch):
        plugin = LegacyIntervalCalibratorPlugin()
        predict_fn = Mock(return_value=np.asarray([0.5]))
        context = make_interval_context("classification", metadata={"predict_function": predict_fn})

        called = {}

        class DummyVennAbers:
            def __init__(self, *args, **kwargs):
                called["args"] = args
                called["kwargs"] = kwargs

        monkeypatch.setattr(
            "calibrated_explanations.calibration.venn_abers.VennAbers",
            DummyVennAbers,
            raising=False,
        )
        calibrator = plugin.create(context)
        assert isinstance(calibrator, DummyVennAbers)
        assert called["args"][0] is context.calibration_splits[0][0]
        assert called["kwargs"]["predict_function"] == predict_fn
