import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from calibrated_explanations.core.prediction.orchestrator import PredictionOrchestrator


@pytest.fixture
def mock_explainer():
    explainer = MagicMock()
    explainer.initialized = True
    explainer.is_multiclass.return_value = False
    explainer.is_fast.return_value = False
    explainer.mode = "classification"
    explainer.num_features = 10
    explainer.categorical_features = []
    explainer.bins = None
    explainer.difficulty_estimator = None
    explainer.plugin_manager.interval_plugin_hints = {}
    explainer.plugin_manager.interval_plugin_fallbacks = {}
    explainer.plugin_manager.interval_plugin_identifiers = {"default": None, "fast": None}
    explainer.plugin_manager.telemetry_interval_sources = {"default": None, "fast": None}
    explainer.plugin_manager.interval_preferred_identifier = {"default": None, "fast": None}
    explainer.plugin_manager.interval_context_metadata = {"default": {}, "fast": {}}
    explainer.plugin_manager.coerce_plugin_override.return_value = None
    explainer.plugin_manager.fast_interval_plugin_override = None
    explainer.plugin_manager.interval_plugin_override = None
    explainer.perf_cache = None
    return explainer


@pytest.fixture
def orchestrator(mock_explainer):
    return PredictionOrchestrator(mock_explainer)


def test_predict_impl_fast_binary_classification(orchestrator, mock_explainer):
    """Test _predict_impl fast path for binary classification (lines 212-215)."""
    mock_explainer.is_fast.return_value = True
    mock_explainer.mode = "classification"
    mock_explainer.is_multiclass.return_value = False

    # Mock interval_learner as a list for fast mode (one per feature)
    mock_learner = MagicMock()
    mock_learner.predict_proba.return_value = (
        np.array([[0.1, 0.9]]),  # predict
        np.array([[0.05, 0.85]]),  # low
        np.array([[0.15, 0.95]]),  # high
    )
    # Simulate list of learners
    mock_explainer.interval_learner = [mock_learner] * 11  # index by feature

    x = np.array([[1, 2]])
    predict, low, high, classes = orchestrator.predict(x, feature=0)

    assert predict[0] == 0.9
    # low is (n_samples, n_classes), so low[0] is [0.05, 0.85]
    assert np.allclose(low[0], np.array([0.05, 0.85]))

    mock_learner.predict_proba.assert_called_once()


def test_predict_impl_fast_probabilistic_regression(orchestrator, mock_explainer):
    """Test _predict_impl fast path for probabilistic regression (lines 314-317)."""
    mock_explainer.is_fast.return_value = True
    mock_explainer.mode = "regression"

    mock_learner = MagicMock()
    mock_learner.predict_probability.return_value = (
        np.array([0.5]),
        np.array([0.4]),
        np.array([0.6]),
        None,
    )
    mock_explainer.interval_learner = [mock_learner] * 11

    x = np.array([[1, 2]])
    threshold = 0.5

    result = orchestrator.predict(x, threshold=threshold, feature=0)

    assert result[0][0] == 0.5
    mock_learner.predict_probability.assert_called_once()


def test_resolve_interval_plugin_object_override(orchestrator, mock_explainer):
    """Test _resolve_interval_plugin with object override (lines 488-490)."""
    mock_plugin = MagicMock()
    mock_plugin.plugin_meta = {"name": "custom_plugin"}

    mock_explainer.plugin_manager.coerce_plugin_override.return_value = mock_plugin

    plugin, identifier = orchestrator.resolve_interval_plugin(fast=False)

    assert plugin == mock_plugin
    assert identifier == "custom_plugin"


def test_resolve_interval_plugin_missing_in_chain(orchestrator, mock_explainer):
    """Test _resolve_interval_plugin with missing plugin in chain (lines 560-565)."""
    # Setup a chain where the first one is missing, second one works
    mock_explainer.plugin_manager.interval_plugin_fallbacks = {
        "default": ["missing_plugin", "valid_plugin"]
    }

    valid_plugin_mock = MagicMock()
    valid_plugin_mock.plugin_meta = {
        "name": "valid_plugin",
        "schema_version": 1,
        "modes": ("classification",),
        "capabilities": ("interval:classification",),
    }

    with patch(
        "calibrated_explanations.core.prediction.orchestrator.find_interval_descriptor",
        return_value=None,
    ), patch(
        "calibrated_explanations.core.prediction.orchestrator.find_interval_plugin",
        return_value=None,
    ), patch(
        "calibrated_explanations.core.prediction.orchestrator.find_interval_plugin_trusted",
        side_effect=[None, valid_plugin_mock],
    ), patch(
        "calibrated_explanations.core.prediction.orchestrator.is_identifier_denied",
        return_value=False,
    ):
        # We need the second one to be found
        # The first call to find_interval_plugin_trusted returns None (for missing_plugin)
        # The second call returns a mock (for valid_plugin)

        plugin, identifier = orchestrator.resolve_interval_plugin(fast=False)

        assert identifier == "valid_plugin"


def test_obtain_interval_calibrator_fast_metadata(orchestrator, mock_explainer):
    """Test obtain_interval_calibrator fast path metadata logic (lines 700-704)."""
    mock_explainer.is_fast.return_value = True

    mock_plugin = MagicMock()
    mock_calibrator = MagicMock()
    mock_plugin.create.return_value = mock_calibrator

    with patch.object(
        orchestrator, "resolve_interval_plugin", return_value=(mock_plugin, "test_plugin")
    ):
        calibrator, identifier = orchestrator.obtain_interval_calibrator(fast=True, metadata={})

        assert calibrator == mock_calibrator
        # Check if metadata was updated correctly
        context_metadata = mock_explainer.plugin_manager.interval_context_metadata["fast"]
        assert "fast_calibrators" in context_metadata
        assert context_metadata["fast_calibrators"] == (mock_calibrator,)


def test_capture_interval_calibrators_fast_sequence(orchestrator):
    """Test _capture_interval_calibrators with sequence (lines 730)."""
    context = MagicMock()
    context.metadata = {}

    calibrators = [MagicMock(), MagicMock()]

    orchestrator.capture_interval_calibrators(context=context, calibrator=calibrators, fast=True)

    assert "fast_calibrators" in context.metadata
    assert context.metadata["fast_calibrators"] == tuple(calibrators)


def test_capture_interval_calibrators_not_dict(orchestrator):
    """Test _capture_interval_calibrators when metadata is not dict (line 723)."""
    context = MagicMock()
    context.metadata = "not a dict"

    # Should just return without error
    orchestrator.capture_interval_calibrators(context=context, calibrator=None, fast=True)
