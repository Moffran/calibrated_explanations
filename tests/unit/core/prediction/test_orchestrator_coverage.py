import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from calibrated_explanations.core.prediction.orchestrator import PredictionOrchestrator
from calibrated_explanations.utils.exceptions import (
    DataShapeError,
)


@pytest.fixture
def mock_explainer():
    explainer = MagicMock()
    explainer.plugin_manager = MagicMock()
    explainer.perf_cache = None
    explainer.mode = "classification"
    explainer.is_multiclass.return_value = False
    explainer.is_fast.return_value = False
    # Use a simple attribute for initialized to allow toggling in tests
    explainer.initialized = True
    return explainer


@pytest.fixture
def orchestrator(mock_explainer):
    with patch(
        "calibrated_explanations.core.prediction.interval_registry.IntervalRegistry"
    ) as mock_registry:
        orchestrator = PredictionOrchestrator(mock_explainer)
        orchestrator.interval_registry = mock_registry.return_value
        return orchestrator


def test_initialize_chains(orchestrator, mock_explainer):
    orchestrator.initialize_chains()
    mock_explainer.plugin_manager.initialize_chains.assert_called_once()


def test_predict_caching(orchestrator, mock_explainer):
    x = np.array([[1, 2]])
    mock_cache = MagicMock()
    mock_cache.enabled = True
    mock_cache.get.return_value = "cached_result"
    mock_explainer.perf_cache = mock_cache

    result = orchestrator.predict(x)
    assert result == "cached_result"
    mock_cache.get.assert_called_once()


def test_predict_caching_miss(orchestrator, mock_explainer):
    x = np.array([[1, 2]])
    mock_cache = MagicMock()
    mock_cache.enabled = True
    mock_cache.get.return_value = None
    mock_explainer.perf_cache = mock_cache

    with patch.object(orchestrator, "predict_impl") as mock_impl:
        mock_impl.return_value = (np.array([0.5]), np.array([0.4]), np.array([0.6]), None)

        result = orchestrator.predict(x)

        mock_cache.get.assert_called_once()
        mock_impl.assert_called_once()
        mock_cache.set.assert_called_once()
        assert result == mock_impl.return_value


def testpredict_impl_regression_crepes_error(orchestrator, mock_explainer, enable_fallbacks):
    with patch(
        "calibrated_explanations.core.prediction.interval_registry.IntervalRegistry"
    ) as mock_registry:
        orchestrator = PredictionOrchestrator(mock_explainer)
        orchestrator.interval_registry = mock_registry.return_value
    mock_explainer.mode = "regression"
    mock_explainer.is_fast.return_value = False
    mock_explainer.suppress_crepes_errors = True

    mock_learner = MagicMock()
    mock_learner.predict_uncertainty.side_effect = ValueError("crepes error")
    mock_explainer.interval_learner = mock_learner

    x = np.array([[1, 2]])
    with pytest.warns(UserWarning, match="crepes produced an unexpected result"):
        predict, low, high, classes = orchestrator.predict(x)

    assert np.allclose(predict, [0])
    assert np.allclose(low, [0])
    assert np.allclose(high, [0])


def testpredict_impl_regression_crepes_error_reraise(orchestrator, mock_explainer):
    mock_explainer.mode = "regression"
    mock_explainer.is_fast.return_value = False
    mock_explainer.suppress_crepes_errors = False

    mock_learner = MagicMock()
    mock_learner.predict_uncertainty.side_effect = ValueError("crepes error")
    mock_explainer.interval_learner = mock_learner

    x = np.array([[1, 2]])
    with pytest.raises(ValueError, match="crepes error"):
        orchestrator.predict_impl(x)


def testpredict_impl_regression_probabilistic_crepes_error_suppress(
    orchestrator, mock_explainer, enable_fallbacks
):
    with patch(
        "calibrated_explanations.core.prediction.interval_registry.IntervalRegistry"
    ) as mock_registry:
        orchestrator = PredictionOrchestrator(mock_explainer)
        orchestrator.interval_registry = mock_registry.return_value
    mock_explainer.mode = "regression"
    mock_explainer.is_fast.return_value = False
    mock_explainer.suppress_crepes_errors = True

    mock_learner = MagicMock()
    mock_learner.predict_probability.side_effect = ValueError("crepes error")
    mock_explainer.interval_learner = mock_learner

    x = np.array([[1, 2]])
    with pytest.warns(UserWarning, match="crepes produced an unexpected result"):
        predict, low, high, classes = orchestrator.predict_impl(x, threshold=0.5)

    assert np.allclose(predict, [0])


def testpredict_impl_regression_probabilistic_crepes_error_reraise(orchestrator, mock_explainer):
    mock_explainer.mode = "regression"
    mock_explainer.is_fast.return_value = False
    mock_explainer.suppress_crepes_errors = False

    mock_learner = MagicMock()
    mock_learner.predict_probability.side_effect = ValueError("crepes error")
    mock_explainer.interval_learner = mock_learner

    x = np.array([[1, 2]])
    # It raises DataShapeError wrapping the original error
    with pytest.raises(DataShapeError, match="Error while computing prediction intervals"):
        orchestrator.predict_impl(x, threshold=0.5)


def test_validate_prediction_result_none(orchestrator):
    result = (None, None, None, None)
    assert orchestrator.validate_prediction_result(result) is None


def test_validate_prediction_result_empty(orchestrator):
    result = (np.array([]), np.array([]), np.array([]), None)
    assert orchestrator.validate_prediction_result(result) is None


def testpredict_impl_unknown_mode(orchestrator, mock_explainer):
    mock_explainer.mode = "unknown"
    mock_explainer.is_fast.return_value = False

    x = np.array([[1, 2]])
    result = orchestrator.predict_impl(x)
    assert result == (None, None, None, None)
