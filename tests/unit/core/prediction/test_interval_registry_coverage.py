import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from calibrated_explanations.core.prediction.interval_registry import IntervalRegistry
from calibrated_explanations.core.exceptions import ConfigurationError
import calibrated_explanations.calibration.venn_abers as venn_module


@pytest.fixture
def mock_explainer():
    explainer = MagicMock()
    explainer.difficulty_estimator = None
    explainer.is_fast.return_value = False
    explainer.mode = "classification"
    explainer.x_cal = np.array([[1]])
    explainer.y_cal = np.array([0])
    explainer.learner = MagicMock()
    explainer.bins = None
    explainer.predict_function = MagicMock()
    return explainer


@pytest.fixture
def registry(mock_explainer):
    return IntervalRegistry(mock_explainer)


def test_get_sigma_test_no_estimator(registry):
    x = np.array([[1], [2]])
    sigma = registry.get_sigma_test(x)
    assert np.array_equal(sigma, np.ones(2))


def test_get_sigma_test_with_estimator(registry, mock_explainer):
    mock_estimator = MagicMock()
    mock_estimator.apply.return_value = np.array([0.5, 0.5])
    mock_explainer.difficulty_estimator = mock_estimator

    x = np.array([[1], [2]])
    sigma = registry.get_sigma_test(x)

    mock_estimator.apply.assert_called_once_with(x)
    assert np.array_equal(sigma, np.array([0.5, 0.5]))


def test_constant_sigma_scalar(registry):
    # Although type hint says np.ndarray, it handles other types
    sigma = registry._constant_sigma(1)
    assert np.array_equal(sigma, np.ones(1))


def test_update_fast_explainer_raises(registry, mock_explainer):
    mock_explainer.is_fast.return_value = True
    with pytest.raises(ConfigurationError, match="Fast explanations are not supported"):
        registry.update(np.array([]), np.array([]))


def test_update_classification(registry, mock_explainer):
    mock_explainer.mode = "classification"
    # Patch the source where VennAbers is imported from in the method
    with patch.object(venn_module, "VennAbers") as mock_venn_abers:
        registry.update(np.array([]), np.array([]))
        mock_venn_abers.assert_called_once()
        assert registry.interval_learner == mock_venn_abers.return_value


def test_update_regression(registry, mock_explainer):
    mock_explainer.mode = "regression"
    mock_learner = MagicMock()
    registry.interval_learner = mock_learner

    xs = np.array([[1]])
    ys = np.array([1])
    registry.update(xs, ys)

    mock_learner.insert_calibration.assert_called_once_with(xs, ys, bins=None)


def test_update_regression_fast_list_raises(registry, mock_explainer):
    mock_explainer.mode = "regression"
    registry.interval_learner = []  # List implies fast mode structure in some contexts

    with pytest.raises(ConfigurationError, match="Fast explanations are not supported"):
        registry.update(np.array([]), np.array([]))


def test_initialize(registry):
    with patch(
        "calibrated_explanations.core.prediction.interval_registry.initialize_interval_learner"
    ) as mock_init:
        registry.initialize()
        mock_init.assert_called_once_with(registry.explainer)


def test_initialize_for_fast_explainer(registry):
    with patch(
        "calibrated_explanations.core.prediction.interval_registry.initialize_interval_learner_for_fast_explainer"
    ) as mock_init:
        registry.initialize_for_fast_explainer()
        mock_init.assert_called_once_with(registry.explainer)
