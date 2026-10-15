import pytest
from unittest.mock import MagicMock
import numpy as np
from calibrated_explanations.core.prediction.interval_registry import IntervalRegistry


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


def test_constant_sigma_scalar(registry):
    # Although type hint says np.ndarray, it handles other types
    sigma = registry.constant_sigma(1)
    assert np.array_equal(sigma, np.ones(1))
