import numpy as np
import copy
from unittest.mock import MagicMock
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer


def test_calibrated_explainer_deepcopy():
    # Setup a basic explainer
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])

    # We need a learner that has predict_proba if mode is classification
    class MockLearner:
        def __init__(self):
            self.fitted = True

        def predict_proba(self, x):
            return np.array([[0.8, 0.2], [0.3, 0.7]])

        def predict(self, x):
            return np.array([0, 1])

        def fit(self, x, y):
            pass

    learner = MockLearner()
    explainer = CalibratedExplainer(learner, x_cal, y_cal, mode="classification")

    # Perform deepcopy
    explainer_copy = copy.deepcopy(explainer)

    # Verify basic attributes are copied
    assert explainer_copy is not explainer
    assert explainer_copy.mode == explainer.mode
    np.testing.assert_array_equal(explainer_copy.x_cal, explainer.x_cal)
    np.testing.assert_array_equal(explainer_copy.y_cal, explainer.y_cal)

    # Verify that internal state is preserved
    assert (
        explainer_copy.learner is explainer.learner
    )  # Learner is usually not deepcopied if it's a model


def test_calibrated_explainer_close():
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])

    class MockLearner:
        def __init__(self):
            self.fitted = True

        def predict_proba(self, x):
            return np.array([[0.8, 0.2], [0.3, 0.7]])

        def predict(self, x):
            return np.array([0, 1])

        def fit(self, x, y):
            pass

    learner = MockLearner()
    explainer = CalibratedExplainer(learner, x_cal, y_cal, mode="classification")

    # Test close when no pool is initialized
    explainer.close()
    assert getattr(explainer, "_perf_parallel", None) is None

    # Test close when pool is initialized (mocking ParallelExecutor)
    from unittest.mock import MagicMock

    mock_perf = MagicMock()
    setattr(explainer, "_perf_parallel", mock_perf)
    explainer.close()
    mock_perf.__exit__.assert_called_once()
    assert getattr(explainer, "_perf_parallel", None) is None


def test_calibrated_explainer_context_manager():
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])

    class MockLearner:
        def __init__(self):
            self.fitted = True

        def predict_proba(self, x):
            return np.array([[0.8, 0.2], [0.3, 0.7]])

        def predict(self, x):
            return np.array([0, 1])

        def fit(self, x, y):
            pass

    learner = MockLearner()
    with CalibratedExplainer(learner, x_cal, y_cal, mode="classification") as explainer:
        assert isinstance(explainer, CalibratedExplainer)
        # initialize_pool should have been called

    # After context, close should have been called
    assert getattr(explainer, "_perf_parallel", None) is None


def test_calibrated_explainer_deleters():
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])

    class MockLearner:
        def __init__(self):
            self.fitted = True

        def predict_proba(self, x):
            return np.array([[0.8, 0.2], [0.3, 0.7]])

        def predict(self, x):
            return np.array([0, 1])

        def fit(self, x, y):
            pass

    learner = MockLearner()
    explainer = CalibratedExplainer(learner, x_cal, y_cal, mode="classification")

    # Test plugin_manager deleter
    explainer.plugin_manager = "something"
    del explainer.plugin_manager
    assert not hasattr(explainer, "_plugin_manager")

    # Test interval_plugin_hints deleter
    mock_pm = MagicMock()
    explainer.plugin_manager = mock_pm
    del explainer.interval_plugin_hints

    # Test interval_plugin_fallbacks deleter
    del explainer.interval_plugin_fallbacks
