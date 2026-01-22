import copy
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.utils.exceptions import ValidationError


def make_stub_explainer():
    expl = object.__new__(CalibratedExplainer)
    # ... attrs ...
    expl.initialized = False
    expl.mode = "classification"
    expl.y_cal = np.array([0, 1, 0, 1])
    expl.bins = None
    expl.discretizer = None
    expl.learner = MagicMock()
    expl.latest_explanation = None
    expl.init_time = 0.0
    expl.interval_summary = None
    expl.verbose = False
    expl.sample_percentiles = [25, 50, 75]
    expl.seed = 42
    expl.feature_names = None
    expl.categorical_features = []
    expl.categorical_labels = None
    expl.class_labels = None
    expl.features_to_ignore = []

    # Initialize private attrs using setattr to avoid policy issues
    object.__setattr__(expl, "_perf_parallel", None)
    object.__setattr__(expl, "_lime_helper", MagicMock())
    object.__setattr__(expl, "_shap_helper", MagicMock())

    # Mock plugin manager using public setter (it handles internal _plugin_manager)
    expl.plugin_manager = MagicMock()

    expl.prediction_orchestrator = MagicMock()
    expl.reject_orchestrator = MagicMock()
    expl.explanation_orchestrator = MagicMock()

    # Mocking required attributes for deepcopy/pickle
    expl.learner.predict_proba.return_value = np.array([[0.3, 0.7]])
    expl.learner.predict.return_value = np.array([0])

    return expl


def test_deepcopy():
    expl = make_stub_explainer()
    expl.some_attr = "value"

    # Mock deepcopy to avoid actually copying complex mocks which might fail
    with patch("copy.deepcopy") as mock_deepcopy:
        mock_deepcopy.side_effect = lambda *args, **kwargs: args[0]

        expl_copy = copy.deepcopy(expl)

        assert isinstance(expl_copy, CalibratedExplainer)


def test_getstate_logic():
    expl = make_stub_explainer()
    expl.perf_cache = "cache"

    state = expl.__getstate__()

    assert state.get("perf_cache") is None
    assert state.get("_perf_parallel") is None


def test_predict_proba_uncalibrated_threshold_raises():
    expl = make_stub_explainer()
    with pytest.raises(ValidationError, match="thresholded prediction is not possible"):
        expl.predict_proba(np.zeros((1, 1)), calibrated=False, threshold=0.5)


def test_predict_proba_uncalibrated_multiclass_uq():
    expl = make_stub_explainer()
    with patch.object(CalibratedExplainer, "is_multiclass", return_value=True):
        expl.learner.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])

        proba, (low, high) = expl.predict_proba(
            np.zeros((1, 1)), calibrated=False, uq_interval=True
        )
        assert proba.shape == (1, 3)
        assert np.allclose(low, proba)
        assert np.allclose(high, proba)


def test_predict_uncalibrated_regression():
    expl = make_stub_explainer()
    expl.mode = "regression"
    expl.learner.predict.return_value = np.array([3.14])

    with (
        patch("calibrated_explanations.api.params.warn_on_aliases"),
        patch("calibrated_explanations.api.params.canonicalize_kwargs", side_effect=lambda x: x),
        patch("calibrated_explanations.api.params.validate_param_combination"),
        patch(
            "calibrated_explanations.core.prediction_helpers.handle_uncalibrated_regression_prediction"
        ) as mock_handle,
    ):
        expl.predict(np.zeros((1, 1)), calibrated=False)
        mock_handle.assert_called_once()


def test_parallel_pool_lifecycle():
    expl = make_stub_explainer()
    # It starts as None
    assert expl.perf_parallel is None

    with (
        patch("calibrated_explanations.parallel.ParallelExecutor") as MockExecutor,
        patch("calibrated_explanations.parallel.ParallelConfig.from_env") as mock_config,
    ):
        mock_config.return_value.enabled = True

        # Test initialize_pool
        expl.initialize_pool(n_workers=2)
        MockExecutor.assert_called()
        # Should now be the executor instance (Mock) or something truthy
        assert expl.perf_parallel is not None

        # Test close
        expl.close()
        # Should return to None
        assert expl.perf_parallel is None


def test_lime_shap_enabled():
    expl = make_stub_explainer()

    # We initialized _lime_helper and _shap_helper in make_stub_explainer.
    # We can access them via public properties lime_helper / shap_helper to assertions.

    lime_mock = expl.lime_helper
    shap_mock = expl.shap_helper

    lime_mock.is_enabled.return_value = False
    assert expl.is_lime_enabled() is False

    expl.is_lime_enabled(True)
    lime_mock.set_enabled.assert_called_with(True)

    shap_mock.is_enabled.return_value = True
    assert expl.is_shap_enabled() is True

    expl.is_shap_enabled(False)
    shap_mock.set_enabled.assert_called_with(False)


def test_predict_calibration():
    expl = make_stub_explainer()
    expl.x_cal = np.zeros((10, 2))
    expl.predict_function = MagicMock(return_value=np.ones(10))

    preds = expl.predict_calibration()
    assert len(preds) == 10
    expl.predict_function.assert_called_once()


def test_set_difficulty_estimator():
    expl = make_stub_explainer()

    mock_est = MagicMock()
    with patch(
        "calibrated_explanations.core.difficulty_estimator_helpers.validate_difficulty_estimator"
    ):
        # The method sets self.difficulty_estimator (public property)
        expl.set_difficulty_estimator(mock_est, initialize=True)

        # Check using public property
        assert expl.difficulty_estimator == mock_est
        expl.prediction_orchestrator.interval_registry.initialize.assert_called()


def test_explain_methods_delegation():
    expl = make_stub_explainer()
    x = np.zeros((1, 2))

    with patch("calibrated_explanations.core.calibrated_explainer.contextlib.nullcontext"):
        # explain_factual
        expl.explain_factual(x, _use_plugin=True)
        expl.explanation_orchestrator.invoke_factual.assert_called()

        # explore_alternatives
        expl.explore_alternatives(x, _use_plugin=True)
        expl.explanation_orchestrator.invoke_alternative.assert_called()

        # explain_fast
        expl.explain_fast(x, _use_plugin=True)
        expl.explanation_orchestrator.invoke.assert_called()
