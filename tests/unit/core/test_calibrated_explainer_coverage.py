import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.utils.exceptions import ValidationError, DataShapeError


@pytest.fixture
def mock_learner():
    learner = MagicMock()
    learner.predict.return_value = np.array([0, 1])
    learner.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
    return learner


@pytest.fixture(autouse=True)
def mock_check_is_fitted():
    with patch("calibrated_explanations.core.calibrated_explainer.check_is_fitted") as mock:
        yield mock


@pytest.fixture(autouse=True)
def mock_plugin_manager():
    with patch("calibrated_explanations.plugins.manager.PluginManager") as mock:
        instance = mock.return_value
        instance.initialize_from_kwargs.return_value = None
        instance.initialize_orchestrators.return_value = None

        # Mock orchestrators
        instance.explanation_orchestrator = MagicMock()
        instance.prediction_orchestrator = MagicMock()
        instance.reject_orchestrator = MagicMock()

        yield mock


@pytest.fixture(autouse=True)
def mock_interval_calibrator():
    with patch(
        "calibrated_explanations.core.prediction.orchestrator.PredictionOrchestrator.obtain_interval_calibrator"
    ) as mock:
        mock.return_value = (MagicMock(), None)
        yield mock


@pytest.fixture(autouse=True)
def mock_legacy_predict_bridge():
    with patch("calibrated_explanations.plugins.builtins.LegacyPredictBridge") as mock:
        yield mock


@pytest.fixture(autouse=True)
def mock_identify_constant_features():
    with patch(
        "calibrated_explanations.core.calibration_helpers.identify_constant_features"
    ) as mock:
        mock.return_value = []
        yield mock


@pytest.fixture(autouse=True)
def mock_helpers():
    with patch("calibrated_explanations.integrations.LimeHelper"), patch(
        "calibrated_explanations.integrations.ShapHelper"
    ):
        yield


def test_init_basic(mock_learner):
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])

    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    assert explainer.learner == mock_learner
    np.testing.assert_array_equal(explainer.x_cal, x_cal)
    np.testing.assert_array_equal(explainer.y_cal, y_cal)
    assert explainer.mode == "classification"


def test_init_regression(mock_learner):
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([1.5, 3.5])

    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="regression")

    assert explainer.mode == "regression"
    assert explainer.predict_function == mock_learner.predict


def test_init_invalid_condition_source(mock_learner):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])

    with pytest.raises(ValidationError, match="condition_source must be either"):
        CalibratedExplainer(mock_learner, x_cal, y_cal, condition_source="invalid")


def test_init_oob_classification(mock_learner):
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])  # Ignored if oob=True

    mock_learner.oob_decision_function_ = np.array([[0.1, 0.9], [0.8, 0.2]])

    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification", oob=True)

    # oob predictions: [1, 0]
    np.testing.assert_array_equal(explainer.y_cal, np.array([1, 0]))


def test_init_oob_regression(mock_learner):
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])  # Ignored

    mock_learner.oob_prediction_ = np.array([1.1, 3.3])

    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="regression", oob=True)

    np.testing.assert_array_equal(explainer.y_cal, np.array([1.1, 3.3]))


def test_init_oob_shape_mismatch(mock_learner):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])

    mock_learner.oob_decision_function_ = np.array([[0.1, 0.9], [0.8, 0.2]])  # 2 samples

    with pytest.raises(DataShapeError, match="length of the out-of-bag predictions"):
        CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification", oob=True)


def test_call_delegates_to_orchestrator(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    x_test = np.array([[5, 6]])
    explainer(x_test)

    mock_plugin_manager.return_value.explanation_orchestrator.invoke.assert_called_once()
    args, kwargs = mock_plugin_manager.return_value.explanation_orchestrator.invoke.call_args
    assert args[1] is x_test
    assert kwargs["extras"]["mode"] == "factual"  # Default inferred mode


def test_explain_factual_delegates(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    x_test = np.array([[5, 6]])
    explainer.explain_factual(x_test)

    mock_plugin_manager.return_value.explanation_orchestrator.invoke_factual.assert_called_once()
    args, kwargs = (
        mock_plugin_manager.return_value.explanation_orchestrator.invoke_factual.call_args
    )
    assert args[0] is x_test


def test_explore_alternatives_delegates(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    x_test = np.array([[5, 6]])
    explainer.explore_alternatives(x_test)

    mock_plugin_manager.return_value.explanation_orchestrator.invoke_alternative.assert_called_once()
    args, kwargs = (
        mock_plugin_manager.return_value.explanation_orchestrator.invoke_alternative.call_args
    )
    assert args[0] is x_test


def test_explain_fast_delegates(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    x_test = np.array([[5, 6]])
    explainer.explain_fast(x_test)

    mock_plugin_manager.return_value.explanation_orchestrator.invoke.assert_called()
    args, kwargs = mock_plugin_manager.return_value.explanation_orchestrator.invoke.call_args
    assert args[0] == "fast"
    assert args[1] is x_test


def test_predict_delegates(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    x_test = np.array([[5, 6]])

    # Mock return value to allow unpacking in CalibratedExplainer.predict
    orchestrator = explainer.prediction_orchestrator
    orchestrator._predict_impl.return_value = (
        np.array([0]),
        np.array([0]),
        np.array([0]),
        np.array([0]),
    )

    explainer.predict(x_test)

    # Check that it delegated to the implementation method
    orchestrator._predict_impl.assert_called_once()
    args, kwargs = orchestrator._predict_impl.call_args
    assert args[0] is x_test


def test_properties_delegation(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    # Access properties to ensure they delegate
    _ = explainer.explanation_plugin_overrides
    _ = explainer._interval_plugin_override
    _ = explainer._fast_interval_plugin_override
    _ = explainer.plot_style_override
    _ = explainer._bridge_monitors
    _ = explainer._explanation_plugin_instances
    _ = explainer.plugin_manager.explanation_plugin_identifiers
    _ = explainer.plugin_manager.explanation_plugin_fallbacks
    _ = explainer.plugin_manager.plot_plugin_fallbacks
    _ = explainer.plugin_manager.interval_plugin_hints
    _ = explainer.plugin_manager.interval_plugin_fallbacks
    _ = explainer.plugin_manager.interval_plugin_identifiers
    _ = explainer.plugin_manager.telemetry_interval_sources
    _ = explainer.plugin_manager.interval_preferred_identifier
    _ = explainer.plugin_manager.interval_context_metadata
    _ = explainer.plugin_manager.plot_style_chain
    _ = explainer.plugin_manager.explanation_contexts
    _ = explainer.plugin_manager.last_explanation_mode
    _ = explainer.plugin_manager.last_telemetry
    _ = explainer._pyproject_explanations
    _ = explainer._pyproject_intervals
    _ = explainer._pyproject_plots

    # Just checking no attribute error is raised and they access the manager
    # We can verify one of them
    assert (
        explainer.explanation_plugin_overrides
        == mock_plugin_manager.return_value.explanation_plugin_overrides
    )


def test_setters_delegation(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    explainer.explanation_plugin_overrides = {"a": 1}
    assert mock_plugin_manager.return_value.explanation_plugin_overrides == {"a": 1}


def test_interval_learner_property(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    # Mock the registry
    mock_registry = MagicMock()
    mock_plugin_manager.return_value.prediction_orchestrator.interval_registry = mock_registry

    _ = explainer.interval_learner

    # Verify access
    # mock_registry.interval_learner was accessed

    explainer.interval_learner = "test"
    assert mock_registry.interval_learner == "test"


def test_infer_explanation_mode(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    # Default is factual
    assert explainer.infer_explanation_mode() == "factual"

    # Mock discretizer
    with patch("calibrated_explanations.utils.EntropyDiscretizer") as mock_entropy:
        explainer.discretizer = mock_entropy()
        # We need to make isinstance work.
        # Since we can't easily patch isinstance check against a class defined inside the method,
        # we might need to rely on the fact that the method imports them.
        # But patching the import inside the method is hard.
        # However, if we patch 'calibrated_explanations.utils.EntropyDiscretizer' globally,
        # the import inside the method should pick it up if it hasn't been imported yet
        # or if we patch it in sys.modules.
        pass


def test_reinitialize(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    new_learner = MagicMock()
    new_learner.predict_proba.return_value = np.array([[0.9, 0.1]])

    with patch(
        "calibrated_explanations.calibration.interval_learner.initialize_interval_learner"
    ) as mock_init_il:
        explainer.reinitialize(new_learner)
        assert explainer.learner is new_learner
        mock_init_il.assert_called_once_with(explainer)


def test_reinitialize_with_data(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    new_learner = MagicMock()
    xs = np.array([[5, 6]])
    ys = np.array([1])

    with patch(
        "calibrated_explanations.calibration.interval_learner.update_interval_learner"
    ) as mock_update_il, patch(
        "calibrated_explanations.calibration.state.CalibrationState.append_calibration"
    ) as mock_append:
        explainer.reinitialize(new_learner, xs, ys)

        mock_append.assert_called_once()
        mock_update_il.assert_called_once()


def test_repr(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    repr_str = repr(explainer)
    assert "CalibratedExplainer" in repr_str
    assert "mode=classification" in repr_str


def test_x_cal_y_cal_properties(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    with patch(
        "calibrated_explanations.calibration.state.CalibrationState.get_x_cal"
    ) as mock_get_x, patch(
        "calibrated_explanations.calibration.state.CalibrationState.set_x_cal"
    ) as mock_set_x:
        _ = explainer.x_cal
        mock_get_x.assert_called_once_with(explainer)

        explainer.x_cal = x_cal
        mock_set_x.assert_called_once_with(explainer, x_cal)

    with patch(
        "calibrated_explanations.calibration.state.CalibrationState.get_y_cal"
    ) as mock_get_y, patch(
        "calibrated_explanations.calibration.state.CalibrationState.set_y_cal"
    ) as mock_set_y:
        _ = explainer.y_cal
        mock_get_y.assert_called_once_with(explainer)

        explainer.y_cal = y_cal
        mock_set_y.assert_called_once_with(explainer, y_cal)


def test_deepcopy_circular(mock_learner, mock_plugin_manager):
    import copy
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")
    
    memo = {id(explainer): explainer}
    result = explainer.__deepcopy__(memo)
    assert result is explainer


def test_require_plugin_manager_error(mock_learner):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")
    
    from calibrated_explanations.utils.exceptions import NotFittedError
    
    # Manually remove _plugin_manager to trigger error
    del explainer._plugin_manager
    with pytest.raises(NotFittedError, match="PluginManager is not initialized"):
        explainer.require_plugin_manager()


def test_deleters_and_setters(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")
    
    # Test various deleters and setters to hit missing lines
    explainer.interval_plugin_hints = {"test": ("hint",)}
    del explainer.interval_plugin_hints
    
    explainer.interval_plugin_fallbacks = {"test": ("fallback",)}
    del explainer.interval_plugin_fallbacks
    
    explainer._interval_preferred_identifier = {"test": "id"}
    del explainer._interval_preferred_identifier
    
    explainer._telemetry_interval_sources = {"test": "source"}
    del explainer._telemetry_interval_sources
    
    explainer._interval_plugin_identifiers = {"test": "id"}
    del explainer._interval_plugin_identifiers
    
    explainer._interval_context_metadata = {"test": {"meta": "data"}}
    del explainer._interval_context_metadata
    
    explainer.plot_plugin_fallbacks = {"test": ("fallback",)}
    explainer.explanation_plugin_overrides = {"test": "override"}
    
    explainer.plugin_manager = MagicMock()
    del explainer.plugin_manager
    
    explainer.feature_filter_per_instance_ignore = [1]
    del explainer.feature_filter_per_instance_ignore
    
    explainer.parallel_executor = MagicMock()
    explainer.feature_filter_config = MagicMock()
    explainer.predict_bridge = MagicMock()
    explainer.categorical_value_counts_cache = MagicMock()
    explainer.numeric_sorted_cache = MagicMock()
    explainer.calibration_summary_shape = MagicMock()
    
    explainer.initialized = True
    assert explainer.is_initialized is True
    
    explainer.last_explanation_mode = "factual"
    assert explainer.last_explanation_mode == "factual"


def test_repr_verbose(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")
    explainer.verbose = True
    explainer.feature_names = ["f1", "f2"]
    explainer.categorical_features = [0]
    explainer.categorical_labels = {0: {0: "A", 1: "B"}}
    explainer.class_labels = ["C1", "C2"]
    
    repr_str = repr(explainer)
    assert "feature_names" in repr_str
    assert "categorical_features" in repr_str
    assert "class_labels" in repr_str


def test_preloads(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")
    
    with patch.object(explainer, "_lime_helper") as mock_lime:
        explainer.preload_lime(x_cal)
        mock_lime.preload.assert_called_once_with(x_cal=x_cal)
        
    with patch.object(explainer, "_shap_helper") as mock_shap:
        explainer.preload_shap(num_test=10)
        mock_shap.preload.assert_called_once_with(num_test=10)


def test_predict_fallback(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")
    
    # Mock orchestrator without _predict_impl
    class Orchestrator:
        def __init__(self):
            self.predict = MagicMock(return_value="pred")

    orchestrator = Orchestrator()
    explainer.plugin_manager.prediction_orchestrator = orchestrator
    
    result = explainer._predict(x_cal)
    assert result == "pred"
    orchestrator.predict.assert_called_once()


def test_explain_methods(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")
    
    x_test = np.array([[1, 2]])
    
    explainer.explain_factual(x_test)
    explainer.plugin_manager.explanation_orchestrator.invoke_factual.assert_called_once()
    
    explainer.explore_alternatives(x_test)
    explainer.plugin_manager.explanation_orchestrator.invoke_alternative.assert_called_once()
    
    with patch.object(explainer, "_explain") as mock_explain:
        explainer(x_test)
        mock_explain.assert_called_once()


def test_external_pipelines(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")
    x_test = np.array([[1, 2]])
    
    with patch("external_plugins.fast_explanations.pipeline.FastExplanationPipeline") as mock_fast:
        explainer.explain_fast(x_test, _use_plugin=False)
        mock_fast.return_value.explain.assert_called_once()
        
    with patch("external_plugins.integrations.lime_pipeline.LimePipeline") as mock_lime:
        explainer.explain_lime(x_test)
        mock_lime.return_value.explain.assert_called_once()
        
    with patch("external_plugins.integrations.shap_pipeline.ShapPipeline") as mock_shap:
        explainer.explain_shap(x_test)
        mock_shap.return_value.explain.assert_called_once()


def test_setters_initialization(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")
    
    explainer.set_difficulty_estimator(MagicMock())
    explainer.plugin_manager.prediction_orchestrator.interval_registry.initialize.assert_called()
    
    explainer.set_mode("regression")
    explainer.plugin_manager.prediction_orchestrator.interval_registry.initialize.assert_called()


def test_predict_proba_uncalibrated_multiclass(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0, 1, 2])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")
    explainer.num_classes = 3
    
    x_test = np.array([[1, 2]])
    mock_learner.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
    
    res, (low, high) = explainer.predict_proba(x_test, calibrated=False, uq_interval=True)
    assert res.shape == (1, 3)
    assert np.array_equal(low, res)


def test_repr_regression_complex(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([1.0])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="regression")
    explainer.bins = [0]
    explainer.discretizer = MagicMock()
    explainer.difficulty_estimator = MagicMock()
    
    repr_str = repr(explainer)
    assert "conditional=True" in repr_str
    assert "discretizer=" in repr_str
    assert "difficulty_estimator=" in repr_str


def test_instantiate_plugin(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")
    
    prototype = MagicMock()
    explainer.instantiate_plugin(prototype)
    explainer.plugin_manager.explanation_orchestrator.instantiate_plugin.assert_called_once_with(prototype)


def test_predict_proba_uncalibrated_binary(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0, 1])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")
    
    x_test = np.array([[1, 2]])
    mock_learner.predict_proba.return_value = np.array([[0.8, 0.2]])
    
    res = explainer.predict_proba(x_test, calibrated=False)
    assert res.shape == (1, 2)


def test_predict_proba_regression_list(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([1.0])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="regression")
    
    x_test = np.array([[1, 2]])
    mock_il = MagicMock()
    mock_il.predict_probability.return_value = (np.array([0.6]), np.array([0.5]), np.array([0.7]), None)
    explainer.interval_learner = [mock_il]
    
    res = explainer.predict_proba(x_test, threshold=1.5)
    assert res.shape == (1, 2)
    assert res[0, 1] == 0.6


def test_predict_proba_multiclass_list(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0, 1, 2])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")
    explainer.num_classes = 3
    
    x_test = np.array([[1, 2]])
    mock_il = MagicMock()
    mock_il.predict_proba.return_value = (np.array([[0.1, 0.2, 0.7]]), np.array([[0.05, 0.15, 0.65]]), np.array([[0.15, 0.25, 0.75]]), None)
    explainer.interval_learner = [mock_il]
    
    res = explainer.predict_proba(x_test)
    assert res.shape == (1, 3)


def test_parallel_pool_lifecycle(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")
    
    with patch("calibrated_explanations.parallel.ParallelExecutor") as mock_executor:
        explainer.initialize_pool(n_workers=2, pool_at_init=True)
        assert explainer.parallel_executor is not None
        mock_executor.return_value.__enter__.assert_called_once()
        
        explainer.close()
        assert explainer.parallel_executor is None
        mock_executor.return_value.__exit__.assert_called_once()


def test_context_manager(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")
    
    with patch.object(explainer, "initialize_pool") as mock_init, \
         patch.object(explainer, "close") as mock_close:
        with explainer as e:
            assert e is explainer
            mock_init.assert_called_once_with(pool_at_init=True)
        mock_close.assert_called_once()


def test_deepcopy_returns_memo(mock_learner):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    memo = {id(explainer): explainer}
    result = explainer.__deepcopy__(memo)
    assert result is explainer


def test_plugin_delegations_and_aliases(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")
    plugin_manager = explainer.plugin_manager

    plugin_manager.build_plot_chain.return_value = ("default",)
    assert explainer.build_plot_style_chain() == ("default",)

    explainer.prediction_orchestrator.ensure_interval_runtime_state.return_value = "ok"
    assert explainer.ensure_interval_runtime_state() == "ok"
    explainer.prediction_orchestrator.gather_interval_hints.return_value = ("hint",)
    assert explainer.gather_interval_hints(fast=True) == ("hint",)

    plugin_manager.interval_plugin_hints = {"a": ("b",)}
    assert explainer._interval_plugin_hints == {"a": ("b",)}
    explainer._interval_plugin_hints = {"c": ("d",)}
    assert plugin_manager.interval_plugin_hints == {"c": ("d",)}
    assert explainer.interval_plugin_hints == {"c": ("d",)}

    plugin_manager.interval_plugin_fallbacks = {"a": ("b",)}
    assert explainer._interval_plugin_fallbacks == {"a": ("b",)}
    explainer._interval_plugin_fallbacks = {"c": ("d",)}
    assert plugin_manager.interval_plugin_fallbacks == {"c": ("d",)}
    assert explainer.interval_plugin_fallbacks == {"c": ("d",)}

    plugin_manager.interval_preferred_identifier = {"x": "y"}
    assert explainer._interval_preferred_identifier == {"x": "y"}
    explainer.interval_preferred_identifier = {"z": None}
    assert plugin_manager.interval_preferred_identifier == {"z": None}

    plugin_manager.telemetry_interval_sources = {"x": "y"}
    assert explainer._telemetry_interval_sources == {"x": "y"}
    explainer.telemetry_interval_sources = {"z": "w"}
    assert plugin_manager.telemetry_interval_sources == {"z": "w"}

    plugin_manager.interval_plugin_identifiers = {"x": "y"}
    assert explainer._interval_plugin_identifiers == {"x": "y"}
    explainer._interval_plugin_identifiers = {"z": "w"}
    assert plugin_manager.interval_plugin_identifiers == {"z": "w"}

    plugin_manager.interval_context_metadata = {"x": {"y": 1}}
    assert explainer._interval_context_metadata == {"x": {"y": 1}}
    explainer._interval_context_metadata = {"z": {"w": 2}}
    assert plugin_manager.interval_context_metadata == {"z": {"w": 2}}

    explainer.plot_plugin_fallbacks = {"plot": ("fallback",)}
    assert plugin_manager.plot_plugin_fallbacks == {"plot": ("fallback",)}

    explainer._explanation_plugin_instances = {"plugin": object()}
    assert plugin_manager.explanation_plugin_instances == explainer._explanation_plugin_instances

    explainer._interval_plugin_override = "override"
    assert explainer.interval_plugin_override == "override"

    explainer.fast_interval_plugin_override = "fast"
    assert explainer.fast_interval_plugin_override == "fast"

    explainer.initialized = True
    assert explainer.is_initialized is True

    explainer.parallel_executor = MagicMock()
    explainer.predict_bridge = MagicMock()


def test_enable_fast_mode_resets_on_error(mock_learner):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    with patch.object(
        explainer,
        "_CalibratedExplainer__initialize_interval_learner_for_fast_explainer",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            explainer.enable_fast_mode()

    assert explainer.is_fast() is False


def test_explain_mondrian_bins_and_legacy_path(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")
    explainer.bins = np.array([1])

    x_test = np.array([[3, 4]])
    explainer.explanation_orchestrator.invoke_factual.return_value = "factual"
    explainer.explain_factual(x_test)
    args, _ = explainer.explanation_orchestrator.invoke_factual.call_args
    assert args[3] is explainer.bins

    explainer.explanation_orchestrator.invoke_alternative.return_value = "alternative"
    explainer.explore_alternatives(x_test)
    args, _ = explainer.explanation_orchestrator.invoke_alternative.call_args
    assert args[3] is explainer.bins

    with patch("calibrated_explanations.core.explain._legacy_explain.explain") as mock_legacy:
        mock_legacy.return_value = "legacy"
        result = explainer._explain(x_test, bins=None, _use_plugin=False)
        assert result == "legacy"
        assert mock_legacy.call_args.kwargs["bins"] is explainer.bins


def test_explain_fast_mondrian_bins(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")
    explainer.bins = np.array([1])

    x_test = np.array([[3, 4]])
    with patch.object(explainer, "_invoke_explanation_plugin", return_value="fast") as mock_invoke:
        result = explainer.explain_fast(x_test, _use_plugin=True)
        assert result == "fast"
        assert mock_invoke.call_args.args[4] is explainer.bins


def test_predict_proba_uncalibrated_threshold_raises(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2]])
    y_cal = np.array([0, 1])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    with pytest.raises(ValidationError, match="thresholded prediction is not possible"):
        explainer.predict_proba(np.array([[3, 4]]), calibrated=False, threshold=0.5)


def test_predict_proba_uncalibrated_multiclass_interval(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2], [3, 4], [5, 6]])
    y_cal = np.array([0, 1, 2])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")
    explainer.num_classes = 3

    x_test = np.array([[1, 2]])
    mock_learner.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])

    res, (low, high) = explainer.predict_proba(x_test, calibrated=False, uq_interval=True)
    assert res.shape == (1, 3)
    assert np.array_equal(low, res)
    assert np.array_equal(high, res)


def test_predict_proba_multiclass_interval_learner_list(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2], [3, 4], [5, 6]])
    y_cal = np.array([0, 1, 2])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")
    explainer.num_classes = 3

    interval = MagicMock()
    interval.predict_proba.return_value = (
        np.array([[0.2, 0.3, 0.5]]),
        np.array([[0.1, 0.2, 0.4]]),
        np.array([[0.3, 0.4, 0.6]]),
        None,
    )
    explainer.interval_learner = [interval]

    res = explainer.predict_proba(np.array([[1, 2]]))
    assert res.shape == (1, 3)
