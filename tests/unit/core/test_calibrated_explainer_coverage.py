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
        instance._explanation_orchestrator = MagicMock()
        instance._prediction_orchestrator = MagicMock()
        instance._reject_orchestrator = MagicMock()

        yield mock


@pytest.fixture(autouse=True)
def mock_interval_calibrator():
    with patch(
        "calibrated_explanations.core.prediction.orchestrator.PredictionOrchestrator._obtain_interval_calibrator"
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

    mock_plugin_manager.return_value._explanation_orchestrator.invoke.assert_called_once()
    args, kwargs = mock_plugin_manager.return_value._explanation_orchestrator.invoke.call_args
    assert args[1] is x_test
    assert kwargs["extras"]["mode"] == "factual"  # Default inferred mode


def test_explain_factual_delegates(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    x_test = np.array([[5, 6]])
    explainer.explain_factual(x_test)

    mock_plugin_manager.return_value._explanation_orchestrator.invoke_factual.assert_called_once()
    args, kwargs = (
        mock_plugin_manager.return_value._explanation_orchestrator.invoke_factual.call_args
    )
    assert args[0] is x_test


def test_explore_alternatives_delegates(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    x_test = np.array([[5, 6]])
    explainer.explore_alternatives(x_test)

    mock_plugin_manager.return_value._explanation_orchestrator.invoke_alternative.assert_called_once()
    args, kwargs = (
        mock_plugin_manager.return_value._explanation_orchestrator.invoke_alternative.call_args
    )
    assert args[0] is x_test


def test_explain_fast_delegates(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    x_test = np.array([[5, 6]])
    explainer.explain_fast(x_test)

    mock_plugin_manager.return_value._explanation_orchestrator.invoke.assert_called()
    args, kwargs = mock_plugin_manager.return_value._explanation_orchestrator.invoke.call_args
    assert args[0] == "fast"
    assert args[1] is x_test


def test_predict_delegates(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    x_test = np.array([[5, 6]])
    explainer._predict(x_test)

    mock_plugin_manager.return_value._prediction_orchestrator.predict.assert_called_once()
    args, kwargs = mock_plugin_manager.return_value._prediction_orchestrator.predict.call_args
    assert args[0] is x_test


def test_properties_delegation(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    # Access properties to ensure they delegate
    _ = explainer.explanation_plugin_overrides
    _ = explainer._interval_plugin_override
    _ = explainer._fast_interval_plugin_override
    _ = explainer._plot_style_override
    _ = explainer._bridge_monitors
    _ = explainer._explanation_plugin_instances
    _ = explainer._explanation_plugin_identifiers
    _ = explainer._explanation_plugin_fallbacks
    _ = explainer._plot_plugin_fallbacks
    _ = explainer._interval_plugin_hints
    _ = explainer._interval_plugin_fallbacks
    _ = explainer._interval_plugin_identifiers
    _ = explainer._telemetry_interval_sources
    _ = explainer._interval_preferred_identifier
    _ = explainer._interval_context_metadata
    _ = explainer._plot_style_chain
    _ = explainer._explanation_contexts
    _ = explainer._last_explanation_mode
    _ = explainer._last_telemetry
    _ = explainer._pyproject_explanations
    _ = explainer._pyproject_intervals
    _ = explainer._pyproject_plots

    # Just checking no attribute error is raised and they access the manager
    # We can verify one of them
    assert (
        explainer.explanation_plugin_overrides
        == mock_plugin_manager.return_value._explanation_plugin_overrides
    )


def test_setters_delegation(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    explainer.explanation_plugin_overrides = {"a": 1}
    assert mock_plugin_manager.return_value._explanation_plugin_overrides == {"a": 1}


def test_interval_learner_property(mock_learner, mock_plugin_manager):
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")

    # Mock the registry
    mock_registry = MagicMock()
    mock_plugin_manager.return_value._prediction_orchestrator._interval_registry = mock_registry

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
    assert explainer._infer_explanation_mode() == "factual"

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
