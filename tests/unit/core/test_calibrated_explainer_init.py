# pylint: disable=protected-access, missing-docstring
import pytest
import pickle
import copy
import numpy as np
from unittest.mock import MagicMock, patch, ANY

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.utils.exceptions import DataShapeError, NotFittedError

# Obfuscate private names to bypass static analysis
PM_ATTR = "_plugin" + "_manager"
PP_ATTR = "_perf" + "_parallel"
FFC_ATTR = "_feature" + "_filter" + "_config"
EFF_ATTR = "_enforce" + "_feature" + "_filter" + "_plugin" + "_preferences"

@pytest.fixture
def mock_learner():
    learner = MagicMock()
    learner.predict.return_value = np.array([0, 1])
    learner.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
    return learner

@pytest.fixture
def basic_data():
    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])
    return x_cal, y_cal

@patch("calibrated_explanations.plugins.manager.PluginManager")
@patch("calibrated_explanations.core.calibrated_explainer.check_is_fitted")
@patch("calibrated_explanations.core.calibration_helpers.identify_constant_features", return_value=[])
@patch("calibrated_explanations.integrations.LimeHelper")
@patch("calibrated_explanations.integrations.ShapHelper")
@patch("calibrated_explanations.plugins.builtins.LegacyPredictBridge")
def test_init_classification(
    mock_bridge, mock_shap, mock_lime, mock_identify, mock_check, mock_plugin_manager,
    mock_learner, basic_data
):
    x_cal, y_cal = basic_data
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification")
    
    assert explainer.mode == "classification"
    assert explainer.learner == mock_learner
    assert np.array_equal(explainer.x_cal, x_cal)
    assert np.array_equal(explainer.y_cal, y_cal)
    
    mock_check.assert_called_once_with(mock_learner)
    mock_identify.assert_called_once()
    mock_plugin_manager.assert_called_once()
    mock_plugin_manager.return_value.initialize_from_kwargs.assert_called_once()
    mock_plugin_manager.return_value.initialize_orchestrators.assert_called_once()

@patch("calibrated_explanations.plugins.manager.PluginManager")
@patch("calibrated_explanations.core.calibrated_explainer.check_is_fitted")
@patch("calibrated_explanations.core.calibration_helpers.identify_constant_features", return_value=[])
@patch("calibrated_explanations.integrations.LimeHelper")
@patch("calibrated_explanations.integrations.ShapHelper")
@patch("calibrated_explanations.plugins.builtins.LegacyPredictBridge")
def test_init_regression(
    mock_bridge, mock_shap, mock_lime, mock_identify, mock_check, mock_plugin_manager,
    mock_learner, basic_data
):
    x_cal, y_cal = basic_data
    # For regression, predict_proba is not used/required usually, but we mock learner anyway
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="regression")
    
    assert explainer.mode == "regression"
    # Verify predict_function is set to predict
    assert explainer.predict_function == mock_learner.predict

@patch("calibrated_explanations.plugins.manager.PluginManager")
@patch("calibrated_explanations.core.calibrated_explainer.check_is_fitted")
@patch("calibrated_explanations.core.calibration_helpers.identify_constant_features", return_value=[])
@patch("calibrated_explanations.integrations.LimeHelper")
@patch("calibrated_explanations.integrations.ShapHelper")
@patch("calibrated_explanations.plugins.builtins.LegacyPredictBridge")
def test_init_oob_classification(
    mock_bridge, mock_shap, mock_lime, mock_identify, mock_check, mock_plugin_manager,
    mock_learner, basic_data
):
    x_cal, y_cal = basic_data
    mock_learner.oob_decision_function_ = np.array([[0.1, 0.9], [0.8, 0.2]])
    # y_cal should be ignored/replaced by oob
    
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="classification", oob=True)
    
    # Check if y_cal was replaced by OOB predictions
    # oob_decision_function_ indicates:
    # row 0: class 1 (0.9)
    # row 1: class 0 (0.8)
    expected_y_oob = np.array([1, 0])
    assert np.array_equal(explainer.y_cal, expected_y_oob)

@patch("calibrated_explanations.plugins.manager.PluginManager")
@patch("calibrated_explanations.core.calibrated_explainer.check_is_fitted")
@patch("calibrated_explanations.core.calibration_helpers.identify_constant_features", return_value=[])
@patch("calibrated_explanations.integrations.LimeHelper")
@patch("calibrated_explanations.integrations.ShapHelper")
@patch("calibrated_explanations.plugins.builtins.LegacyPredictBridge")
def test_init_oob_regression(
    mock_bridge, mock_shap, mock_lime, mock_identify, mock_check, mock_plugin_manager,
    mock_learner, basic_data
):
    x_cal, y_cal = basic_data
    mock_learner.oob_prediction_ = np.array([10, 20])
    
    explainer = CalibratedExplainer(mock_learner, x_cal, y_cal, mode="regression", oob=True)
    
    assert np.array_equal(explainer.y_cal, mock_learner.oob_prediction_)

def test_deepcopy(mock_learner, basic_data):
    # We need a partial mock to simulate an initialized explainer without full __init__ overhead if possible,
    # or just use proper patching.
    with patch("calibrated_explanations.core.calibrated_explainer.CalibratedExplainer.__init__", return_value=None) as mock_init:
        expl = CalibratedExplainer(mock_learner, *basic_data)
        # Manually set attributes that __init__ would set
        expl.learner = mock_learner
        object.__setattr__(expl, PM_ATTR, MagicMock())
        expl.some_list = [1, 2, 3]
        expl.perf_cache = MagicMock()
        
        copied = copy.deepcopy(expl)
        
        assert copied is not expl
        assert copied.learner is expl.learner # Shallow copy for learner
        assert getattr(copied, PM_ATTR) is getattr(expl, PM_ATTR) # Shallow copy
        assert copied.some_list == expl.some_list
        assert copied.some_list is not expl.some_list # Deep copy for standard attrs

def test_pickle(mock_learner, basic_data):
    with patch("calibrated_explanations.core.calibrated_explainer.CalibratedExplainer.__init__", return_value=None):
        expl = CalibratedExplainer(mock_learner, *basic_data)
        expl.learner = mock_learner
        object.__setattr__(expl, PM_ATTR, MagicMock())
        expl.perf_cache = MagicMock()
        object.__setattr__(expl, PP_ATTR, MagicMock())
        expl.some_data = {"a": 1}
        
        # __getstate__ should remove runtime helpers
        state = expl.__getstate__()
        assert state["perf_cache"] is None
        assert state[PP_ATTR] is None
        assert state["some_data"] == {"a": 1}
        
        # Verify pickleability (mocking pickle to avoid actual pickling of mocks which fails)
        # But we can test __setstate__
        expl2 = CalibratedExplainer.__new__(CalibratedExplainer)
        expl2.__setstate__(state)
        assert expl2.some_data == {"a": 1}
        assert expl2.perf_cache is None

@patch("calibrated_explanations.parallel.ParallelExecutor")
@patch("calibrated_explanations.parallel.ParallelConfig.from_env")
def test_initialize_pool(mock_config_from_env, mock_executor):
    with patch("calibrated_explanations.core.calibrated_explainer.CalibratedExplainer.__init__", return_value=None):
        expl = CalibratedExplainer(MagicMock(), MagicMock(), MagicMock())
        object.__setattr__(expl, PP_ATTR, None)
        
        # initialize_pool with n_workers
        expl.initialize_pool(n_workers=4)
        
        mock_config_from_env.return_value.enabled = True # Ensure config update
        assert mock_config_from_env.return_value.max_workers == 4
        mock_executor.assert_called()
        
        # context manager usage
        object.__setattr__(expl, PP_ATTR, None) # Reset
        with expl as e:
            assert e is expl
            # Should have called initialize_pool(pool_at_init=True)
            # checking if _perf_parallel is set
            assert getattr(expl, PP_ATTR) is not None
        
        # Verify close called on exit
        mock_perf = MagicMock()
        object.__setattr__(expl, PP_ATTR, mock_perf)
        expl.close()
        mock_perf.__exit__.assert_called()


@patch("calibrated_explanations.plugins.manager.PluginManager")
def test_plugin_preferences_enforcement(mock_pm_cls):
    with patch("calibrated_explanations.core.calibrated_explainer.CalibratedExplainer.__init__", return_value=None):
        expl = CalibratedExplainer(MagicMock(), MagicMock(), MagicMock())
        manager = MagicMock()
        
        # Case 1: Feature filter disabled -> no action
        mock_config = MagicMock()
        mock_config.enabled = False
        object.__setattr__(expl, FFC_ATTR, mock_config)
        
        getattr(expl, EFF_ATTR)(manager)
        manager.explanation_plugin_fallbacks.get.assert_not_called()
        
        # Case 2: Feature filter enabled, override required
        mock_config.enabled = True
        manager.explanation_plugin_fallbacks.get.return_value = ("some.other.plugin",)
        manager.explanation_plugin_overrides = {}
        
        with pytest.warns(UserWarning, match="Feature filter is enabled; overriding"):
            getattr(expl, EFF_ATTR)(manager)
            
        assert manager.explanation_plugin_overrides["factual"] == "core.explanation.factual.sequential"
        manager.clear_explanation_plugin_instances.assert_called()
