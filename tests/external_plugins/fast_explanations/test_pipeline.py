from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.explanations import CalibratedExplanations
from external_plugins.fast_explanations.pipeline import FastExplanationPipeline


@pytest.fixture
def explainer():
    explainer = MagicMock(spec=CalibratedExplainer)
    explainer.num_features = 3
    explainer.feature_names = ["f1", "f2", "f3"]
    explainer.x_cal = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ])
    explainer.y_cal = np.array([0, 1])
    explainer.scaled_y_cal = np.array([0, 1])
    explainer.features_to_ignore = []
    explainer.mode = "classification"
    explainer.is_multiclass.return_value = False
    explainer.is_mondrian.return_value = False
    explainer.is_fast.return_value = True
    explainer.condition_source = "observed"
    explainer.categorical_features = []
    explainer.categorical_labels = None
    
    # Prevent deepcopy to allow FrozenCalibratedExplainer to work with this mock
    explainer.__deepcopy__ = MagicMock(side_effect=RuntimeError("Do not copy me"))
    
    return explainer

@pytest.fixture
def pipeline(explainer):
    return FastExplanationPipeline(explainer)

def test_preprocess_identifies_constant_columns(pipeline, explainer):
    explainer.x_cal = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 5.0, 3.0],
        [1.0, 8.0, 3.0],
    ])
    # Column 0 (all 1.0) and 2 (all 3.0) are constant
    pipeline.preprocess()
    assert explainer.features_to_ignore == [0, 2]

def test_discretize(pipeline, explainer):
    explainer.discretizer = MagicMock()
    explainer.discretizer.to_discretize = [1]
    explainer.discretizer.mins = {1: [-np.inf, 1.0, 5.0, 10.0]}
    explainer.discretizer.means = {1: [0.5, 3.0, 7.5, 12.0]}
    
    x_test = np.array([[1.0, 2.5, 3.0]])
    
    discretized = pipeline.discretize(x_test)
    
    assert discretized.shape == x_test.shape
    assert discretized[0, 0] == 1.0
    assert discretized[0, 2] == 3.0
    assert discretized[0, 1] != 2.5

def test_explain_validates_fast_mode(pipeline, explainer):
    explainer.is_fast.return_value = False
    explainer.enable_fast_mode.side_effect = Exception("Failed to enable")
    
    with pytest.raises(Exception, match="Fast explanations are only possible"):
        pipeline.explain(np.array([[1, 2, 3]]))

def test_explain_validates_input_shape(pipeline):
    with pytest.raises(Exception, match="number of features"):
        pipeline.explain(np.array([[1, 2]])) 

def test_rule_boundaries_single_instance(pipeline, explainer):
    explainer.discretizer = MagicMock()
    explainer.discretizer.to_discretize = [0]
    explainer.discretizer.mins = {0: [-np.inf, 0, 5, 10]}
    explainer.discretizer.maxs = {0: [0, 5, 10, np.inf]}
    explainer.discretizer.means = {0: [-1, 2.5, 7.5, 12]}
    
    instance = np.array([2.5, 2.0, 3.0]) 
    
    boundaries = pipeline.rule_boundaries(instance)
    
    assert len(boundaries) == 3
    assert boundaries[0] == [0, 5]
    assert boundaries[1] == [2.0, 2.0]

def test_explain_happy_path_classification(pipeline, explainer):
    x_test = np.array([[1.0, 2.0, 3.0]])
    
    # Mock predictions
    n_samples = x_test.shape[0]
    n_features = 3
    predict = np.array([0.8])
    low = np.array([0.7])
    high = np.array([0.9])
    predicted_class = np.array([1])
    
    explainer.predict_calibrated.return_value = (predict, low, high, predicted_class)
    
    # Mock interval learner for full probs
    mock_learner = MagicMock()
    mock_learner.predict_proba.return_value = np.array([[0.2, 0.8]])
    explainer.interval_learner = {n_features: mock_learner}
    
    # Mock compute_feature_effects
    # It returns a list of tuples:
    # (feature_index, weights_predict, weights_low, weights_high, local_predict, local_low, local_high)
    
    # Patch deepcopy to return the original explainer, avoiding Mock copy issues
    with patch("external_plugins.fast_explanations.pipeline.compute_feature_effects") as mock_compute, \
         patch("calibrated_explanations.explanations.explanations.deepcopy", side_effect=lambda x: x):

        mock_compute.return_value = [
            (0, [0.1], [0.1], [0.1], [0], [0], [0]),
            (1, [0.1], [0.1], [0.1], [0], [0], [0]),
            (2, [0.1], [0.1], [0.1], [0], [0], [0]),
        ]
        
        explanation = pipeline.explain(x_test)
        
        assert isinstance(explanation, CalibratedExplanations)
        assert explainer.predict_calibrated.called
        assert mock_compute.called

def test_explain_happy_path_regression(pipeline, explainer):
    explainer.mode = "regression"
    x_test = np.array([[1.0, 2.0, 3.0]])
    
    predict = np.array([0.5])
    low = np.array([0.4])
    high = np.array([0.6])
    predicted_class = np.ones(1) # For regression class is ignored usually
    
    explainer.predict_calibrated.return_value = (predict, low, high, predicted_class)
    
    with patch("external_plugins.fast_explanations.pipeline.compute_feature_effects") as mock_compute, \
         patch("calibrated_explanations.explanations.explanations.deepcopy", side_effect=lambda x: x):
         
        mock_compute.return_value = [
            (0, [0.1], [0.1], [0.1], [0], [0], [0]),
            (1, [0.1], [0.1], [0.1], [0], [0], [0]),
            (2, [0.1], [0.1], [0.1], [0], [0], [0]),
        ]
        
        explanation = pipeline.explain(x_test, threshold=0.5)
        # Threshold for regression is valid
        assert explanation is not None

def test_rule_boundaries_batch(pipeline, explainer):
    explainer.discretizer = MagicMock()
    explainer.discretizer.to_discretize = [0]
    explainer.discretizer.mins = {0: [-np.inf, 0, 5, 10]}
    explainer.discretizer.maxs = {0: [0, 5, 10, np.inf]}
    explainer.discretizer.means = {0: [-1, 2.5, 7.5, 12]}
    
    instances = np.array([
        [2.5, 2.0, 3.0], 
        [7.5, 2.0, 3.0]
    ])
    
    boundaries = pipeline.rule_boundaries(instances)
    
    assert boundaries.shape[0] == 2
    assert boundaries.shape[1] == 3
    # Instance 0 (2.5 -> bin 0-5)
    assert np.all(boundaries[0][0] == [0, 5])
    # Instance 1 (7.5 -> bin 5-10)
    assert np.all(boundaries[1][0] == [5, 10])

def test_explain_1d_input(pipeline, explainer):
    x_test = np.array([1.0, 2.0, 3.0]) # 1D
    
    predict = np.array([0.8])
    low = np.array([0.7])
    high = np.array([0.9])
    predicted_class = np.array([1])
    explainer.predict_calibrated.return_value = (predict, low, high, predicted_class)
    
    explanation = pipeline.explain(x_test)
    assert explanation is not None

def test_explain_mondrian_bins_mismatch(pipeline, explainer):
    explainer.is_mondrian.return_value = True
    x_test = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    
    # bins length 1 vs 2 instances
    with pytest.raises(Exception, match="length of the bins"):
         pipeline.explain(x_test, bins=[1])

def test_explain_regression_no_threshold(pipeline, explainer):
    explainer.mode = "regression"
    x_test = np.array([[1.0, 2.0, 3.0]])
    
    predict = np.array([0.5])
    low = np.array([0.4])
    high = np.array([0.6])
    predicted_class = np.array([1])
    explainer.predict_calibrated.return_value = (predict, low, high, predicted_class)
    
    # Should work without threshold
    explanation = pipeline.explain(x_test, threshold=None)
    assert explanation is not None



