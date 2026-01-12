from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.exceptions import ValidationError, DataShapeError, ConfigurationError
from external_plugins.integrations.lime_pipeline import LimePipeline

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
    explainer.categorical_features = []
    explainer.categorical_labels = None
    explainer.is_mondrian.return_value = False
    explainer.is_multiclass.return_value = False
    explainer.is_fast.return_value = False # LIME is not fast mode
    explainer.condition_source = "observed"
    # LIME pipeline might access this
    explainer.discretizer = None 
    explainer.learner = MagicMock()
    # Mock rule_boundaries to return 3 rules (one per feature)
    explainer.rule_boundaries.return_value = [["-inf", "inf"]] * 3
    
    # Prevent deepcopy to ensure FrozenCalibratedExplainer uses this original instance
    # instead of creating a fresh Mock that loses our configured attributes
    explainer.__deepcopy__ = MagicMock(side_effect=RuntimeError("Do not copy me"))

    return explainer

@pytest.fixture
def pipeline_fixture(explainer):
    # Patch LimeHelper for the duration of the test
    with patch("external_plugins.integrations.lime_pipeline.LimeHelper") as MockLimeHelper:
        mock_helper_instance = MockLimeHelper.return_value
        # preload returns (lime_explainer, reference_explanation)
        mock_helper_instance.preload.return_value = (MagicMock(), MagicMock())
        
        pipeline = LimePipeline(explainer)
        yield pipeline, mock_helper_instance

def test_explain_validates_mondrian_bins(pipeline_fixture, explainer):
    pipeline, _ = pipeline_fixture
    explainer.is_mondrian.return_value = True
    x_test = np.array([[1.0, 2.0, 3.0]])
    
    with pytest.raises(ValidationError, match="bins parameter must be specified"):
        pipeline.explain(x_test, bins=None)

    with pytest.raises(DataShapeError, match="length of the bins parameter"):
        pipeline.explain(x_test, bins=[1, 2])

def test_explain_validates_threshold_regression(pipeline_fixture, explainer):
    pipeline, _ = pipeline_fixture
    explainer.mode = "classification"
    x_test = np.array([[1.0, 2.0, 3.0]])
    
    with pytest.raises(ValidationError, match="threshold parameter is only supported"):
        pipeline.explain(x_test, threshold=0.5)

def test_explain_validates_input_shape(pipeline_fixture):
    pipeline, _ = pipeline_fixture
    with pytest.raises(DataShapeError, match="number of features"):
        pipeline.explain(np.array([[1, 2]]))

def test_explain_happy_path_classification(pipeline_fixture, explainer):
    pipeline, mock_helper = pipeline_fixture
    x_test = np.array([[1.0, 2.0, 3.0]])
    explainer.predict_calibrated.return_value = (
        np.array([0.5]), np.array([0.4]), np.array([0.6]), np.array([1])
    )
    
    # Configure mock
    mime_lime, _ = mock_helper.preload.return_value
    mime_lime.explain_instance.return_value = MagicMock()
    
    explanation = pipeline.explain(x_test)
    assert explanation is not None

def test_explain_validates_pandas(pipeline_fixture, explainer):
    pipeline, _ = pipeline_fixture
    import pandas as pd
    x_test = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["f1", "f2", "f3"])
    explainer.predict_calibrated.return_value = (
        np.array([0.5]), np.array([0.4]), np.array([0.6]), np.array([1])
    )
    pipeline.explain(x_test)

def test_explain_mondrian_valid(pipeline_fixture, explainer):
    pipeline, _ = pipeline_fixture
    explainer.is_mondrian.return_value = True
    x_test = np.array([[1.0, 2.0, 3.0]])
    explainer.predict_calibrated.return_value = (
        np.array([0.5]), np.array([0.4]), np.array([0.6]), np.array([1])
    )
    pipeline.explain(x_test, bins=[1])

def test_explain_regression_threshold(pipeline_fixture, explainer):
    pipeline, _ = pipeline_fixture
    explainer.mode = "regression"
    x_test = np.array([[1.0, 2.0, 3.0]])
    explainer.predict_calibrated.return_value = (
        np.array([0.5]), np.array([0.4]), np.array([0.6]), np.array([1])
    )
    pipeline.explain(x_test, threshold=0.5)

def test_explain_raises_configuration_error_if_lime_missing(pipeline_fixture, explainer):
    pipeline, mock_helper = pipeline_fixture
    mock_helper.preload.return_value = (None, None)
    
    x_test = np.array([[1.0, 2.0, 3.0]])
    explainer.predict_calibrated.return_value = (
        np.array([0.5]), np.array([0.4]), np.array([0.6]), np.array([1])
    )
    
    with pytest.raises(ConfigurationError, match="LIME integration requested"):
        pipeline.explain(x_test)

def test_explain_multiclass(pipeline_fixture, explainer):
    pipeline, _ = pipeline_fixture
    explainer.is_multiclass.return_value = True
    x_test = np.array([[1.0, 2.0, 3.0]])
    explainer.predict_calibrated.return_value = (
        np.array([0.5]), np.array([0.4]), np.array([0.6]), np.array([1])
    )
    pipeline.explain(x_test)

def test_explain_1d_input(pipeline_fixture, explainer):
    pipeline, mock_helper = pipeline_fixture
    x_test = np.array([1.0, 2.0, 3.0])  # 1D array
    
    explainer.predict_calibrated.return_value = (
        np.array([0.5]), np.array([0.4]), np.array([0.6]), np.array([1])
    )
    
    # Configure mock
    mime_lime, _ = mock_helper.preload.return_value
    mime_lime.explain_instance.return_value = MagicMock()
    
    explanation = pipeline.explain(x_test)
    assert explanation is not None

def test_explain_regression_no_threshold(pipeline_fixture, explainer):
    pipeline, mock_helper = pipeline_fixture
    explainer.mode = "regression"
    x_test = np.array([[1.0, 2.0, 3.0]])
    
    explainer.predict_calibrated.return_value = (
        np.array([0.5]), np.array([0.4]), np.array([0.6]), np.array([1])
    )
    
    # Configure mock
    mime_lime, _ = mock_helper.preload.return_value
    mime_lime.explain_instance.return_value = MagicMock()
    
    pipeline.explain(x_test, threshold=None)
 
