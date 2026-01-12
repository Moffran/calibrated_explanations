# tests/unit/external_plugins/test_shap_pipeline.py
import pytest
from unittest.mock import Mock, patch
from external_plugins.integrations.shap_pipeline import ShapPipeline
from calibrated_explanations.utils.exceptions import ConfigurationError

class TestShapPipeline:
    @pytest.fixture
    def explainer(self):
        return Mock()
    
    @pytest.fixture
    def shap_helper_mock(self):
        # Patch where it is imported in the SUT
        with patch('external_plugins.integrations.shap_pipeline.ShapHelper') as mock:
            yield mock

    def test_init(self, explainer):
        pipeline = ShapPipeline(explainer)
        assert pipeline.explainer == explainer

    def test_is_shap_enabled_lazy_init(self, explainer, shap_helper_mock):
        pipeline = ShapPipeline(explainer)
        # Should initialize helper
        pipeline.is_shap_enabled()
        shap_helper_mock.assert_called_once_with(explainer)
        shap_helper_mock.return_value.is_enabled.assert_called()

    def test_is_shap_enabled_set_true(self, explainer, shap_helper_mock):
        pipeline = ShapPipeline(explainer)
        pipeline.is_shap_enabled(True)
        shap_helper_mock.return_value.set_enabled.assert_called_with(True)
        shap_helper_mock.return_value.is_enabled.assert_called()

    def test_is_shap_enabled_already_initialized(self, explainer, shap_helper_mock):
        pipeline = ShapPipeline(explainer)
        
        # First call initializes
        pipeline.is_shap_enabled()
        shap_helper_mock.assert_called_once()
        
        # Second call should not re-initialize (still called once)
        pipeline.is_shap_enabled()
        shap_helper_mock.assert_called_once()
        
        # But method on helper is called twice
        assert shap_helper_mock.return_value.is_enabled.call_count == 2

    def test_preload_shap(self, explainer, shap_helper_mock):
        pipeline = ShapPipeline(explainer)
        pipeline.preload_shap(num_test=10)
        shap_helper_mock.assert_called_once_with(explainer)
        shap_helper_mock.return_value.preload.assert_called_with(num_test=10)

    def test_explain_success(self, explainer, shap_helper_mock):
        pipeline = ShapPipeline(explainer)
        
        # Setup mock return for preload
        mock_shap_explainer = Mock()
        mock_shap_explainer.return_value = "explanation_result"
        shap_helper_mock.return_value.preload.return_value = (mock_shap_explainer, "ref")
        
        x_test = [1, 2, 3]
        result = pipeline.explain(x_test, check_additivity=False)
        
        assert result == "explanation_result"
        # preload called with len(x_test)
        shap_helper_mock.return_value.preload.assert_called_with(num_test=3)
        # explainer called with x_test and kwargs
        mock_shap_explainer.assert_called_with(x_test, check_additivity=False)

    def test_explain_failure_missing_dep(self, explainer, shap_helper_mock):
        pipeline = ShapPipeline(explainer)
        
        # Simulate SHAP missing (preload returns None, None)
        shap_helper_mock.return_value.preload.return_value = (None, None)
        
        with pytest.raises(ConfigurationError, match="SHAP integration requested"):
            pipeline.explain([1])
