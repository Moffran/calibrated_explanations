"""module containing wrappers for the LIME and SHAP explainers to use the CalibratedExplainer
"""
# pylint: disable=invalid-name, line-too-long, super-init-not-called, arguments-differ, unused-argument, too-many-arguments
# flake8: noqa: E501
from lime.lime_tabular import LimeTabularExplainer

from .core import CalibratedExplainer
from .utils import safe_import

shap = safe_import('shap')

class CalibratedAsLimeTabularExplainer(LimeTabularExplainer):
    '''
    Wrapper for LimeTabularExplainer to use calibrated explainer
    '''
    def __init__(self,
                 training_data,
                 mode="classification",
                 training_labels=None,
                 feature_names=None,
                 categorical_features=None,
                 discretizer='binaryEntropy',
                 **kwargs):
        self.calibrated_explainer = None
        self.mode = mode
        self.training_data = training_data
        self.training_labels = training_labels
        self.discretizer = discretizer
        self.categorical_features = categorical_features
        self.feature_names = feature_names
        assert training_labels is not None, "Calibrated Explanations requires training labels"

    def explain_instance(self, data_row, classifier, **kwargs):
        if self.calibrated_explainer is None:
            if self.mode == "classification":
                assert 'predict_proba' in dir(classifier), "The classifier must have a predict_proba method."
            else:
                assert 'predict' in dir(classifier), "The classifier must have a predict method."
            self.calibrated_explainer = CalibratedExplainer(classifier, self.training_data, self.training_labels, feature_names=self.feature_names, categorical_features=self.categorical_features, mode=self.mode)
        explanation = self.calibrated_explainer.explain_factual(data_row).as_lime()[0]
        self.discretizer = self.calibrated_explainer.discretizer
        return explanation

# pylint: disable=too-few-public-methods
class CalibratedAsShapExplainer(shap.Explainer):
    '''
    Wrapper for the CalibratedExplainer to be used as a shap explainer.
    The masker must contain a data and a target field for the calibration set.
    The model must have a predict_proba method.
    '''
    def __init__(self, model, calibration, feature_names=None, mode="classification", **kwargs):
        assert 'data' in calibration and 'target' in calibration, "The calibration must contain a data and a target field for the calibration set."
        if mode == "classification":
            assert 'predict_proba' in dir(model), "The classifier must have a predict_proba method."
        else:
            assert 'predict' in dir(model), "The classifier must have a predict method."
        self.calibrated_explainer = CalibratedExplainer(model, calibration['data'], calibration['target'], feature_names=feature_names, mode=mode)

    def __call__(self, *args, **kwargs):
        return self.calibrated_explainer.explain_factual(*args, **kwargs).as_shap()
    