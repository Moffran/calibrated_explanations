from lime.lime_tabular import LimeTabularExplainer
from shap import Explainer

from .core import CalibratedExplainer


# def preload_LIME(ce: CalibratedExplainer) -> None:
#     if not ce.is_LIME_enabled():
#         ce.lime = LimeTabularExplainer(ce.calX[:1,:], feature_names=ce.feature_names, class_names=['0','1'], mode=ce.mode)
#         ce.lime_exp = ce.lime.explain_instance(ce.calX[0,:], ce.va_model.predict_proba, num_features=ce.num_features)
#         ce.is_LIME_enabled(True)
#     return ce.lime, ce.lime_exp
    
# def preload_SHAP(ce: CalibratedExplainer) -> None:
#     if not ce.is_SHAP_enabled():
#         f = lambda x: ce.va_model.predict_proba(x)[:,1]
#         ce.shap = Explainer(f, ce.calX[:1,:], feature_names=ce.feature_names)
#         ce.shap_exp = ce.shap(ce.calX[0,:].reshape(1,-1))  
#         ce.is_SHAP_enabled(True)
#     return ce.shap, ce.shap_exp


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
        self.training_data = training_data
        self.training_labels = training_labels
        self.discretizer = discretizer
        self.categorical_features = categorical_features
        self.feature_names = feature_names
        assert training_labels is not None, "Calibrated Explanations requires training labels"

    def explain_instance(self, data_row, classifier, **kwargs):
        if self.calibrated_explainer is None:
            assert 'predict_proba' in dir(classifier), "The classifier must have a predict_proba method."
            self.calibrated_explainer = CalibratedExplainer(classifier, self.training_data, self.training_labels, self.feature_names, self.discretizer, self.categorical_features,)
            self.discretizer = self.calibrated_explainer.discretizer
        return self.calibrated_explainer(data_row).as_lime()[0]
    
class CalibratedAsShapExplainer(Explainer):
    '''
    Wrapper for the CalibratedExplainer to be used as a shap explainer.
    The masker must contain a data and a target field for the calibration set.
    The model must have a predict_proba method.
    '''
    def __init__(self, model, masker, feature_names=None, **kwargs):
        assert ['data','target'] in dir(masker), "The masker must contain a data and a target field for the calibration set."
        assert 'predict_proba' in dir(model), "The model must have a predict_proba method."
        self.calibrated_explainer = CalibratedExplainer(model, masker.data, masker.labels, feature_names)

    def __call__(self, *args, **kwargs):
        return self.calibrated_explainer(*args, **kwargs).as_shap()