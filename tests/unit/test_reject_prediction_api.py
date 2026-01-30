from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer


def test_prediction_internals_are_removed_from_calibrated_explainer():
    assert not hasattr(CalibratedExplainer, "predict_internal")
    assert not hasattr(CalibratedExplainer, "predict_calibrated")
    assert not hasattr(CalibratedExplainer, "_predict")
