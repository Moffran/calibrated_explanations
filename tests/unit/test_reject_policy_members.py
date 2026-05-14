from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer


def test_removed_explanation_plugin_delegator_fails_closed():
    assert not hasattr(CalibratedExplainer, "invoke_explanation_plugin")


def test_predict_internal_delegators_remain_removed():
    assert not hasattr(CalibratedExplainer, "predict_internal")
    assert not hasattr(CalibratedExplainer, "predict_calibrated")
    assert not hasattr(CalibratedExplainer, "_predict")
