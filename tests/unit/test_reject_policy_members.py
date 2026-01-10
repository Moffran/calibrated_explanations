from types import SimpleNamespace

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy
from calibrated_explanations.explanations.reject import RejectResult


class DummyExplanationOrch:
    def __init__(self):
        self.invoked = False

    def invoke(
        self,
        mode,
        x,
        threshold,
        low_high_percentiles,
        bins,
        features_to_ignore,
        extras=None,
        reject_policy=None,
        _ce_skip_reject=False,
    ):
        self.invoked = True
        return {"legacy": True, "x": x}


class DummyPredOrch:
    def predict_internal(self, x, **kw):
        return ("PRED", None, None, None)


class DummyRejectOrch:
    def __init__(self, return_value=None):
        self.called = False
        self.args = None
        self.last_policy = None
        self.return_value = return_value or RejectResult(prediction=None, explanation=None, rejected=None, policy=RejectPolicy.PREDICT_AND_FLAG, metadata={})

    def apply_policy(self, policy, x, explain_fn=None, bins=None, **kwargs):
        self.called = True
        self.last_policy = policy
        self.args = dict(policy=policy, x=x, bins=bins)
        # If an explain_fn/predict_fn was provided, call it to emulate inner behavior
        if explain_fn is not None:
            try:
                inner = explain_fn(x)
            except Exception:
                inner = None
            return self.return_value
        return self.return_value


def make_dummy_explain_self():
    obj = SimpleNamespace()
    obj.explanation_orchestrator = DummyExplanationOrch()
    obj.reject_orchestrator = DummyRejectOrch()
    obj.plugin_manager = SimpleNamespace(initialize_orchestrators=lambda: None)
    obj.default_reject_policy = None
    return obj


def make_dummy_predict_self():
    obj = SimpleNamespace()
    obj.prediction_orchestrator = DummyPredOrch()
    obj.reject_orchestrator = DummyRejectOrch()
    obj.plugin_manager = SimpleNamespace(initialize_orchestrators=lambda: None)
    obj.default_reject_policy = None
    return obj


def test_invoke_supports_all_policies():
    for policy in list(RejectPolicy):
        s = make_dummy_explain_self()
        # Call the unbound method
        res = CalibratedExplainer.invoke_explanation_plugin(
            s,
            mode="factual",
            x=[1],
            threshold=None,
            low_high_percentiles=None,
            bins=None,
            features_to_ignore=None,
            extras=None,
            reject_policy=policy,
        )

        if policy is RejectPolicy.NONE:
            assert s.explanation_orchestrator.invoked is True
            assert isinstance(res, dict) and res.get("legacy") is True
        else:
            assert s.reject_orchestrator.called is True
            assert isinstance(res, RejectResult)
            assert s.reject_orchestrator.last_policy == policy


def test_predict_supports_all_policies():
    for policy in list(RejectPolicy):
        s = make_dummy_predict_self()
        res = CalibratedExplainer.predict_internal(
            s,
            x=[1, 2],
            reject_policy=policy,
        )

        if policy is RejectPolicy.NONE:
            assert isinstance(res, tuple)
        else:
            assert s.reject_orchestrator.called is True
            assert isinstance(res, RejectResult)
            assert s.reject_orchestrator.last_policy == policy
