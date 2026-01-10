from types import SimpleNamespace

from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy
from calibrated_explanations.explanations.reject import RejectResult


class DummyPredOrch:
    def predict_internal(self, x, **kw):
        return ("PRED", None, None, None)


class DummyRejectOrch:
    def __init__(self):
        self.called = False
        self.last_policy = None

    def apply_policy(self, policy, x, explain_fn=None, **kw):
        self.called = True
        self.last_policy = policy
        # call the provided predict fn to produce the normal prediction
        pred = None
        if explain_fn is not None:
            pred = explain_fn(x)
        return RejectResult(prediction=pred, explanation=None, rejected=False, policy=policy, metadata={})


def test_predict_delegates_to_reject_orchestrator_for_non_none_policy():
    dummy = SimpleNamespace()
    dummy.prediction_orchestrator = DummyPredOrch()
    dummy.reject_orchestrator = DummyRejectOrch()
    dummy.plugin_manager = SimpleNamespace(initialize_orchestrators=lambda: None)

    # Use the unbound method from CalibratedExplainer to simulate instance call
    result = CalibratedExplainer.predict_internal(
        dummy, x=[1, 2, 3], reject_policy=RejectPolicy.PREDICT_AND_FLAG
    )

    assert isinstance(result, RejectResult)
    assert dummy.reject_orchestrator.called is True
    assert dummy.reject_orchestrator.last_policy == RejectPolicy.PREDICT_AND_FLAG


def test_predict_uses_explainer_default_policy_when_present():
    dummy = SimpleNamespace()
    dummy.prediction_orchestrator = DummyPredOrch()
    dummy.reject_orchestrator = DummyRejectOrch()
    dummy.plugin_manager = SimpleNamespace(initialize_orchestrators=lambda: None)
    dummy.default_reject_policy = RejectPolicy.EXPLAIN_ALL

    result = CalibratedExplainer.predict_internal(dummy, x=[10])

    assert isinstance(result, RejectResult)
    assert dummy.reject_orchestrator.called is True
    assert dummy.reject_orchestrator.last_policy == RejectPolicy.EXPLAIN_ALL


def test_per_call_none_overrides_explainer_default_policy():
    dummy = SimpleNamespace()
    dummy.prediction_orchestrator = DummyPredOrch()
    dummy.reject_orchestrator = DummyRejectOrch()
    dummy.plugin_manager = SimpleNamespace(initialize_orchestrators=lambda: None)
    dummy.default_reject_policy = RejectPolicy.EXPLAIN_ALL

    result = CalibratedExplainer.predict_internal(
        dummy, x=[10], reject_policy=RejectPolicy.NONE
    )

    assert isinstance(result, tuple)
    assert dummy.reject_orchestrator.called is False
