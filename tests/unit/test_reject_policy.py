from calibrated_explanations.core.reject.policy import RejectPolicy, is_policy_enabled
from calibrated_explanations.explanations.reject import RejectResult
from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer
from calibrated_explanations.core.reject.orchestrator import RejectOrchestrator
from types import SimpleNamespace


def test_policy_enum_and_helper():
    assert RejectPolicy.NONE.name == "NONE"
    assert is_policy_enabled(RejectPolicy.NONE) is False
    assert is_policy_enabled(RejectPolicy.PREDICT_AND_FLAG) is True


def test_reject_result_dataclass():
    r = RejectResult(prediction=[1, 0], explanation=None, rejected=[False, True], policy=RejectPolicy.PREDICT_AND_FLAG, metadata={"x": 1})
    assert r.policy == RejectPolicy.PREDICT_AND_FLAG
    assert r.rejected[1] is True


def test_invoke_policy_initializes_orchestrator():
    # Minimal explainer stub that provides required attributes
    stub = SimpleNamespace(x_cal=[], y_cal=[], mode='classification', interval_learner=SimpleNamespace())
    expl = SimpleNamespace()
    expl.reject_learner = None
    # Create RejectOrchestrator with a lightweight explainer stub
    ro = RejectOrchestrator(stub)
    # Applying NONE returns a RejectResult with policy NONE
    res = ro.apply_policy(RejectPolicy.NONE, x=[1,2], explain_fn=None)
    assert res.policy == RejectPolicy.NONE
