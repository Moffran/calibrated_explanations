import pytest
from types import SimpleNamespace

from calibrated_explanations.core.reject.orchestrator import RejectOrchestrator
from calibrated_explanations.core.reject.policy import RejectPolicy
from calibrated_explanations.explanations.reject import RejectResult


def make_stub():
    stub = SimpleNamespace()
    stub.x_cal = []
    stub.y_cal = []
    stub.mode = "classification"
    stub.interval_learner = SimpleNamespace()
    stub.predict = lambda x, **kw: [0 for _ in x]
    return stub


def test_apply_policy_none_and_invalid_policy_use_builtin():
    stub = make_stub()
    ro = RejectOrchestrator(stub)

    # explicit NONE -> no action via public apply_policy
    res = ro.apply_policy(RejectPolicy.NONE, x=[1, 2, 3])
    assert isinstance(res, RejectResult)
    assert res.prediction is None and res.explanation is None and res.rejected is None

    # unknown policy string is treated as NONE
    res2 = ro.apply_policy("not-a-policy", x=[1])
    assert isinstance(res2, RejectResult)
    assert res2.prediction is None and res2.rejected is None


def test_register_strategy_validation_errors():
    stub = make_stub()
    ro = RejectOrchestrator(stub)

    with pytest.raises(ValueError):
        ro.register_strategy("", lambda: None)

    with pytest.raises(ValueError):
        ro.register_strategy("valid.name", 123)
