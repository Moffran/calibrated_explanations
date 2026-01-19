import pytest

from calibrated_explanations.core.reject.orchestrator import RejectOrchestrator
from calibrated_explanations.core.reject.policy import RejectPolicy
from calibrated_explanations.explanations.reject import RejectResult
from calibrated_explanations.utils.exceptions import ValidationError


class DummyExplainer:
    def __init__(self):
        self.mode = "classification"


def test_register_and_resolve_strategy_and_apply():
    expl = DummyExplainer()
    ro = RejectOrchestrator(expl)

    # builtin.default should be registered
    default = ro.resolve_strategy(None)
    assert callable(default)

    # apply NONE policy returns empty envelope
    res = ro.apply_policy(RejectPolicy.NONE, x=[1, 2, 3])
    assert isinstance(res, RejectResult)
    assert res.prediction is None and res.explanation is None

    # register a custom strategy
    def strategy(policy, x, **kwargs):
        return RejectResult(
            prediction=[42],
            explanation=None,
            rejected=[False],
            policy=policy,
            metadata={"ok": True},
        )

    ro.register_strategy("my.strategy", strategy)
    resolved = ro.resolve_strategy("my.strategy")
    assert resolved is strategy

    out = ro.apply_policy(RejectPolicy.PREDICT_AND_FLAG, x=[1], strategy="my.strategy")
    assert out.prediction == [42]
    assert out.metadata == {"ok": True}


def test_register_strategy_invalid_inputs():
    expl = DummyExplainer()
    ro = RejectOrchestrator(expl)
    with pytest.raises(ValidationError):
        ro.register_strategy("", lambda: None)
    with pytest.raises(ValidationError):
        ro.register_strategy("name", object())


def test_resolve_unknown_strategy_raises():
    expl = DummyExplainer()
    ro = RejectOrchestrator(expl)
    with pytest.raises(KeyError):
        ro.resolve_strategy("no.such.strategy")
