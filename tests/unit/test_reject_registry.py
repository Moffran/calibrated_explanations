import pytest
from types import SimpleNamespace

from calibrated_explanations.core.reject.orchestrator import RejectOrchestrator
from calibrated_explanations.core.reject.policy import RejectPolicy
from calibrated_explanations.explanations.reject import RejectResult


def make_stub():
    # Minimal explainer stub sufficient for RejectOrchestrator
    stub = SimpleNamespace()
    stub.x_cal = []
    stub.y_cal = []
    stub.mode = "classification"
    stub.interval_learner = SimpleNamespace()
    # simple predict to avoid AttributeErrors if invoked
    stub.predict = lambda x, **kw: [0 for _ in x]
    return stub


def test_builtin_default_registered():
    stub = make_stub()
    ro = RejectOrchestrator(stub)
    # builtin.default should resolve via the public resolver
    strat = ro.resolve_strategy(None)
    assert callable(strat)
    # also resolve by explicit identifier
    strat2 = ro.resolve_strategy("builtin.default")
    assert strat2 is strat


def test_register_custom_strategy_and_apply():
    stub = make_stub()
    ro = RejectOrchestrator(stub)

    def custom_strategy(policy, x, explain_fn=None, **kwargs):
        # Return a recognizable RejectResult
        return RejectResult(
            prediction=["ok"],
            explanation=None,
            rejected=[True],
            policy=RejectPolicy.PREDICT_AND_FLAG,
            metadata={"custom": True},
        )

    ro.register_strategy("custom.foo", custom_strategy)
    res = ro.apply_policy(
        RejectPolicy.PREDICT_AND_FLAG, x=[1], explain_fn=None, strategy="custom.foo"
    )
    assert isinstance(res, RejectResult)
    assert res.metadata is not None and res.metadata.get("custom") is True


def test_resolve_unknown_raises():
    stub = make_stub()
    ro = RejectOrchestrator(stub)
    with pytest.raises(KeyError):
        ro.resolve_strategy("no.such.strategy")
