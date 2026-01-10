from calibrated_explanations.core.explain.orchestrator import ExplanationOrchestrator
from calibrated_explanations.core.reject.policy import RejectPolicy


class FakeRejectOrch:
    def __init__(self, value):
        self.value = value

    def apply_policy(self, policy, x, **kwargs):
        return {"policy": policy, "x": x, "value": self.value}


class FakeExplainer:
    def __init__(self, value="sentinel"):
        self.reject_orchestrator = FakeRejectOrch(value)
        self.default_reject_policy = RejectPolicy.NONE


def test_invoke_short_circuits_to_reject_orchestrator():
    fe = FakeExplainer(value=123)
    eo = ExplanationOrchestrator(fe)
    out = eo.invoke(
        mode="factual",
        x=[1, 2, 3],
        threshold=None,
        low_high_percentiles=None,
        bins=None,
        features_to_ignore=None,
        reject_policy=RejectPolicy.PREDICT_AND_FLAG,
    )
    assert out["value"] == 123
    assert out["policy"] == RejectPolicy.PREDICT_AND_FLAG


def test_invoke_applies_default_policy_when_not_supplied():
    fe = FakeExplainer(value="default")
    fe.default_reject_policy = RejectPolicy.EXPLAIN_ALL
    eo = ExplanationOrchestrator(fe)
    out = eo.invoke(
        mode="factual",
        x=[4, 5],
        threshold=None,
        low_high_percentiles=None,
        bins=None,
        features_to_ignore=None,
        reject_policy=None,
    )
    assert out["value"] == "default"
    assert out["policy"] == RejectPolicy.EXPLAIN_ALL
