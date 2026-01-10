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


class DummyRejectOrch:
    def __init__(self, return_value=None):
        self.called = False
        self.args = None
        self.return_value = return_value or RejectResult(prediction=None, explanation=None, rejected=None, policy=RejectPolicy.PREDICT_AND_FLAG, metadata={})

    def apply_policy(self, policy, x, explain_fn=None, bins=None, **kwargs):
        self.called = True
        self.args = dict(policy=policy, x=x, bins=bins)
        return self.return_value


def make_dummy_self():
    # Create a minimal object with the attributes used by invoke_explanation_plugin
    obj = SimpleNamespace()
    obj.explanation_orchestrator = DummyExplanationOrch()
    obj.reject_orchestrator = DummyRejectOrch()
    obj.plugin_manager = SimpleNamespace(initialize_orchestrators=lambda: None)
    obj.default_reject_policy = RejectPolicy.NONE
    return obj


def test_invoke_uses_legacy_when_none():
    s = make_dummy_self()
    # Call the unbound method with our dummy self
    result = CalibratedExplainer.invoke_explanation_plugin(
        s, mode="factual", x=[1, 2], threshold=None, low_high_percentiles=None, bins=None, features_to_ignore=None, extras=None, reject_policy=None
    )
    assert s.explanation_orchestrator.invoked is True
    assert result.get("legacy") is True


def test_invoke_delegates_to_reject_orchestrator_on_per_call():
    s = make_dummy_self()
    res = CalibratedExplainer.invoke_explanation_plugin(
        s, mode="factual", x=[1, 2], threshold=None, low_high_percentiles=None, bins=None, features_to_ignore=None, extras=None, reject_policy=RejectPolicy.PREDICT_AND_FLAG
    )
    # When policy non-NONE, the dummy reject orchestrator should be called and return its RejectResult
    assert s.reject_orchestrator.called is True
    assert isinstance(res, RejectResult)


def test_explainer_default_policy_used_when_no_per_call():
    s = make_dummy_self()
    s.default_reject_policy = RejectPolicy.EXPLAIN_ALL
    res = CalibratedExplainer.invoke_explanation_plugin(
        s, mode="factual", x=[3, 4], threshold=None, low_high_percentiles=None, bins=None, features_to_ignore=None, extras=None, reject_policy=None
    )
    assert s.reject_orchestrator.called is True
    assert isinstance(res, RejectResult)


def test_per_call_none_overrides_default_policy():
    s = make_dummy_self()
    s.default_reject_policy = RejectPolicy.EXPLAIN_ALL

    res = CalibratedExplainer.invoke_explanation_plugin(
        s,
        mode="factual",
        x=[5, 6],
        threshold=None,
        low_high_percentiles=None,
        bins=None,
        features_to_ignore=None,
        extras=None,
        reject_policy=RejectPolicy.NONE,
    )

    assert isinstance(res, dict)
    assert s.explanation_orchestrator.invoked is True
    assert s.reject_orchestrator.called is False
