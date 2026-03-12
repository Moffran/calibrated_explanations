"""Tests for robust reject policy spec resolution formats."""

from __future__ import annotations

import pytest

from calibrated_explanations.core.reject.orchestrator import (
    resolve_effective_reject_policy,
    resolve_policy_spec,
)
from calibrated_explanations.core.reject.policy import RejectPolicy
from calibrated_explanations.explanations.reject import RejectPolicySpec
from calibrated_explanations.utils.exceptions import ValidationError


class DummyRejectOrchestrator:
    def __init__(self, explainer):
        self.explainer = explainer

    def initialize_reject_learner(self, ncf=None, w=0.5):
        self.explainer.calls.append((ncf, w))
        self.explainer.reject_ncf = ncf
        self.explainer.reject_ncf_w = w
        return self


class DummyPluginManager:
    def __init__(self, explainer):
        self.explainer = explainer

    def initialize_orchestrators(self):
        if self.explainer.reject_orchestrator is None:
            self.explainer.reject_orchestrator = DummyRejectOrchestrator(self.explainer)


class DummyExplainer:
    def __init__(self):
        self.reject_ncf = "default"
        self.reject_ncf_w = 0.0
        self.calls = []
        self.reject_orchestrator = DummyRejectOrchestrator(self)
        self.plugin_manager = DummyPluginManager(self)


def test_resolve_policy_spec_accepts_policy_spec():
    explainer = DummyExplainer()
    spec = RejectPolicySpec.flag(ncf="default", w=0.33)
    resolved = resolve_policy_spec(spec, explainer)
    assert resolved is RejectPolicy.FLAG
    assert explainer.calls == []


def test_resolve_policy_spec_accepts_dict_and_json_and_value_string():
    explainer = DummyExplainer()

    resolved_dict = resolve_policy_spec(
        {"policy": "flag", "ncf": "default", "w": 0.33},
        explainer,
    )
    assert resolved_dict is RejectPolicy.FLAG
    assert explainer.calls == []

    with pytest.raises(ValidationError, match="JSON string reject policy payloads are unsupported"):
        _ = resolve_policy_spec('{"policy": "flag", "ncf": "entropy", "w": 0.4}', explainer)

    with pytest.raises(ValidationError, match="Unknown reject policy string"):
        _ = resolve_policy_spec("only_rejected[ncf=margin,w=0.25]", explainer)


def test_resolve_policy_spec_accepts_plain_policy_string_and_enum():
    explainer = DummyExplainer()
    assert resolve_policy_spec("flag", explainer) is RejectPolicy.FLAG
    assert resolve_policy_spec("FLAG", explainer) is RejectPolicy.FLAG
    assert resolve_policy_spec(RejectPolicy.FLAG, explainer) is RejectPolicy.FLAG


def test_resolve_policy_spec_accepts_whitespace_value_string():
    explainer = DummyExplainer()
    with pytest.raises(ValidationError, match="Unknown reject policy string"):
        _ = resolve_policy_spec(" FLAG [ ncf = margin , w = .5 ] ", explainer)


def test_resolve_policy_spec_invalid_json_raises_validation_error():
    explainer = DummyExplainer()
    with pytest.raises(ValidationError, match="JSON string reject policy payloads are unsupported"):
        _ = resolve_policy_spec('{"policy": "flag",', explainer)


def test_resolve_policy_spec_accepts_none():
    explainer = DummyExplainer()
    assert resolve_policy_spec(None, explainer) is None


def test_resolve_policy_spec_unsupported_type_raises_validation_error():
    explainer = DummyExplainer()
    with pytest.raises(ValidationError, match="Unsupported reject_policy input type"):
        _ = resolve_policy_spec(123, explainer)


def test_resolve_policy_spec_initializes_orchestrator_when_missing():
    explainer = DummyExplainer()
    explainer.reject_orchestrator = None
    resolved = resolve_policy_spec({"policy": "flag", "ncf": "ensured", "w": 0.33}, explainer)
    assert resolved is RejectPolicy.FLAG
    assert explainer.calls == [("ensured", 0.33)]
    assert explainer.reject_orchestrator is not None


def test_resolve_policy_spec_does_not_reinit_when_only_ignored_w_differs():
    explainer = DummyExplainer()
    explainer.reject_ncf = "default"
    explainer.reject_ncf_w = 0.0

    resolved = resolve_policy_spec({"policy": "flag", "ncf": "default", "w": 0.91}, explainer)
    assert resolved is RejectPolicy.FLAG
    assert explainer.calls == []


def test_resolve_policy_spec_maps_entropy_to_default():
    explainer = DummyExplainer()
    resolved = resolve_policy_spec({"policy": "flag", "ncf": "entropy", "w": 0.4}, explainer)
    assert resolved is RejectPolicy.FLAG
    assert explainer.calls == []


@pytest.mark.parametrize("ncf", ["hinge", "margin"])
def test_resolve_policy_spec_rejects_removed_explicit_ncfs(ncf):
    explainer = DummyExplainer()
    with pytest.raises(ValidationError, match="no longer supported"):
        _ = resolve_policy_spec({"policy": "flag", "ncf": ncf, "w": 0.5}, explainer)


def test_resolve_effective_reject_policy_uses_explicit_policy():
    explainer = DummyExplainer()
    res = resolve_effective_reject_policy(
        "flag",
        explainer,
        default_policy=RejectPolicy.NONE,
    )
    assert res.policy is RejectPolicy.FLAG
    assert res.used_default is False
    assert res.fallback_used is False
    assert res.reason is None


def test_resolve_effective_reject_policy_invalid_explicit_fails_fast():
    explainer = DummyExplainer()
    with pytest.raises(ValidationError, match="Unknown reject policy string"):
        _ = resolve_effective_reject_policy(
            "not-a-policy",
            explainer,
            default_policy=RejectPolicy.NONE,
        )


def test_resolve_effective_reject_policy_invalid_default_falls_back_with_warning():
    explainer = DummyExplainer()
    with pytest.warns(UserWarning, match="Invalid default_reject_policy"):
        res = resolve_effective_reject_policy(
            None,
            explainer,
            default_policy="not-a-policy",
        )
    assert res.policy is RejectPolicy.NONE
    assert res.used_default is True
    assert res.fallback_used is True
    assert res.reason == "invalid_default_reject_policy"
