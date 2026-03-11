"""Integration tests for reject-aware alternative explanations."""

from __future__ import annotations

import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy
from calibrated_explanations.explanations.explanations import AlternativeExplanations
from calibrated_explanations.explanations.reject import RejectAlternativeExplanations


@pytest.mark.integration
def test_reject_alternative_wrap_and_end_to_end():
    x, y = make_classification(n_samples=150, n_features=6, random_state=9)
    x_proper, x_cal, y_proper, y_cal = train_test_split(x, y, test_size=0.4, random_state=9)

    model = RandomForestClassifier(n_estimators=12, random_state=9)
    wrapper = WrapCalibratedExplainer(model)
    wrapper.fit(x_proper, y_proper)
    wrapper.calibrate(x_cal, y_cal, seed=9)
    wrapper.explainer.reject_orchestrator.initialize_reject_learner()

    alternatives = wrapper.explore_alternatives(x_cal[:10], reject_policy=RejectPolicy.FLAG)

    assert isinstance(alternatives, RejectAlternativeExplanations)
    assert "reject_rate" in alternatives.metadata

    ensured = alternatives.ensured_explanations()
    assert isinstance(ensured, RejectAlternativeExplanations)
    assert ensured.policy is RejectPolicy.FLAG


@pytest.mark.integration
def test_reject_alternative_getitem_wraps_plain_collection_result(monkeypatch):
    x, y = make_classification(n_samples=120, n_features=6, random_state=21)
    x_proper, x_cal, y_proper, y_cal = train_test_split(x, y, test_size=0.4, random_state=21)
    model = RandomForestClassifier(n_estimators=10, random_state=21)
    wrapper = WrapCalibratedExplainer(model)
    wrapper.fit(x_proper, y_proper)
    wrapper.calibrate(x_cal, y_cal, seed=21)
    wrapper.explainer.reject_orchestrator.initialize_reject_learner()

    reject_alt = wrapper.explore_alternatives(x_cal[:8], reject_policy=RejectPolicy.FLAG)
    plain_alt = wrapper.explore_alternatives(x_cal[:2])
    monkeypatch.setattr(AlternativeExplanations, "__getitem__", lambda _self, _key: plain_alt)

    wrapped = RejectAlternativeExplanations.__getitem__(reject_alt, slice(0, 2))
    assert isinstance(wrapped, RejectAlternativeExplanations)
    assert wrapped.policy is RejectPolicy.FLAG
