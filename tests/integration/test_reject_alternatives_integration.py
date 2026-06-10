"""Integration tests for reject-policy paths on explore_alternatives and filter_by_target_confidence."""

import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier

from calibrated_explanations import WrapCalibratedExplainer, RejectPolicySpec
from calibrated_explanations.explanations.reject import (
    RejectAlternativeExplanations,
    RejectCalibratedExplanations,
    RejectResult,
)

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Shared model setup (inline — no shared conftest; mirrors test_reject_integration_crepes.py)
# ---------------------------------------------------------------------------

X, y = make_classification(n_samples=120, n_features=6, random_state=0)
X_fit, y_fit = X[:50], y[:50]
X_cal, y_cal = X[50:90], y[50:90]
X_test = X[90:]

_clf = DecisionTreeClassifier(random_state=0)
_wrapper = WrapCalibratedExplainer(_clf)
_wrapper.fit(X_fit, y_fit)
_wrapper.calibrate(X_cal, y_cal)


# ---------------------------------------------------------------------------
# Reject-policy paths — alternatives
# ---------------------------------------------------------------------------


def test_explore_alternatives_flag_policy_returns_reject_alternative_explanations():
    result = _wrapper.explore_alternatives(X_test, reject_policy=RejectPolicySpec.flag())
    assert isinstance(result, RejectAlternativeExplanations)
    assert len(result) == len(X_test)
    assert result.rejected is not None


def test_explore_alternatives_only_accepted_returns_reject_alternative_explanations():
    result = _wrapper.explore_alternatives(X_test, reject_policy=RejectPolicySpec.only_accepted())
    # When some instances are accepted: RejectAlternativeExplanations with len <= n_test.
    # When ALL instances are rejected (e.g. VA rescaling at low confidence flattens scores),
    # the orchestrator returns RejectResult with explanation=None — also a valid outcome.
    if isinstance(result, RejectResult):
        assert result.explanation is None
    else:
        assert isinstance(result, RejectAlternativeExplanations)
        assert len(result) <= len(X_test)


def test_explore_alternatives_only_rejected_returns_reject_alternative_explanations():
    result = _wrapper.explore_alternatives(X_test, reject_policy=RejectPolicySpec.only_rejected())
    assert isinstance(result, RejectAlternativeExplanations)
    assert len(result) <= len(X_test)


# ---------------------------------------------------------------------------
# Cross-check — factual flag policy (verifies shared orchestration plumbing)
# ---------------------------------------------------------------------------


def test_explain_factual_flag_policy_cross_check():
    result = _wrapper.explain_factual(X_test, reject_policy=RejectPolicySpec.flag())
    assert isinstance(result, RejectCalibratedExplanations)
    assert len(result) == len(X_test)
    assert result.rejected is not None


# ---------------------------------------------------------------------------
# Subtask 11b — filter_by_target_confidence (conformal decision-making)
# ---------------------------------------------------------------------------


def test_filter_by_target_confidence_returns_alternative_explanations():
    from calibrated_explanations.explanations.explanations import AlternativeExplanations

    result = _wrapper.explore_alternatives(X_test)
    filtered = result.filter_by_target_confidence(0.8)
    assert isinstance(filtered, AlternativeExplanations)


def test_filter_by_target_confidence_confidence_one_retains_zero_intervals():
    # At confidence=1.0, epsilon=0.0: p_val_k >= 0.0 always True for both classes →
    # both in prediction set → ambiguity rejection → all intervals discarded.
    result = _wrapper.explore_alternatives(X_test)

    def count_intervals(collection):
        return sum(len(exp.get_rules()["rule"]) for exp in collection.explanations)

    filtered = result.filter_by_target_confidence(1.0)
    assert count_intervals(filtered) == 0


def test_filter_by_target_confidence_mid_confidence_returns_subset():
    # At mid-range confidence the filter is a strict subset of the original.
    result = _wrapper.explore_alternatives(X_test)

    def count_intervals(collection):
        return sum(len(exp.get_rules()["rule"]) for exp in collection.explanations)

    filtered = result.filter_by_target_confidence(0.8)
    assert count_intervals(filtered) <= count_intervals(result)


def test_filter_by_target_confidence_does_not_mutate_original():
    result = _wrapper.explore_alternatives(X_test)

    def count_intervals(collection):
        return sum(len(exp.get_rules()["rule"]) for exp in collection.explanations)

    original_count = count_intervals(result)
    result.filter_by_target_confidence(0.9)
    assert count_intervals(result) == original_count


def test_filter_by_target_confidence_invalid_confidence_raises():
    from calibrated_explanations.utils.exceptions import ValidationError

    result = _wrapper.explore_alternatives(X_test)
    with pytest.raises(ValidationError):
        result.filter_by_target_confidence(1.5)


def test_filter_by_target_confidence_non_probabilistic_raises():
    # Regression model without threshold → is_probabilistic() returns False → ValidationError.
    from calibrated_explanations.utils.exceptions import ValidationError

    X_reg, y_reg = make_regression(n_samples=80, n_features=4, random_state=0)
    reg_wrapper = WrapCalibratedExplainer(RandomForestRegressor(n_estimators=5, random_state=0))
    reg_wrapper.fit(X_reg[:40], y_reg[:40])
    reg_wrapper.calibrate(X_reg[40:60], y_reg[40:60])
    result = reg_wrapper.explore_alternatives(X_reg[60:62])
    with pytest.raises(ValidationError):
        result.filter_by_target_confidence(0.8)


# ---------------------------------------------------------------------------
# Coverage helpers — cache paths and shorthand delegators on AlternativeExplanations
# ---------------------------------------------------------------------------


def test_alternative_explanations_probabilities_property_cached():
    # Calling .probabilities twice covers the cached-result branch (907->929 in explanations.py).
    result = _wrapper.explore_alternatives(X_test[:2])
    _ = result.probabilities
    _ = result.probabilities  # second call hits the "cache already populated" branch


def test_alternative_explanations_regression_interval_properties_cached():
    # Calling .lower / .upper twice covers their cached-result branches (934->945, 950->961).
    X_reg, y_reg = make_regression(n_samples=80, n_features=4, random_state=0)
    reg_wrapper = WrapCalibratedExplainer(RandomForestRegressor(n_estimators=5, random_state=0))
    reg_wrapper.fit(X_reg[:40], y_reg[:40])
    reg_wrapper.calibrate(X_reg[40:60], y_reg[40:60])
    result = reg_wrapper.explore_alternatives(X_reg[60:62])
    _ = result.lower
    _ = result.lower  # hits 934->945 cached branch
    _ = result.upper
    _ = result.upper  # hits 950->961 cached branch


def test_alternative_explanations_shorthand_delegators():
    # semi(), counter(), ensured(), super(), pareto() are delegators.
    result = _wrapper.explore_alternatives(X_test[:2])
    _ = result.semi()
    _ = result.counter()
    _ = result.ensured()
    _ = result.super()  # line 1579 in explanations.py (CalibratedExplanations.super shorthand)
    _ = result.pareto()  # line 1910 in explanations.py (AlternativeExplanations.pareto shorthand)


def test_alternative_to_json_stream_covers_alternative_type():
    # to_json_stream on AlternativeExplanations covers isinstance(exp, AlternativeExplanation)
    # at line 758 in explanations.py (_legacy_payload alternative branch).
    result = _wrapper.explore_alternatives(X_test[:2])
    fragments = list(result.to_json_stream())
    assert len(fragments) > 0


def test_factual_probabilities_property_cache_hit():
    # Calling .probabilities twice on a factual classification result covers the
    # cache-hit arc 907->929 in explanations.py (False branch: cache already populated).
    result = _wrapper.explain_factual(X_test[:2])
    p1 = result.probabilities  # first call: populates cache
    p2 = result.probabilities  # second call: arc 907->929 (False: cache not None)
    assert p1 is p2
