"""Integration tests for Solution 1: Transparent Reject Integration."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations import WrapCalibratedExplainer
from calibrated_explanations.core.reject.policy import RejectPolicy
from calibrated_explanations.explanations.reject import RejectCalibratedExplanations
from calibrated_explanations.explanations.explanation import FactualExplanation

pytestmark = pytest.mark.integration


def test_should_return_rejected_collection_subclass_when_explain_factual_called_with_reject_policy():
    # Arrange
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.5, random_state=42)

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)

    w = WrapCalibratedExplainer(clf)
    w.calibrate(X_cal, y_cal)

    # Initialize reject learner to ensure reject logic runs
    w.explainer.reject_orchestrator.initialize_reject_learner()

    X_test = X_cal[:5]

    # Act
    res = w.explain_factual(X_test, reject_policy=RejectPolicy.FLAG)

    # Assert
    assert isinstance(res, RejectCalibratedExplanations)
    assert hasattr(res, "ambiguity_mask")
    assert isinstance(res.ambiguity_mask, np.ndarray)
    assert len(res.ambiguity_mask) == 5
    assert len(res.rejected) == len(res.explanations)
    assert res.metadata["source_indices"] == list(range(5))
    assert res.metadata["original_count"] == 5

    # Check indexing
    item = res[0]
    assert isinstance(item, FactualExplanation)
    # Check that item has reject context (attached by orchestrator)
    if hasattr(item, "reject_context"):
        assert item.reject_context is not None

    # Check slicing behavior (Mixin logic)
    subset = res[1:3]
    assert isinstance(subset, RejectCalibratedExplanations)
    assert len(subset.explanations) == 2
    assert len(subset.ambiguity_mask) == 2
    assert subset.metadata["source_indices"] == res.metadata["source_indices"][1:3]
    assert subset.metadata["original_count"] == res.metadata["original_count"]
    # Verify values match
    assert subset.ambiguity_mask[0] == res.ambiguity_mask[1]

    # Check add_conjunctions support (inherited)
    res_conj = res.add_conjunctions(n_top_features=2)
    assert res_conj is res  # returns self
    assert res[0].has_conjunctive_rules

    # Check plot support (inherited)
    try:
        res.plot(show=False)
    except Exception as e:
        pytest.fail(f"plot() failed on RejectCalibratedExplanations: {e}")


def test_subset_policy_returns_alignment_safe_reject_wrapper():
    X, y = make_classification(n_samples=140, n_features=4, random_state=7)
    X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.5, random_state=7)

    clf = RandomForestClassifier(n_estimators=12, random_state=7)
    clf.fit(X_train, y_train)

    w = WrapCalibratedExplainer(clf)
    w.calibrate(X_cal, y_cal)
    w.explainer.reject_orchestrator.initialize_reject_learner()

    X_test = X_cal[:10]
    res = w.explain_factual(X_test, reject_policy=RejectPolicy.ONLY_ACCEPTED)
    assert isinstance(res, RejectCalibratedExplanations)

    if res.rejected is not None:
        assert len(res.rejected) == len(res.explanations)
    if res.ambiguity_mask is not None:
        assert len(res.ambiguity_mask) == len(res.explanations)
    if res.novelty_mask is not None:
        assert len(res.novelty_mask) == len(res.explanations)

    assert "source_indices" in res.metadata
    assert "original_count" in res.metadata
    assert res.metadata["original_count"] == 10
    assert len(res.metadata["source_indices"]) == len(res.explanations)


def test_subset_wrapper_indexing_and_slicing_stays_consistent():
    X, y = make_classification(n_samples=120, n_features=4, random_state=11)
    X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.5, random_state=11)

    clf = RandomForestClassifier(n_estimators=10, random_state=11)
    clf.fit(X_train, y_train)

    w = WrapCalibratedExplainer(clf)
    w.calibrate(X_cal, y_cal)
    w.explainer.reject_orchestrator.initialize_reject_learner()

    res = w.explain_factual(X_cal[:8], reject_policy=RejectPolicy.ONLY_REJECTED)
    assert isinstance(res, RejectCalibratedExplanations)

    if len(res.explanations) > 0:
        _ = res[0]
    sliced = res[: min(2, len(res.explanations))]
    assert isinstance(sliced, RejectCalibratedExplanations)
    if sliced.rejected is not None:
        assert len(sliced.rejected) == len(sliced.explanations)


def test_repeated_calls_and_call_order_keep_subset_mappings_deterministic():
    X, y = make_classification(n_samples=150, n_features=4, random_state=23)
    X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.5, random_state=23)

    clf = RandomForestClassifier(n_estimators=12, random_state=23)
    clf.fit(X_train, y_train)

    w = WrapCalibratedExplainer(clf)
    w.calibrate(X_cal, y_cal)
    w.explainer.reject_orchestrator.initialize_reject_learner()

    X_test = X_cal[:12]

    accepted_first = w.explain_factual(X_test, reject_policy=RejectPolicy.ONLY_ACCEPTED)
    rejected_second = w.explain_factual(X_test, reject_policy=RejectPolicy.ONLY_REJECTED)

    accepted_again = w.explain_factual(X_test, reject_policy=RejectPolicy.ONLY_ACCEPTED)
    rejected_again = w.explain_factual(X_test, reject_policy=RejectPolicy.ONLY_REJECTED)

    assert accepted_first.metadata["source_indices"] == accepted_again.metadata["source_indices"]
    assert rejected_second.metadata["source_indices"] == rejected_again.metadata["source_indices"]
    np.testing.assert_array_equal(accepted_first.rejected, accepted_again.rejected)
    np.testing.assert_array_equal(rejected_second.rejected, rejected_again.rejected)
    assert (
        accepted_first.metadata["original_count"]
        == rejected_second.metadata["original_count"]
        == 12
    )
