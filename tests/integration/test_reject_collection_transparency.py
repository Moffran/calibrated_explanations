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


def should_return_rejected_collection_subclass_when_explain_factual_called_with_reject_policy():
    # Arrange
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.5, random_state=42)

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)

    w = WrapCalibratedExplainer(clf)
    w.calibrate(X_cal, y_cal)

    # Initialize reject learner to ensure reject logic runs
    w.initialize_reject_learner()

    X_test = X_cal[:5]

    # Act
    res = w.explain_factual(X_test, reject_policy=RejectPolicy.FLAG)

    # Assert
    assert isinstance(res, RejectCalibratedExplanations)
    assert hasattr(res, "ambiguity_mask")
    assert isinstance(res.ambiguity_mask, np.ndarray)
    assert len(res.ambiguity_mask) == 5

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
