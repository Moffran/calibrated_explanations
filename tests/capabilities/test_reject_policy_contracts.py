"""Capability contract tests for reject/defer policy explanations.

Requirements verified:
  CE-REQ-REJECT-API-001 — Reject policy API contract (CE-CAP-REJECT-001)

These tests verify the observable public-API behavior stated in those requirements.
They do not prove that rejection tags are statistically optimal or that coverage
guarantees hold for any particular reject threshold.
See development/capabilities/requirements/CE-REQ-REJECT-API-001.md for the full
assumption boundary.
"""

from __future__ import annotations

import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations import RejectPolicySpec
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer

_RNG_SEED = 42
_N_SAMPLES = 120
_N_FEATURES = 4
_N_TEST = 3


@pytest.fixture(scope="module")
def reject_explainer():
    """Return a fitted and calibrated WrapCalibratedExplainer for reject policy tests."""
    X, y = make_classification(
        n_samples=_N_SAMPLES,
        n_features=_N_FEATURES,
        n_informative=3,
        n_redundant=1,
        random_state=_RNG_SEED,
    )
    X_train_cal, X_test, y_train_cal, _ = train_test_split(
        X, y, test_size=_N_TEST, random_state=_RNG_SEED
    )
    X_proper, X_cal, y_proper, y_cal = train_test_split(
        X_train_cal, y_train_cal, test_size=0.35, random_state=_RNG_SEED
    )
    explainer = WrapCalibratedExplainer(
        RandomForestClassifier(n_estimators=10, random_state=_RNG_SEED)
    )
    explainer.fit(X_proper, y_proper)
    explainer.calibrate(X_cal, y_cal)
    return explainer, X_test


# ---------------------------------------------------------------------------
# CE-REQ-REJECT-API-001 — Reject policy API contract
# ---------------------------------------------------------------------------


def test_should_return_explanations_when_reject_policy_flag_provided(
    reject_explainer,
):
    """Verify CE-REQ-REJECT-API-001: explain_factual with RejectPolicySpec.flag() returns valid collection.

    Acceptance criteria (from CE-REQ-REJECT-API-001):
    - explain_factual(X_test, reject_policy=RejectPolicySpec.flag()) completes without error.
    - The result is not None.
    - len(result) == len(X_test).
    """
    explainer, X_test = reject_explainer

    result = explainer.explain_factual(X_test, reject_policy=RejectPolicySpec.flag())

    assert (
        result is not None
    ), "CE-REQ-REJECT-API-001: explain_factual with RejectPolicySpec.flag() must return non-None"
    assert len(result) == len(
        X_test
    ), f"CE-REQ-REJECT-API-001: len(result)={len(result)} != len(X_test)={len(X_test)}"
