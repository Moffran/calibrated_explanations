"""Capability contract tests for guarded (in-distribution filtered) explanations.

Requirements verified:
  CE-REQ-GUARD-API-001 — Guarded explanation API contract (CE-CAP-GUARD-001)

These tests verify the observable public-API behavior stated in those requirements.
They do not prove that GuardedOptions identifies all out-of-distribution instances.
See development/capabilities/requirements/CE-REQ-GUARD-API-001.md for the full
assumption boundary.
"""

from __future__ import annotations

import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations import GuardedOptions
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer

_RNG_SEED = 42
_N_SAMPLES = 120
_N_FEATURES = 4
_N_TEST = 3


@pytest.fixture(scope="module")
def guarded_explainer():
    """Return a fitted and calibrated WrapCalibratedExplainer for guarded explanation tests."""
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
# CE-REQ-GUARD-API-001 — Guarded explanation API contract
# ---------------------------------------------------------------------------


def test_should_return_explanations_when_guarded_options_provided(
    guarded_explainer,
):
    """Verify CE-REQ-GUARD-API-001: explain_factual with GuardedOptions returns valid collection.

    Acceptance criteria (from CE-REQ-GUARD-API-001):
    - explain_factual(X_test, guarded_options=GuardedOptions()) completes without error.
    - The result is not None.
    - len(result) == len(X_test).
    """
    explainer, X_test = guarded_explainer

    result = explainer.explain_factual(X_test, guarded_options=GuardedOptions())

    assert (
        result is not None
    ), "CE-REQ-GUARD-API-001: explain_factual with GuardedOptions must return non-None"
    assert len(result) == len(
        X_test
    ), f"CE-REQ-GUARD-API-001: len(result)={len(result)} != len(X_test)={len(X_test)}"
