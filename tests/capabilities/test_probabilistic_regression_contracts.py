"""Capability contract tests for probabilistic regression threshold queries.

Requirements verified:
  CE-REQ-PRED-PROB-API-001 — Probabilistic regression threshold query API contract
                              (CE-CAP-PRED-PROB-001)

These tests verify the observable public-API behavior stated in those requirements.
They do not prove CPS coverage guarantees or frequency-calibration of P(Y > threshold | X).
See development/capabilities/requirements/CE-REQ-PRED-PROB-API-001.md for the full
assumption boundary.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer

_RNG_SEED = 42
_N_SAMPLES = 120
_N_FEATURES = 4
_N_TEST = 5


@pytest.fixture(scope="module")
def regression_explainer():
    """Return a fitted and calibrated WrapCalibratedExplainer for regression."""
    X, y = make_regression(
        n_samples=_N_SAMPLES,
        n_features=_N_FEATURES,
        n_informative=3,
        random_state=_RNG_SEED,
        noise=10.0,
    )
    X_train_cal, X_test, y_train_cal, y_cal_all = train_test_split(
        X, y, test_size=_N_TEST, random_state=_RNG_SEED
    )
    X_proper, X_cal, y_proper, y_cal = train_test_split(
        X_train_cal, y_train_cal, test_size=0.35, random_state=_RNG_SEED
    )
    explainer = WrapCalibratedExplainer(
        RandomForestRegressor(n_estimators=10, random_state=_RNG_SEED)
    )
    explainer.fit(X_proper, y_proper)
    explainer.calibrate(X_cal, y_cal, mode="regression")
    y_threshold = float(np.median(y_proper))
    return explainer, X_test, y_threshold


# ---------------------------------------------------------------------------
# CE-REQ-PRED-PROB-API-001 — Probabilistic regression threshold query API contract
# ---------------------------------------------------------------------------


def test_should_return_bounded_probabilities_when_regression_threshold_query(
    regression_explainer,
):
    """Verify CE-REQ-PRED-PROB-API-001: predict_proba with threshold returns values in [0, 1].

    Acceptance criteria (from CE-REQ-PRED-PROB-API-001):
    - predict_proba(X_test, threshold=y_threshold) completes without error.
    - All returned values are in [0, 1].
    """
    explainer, X_test, y_threshold = regression_explainer

    result = explainer.predict_proba(X_test, threshold=y_threshold)

    assert result is not None, "predict_proba with threshold must return a non-None object"
    arr = np.asarray(result)
    assert (
        arr.min() >= 0.0
    ), f"CE-REQ-PRED-PROB-API-001: probabilities must be >= 0, got min={arr.min()}"
    assert (
        arr.max() <= 1.0
    ), f"CE-REQ-PRED-PROB-API-001: probabilities must be <= 1, got max={arr.max()}"


def test_should_return_correct_length_when_regression_threshold_query(
    regression_explainer,
):
    """Verify CE-REQ-PRED-PROB-API-001: predict_proba with threshold returns correct length.

    Acceptance criterion (from CE-REQ-PRED-PROB-API-001):
    - len(result) == len(X_test).
    """
    explainer, X_test, y_threshold = regression_explainer

    result = explainer.predict_proba(X_test, threshold=y_threshold)

    assert len(result) == len(
        X_test
    ), f"CE-REQ-PRED-PROB-API-001: len(result)={len(result)} != len(X_test)={len(X_test)}"
