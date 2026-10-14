"""Capability contract tests for probabilistic regression threshold queries.

Requirements verified:
  CE-REQ-PRED-PROB-API-001 — Probabilistic regression threshold query API contract
                              (CE-CAP-PRED-PROB-001)
  CE-REQ-PRED-PROB-INSTANCE-001 — Instance-wise (batch-invariant) outputs
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

# conftest.py adds development/capabilities/verification/tif/ to sys.path.
from tif_prob_regression_instance import run_prob_regression_instance_tif_scenario

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
    assert arr.min() >= 0.0, (
        f"CE-REQ-PRED-PROB-API-001: probabilities must be >= 0, got min={arr.min()}"
    )
    assert arr.max() <= 1.0, (
        f"CE-REQ-PRED-PROB-API-001: probabilities must be <= 1, got max={arr.max()}"
    )


def test_should_return_correct_length_when_regression_threshold_query(
    regression_explainer,
):
    """Verify CE-REQ-PRED-PROB-API-001: predict_proba with threshold returns correct length.

    Acceptance criterion (from CE-REQ-PRED-PROB-API-001):
    - len(result) == len(X_test).
    """
    explainer, X_test, y_threshold = regression_explainer

    result = explainer.predict_proba(X_test, threshold=y_threshold)

    assert len(result) == len(X_test), (
        f"CE-REQ-PRED-PROB-API-001: len(result)={len(result)} != len(X_test)={len(X_test)}"
    )


def test_should_return_event_probability_in_column_one_when_regression_threshold_query(
    regression_explainer,
):
    """Verify the public threshold-query payload shape used by quickstart examples."""
    explainer, X_test, y_threshold = regression_explainer

    probabilities, (low, high) = explainer.predict_proba(
        X_test, threshold=y_threshold, uq_interval=True
    )

    assert probabilities.shape == (len(X_test), 2)
    assert low.shape == (len(X_test),)
    assert high.shape == (len(X_test),)
    np.testing.assert_allclose(probabilities[:, 0] + probabilities[:, 1], 1.0, atol=1e-9)
    assert np.isscalar(probabilities[0, 1])
    assert low[0] <= probabilities[0, 1] <= high[0]


# ---------------------------------------------------------------------------
# CE-REQ-PRED-PROB-INSTANCE-001 — Instance-wise probabilistic regression contract
# ---------------------------------------------------------------------------


def test_should_return_batch_invariant_probabilities_when_regression_threshold_query():
    """Verify CE-REQ-PRED-PROB-INSTANCE-001: outputs do not depend on batch composition.

    Acceptance criteria (from CE-REQ-PRED-PROB-INSTANCE-001):
    - Batched outputs equal one-at-a-time outputs exactly (point, low, high).
    - Reversed-batch outputs, mapped back, equal one-at-a-time outputs exactly.
    """
    obs = run_prob_regression_instance_tif_scenario(threshold=0.0)

    assert obs.max_abs_batch_deviation == 0.0, (
        "CE-REQ-PRED-PROB-INSTANCE-001: batched output differs from one-at-a-time "
        f"output by {obs.max_abs_batch_deviation}"
    )
    assert obs.max_abs_permutation_deviation == 0.0, (
        "CE-REQ-PRED-PROB-INSTANCE-001: reversed-batch output differs from one-at-a-time "
        f"output by {obs.max_abs_permutation_deviation}"
    )
