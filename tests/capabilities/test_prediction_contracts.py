"""Capability contract tests for uncertainty interval predictions.

Requirements verified:
  CE-REQ-PRED-API-001           — Uncertainty interval API contract (CE-CAP-PRED-001)
  CE-REQ-PRED-INTERVAL-BOUNDS-001 — Conformal interval percentile selection (CE-CAP-PRED-001)

These tests verify the observable public-API behavior stated in those requirements.
They do not prove statistical coverage guarantees. The tests verify the API contract
and structural invariants (return shape, low <= high, percentile monotonicity) only.
See the requirement files under development/capabilities/requirements/ for the full
assumption boundary.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer

_RNG_SEED = 42
_N_SAMPLES = 120
_N_FEATURES = 4
_N_TEST = 4


@pytest.fixture(scope="module")
def binary_classification_explainer():
    """Return a fitted and calibrated WrapCalibratedExplainer for binary classification."""
    X, y = make_classification(
        n_samples=_N_SAMPLES,
        n_features=_N_FEATURES,
        n_informative=3,
        n_redundant=1,
        random_state=_RNG_SEED,
    )
    X_train_cal, X_test, y_train_cal, _y_test = train_test_split(
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


@pytest.fixture(scope="module")
def regression_explainer():
    """Return a fitted and calibrated WrapCalibratedExplainer for regression."""
    X, y = make_regression(
        n_samples=_N_SAMPLES,
        n_features=_N_FEATURES,
        n_informative=3,
        noise=0.1,
        random_state=_RNG_SEED,
    )
    X_train_cal, X_test, y_train_cal, _y_test = train_test_split(
        X, y, test_size=_N_TEST, random_state=_RNG_SEED
    )
    X_proper, X_cal, y_proper, y_cal = train_test_split(
        X_train_cal, y_train_cal, test_size=0.35, random_state=_RNG_SEED
    )
    explainer = WrapCalibratedExplainer(
        RandomForestRegressor(n_estimators=10, random_state=_RNG_SEED)
    )
    explainer.fit(X_proper, y_proper)
    explainer.calibrate(X_cal, y_cal)
    return explainer, X_test


# ---------------------------------------------------------------------------
# CE-REQ-PRED-API-001 — Uncertainty interval API contract
# ---------------------------------------------------------------------------


def test_should_return_uncertainty_interval_when_uq_interval_true_classification(
    binary_classification_explainer,
):
    """Verify CE-REQ-PRED-API-001: predict(uq_interval=True) returns (y_hat, (low, high)) for classification.

    Acceptance criteria (from CE-REQ-PRED-API-001):
    - Returns a 2-tuple (y_hat, (low, high)).
    - len(y_hat) == len(X_test).
    - low and high are not None.
    - For all i: low[i] <= high[i].
    """
    explainer, X_test = binary_classification_explainer

    result = explainer.predict(X_test, uq_interval=True)

    assert isinstance(
        result, tuple
    ), "CE-REQ-PRED-API-001: predict(uq_interval=True) must return a tuple"
    assert len(result) == 2, "CE-REQ-PRED-API-001: tuple must have 2 elements (y_hat, (low, high))"
    y_hat, bounds = result
    assert isinstance(bounds, tuple), "CE-REQ-PRED-API-001: bounds must be a tuple (low, high)"
    assert len(bounds) == 2, "CE-REQ-PRED-API-001: bounds must have exactly 2 elements"
    low, high = bounds

    assert low is not None, "CE-REQ-PRED-API-001: low must not be None"
    assert high is not None, "CE-REQ-PRED-API-001: high must not be None"
    assert len(y_hat) == len(
        X_test
    ), f"CE-REQ-PRED-API-001: len(y_hat)={len(y_hat)} != len(X_test)={len(X_test)}"
    assert len(low) == len(
        X_test
    ), f"CE-REQ-PRED-API-001: len(low)={len(low)} != len(X_test)={len(X_test)}"
    assert len(high) == len(
        X_test
    ), f"CE-REQ-PRED-API-001: len(high)={len(high)} != len(X_test)={len(X_test)}"
    for i in range(len(X_test)):
        assert (
            low[i] <= high[i]
        ), f"CE-REQ-PRED-API-001: low[{i}]={low[i]} > high[{i}]={high[i]} — interval bounds must be ordered"


def test_should_return_uncertainty_interval_when_uq_interval_true_regression(
    regression_explainer,
):
    """Verify CE-REQ-PRED-API-001: predict(uq_interval=True) returns (y_hat, (low, high)) for regression.

    Acceptance criteria (from CE-REQ-PRED-API-001):
    - Returns a 2-tuple (y_hat, (low, high)).
    - len(y_hat) == len(X_test).
    - low and high are not None.
    - For all i: low[i] <= high[i].
    """
    explainer, X_test = regression_explainer

    result = explainer.predict(X_test, uq_interval=True)

    assert isinstance(
        result, tuple
    ), "CE-REQ-PRED-API-001: predict(uq_interval=True) must return a tuple"
    assert len(result) == 2, "CE-REQ-PRED-API-001: tuple must have 2 elements (y_hat, (low, high))"
    y_hat, bounds = result
    assert isinstance(bounds, tuple), "CE-REQ-PRED-API-001: bounds must be a tuple (low, high)"
    low, high = bounds

    assert low is not None, "CE-REQ-PRED-API-001: low must not be None"
    assert high is not None, "CE-REQ-PRED-API-001: high must not be None"
    assert len(y_hat) == len(
        X_test
    ), f"CE-REQ-PRED-API-001: len(y_hat)={len(y_hat)} != len(X_test)={len(X_test)}"
    assert len(low) == len(
        X_test
    ), f"CE-REQ-PRED-API-001: len(low)={len(low)} != len(X_test)={len(X_test)}"
    assert len(high) == len(
        X_test
    ), f"CE-REQ-PRED-API-001: len(high)={len(high)} != len(X_test)={len(X_test)}"
    for i in range(len(X_test)):
        assert (
            low[i] <= high[i]
        ), f"CE-REQ-PRED-API-001: low[{i}]={low[i]} > high[{i}]={high[i]} — interval bounds must be ordered"


# ---------------------------------------------------------------------------
# CE-REQ-PRED-INTERVAL-BOUNDS-001 — Conformal interval percentile selection
# ---------------------------------------------------------------------------


def test_should_accept_low_high_percentiles_and_return_ordered_bounds_when_regression(
    regression_explainer,
):
    """Verify CE-REQ-PRED-INTERVAL-BOUNDS-001: low_high_percentiles is accepted and bounds are ordered.

    Acceptance criteria (from CE-REQ-PRED-INTERVAL-BOUNDS-001):
    - predict(X, uq_interval=True, low_high_percentiles=(10, 90)) completes without error.
    - Returns a 2-tuple (y_hat, (low, high)).
    - len(y_hat) == len(X_test).
    - For all i: low[i] <= high[i].
    """
    explainer, X_test = regression_explainer

    result = explainer.predict(X_test, uq_interval=True, low_high_percentiles=(10, 90))

    assert isinstance(
        result, tuple
    ), "CE-REQ-PRED-INTERVAL-BOUNDS-001: predict with low_high_percentiles must return a tuple"
    assert (
        len(result) == 2
    ), "CE-REQ-PRED-INTERVAL-BOUNDS-001: tuple must have 2 elements (y_hat, (low, high))"
    y_hat, bounds = result
    assert isinstance(
        bounds, tuple
    ), "CE-REQ-PRED-INTERVAL-BOUNDS-001: bounds must be a tuple (low, high)"
    low, high = bounds
    assert low is not None, "CE-REQ-PRED-INTERVAL-BOUNDS-001: low must not be None"
    assert high is not None, "CE-REQ-PRED-INTERVAL-BOUNDS-001: high must not be None"
    assert len(y_hat) == len(
        X_test
    ), f"CE-REQ-PRED-INTERVAL-BOUNDS-001: len(y_hat)={len(y_hat)} != len(X_test)={len(X_test)}"
    for i in range(len(X_test)):
        assert low[i] <= high[i], (
            f"CE-REQ-PRED-INTERVAL-BOUNDS-001: low[{i}]={low[i]} > high[{i}]={high[i]} — "
            "interval bounds must be ordered"
        )


def test_should_return_narrower_interval_when_percentiles_are_closer(
    regression_explainer,
):
    """Verify CE-REQ-PRED-INTERVAL-BOUNDS-001: a contracted percentile range produces narrower bounds.

    Acceptance criterion (from CE-REQ-PRED-INTERVAL-BOUNDS-001):
    - (5, 95) interval is at least as wide as (10, 90) interval for every instance:
      low_5_95[i] <= low_10_90[i] and high_10_90[i] <= high_5_95[i].
    """
    explainer, X_test = regression_explainer

    _, (low_wide, high_wide) = explainer.predict(
        X_test, uq_interval=True, low_high_percentiles=(5, 95)
    )
    _, (low_narrow, high_narrow) = explainer.predict(
        X_test, uq_interval=True, low_high_percentiles=(10, 90)
    )

    for i in range(len(X_test)):
        assert low_wide[i] <= low_narrow[i], (
            f"CE-REQ-PRED-INTERVAL-BOUNDS-001: (5,95) lower bound {low_wide[i]} > "
            f"(10,90) lower bound {low_narrow[i]} at index {i} — wider percentiles must produce lower or equal lower bound"
        )
        assert high_narrow[i] <= high_wide[i], (
            f"CE-REQ-PRED-INTERVAL-BOUNDS-001: (10,90) upper bound {high_narrow[i]} > "
            f"(5,95) upper bound {high_wide[i]} at index {i} — wider percentiles must produce higher or equal upper bound"
        )


def test_should_accept_neg_inf_lower_bound_and_return_constant_floor_when_regression(
    regression_explainer,
):
    """Verify CE-REQ-PRED-INTERVAL-BOUNDS-001: -np.inf lower percentile produces a one-sided upper-bounded interval.

    Acceptance criteria (from CE-REQ-PRED-INTERVAL-BOUNDS-001 §6):
    - predict(X, uq_interval=True, low_high_percentiles=(-np.inf, 90)) completes without error.
    - All lower bounds equal min(y_cal) — a constant floor.
    - For all i: low[i] <= high[i].
    """
    explainer, X_test = regression_explainer

    result = explainer.predict(X_test, uq_interval=True, low_high_percentiles=(-np.inf, 90))

    assert isinstance(
        result, tuple
    ), "CE-REQ-PRED-INTERVAL-BOUNDS-001: predict with -np.inf lower bound must return a tuple"
    _, (low, high) = result
    assert low is not None, "CE-REQ-PRED-INTERVAL-BOUNDS-001: low must not be None"
    assert high is not None, "CE-REQ-PRED-INTERVAL-BOUNDS-001: high must not be None"

    expected_floor = low[0]
    for i in range(len(X_test)):
        assert low[i] == expected_floor, (
            f"CE-REQ-PRED-INTERVAL-BOUNDS-001: lower bound at index {i} ({low[i]}) differs from "
            f"constant floor ({expected_floor}) — -np.inf must produce a uniform lower bound"
        )
        assert low[i] <= high[i], (
            f"CE-REQ-PRED-INTERVAL-BOUNDS-001: low[{i}]={low[i]} > high[{i}]={high[i]} — "
            "one-sided interval bounds must still be ordered"
        )


def test_should_accept_pos_inf_upper_bound_and_return_constant_ceiling_when_regression(
    regression_explainer,
):
    """Verify CE-REQ-PRED-INTERVAL-BOUNDS-001: np.inf upper percentile produces a one-sided lower-bounded interval.

    Acceptance criteria (from CE-REQ-PRED-INTERVAL-BOUNDS-001 §7):
    - predict(X, uq_interval=True, low_high_percentiles=(10, np.inf)) completes without error.
    - All upper bounds equal max(y_cal) — a constant ceiling.
    - For all i: low[i] <= high[i].
    """
    explainer, X_test = regression_explainer

    result = explainer.predict(X_test, uq_interval=True, low_high_percentiles=(10, np.inf))

    assert isinstance(
        result, tuple
    ), "CE-REQ-PRED-INTERVAL-BOUNDS-001: predict with np.inf upper bound must return a tuple"
    _, (low, high) = result
    assert low is not None, "CE-REQ-PRED-INTERVAL-BOUNDS-001: low must not be None"
    assert high is not None, "CE-REQ-PRED-INTERVAL-BOUNDS-001: high must not be None"

    expected_ceiling = high[0]
    for i in range(len(X_test)):
        assert high[i] == expected_ceiling, (
            f"CE-REQ-PRED-INTERVAL-BOUNDS-001: upper bound at index {i} ({high[i]}) differs from "
            f"constant ceiling ({expected_ceiling}) — np.inf must produce a uniform upper bound"
        )
        assert low[i] <= high[i], (
            f"CE-REQ-PRED-INTERVAL-BOUNDS-001: low[{i}]={low[i]} > high[{i}]={high[i]} — "
            "one-sided interval bounds must still be ordered"
        )


def test_should_return_predict_proba_interval_when_uq_interval_true_classification(
    binary_classification_explainer,
):
    """Verify CE-REQ-PRED-API-001: predict_proba(uq_interval=True) returns (proba, (low, high)).

    Acceptance criteria (from CE-REQ-PRED-API-001):
    - Returns a 2-tuple (proba, (low, high)).
    - len(proba) == len(X_test).
    - low and high are not None.
    - For all i: low[i] <= high[i].
    """
    explainer, X_test = binary_classification_explainer

    result = explainer.predict_proba(X_test, uq_interval=True)

    assert isinstance(
        result, tuple
    ), "CE-REQ-PRED-API-001: predict_proba(uq_interval=True) must return a tuple"
    assert len(result) == 2, "CE-REQ-PRED-API-001: tuple must have 2 elements (proba, (low, high))"
    proba, bounds = result
    assert isinstance(bounds, tuple), "CE-REQ-PRED-API-001: bounds must be a tuple (low, high)"
    low, high = bounds

    assert low is not None, "CE-REQ-PRED-API-001: low must not be None"
    assert high is not None, "CE-REQ-PRED-API-001: high must not be None"
    assert len(proba) == len(
        X_test
    ), f"CE-REQ-PRED-API-001: len(proba)={len(proba)} != len(X_test)={len(X_test)}"
    for i in range(len(X_test)):
        assert (
            low[i] <= high[i]
        ), f"CE-REQ-PRED-API-001: low[{i}]={low[i]} > high[{i}]={high[i]} — probability interval bounds must be ordered"
