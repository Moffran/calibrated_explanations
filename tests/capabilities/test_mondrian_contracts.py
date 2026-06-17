"""Capability contract tests for Mondrian conditional calibration.

Requirements verified:
  CE-REQ-MOND-API-001 — Mondrian calibration API contract (CE-CAP-MOND-001)

These tests verify the observable public-API behavior stated in those requirements.
They do not prove conditional validity guarantees within each Mondrian category.
See development/capabilities/requirements/CE-REQ-MOND-API-001.md for the full
assumption boundary.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer

_RNG_SEED = 42
_N_SAMPLES = 120
_N_FEATURES = 4
_N_TEST = 3


@pytest.fixture(scope="module")
def mondrian_setup():
    """Return fitted WrapCalibratedExplainer, calibration data, and a simple Mondrian fn."""
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

    # Mondrian categorizer: partition by sign of first feature (2 categories: 0, 1)
    def mondrian_fn(x):
        return (np.asarray(x)[:, 0] >= 0).astype(int)

    return explainer, X_cal, y_cal, mondrian_fn


# ---------------------------------------------------------------------------
# CE-REQ-MOND-API-001 — Mondrian calibration API contract
# ---------------------------------------------------------------------------


def test_should_calibrate_when_mondrian_categorizer_provided(
    mondrian_setup,
):
    """Verify CE-REQ-MOND-API-001: calibrate with mc= completes and reports calibrated=True.

    Acceptance criteria (from CE-REQ-MOND-API-001):
    - calibrate(X_cal, y_cal, mc=mondrian_fn) completes without error.
    - After calibration, wrapper.calibrated is True.
    """
    explainer, X_cal, y_cal, mondrian_fn = mondrian_setup

    explainer.calibrate(X_cal, y_cal, mc=mondrian_fn)

    assert (
        explainer.calibrated is True
    ), "CE-REQ-MOND-API-001: explainer.calibrated must be True after Mondrian calibration"
