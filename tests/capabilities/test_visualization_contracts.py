"""Capability contract tests for visualization output (smoke tests).

Requirements verified:
  CE-REQ-VIZ-SMOKE-001 — Visualization smoke test contract (CE-CAP-VIZ-001)

These tests verify only that the plot API does not raise exceptions in a headless
(Agg) backend environment. Visual correctness is NOT verified.
See development/capabilities/requirements/CE-REQ-VIZ-SMOKE-001.md for the full
assumption boundary.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # Must be set before any other matplotlib import

import matplotlib.pyplot as plt
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer

_RNG_SEED = 42
_N_SAMPLES = 120
_N_FEATURES = 4
_N_TEST = 2


@pytest.fixture(scope="module")
def viz_explainer():
    """Return a fitted and calibrated explainer and factual explanations for plot tests."""
    X, y = make_classification(
        n_samples=_N_SAMPLES,
        n_features=_N_FEATURES,
        n_informative=3,
        n_redundant=1,
        random_state=_RNG_SEED,
    )
    X_train_cal, X_test, y_train_cal, y_test = train_test_split(
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
    explanations = explainer.explain_factual(X_test)
    return explanations, X_test, y_test


# ---------------------------------------------------------------------------
# CE-REQ-VIZ-SMOKE-001 — Visualization smoke test contract
# ---------------------------------------------------------------------------


def test_should_not_raise_when_plot_called_with_agg_backend(
    viz_explainer,
):
    """Verify CE-REQ-VIZ-SMOKE-001: CalibratedExplanations.plot() does not raise in Agg mode.

    Acceptance criterion (from CE-REQ-VIZ-SMOKE-001):
    - explanations.plot() completes without raising an exception.

    This is a no-raise smoke test. Visual correctness is not asserted.
    """
    explanations, _, _ = viz_explainer

    try:
        explanations.plot(show=False)
    finally:
        plt.close("all")
