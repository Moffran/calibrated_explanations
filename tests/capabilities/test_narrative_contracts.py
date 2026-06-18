"""Capability contract tests for explanation narrative generation.

Requirements verified:
  CE-REQ-NARR-API-001 — Narrative output API contract (CE-CAP-NARR-001)

These tests verify the observable public-API behavior stated in those requirements.
They do not prove narrative quality, fluency, or domain correctness.
See development/capabilities/requirements/CE-REQ-NARR-API-001.md for the full
assumption boundary.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "yaml",
    reason="pyyaml not installed; skipping narrative contract tests (requires 'narrative' extra)",
)

from sklearn.datasets import make_classification  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer  # noqa: E402

_RNG_SEED = 42
_N_SAMPLES = 120
_N_FEATURES = 4
_N_TEST = 3


@pytest.fixture(scope="module")
def factual_explanations():
    """Return a factual explanation collection from a fitted and calibrated explainer."""
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
    return explainer.explain_factual(X_test)


# ---------------------------------------------------------------------------
# CE-REQ-NARR-API-001 — Narrative output API contract
# ---------------------------------------------------------------------------


def test_should_return_non_empty_string_when_narrative_text_format(
    factual_explanations,
):
    """Verify CE-REQ-NARR-API-001: to_narrative(output_format='text') returns non-empty string.

    Acceptance criteria (from CE-REQ-NARR-API-001):
    - to_narrative(output_format='text') completes without error.
    - The result is not None.
    - isinstance(result, str) is True.
    - len(result) > 0.
    """
    result = factual_explanations.to_narrative(output_format="text")

    assert (
        result is not None
    ), "CE-REQ-NARR-API-001: to_narrative(output_format='text') must return non-None"
    assert isinstance(result, str), f"CE-REQ-NARR-API-001: result must be str, got {type(result)}"
    assert len(result) > 0, "CE-REQ-NARR-API-001: to_narrative text output must be non-empty"
