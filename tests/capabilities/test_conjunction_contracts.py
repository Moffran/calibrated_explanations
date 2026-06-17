"""Capability contract tests for conjunctive multi-feature explanation rules.

Requirement verified:
  CE-REQ-EXPL-CONJ-001 — Conjunction API contract: collection and individual
                          (CE-CAP-EXPL-CONJ-001)

  applicable_on: collection (CalibratedExplanations, AlternativeExplanations)
                 and individual (FactualExplanation, AlternativeExplanation)

These tests verify the observable public-API behavior stated in that requirement.
They do not assert that conjunctions produce better explanations than single-feature
rules. See development/capabilities/requirements/ for the full assumption boundary.
"""

from __future__ import annotations

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
def explainer_and_data():
    """Return a fitted and calibrated WrapCalibratedExplainer plus test data."""
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
# CE-REQ-EXPL-CONJ-001 — Collection level (applicable_on: collection)
# ---------------------------------------------------------------------------


def test_should_return_conjunctions_when_factual_collection_default_params(
    explainer_and_data,
):
    """Verify CE-REQ-EXPL-CONJ-001: add_conjunctions on factual collection is non-None.

    Acceptance criterion:
    - factual.add_conjunctions() completes without error.
    - Result is not None and len == len(X_test).
    """
    explainer, X_test = explainer_and_data
    factual = explainer.explain_factual(X_test)

    result = factual.add_conjunctions()

    assert (
        result is not None
    ), "CE-REQ-EXPL-CONJ-001: add_conjunctions on factual collection must return non-None"
    assert len(result) == len(
        X_test
    ), f"CE-REQ-EXPL-CONJ-001: len(result)={len(result)} != len(X_test)={len(X_test)}"


def test_should_return_conjunctions_when_alternative_collection_default_params(
    explainer_and_data,
):
    """Verify CE-REQ-EXPL-CONJ-001: add_conjunctions on alternative collection is non-None.

    Acceptance criterion:
    - alternatives.add_conjunctions() completes without error.
    - Result is not None and len == len(X_test).
    """
    explainer, X_test = explainer_and_data
    alternatives = explainer.explore_alternatives(X_test)

    result = alternatives.add_conjunctions()

    assert (
        result is not None
    ), "CE-REQ-EXPL-CONJ-001: add_conjunctions on alternative collection must return non-None"
    assert len(result) == len(
        X_test
    ), f"CE-REQ-EXPL-CONJ-001: len(result)={len(result)} != len(X_test)={len(X_test)}"


def test_should_return_conjunctions_when_alternative_collection_max_rule_size_one(
    explainer_and_data,
):
    """Verify CE-REQ-EXPL-CONJ-001: add_conjunctions with max_rule_size=1 returns non-None.

    max_rule_size=1 disables conjunction generation (single-feature rules only).
    API must still complete without error.
    """
    explainer, X_test = explainer_and_data
    alternatives = explainer.explore_alternatives(X_test)

    result = alternatives.add_conjunctions(max_rule_size=1)

    assert (
        result is not None
    ), "CE-REQ-EXPL-CONJ-001: add_conjunctions(max_rule_size=1) must return non-None"
    assert len(result) == len(
        X_test
    ), f"CE-REQ-EXPL-CONJ-001: len(result)={len(result)} != len(X_test)={len(X_test)}"


# ---------------------------------------------------------------------------
# CE-REQ-EXPL-CONJ-001 — Individual level (applicable_on: individual)
# ---------------------------------------------------------------------------


def test_should_return_conjunctions_when_individual_factual_explanation(
    explainer_and_data,
):
    """Verify CE-REQ-EXPL-CONJ-001: add_conjunctions on individual FactualExplanation.

    Acceptance criterion:
    - factual[0].add_conjunctions() completes without error.
    - Result is not None.
    """
    explainer, X_test = explainer_and_data
    factual = explainer.explain_factual(X_test)

    result = factual[0].add_conjunctions()

    assert (
        result is not None
    ), "CE-REQ-EXPL-CONJ-001: add_conjunctions on individual FactualExplanation must return non-None"


def test_should_return_conjunctions_when_individual_alternative_explanation(
    explainer_and_data,
):
    """Verify CE-REQ-EXPL-CONJ-001: add_conjunctions on individual AlternativeExplanation.

    Acceptance criterion:
    - alternatives[0].add_conjunctions() completes without error.
    - Result is not None.
    """
    explainer, X_test = explainer_and_data
    alternatives = explainer.explore_alternatives(X_test)

    result = alternatives[0].add_conjunctions()

    assert (
        result is not None
    ), "CE-REQ-EXPL-CONJ-001: add_conjunctions on individual AlternativeExplanation must return non-None"


def test_should_return_conjunctions_when_individual_with_non_default_n_top_features(
    explainer_and_data,
):
    """Verify CE-REQ-EXPL-CONJ-001: add_conjunctions with n_top_features=2, max_rule_size=2.

    Parameter variant: non-default n_top_features limits candidates to 2 features.
    """
    explainer, X_test = explainer_and_data
    alternatives = explainer.explore_alternatives(X_test)

    result = alternatives[0].add_conjunctions(n_top_features=2, max_rule_size=2)

    assert (
        result is not None
    ), "CE-REQ-EXPL-CONJ-001: add_conjunctions(n_top_features=2, max_rule_size=2) must return non-None"
