"""Capability contract tests for alternative explanation filter operations.

Requirements verified:
  CE-REQ-EXPL-FILTER-SUPER-001   — super_explanations / super() API contract
  CE-REQ-EXPL-FILTER-SEMI-001    — semi_explanations / semi() API contract
  CE-REQ-EXPL-FILTER-COUNTER-001 — counter_explanations / counter() API contract
  CE-REQ-EXPL-FILTER-ENSURED-001 — ensured_explanations / ensured() API contract
  CE-REQ-EXPL-FILTER-PARETO-001  — pareto_explanations / pareto() API contract

All requirements derive from CE-CAP-EXPL-FILTER-001.

These tests verify that each filter operation:
- Completes without raising an exception.
- Returns a non-None result at both collection and individual levels.
- Aliases delegate correctly to canonical methods.
- Key parameter variants (only_ensured, include_potential) complete without error.

They do NOT verify statistical optimality of the filtered sets or that filtered
counts are non-zero. See development/capabilities/requirements/CE-REQ-EXPL-FILTER-*.md
for the full assumption boundary.
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
def alternatives():
    """Return an AlternativeExplanations collection from a fitted and calibrated explainer."""
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
    return explainer.explore_alternatives(X_test), X_test


# ===========================================================================
# CE-REQ-EXPL-FILTER-SUPER-001 — super_explanations
# ===========================================================================


def test_should_return_super_explanations_when_default_params_collection(alternatives):
    """Verify CE-REQ-EXPL-FILTER-SUPER-001: super_explanations() on collection is non-None.

    Acceptance criterion:
    - alternatives.super_explanations() completes without error.
    - Result is not None and len == len(X_test).
    """
    alts, X_test = alternatives
    result = alts.super_explanations()
    assert (
        result is not None
    ), "CE-REQ-EXPL-FILTER-SUPER-001: super_explanations() must return non-None"
    assert len(result) == len(X_test)


def test_should_return_super_explanations_when_only_ensured_true_collection(alternatives):
    """Verify CE-REQ-EXPL-FILTER-SUPER-001: super_explanations(only_ensured=True) completes.

    Parameter variant: only_ensured=True adds ensured filter within super set.
    """
    alts, X_test = alternatives
    result = alts.super_explanations(only_ensured=True)
    assert (
        result is not None
    ), "CE-REQ-EXPL-FILTER-SUPER-001: super_explanations(only_ensured=True) must return non-None"
    assert len(result) == len(X_test)


def test_should_return_super_explanations_when_individual_explanation(alternatives):
    """Verify CE-REQ-EXPL-FILTER-SUPER-001: super_explanations() on individual is non-None."""
    alts, _ = alternatives
    result = alts[0].super_explanations()
    assert (
        result is not None
    ), "CE-REQ-EXPL-FILTER-SUPER-001: super_explanations() on individual must return non-None"


def test_should_return_super_explanations_when_alias_super_used(alternatives):
    """Verify CE-REQ-EXPL-FILTER-SUPER-001: .super() alias delegates to super_explanations().

    Both methods must return non-None for the same input.
    """
    alts, X_test = alternatives
    result = alts.super()
    assert result is not None, "CE-REQ-EXPL-FILTER-SUPER-001: .super() alias must return non-None"
    assert len(result) == len(X_test)


# ===========================================================================
# CE-REQ-EXPL-FILTER-SEMI-001 — semi_explanations
# ===========================================================================


def test_should_return_semi_explanations_when_default_params_collection(alternatives):
    """Verify CE-REQ-EXPL-FILTER-SEMI-001: semi_explanations() on collection is non-None."""
    alts, X_test = alternatives
    result = alts.semi_explanations()
    assert (
        result is not None
    ), "CE-REQ-EXPL-FILTER-SEMI-001: semi_explanations() must return non-None"
    assert len(result) == len(X_test)


def test_should_return_semi_explanations_when_only_ensured_true_collection(alternatives):
    """Verify CE-REQ-EXPL-FILTER-SEMI-001: semi_explanations(only_ensured=True) completes."""
    alts, X_test = alternatives
    result = alts.semi_explanations(only_ensured=True)
    assert (
        result is not None
    ), "CE-REQ-EXPL-FILTER-SEMI-001: semi_explanations(only_ensured=True) must return non-None"
    assert len(result) == len(X_test)


def test_should_return_semi_explanations_when_individual_explanation(alternatives):
    """Verify CE-REQ-EXPL-FILTER-SEMI-001: semi_explanations() on individual is non-None."""
    alts, _ = alternatives
    result = alts[0].semi_explanations()
    assert (
        result is not None
    ), "CE-REQ-EXPL-FILTER-SEMI-001: semi_explanations() on individual must return non-None"


def test_should_return_semi_explanations_when_alias_semi_used(alternatives):
    """Verify CE-REQ-EXPL-FILTER-SEMI-001: .semi() alias delegates to semi_explanations()."""
    alts, X_test = alternatives
    result = alts.semi()
    assert result is not None, "CE-REQ-EXPL-FILTER-SEMI-001: .semi() alias must return non-None"
    assert len(result) == len(X_test)


# ===========================================================================
# CE-REQ-EXPL-FILTER-COUNTER-001 — counter_explanations
# ===========================================================================


def test_should_return_counter_explanations_when_default_params_collection(alternatives):
    """Verify CE-REQ-EXPL-FILTER-COUNTER-001: counter_explanations() on collection is non-None."""
    alts, X_test = alternatives
    result = alts.counter_explanations()
    assert (
        result is not None
    ), "CE-REQ-EXPL-FILTER-COUNTER-001: counter_explanations() must return non-None"
    assert len(result) == len(X_test)


def test_should_return_counter_explanations_when_only_ensured_true_collection(alternatives):
    """Verify CE-REQ-EXPL-FILTER-COUNTER-001: counter_explanations(only_ensured=True) completes."""
    alts, X_test = alternatives
    result = alts.counter_explanations(only_ensured=True)
    assert (
        result is not None
    ), "CE-REQ-EXPL-FILTER-COUNTER-001: counter_explanations(only_ensured=True) must return non-None"
    assert len(result) == len(X_test)


def test_should_return_counter_explanations_when_individual_explanation(alternatives):
    """Verify CE-REQ-EXPL-FILTER-COUNTER-001: counter_explanations() on individual is non-None."""
    alts, _ = alternatives
    result = alts[0].counter_explanations()
    assert (
        result is not None
    ), "CE-REQ-EXPL-FILTER-COUNTER-001: counter_explanations() on individual must return non-None"


def test_should_return_counter_explanations_when_alias_counter_used(alternatives):
    """Verify CE-REQ-EXPL-FILTER-COUNTER-001: .counter() alias delegates to counter_explanations()."""
    alts, X_test = alternatives
    result = alts.counter()
    assert (
        result is not None
    ), "CE-REQ-EXPL-FILTER-COUNTER-001: .counter() alias must return non-None"
    assert len(result) == len(X_test)


# ===========================================================================
# CE-REQ-EXPL-FILTER-ENSURED-001 — ensured_explanations
# ===========================================================================


def test_should_return_ensured_explanations_when_default_params_collection(alternatives):
    """Verify CE-REQ-EXPL-FILTER-ENSURED-001: ensured_explanations() on collection is non-None."""
    alts, X_test = alternatives
    result = alts.ensured_explanations()
    assert (
        result is not None
    ), "CE-REQ-EXPL-FILTER-ENSURED-001: ensured_explanations() must return non-None"
    assert len(result) == len(X_test)


def test_should_return_ensured_explanations_when_include_potential_false_collection(alternatives):
    """Verify CE-REQ-EXPL-FILTER-ENSURED-001: ensured_explanations(include_potential=False) completes.

    Parameter variant: include_potential=False excludes potential (one-sided) rules
    from the ensured filter.
    """
    alts, X_test = alternatives
    result = alts.ensured_explanations(include_potential=False)
    assert (
        result is not None
    ), "CE-REQ-EXPL-FILTER-ENSURED-001: ensured_explanations(include_potential=False) must return non-None"
    assert len(result) == len(X_test)


def test_should_return_ensured_explanations_when_individual_explanation(alternatives):
    """Verify CE-REQ-EXPL-FILTER-ENSURED-001: ensured_explanations() on individual is non-None."""
    alts, _ = alternatives
    result = alts[0].ensured_explanations()
    assert (
        result is not None
    ), "CE-REQ-EXPL-FILTER-ENSURED-001: ensured_explanations() on individual must return non-None"


def test_should_return_ensured_explanations_when_alias_ensured_used(alternatives):
    """Verify CE-REQ-EXPL-FILTER-ENSURED-001: .ensured() alias delegates to ensured_explanations()."""
    alts, X_test = alternatives
    result = alts.ensured()
    assert (
        result is not None
    ), "CE-REQ-EXPL-FILTER-ENSURED-001: .ensured() alias must return non-None"
    assert len(result) == len(X_test)


# ===========================================================================
# CE-REQ-EXPL-FILTER-PARETO-001 — pareto_explanations
# ===========================================================================


def test_should_return_pareto_explanations_when_default_params_collection(alternatives):
    """Verify CE-REQ-EXPL-FILTER-PARETO-001: pareto_explanations() on collection is non-None."""
    alts, X_test = alternatives
    result = alts.pareto_explanations()
    assert (
        result is not None
    ), "CE-REQ-EXPL-FILTER-PARETO-001: pareto_explanations() must return non-None"
    assert len(result) == len(X_test)


def test_should_return_pareto_explanations_when_include_potential_false_collection(alternatives):
    """Verify CE-REQ-EXPL-FILTER-PARETO-001: pareto_explanations(include_potential=False) completes.

    Parameter variant: include_potential=False excludes potential rules before Pareto frontier.
    """
    alts, X_test = alternatives
    result = alts.pareto_explanations(include_potential=False)
    assert (
        result is not None
    ), "CE-REQ-EXPL-FILTER-PARETO-001: pareto_explanations(include_potential=False) must return non-None"
    assert len(result) == len(X_test)


def test_should_return_pareto_explanations_when_individual_explanation(alternatives):
    """Verify CE-REQ-EXPL-FILTER-PARETO-001: pareto_explanations() on individual is non-None."""
    alts, _ = alternatives
    result = alts[0].pareto_explanations()
    assert (
        result is not None
    ), "CE-REQ-EXPL-FILTER-PARETO-001: pareto_explanations() on individual must return non-None"


def test_should_return_pareto_explanations_when_alias_pareto_used(alternatives):
    """Verify CE-REQ-EXPL-FILTER-PARETO-001: .pareto() alias delegates to pareto_explanations()."""
    alts, X_test = alternatives
    result = alts.pareto()
    assert result is not None, "CE-REQ-EXPL-FILTER-PARETO-001: .pareto() alias must return non-None"
    assert len(result) == len(X_test)
