# CE-REQ-EXPL-CONJ-001 — Conjunction API Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-EXPL-CONJ-001 |
| obligation_type | api_contract |
| claim_refs | CE-CAP-EXPL-CONJ-001 |
| status | active |
| applicable_on | collection (CalibratedExplanations, AlternativeExplanations) and individual (FactualExplanation, AlternativeExplanation) |
| supersedes | CE-REQ-EXPL-CONJ-COL-001, CE-REQ-EXPL-CONJ-IND-001, CE-REQ-EXPL-CONJ-API-001 |

## Scope

Public API:
- `CalibratedExplanations.add_conjunctions(n_top_features, max_rule_size)` — factual collection
- `AlternativeExplanations.add_conjunctions(n_top_features, max_rule_size)` — alternative collection (inherits from CalibratedExplanations)
- `FactualExplanation.add_conjunctions(n_top_features, max_rule_size)` — individual factual explanation
- `AlternativeExplanation.add_conjunctions(n_top_features, max_rule_size)` — individual alternative explanation

Applicable task types: binary classification, multiclass classification, regression, probabilistic_regression.

Applicable workflow: fit-calibrate-explain_factual-add_conjunctions OR
fit-calibrate-explore_alternatives-add_conjunctions, called on the collection or
on a single item indexed from the collection.

## Observable behavior

**Collection level:** When `explain_factual(X)` or `explore_alternatives(X)` has
returned a collection, calling `add_conjunctions()` on it must:
1. Return without raising an exception for valid inputs.
2. Return a non-`None` object of the same collection type.
3. Support `len()` with `len(result) == len(X)`.

**Individual level:** When `collection[i]` has returned an individual
`FactualExplanation` or `AlternativeExplanation`, calling `add_conjunctions()` on
it must:
1. Return without raising an exception.
2. Return a non-`None` object.

## Acceptance criterion

**Collection — default parameters (n_top_features=5, max_rule_size=2):**

For `factual = explainer.explain_factual(X_test)`:
- `factual.add_conjunctions()` completes without error.
- Result is not `None` and `len(result) == len(X_test)`.

For `alternatives = explainer.explore_alternatives(X_test)`:
- `alternatives.add_conjunctions()` completes without error.
- Result is not `None` and `len(result) == len(X_test)`.

**Collection — parameter variant (max_rule_size=1):**
- `alternatives.add_conjunctions(max_rule_size=1)` completes without error.
- Result is not `None` (max_rule_size=1 disables conjunction generation; single-feature rules remain).

**Individual — FactualExplanation:**
- `factual[0].add_conjunctions()` completes without error.
- Result is not `None`.

**Individual — AlternativeExplanation:**
- `alternatives[0].add_conjunctions()` completes without error.
- Result is not `None`.

**Individual — parameter variant (n_top_features=2, max_rule_size=2):**
- `alternatives[0].add_conjunctions(n_top_features=2, max_rule_size=2)` completes without error.
- Result is not `None`.

## Verification method

Automated pytest tests in `tests/capabilities/`.

Test IDs (in `tests/capabilities/test_conjunction_contracts.py`):
- `test_should_return_conjunctions_when_factual_collection_default_params`
- `test_should_return_conjunctions_when_alternative_collection_default_params`
- `test_should_return_conjunctions_when_alternative_collection_max_rule_size_one`
- `test_should_return_conjunctions_when_individual_factual_explanation`
- `test_should_return_conjunctions_when_individual_alternative_explanation`
- `test_should_return_conjunctions_when_individual_with_non_default_n_top_features`

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| dataset_id | yes |
| random_seed | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies the API contract only. It does not verify:
- That conjunctions are more informative than single-feature rules.
- Conjunction count per instance (may be zero for small feature sets or max_rule_size=1).
- Runtime behavior for large n_top_features or max_rule_size values.

See `CE-CAP-EXPL-CONJ-001` for the full assumption statement.
