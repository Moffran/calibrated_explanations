# CE-REQ-EXPL-FILTER-COUNTER-001 — Counter-Explanations Filter API Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-EXPL-FILTER-COUNTER-001 |
| obligation_type | api_contract |
| claim_refs | CE-CAP-EXPL-FILTER-001 |
| status | active |
| applicable_on | collection (AlternativeExplanations) and individual (AlternativeExplanation) |

## Scope

Public API:
- `AlternativeExplanations.counter_explanations(only_ensured, include_potential, copy)` (collection)
- `AlternativeExplanations.counter(...)` (alias — delegator, same contract)
- `AlternativeExplanation.counter_explanations(only_ensured, include_potential, copy)` (individual)
- `AlternativeExplanation.counter(...)` (alias — delegator, same contract)

Counter-explanations are rules that do **not support** the predicted class (supporting the opposite class or outcome).

Applicable task types: binary classification, multiclass classification, regression.

Applicable workflow: fit-calibrate-explore_alternatives-counter_explanations.

## Observable behavior

**Collection level:** `AlternativeExplanations.counter_explanations()` must:
1. Return without raising an exception.
2. Return a non-`None` `AlternativeExplanations` instance.
3. `len(result) == len(X_test)`.

**Individual level:** `AlternativeExplanation.counter_explanations()` must:
1. Return without raising an exception.
2. Return a non-`None` object.

**Alias:** `.counter()` produces identical results to `.counter_explanations()`.

## Acceptance criterion

**Default parameters (only_ensured=False, include_potential=True):**

Collection: `alternatives.counter_explanations()` → non-None, `len == len(X_test)`.

Individual: `alternatives[0].counter_explanations()` → non-None.

**With only_ensured=True:**

Collection: `alternatives.counter_explanations(only_ensured=True)` → non-None, `len == len(X_test)`.

**Alias check:** `alternatives.counter()` produces a result equal in structure to `alternatives.counter_explanations()`.

## Verification method

Automated pytest tests in `tests/capabilities/`.

Test IDs:
- `test_should_return_counter_explanations_when_default_params_collection`
- `test_should_return_counter_explanations_when_only_ensured_true_collection`
- `test_should_return_counter_explanations_when_individual_explanation`
- `test_should_return_counter_explanations_when_alias_counter_used`

(in `tests/capabilities/test_filter_contracts.py`)

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

This requirement verifies API contract and return type only. It does not verify:
- Semantic correctness of which rules are classified as counter-factual.
- That the filtered count is non-zero.

See `CE-CAP-EXPL-FILTER-001` for the full assumption statement.
