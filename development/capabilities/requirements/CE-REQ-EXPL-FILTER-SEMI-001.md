# CE-REQ-EXPL-FILTER-SEMI-001 — Semi-Explanations Filter API Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-EXPL-FILTER-SEMI-001 |
| obligation_type | api_contract |
| claim_refs | CE-CAP-EXPL-FILTER-001 |
| status | active |
| applicable_on | collection (AlternativeExplanations) and individual (AlternativeExplanation) |

## Scope

Public API:
- `AlternativeExplanations.semi_explanations(only_ensured, include_potential, copy)` (collection)
- `AlternativeExplanations.semi(...)` (alias — delegator, same contract)
- `AlternativeExplanation.semi_explanations(only_ensured, include_potential, copy)` (individual)
- `AlternativeExplanation.semi(...)` (alias — delegator, same contract)

Semi-explanations are rules with **lower probability** supporting the predicted class.

Applicable task types: binary classification, multiclass classification, regression.

Applicable workflow: fit-calibrate-explore_alternatives-semi_explanations.

## Observable behavior

**Collection level:** `AlternativeExplanations.semi_explanations()` must:
1. Return without raising an exception.
2. Return a non-`None` `AlternativeExplanations` instance.
3. `len(result) == len(X_test)`.

**Individual level:** `AlternativeExplanation.semi_explanations()` must:
1. Return without raising an exception.
2. Return a non-`None` object.

**Alias:** `.semi()` produces identical results to `.semi_explanations()`.

## Acceptance criterion

**Default parameters (only_ensured=False, include_potential=True):**

Collection: `alternatives.semi_explanations()` → non-None, `len == len(X_test)`.

Individual: `alternatives[0].semi_explanations()` → non-None.

**With only_ensured=True:**

Collection: `alternatives.semi_explanations(only_ensured=True)` → non-None, `len == len(X_test)`.

**Alias check:** `alternatives.semi()` produces a result equal in structure to `alternatives.semi_explanations()`.

## Verification method

Automated pytest tests in `tests/capabilities/`.

Test IDs:
- `test_should_return_semi_explanations_when_default_params_collection`
- `test_should_return_semi_explanations_when_only_ensured_true_collection`
- `test_should_return_semi_explanations_when_individual_explanation`
- `test_should_return_semi_explanations_when_alias_semi_used`

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
- That semi-explanations are statistically optimal.
- That the filtered count is non-zero.

See `CE-CAP-EXPL-FILTER-001` for the full assumption statement.
