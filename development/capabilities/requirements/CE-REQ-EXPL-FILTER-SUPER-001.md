# CE-REQ-EXPL-FILTER-SUPER-001 — Super-Explanations Filter API Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-EXPL-FILTER-SUPER-001 |
| obligation_type | api_contract |
| claim_refs | CE-CAP-EXPL-FILTER-001 |
| status | active |
| applicable_on | collection (AlternativeExplanations) and individual (AlternativeExplanation) |
| supersedes | CE-REQ-EXPL-ENSURED-API-001 (partial) |

## Scope

Public API:
- `AlternativeExplanations.super_explanations(only_ensured, include_potential, copy)` (collection)
- `AlternativeExplanations.super(...)` (alias — delegator, same contract)
- `AlternativeExplanation.super_explanations(only_ensured, include_potential, copy)` (individual)
- `AlternativeExplanation.super(...)` (alias — delegator, same contract)

Super-explanations are rules with **higher probability** supporting the predicted class.

Applicable task types: binary classification, multiclass classification, regression.

Applicable workflow: fit-calibrate-explore_alternatives-super_explanations.

## Observable behavior

**Collection level:** `AlternativeExplanations.super_explanations()` must:
1. Return without raising an exception.
2. Return a non-`None` `AlternativeExplanations` instance.
3. `len(result) == len(X_test)` (same instance count; rules per instance may be zero).

**Individual level:** `AlternativeExplanation.super_explanations()` must:
1. Return without raising an exception.
2. Return a non-`None` object.

**Alias:** `.super()` produces identical results to `.super_explanations()`.

## Acceptance criterion

**Default parameters (only_ensured=False, include_potential=True):**

Collection: `alternatives.super_explanations()` → non-None, `len == len(X_test)`.

Individual: `alternatives[0].super_explanations()` → non-None.

**With only_ensured=True (applies ensured filter within super set):**

Collection: `alternatives.super_explanations(only_ensured=True)` → non-None, `len == len(X_test)`.

**Alias check:**

`alternatives.super()` produces a result equal in structure to `alternatives.super_explanations()`.

## Verification method

Automated pytest tests in `tests/capabilities/`.

Test IDs:
- `test_should_return_super_explanations_when_default_params_collection`
- `test_should_return_super_explanations_when_only_ensured_true_collection`
- `test_should_return_super_explanations_when_individual_explanation`
- `test_should_return_super_explanations_when_alias_super_used`

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
- That super-explanations contain the statistically optimal rule set.
- That the ensured filter within only_ensured=True is statistically calibrated.
- That the filtered count is non-zero for any given dataset.

See `CE-CAP-EXPL-FILTER-001` for the full assumption statement.
