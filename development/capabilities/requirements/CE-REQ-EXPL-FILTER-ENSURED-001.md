# CE-REQ-EXPL-FILTER-ENSURED-001 — Ensured-Explanations Filter API Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-EXPL-FILTER-ENSURED-001 |
| obligation_type | api_contract |
| claim_refs | CE-CAP-EXPL-FILTER-001 |
| status | active |
| applicable_on | collection (AlternativeExplanations) and individual (AlternativeExplanation) |

## Scope

Public API:
- `AlternativeExplanations.ensured_explanations(include_potential, copy)` (collection)
- `AlternativeExplanations.ensured(...)` (alias — delegator, same contract)
- `AlternativeExplanation.ensured_explanations(include_potential, copy)` (individual)
- `AlternativeExplanation.ensured(...)` (alias — delegator, same contract)

Ensured-explanations are rules where the **calibrated probability interval does not
straddle the decision boundary** — the outcome under the rule is unambiguously assigned.

Note: this is a distinct filter method from the `only_ensured` parameter on
super/semi/counter. It returns rules that are ensured regardless of their rule type.

Applicable task types: binary classification, multiclass classification, regression.

Applicable workflow: fit-calibrate-explore_alternatives-ensured_explanations.

## Observable behavior

**Collection level:** `AlternativeExplanations.ensured_explanations()` must:
1. Return without raising an exception.
2. Return a non-`None` `AlternativeExplanations` instance.
3. `len(result) == len(X_test)`.

**Individual level:** `AlternativeExplanation.ensured_explanations()` must:
1. Return without raising an exception.
2. Return a non-`None` object.

**Alias:** `.ensured()` produces identical results to `.ensured_explanations()`.

## Acceptance criterion

**Default parameters (include_potential=True):**

Collection: `alternatives.ensured_explanations()` → non-None, `len == len(X_test)`.

Individual: `alternatives[0].ensured_explanations()` → non-None.

**With include_potential=False:**

Collection: `alternatives.ensured_explanations(include_potential=False)` → non-None,
`len == len(X_test)`.

**Alias check:** `alternatives.ensured()` produces a result equal in structure to `alternatives.ensured_explanations()`.

## Verification method

Automated pytest tests in `tests/capabilities/`.

Test IDs:
- `test_should_return_ensured_explanations_when_default_params_collection`
- `test_should_return_ensured_explanations_when_include_potential_false_collection`
- `test_should_return_ensured_explanations_when_individual_explanation`
- `test_should_return_ensured_explanations_when_alias_ensured_used`

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
- That the ensured filter correctly identifies calibration-based certainty.
- That the filtered count is non-zero.
- Statistical validity of the ensured boundary.

See `CE-CAP-EXPL-FILTER-001` for the full assumption statement.
