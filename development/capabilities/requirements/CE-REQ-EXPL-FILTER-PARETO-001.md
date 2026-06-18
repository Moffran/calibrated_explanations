# CE-REQ-EXPL-FILTER-PARETO-001 — Pareto-Explanations Filter API Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-EXPL-FILTER-PARETO-001 |
| obligation_type | api_contract |
| claim_refs | CE-CAP-EXPL-FILTER-001 |
| status | active |
| applicable_on | collection (AlternativeExplanations) and individual (AlternativeExplanation) |

## Scope

Public API:
- `AlternativeExplanations.pareto_explanations(include_potential, copy, *, pareto_cost)` (collection)
- `AlternativeExplanations.pareto(...)` (alias — delegator, same contract)
- `AlternativeExplanation.pareto_explanations(include_potential, copy, *, pareto_cost)` (individual)
- `AlternativeExplanation.pareto(...)` (alias — delegator, same contract)

Pareto-explanations are rules on the **Pareto-optimal frontier** of the cost dimension
specified by `pareto_cost` (default: `"uncertainty_width"`).

Applicable task types: binary classification, multiclass classification, regression.

Applicable workflow: fit-calibrate-explore_alternatives-pareto_explanations.

## Observable behavior

**Collection level:** `AlternativeExplanations.pareto_explanations()` must:
1. Return without raising an exception.
2. Return a non-`None` `AlternativeExplanations` instance.
3. `len(result) == len(X_test)`.

**Individual level:** `AlternativeExplanation.pareto_explanations()` must:
1. Return without raising an exception.
2. Return a non-`None` object.

**Alias:** `.pareto()` produces identical results to `.pareto_explanations()`.

## Acceptance criterion

**Default parameters (include_potential=True, pareto_cost="uncertainty_width"):**

Collection: `alternatives.pareto_explanations()` → non-None, `len == len(X_test)`.

Individual: `alternatives[0].pareto_explanations()` → non-None.

**With include_potential=False:**

Collection: `alternatives.pareto_explanations(include_potential=False)` → non-None,
`len == len(X_test)`.

**Alias check:** `alternatives.pareto()` produces a result equal in structure to `alternatives.pareto_explanations()`.

## Verification method

Automated pytest tests in `tests/capabilities/`.

Test IDs:
- `test_should_return_pareto_explanations_when_default_params_collection`
- `test_should_return_pareto_explanations_when_include_potential_false_collection`
- `test_should_return_pareto_explanations_when_individual_explanation`
- `test_should_return_pareto_explanations_when_alias_pareto_used`

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
- Optimality or correctness of the Pareto frontier computation.
- That the filtered count is non-zero.
- Behavior with alternative `pareto_cost` values.

See `CE-CAP-EXPL-FILTER-001` for the full assumption statement.
