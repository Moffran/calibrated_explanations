# CE-REQ-GUARD-API-001 — Guarded Explanation API Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-GUARD-API-001 |
| obligation_type | api_contract |
| claim_refs | CE-CAP-GUARD-001 |
| status | active |

## Scope

Public API: `WrapCalibratedExplainer.explain_factual(X, guarded_options=GuardedOptions())`

Applicable task types: binary classification, multiclass classification, regression.

Applicable workflow: fit-calibrate-explain_factual with GuardedOptions.

## Observable behavior

When `WrapCalibratedExplainer` has been fitted and calibrated, calling
`explain_factual(X, guarded_options=GuardedOptions())` must:

1. Return without raising an exception.
2. Return a collection not equal to `None`.
3. The returned collection supports `len()` with `len(result) == len(X)`.

## Acceptance criterion

For a `WrapCalibratedExplainer` fitted on `X_proper`/`y_proper` and calibrated on
`X_cal`/`y_cal`:

- `explain_factual(X_test, guarded_options=GuardedOptions())` completes without error.
- The result is not `None`.
- `len(result) == len(X_test)`.

## Verification method

Automated pytest test in `tests/capabilities/`.

Test ID:
- `test_should_return_explanations_when_guarded_options_provided`

(in `tests/capabilities/test_guard_contracts.py`)

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

- That the guarded filter correctly identifies all out-of-distribution instances.
- Optimality of the nearest-neighbour in-distribution check.
- Coverage of the guarded explanation set.

Note: `explain_guarded_factual` was REMOVED in v0.11.3. Passing `GuardedOptions` as
a keyword argument is the canonical replacement.

See `CE-CAP-GUARD-001` for the full assumption statement.
