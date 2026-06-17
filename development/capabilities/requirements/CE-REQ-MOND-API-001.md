# CE-REQ-MOND-API-001 — Mondrian Calibration API Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-MOND-API-001 |
| obligation_type | api_contract |
| claim_refs | CE-CAP-MOND-001 |
| status | active |

## Scope

Public API: `WrapCalibratedExplainer.calibrate(X_cal, y_cal, mc=mondrian_fn)`

Applicable task types: binary classification, multiclass classification.

Applicable workflow: fit-calibrate with Mondrian categorizer callable.

## Observable behavior

When `WrapCalibratedExplainer` has been fitted and a callable Mondrian categorizer is
passed to `calibrate(X_cal, y_cal, mc=mondrian_fn)`:

1. `calibrate` must return without raising an exception.
2. The returned wrapper must report as calibrated (`wrapper.calibrated is True`).

## Acceptance criterion

For a `WrapCalibratedExplainer` fitted on `X_proper`/`y_proper`, a calibration set
`X_cal`/`y_cal`, and a callable `mondrian_fn` returning integer category labels:

- `wrapper.calibrate(X_cal, y_cal, mc=mondrian_fn)` completes without error.
- After calibration, `wrapper.calibrated` is `True`.

## Verification method

Automated pytest test in `tests/capabilities/`.

Test ID:
- `test_should_calibrate_when_mondrian_categorizer_provided`

(in `tests/capabilities/test_mondrian_contracts.py`)

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

- Conditional validity within each Mondrian category.
- Per-category calibration accuracy.
- Optimality of the Mondrian category assignment function.
- Coverage guarantees for small per-category calibration sets.

See `CE-CAP-MOND-001` for the full assumption statement.
