# CE-REQ-VIZ-SMOKE-001 — Visualization Smoke Test Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-VIZ-SMOKE-001 |
| obligation_type | empirical_smoke |
| claim_refs | CE-CAP-VIZ-001 |
| status | active |

## Scope

Public API: `CalibratedExplanations.plot()` called on factual explanation output.

Applicable task types: binary classification.

Applicable workflow: fit-calibrate-explain_factual-plot in headless environment.

## Observable behavior

When `explain_factual(X)` has returned a `CalibratedExplanations` instance, calling
`explanations.plot()` with a non-interactive matplotlib backend (Agg) must:

1. Return without raising an exception.
2. Not produce any interactive display output.

## Acceptance criterion

With `matplotlib.use('Agg')` active before import:

- `explanations.plot()` completes without raising an exception.

This is a no-raise smoke test. Visual correctness is not asserted.

## Verification method

Automated pytest test in `tests/capabilities/`.

Test ID:
- `test_should_not_raise_when_plot_called_with_agg_backend`

(in `tests/capabilities/test_visualization_contracts.py`)

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

This requirement verifies no-raise behavior in a headless environment only. It does
not verify:

- Visual correctness of output (colors, layout, labels, axes).
- Rendering behavior across different matplotlib backends.
- Behavior when saving to file (show=False, filename=...).

See `CE-CAP-VIZ-001` for the full assumption statement.
