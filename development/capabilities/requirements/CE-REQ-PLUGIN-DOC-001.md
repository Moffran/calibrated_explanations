# CE-REQ-PLUGIN-DOC-001 — Plugin Protocol Importability Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-PLUGIN-DOC-001 |
| obligation_type | documentation_boundary |
| claim_refs | CE-CAP-PLUGIN-001 |
| status | active |

## Scope

Public API: `calibrated_explanations.plugins.base.ExplainerPlugin`,
`calibrated_explanations.plugins.intervals.IntervalCalibratorPlugin`

Applicable task types: binary classification, multiclass classification, regression.

## Observable behavior

The plugin protocol classes must be importable from `calibrated_explanations.plugins`
submodules and must be inspectable as Protocol types.

1. `from calibrated_explanations.plugins.base import ExplainerPlugin` must succeed.
2. `from calibrated_explanations.plugins.intervals import IntervalCalibratorPlugin` must succeed.
3. Both classes must be callable (can be used as type hints or for runtime isinstance checks
   if runtime_checkable).

## Acceptance criterion

- `from calibrated_explanations.plugins.base import ExplainerPlugin` completes without error.
- `from calibrated_explanations.plugins.intervals import IntervalCalibratorPlugin` completes without error.
- `ExplainerPlugin` is not `None`.
- `IntervalCalibratorPlugin` is not `None`.

## Verification method

Automated pytest test in `tests/capabilities/`.

Test ID:
- `test_should_import_explainer_plugin_protocol_when_plugins_module_available`
- `test_should_import_interval_calibrator_plugin_protocol_when_plugins_module_available`

(in `tests/capabilities/test_plugin_contracts.py`)

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |

## Assumption boundary

This requirement verifies importability only. It does not verify:

- That custom plugin implementations behave correctly.
- That plugin registration through PluginManager is stable across versions.
- Runtime behavior of custom plugins in all explanation scenarios.

See `CE-CAP-PLUGIN-001` for the full assumption statement.
