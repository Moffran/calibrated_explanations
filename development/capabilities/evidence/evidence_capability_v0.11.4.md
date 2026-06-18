# Capability Verification Expanded Evidence

**Task:** v0.11.4 Task 9 — Capability verification — full CE capability surface
**Date:** 2026-06-17
**Status:** CLOSED (expansion of initial scaffold closed same date)

---

## Gate command and result

```
pytest tests/capabilities/ -v
```

**Result:** 21 passed, 0 failed (exit code 1 due to coverage gate — see note below)

### Test output summary

```
tests/capabilities/test_advanced_explanation_contracts.py::test_should_return_ensured_explanations_when_alternatives_available  PASSED
tests/capabilities/test_advanced_explanation_contracts.py::test_should_return_non_none_when_super_explanations_only_ensured  PASSED
tests/capabilities/test_advanced_explanation_contracts.py::test_should_return_conjunctions_when_alternatives_available  PASSED
tests/capabilities/test_classification_contracts.py::test_should_return_bounded_probabilities_when_classification_fitted_and_calibrated  PASSED
tests/capabilities/test_classification_contracts.py::test_should_return_class_labels_when_classification_fitted_and_calibrated  PASSED
tests/capabilities/test_explanation_contracts.py::test_should_produce_factual_explanations_when_fitted_and_calibrated  PASSED
tests/capabilities/test_explanation_contracts.py::test_should_produce_factual_explanations_for_each_instance  PASSED
tests/capabilities/test_explanation_contracts.py::test_should_produce_alternative_explanations_when_fitted_and_calibrated  PASSED
tests/capabilities/test_explanation_contracts.py::test_should_produce_alternative_explanations_for_each_instance  PASSED
tests/capabilities/test_guard_contracts.py::test_should_return_explanations_when_guarded_options_provided  PASSED
tests/capabilities/test_mondrian_contracts.py::test_should_calibrate_when_mondrian_categorizer_provided  PASSED
tests/capabilities/test_narrative_contracts.py::test_should_return_non_empty_string_when_narrative_text_format  PASSED
tests/capabilities/test_plugin_contracts.py::test_should_import_explainer_plugin_protocol_when_plugins_module_available  PASSED
tests/capabilities/test_plugin_contracts.py::test_should_import_interval_calibrator_plugin_protocol_when_plugins_module_available  PASSED
tests/capabilities/test_prediction_contracts.py::test_should_return_uncertainty_interval_when_uq_interval_true_classification  PASSED
tests/capabilities/test_prediction_contracts.py::test_should_return_uncertainty_interval_when_uq_interval_true_regression  PASSED
tests/capabilities/test_prediction_contracts.py::test_should_return_predict_proba_interval_when_uq_interval_true_classification  PASSED
tests/capabilities/test_probabilistic_regression_contracts.py::test_should_return_bounded_probabilities_when_regression_threshold_query  PASSED
tests/capabilities/test_probabilistic_regression_contracts.py::test_should_return_correct_length_when_regression_threshold_query  PASSED
tests/capabilities/test_reject_policy_contracts.py::test_should_return_explanations_when_reject_policy_flag_provided  PASSED
tests/capabilities/test_visualization_contracts.py::test_should_not_raise_when_plot_called_with_agg_backend  PASSED

21 passed in ~9s
```

### Coverage note

Running only `tests/capabilities/` yields ~33.7% package coverage, which fails the
project-wide `--cov-fail-under=90` gate. This is expected: running a subset of the
test suite cannot satisfy the full package coverage requirement. The coverage gate is
satisfied by the full `pytest -q` run (1548+ tests). The release-plan acceptance
criterion for Task 9 is "all tests green", not coverage via the subset command.

Pre-existing test failures in `tests/unit/test_explanations_collection.py`
(`test_legacy_payload_*`) are unrelated and are tracked separately.

---

## Source material

The expanded claims were derived from
`calibrated-explanations-enterprise/docs/marketing/claims/ce_engine.yaml` (14 claims,
CE-PRED-001 through CE-PAR-001). All claims were normalized to CE-native terminology:

- "OSS CE" → removed (not applicable in CE context)
- "enterprise wrappers" / "parity" language → excluded (CEE-owned, not CE-owned)
- `CE-PAR-001` ("Both Adaptive and Governance wrap the OSS CE core…") → excluded entirely
  (this is a CEE parity claim, not a CE capability claim)
- Statistical assumption boundaries added explicitly per project convention

---

## Files changed (expansion — new in this batch)

### Capability claims (10 new, plus 3 from initial scaffold)

| File | Claim ID | Description | Source CE-YAML |
|---|---|---|---|
| `development/capabilities/claims/CE-CAP-PRED-CLASS-001.yaml` | CE-CAP-PRED-CLASS-001 | Calibrated classification predict_proba/predict | CE-PRED-001 |
| `development/capabilities/claims/CE-CAP-PRED-PROB-001.yaml` | CE-CAP-PRED-PROB-001 | Probabilistic regression threshold queries | CE-PRED-003 |
| `development/capabilities/claims/CE-CAP-EXPL-ENSURED-001.yaml` | CE-CAP-EXPL-ENSURED-001 | Uncertainty-filtered (ensured) alternatives | CE-ADV-001 |
| `development/capabilities/claims/CE-CAP-EXPL-CONJ-001.yaml` | CE-CAP-EXPL-CONJ-001 | Conjunctive multi-feature rules | CE-ADV-002 |
| `development/capabilities/claims/CE-CAP-NARR-001.yaml` | CE-CAP-NARR-001 | Human-readable narratives via to_narrative() | CE-ADV-003 |
| `development/capabilities/claims/CE-CAP-GUARD-001.yaml` | CE-CAP-GUARD-001 | Guarded explanations via GuardedOptions | CE-ADV-004 |
| `development/capabilities/claims/CE-CAP-REJECT-001.yaml` | CE-CAP-REJECT-001 | Reject/defer policies via RejectPolicySpec | CE-DECS-001 |
| `development/capabilities/claims/CE-CAP-MOND-001.yaml` | CE-CAP-MOND-001 | Mondrian conditional calibration | CE-DECS-002 |
| `development/capabilities/claims/CE-CAP-PLUGIN-001.yaml` | CE-CAP-PLUGIN-001 | Plugin protocol importability | CE-DECS-003 |
| `development/capabilities/claims/CE-CAP-VIZ-001.yaml` | CE-CAP-VIZ-001 | Visualization smoke (no-raise) | CE-DECS-004 |

Notes:
- CE-PRED-002 (conformal interval regression) is already covered by CE-CAP-PRED-001
  from the initial scaffold (uncertainty intervals via uq_interval=True).
- CE-PAR-001 was explicitly excluded — it is a CEE parity claim (CEE wraps CE core),
  not a CE capability claim.

### Requirements (10 new, plus 3 from initial scaffold)

| File | Req ID | Claim | Obligation type |
|---|---|---|---|
| `development/capabilities/requirements/CE-REQ-PRED-CLASS-API-001.md` | CE-REQ-PRED-CLASS-API-001 | CE-CAP-PRED-CLASS-001 | api_contract |
| `development/capabilities/requirements/CE-REQ-PRED-PROB-API-001.md` | CE-REQ-PRED-PROB-API-001 | CE-CAP-PRED-PROB-001 | api_contract |
| `development/capabilities/requirements/CE-REQ-EXPL-ENSURED-API-001.md` | CE-REQ-EXPL-ENSURED-API-001 | CE-CAP-EXPL-ENSURED-001 | api_contract |
| `development/capabilities/requirements/CE-REQ-EXPL-CONJ-API-001.md` | CE-REQ-EXPL-CONJ-API-001 | CE-CAP-EXPL-CONJ-001 | api_contract |
| `development/capabilities/requirements/CE-REQ-NARR-API-001.md` | CE-REQ-NARR-API-001 | CE-CAP-NARR-001 | api_contract |
| `development/capabilities/requirements/CE-REQ-GUARD-API-001.md` | CE-REQ-GUARD-API-001 | CE-CAP-GUARD-001 | api_contract |
| `development/capabilities/requirements/CE-REQ-REJECT-API-001.md` | CE-REQ-REJECT-API-001 | CE-CAP-REJECT-001 | api_contract |
| `development/capabilities/requirements/CE-REQ-MOND-API-001.md` | CE-REQ-MOND-API-001 | CE-CAP-MOND-001 | api_contract |
| `development/capabilities/requirements/CE-REQ-PLUGIN-DOC-001.md` | CE-REQ-PLUGIN-DOC-001 | CE-CAP-PLUGIN-001 | documentation_boundary |
| `development/capabilities/requirements/CE-REQ-VIZ-SMOKE-001.md` | CE-REQ-VIZ-SMOKE-001 | CE-CAP-VIZ-001 | empirical_smoke |

### Tests (14 new, plus 7 from initial scaffold = 21 total)

| File | Tests | Requirements covered |
|---|---|---|
| `tests/capabilities/test_classification_contracts.py` | 2 | CE-REQ-PRED-CLASS-API-001 |
| `tests/capabilities/test_probabilistic_regression_contracts.py` | 2 | CE-REQ-PRED-PROB-API-001 |
| `tests/capabilities/test_advanced_explanation_contracts.py` | 3 | CE-REQ-EXPL-ENSURED-API-001, CE-REQ-EXPL-CONJ-API-001 |
| `tests/capabilities/test_narrative_contracts.py` | 1 | CE-REQ-NARR-API-001 |
| `tests/capabilities/test_guard_contracts.py` | 1 | CE-REQ-GUARD-API-001 |
| `tests/capabilities/test_reject_policy_contracts.py` | 1 | CE-REQ-REJECT-API-001 |
| `tests/capabilities/test_mondrian_contracts.py` | 1 | CE-REQ-MOND-API-001 |
| `tests/capabilities/test_plugin_contracts.py` | 2 | CE-REQ-PLUGIN-DOC-001 |
| `tests/capabilities/test_visualization_contracts.py` | 1 | CE-REQ-VIZ-SMOKE-001 |

---

## Completion criteria verification

| Criterion | Status |
|---|---|
| Full CE capability surface from ce_engine.yaml source material covered | ✅ (13 claims, excluding CE-PAR-001 which is CEE-owned) |
| All new claims use CE-native terminology (no CEE/enterprise/parity/upstream) | ✅ |
| Statistical assumption boundaries explicit for all claims | ✅ |
| No overclaiming (API tests do not assert calibration validity) | ✅ |
| `pytest tests/capabilities/ -v` all tests green | ✅ (21 passed, 0 failed) |
| ADR-030 test naming (`test_should_<behavior>_when_<condition>`) | ✅ |
| No private member access in tests/capabilities/ | ✅ |
| All tests self-contained (no disk reads, fixed RNG seed=42) | ✅ |

---

## Verification metadata

| Field | Value |
|---|---|
| commit | To be set at commit time |
| package_version | calibrated_explanations 0.11.4-dev |
| test_command | `pytest tests/capabilities/ -v` |
| tests_run | 21 |
| tests_passed | 21 |
| tests_failed | 0 |
| dataset | sklearn make_classification / make_regression (seed=42, in-test, no disk reads) |
| random_seed | 42 |
| date | 2026-06-17 |

---

## Known limitations

1. **Regression interval coverage**: CE-PRED-002 (conformal interval regression) is
   covered by CE-CAP-PRED-001 from the initial scaffold (uq_interval=True).
   No new claim was added to avoid redundancy.

2. **CE-DECS-003 plugin registration**: CE-CAP-PLUGIN-001 covers protocol importability
   only (`documentation_boundary`). Plugin registration through PluginManager uses internal
   APIs not stable for public use; full plugin lifecycle verification is deferred.

3. **Visualization correctness**: CE-CAP-VIZ-001 is an empirical smoke test (no-raise
   only). Visual correctness verification requires manual review and is not automated.

4. **Pre-existing test failures**: `test_legacy_payload_*` in
   `tests/unit/test_explanations_collection.py` fail due to the ADR-008 `legacy_payload`
   deprecation (pre-existing). Not introduced by Task 9.
