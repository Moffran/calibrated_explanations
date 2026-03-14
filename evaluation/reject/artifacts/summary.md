# Reject Evaluation Summary

This report aggregates the outcome of the real `WrapCalibratedExplainer` reject evaluation suite.

## Core Research Scenarios (1–6)

### Scenario 1 — Binary marginal coverage sweep — RQ1: Binary marginal coverage preservation

- **Status**: formal_target

- Coverage is reported as standard label-set coverage from the conformal prediction sets.
- Accepted accuracy is included as a separate empirical metric and not treated as the conformal guarantee.
- Observed coverage violations: 31/78 (0.3974).
- Structural violations (CI upper bound < 1-epsilon, cannot be attributed to finite-sample noise): 4/78.
Outcome snapshot:
- **datasets**: 26
- **rows**: 78
- **violation_rate**: 0.3974
- **structural_violations**: 4
- **mean_coverage**: 0.9424

Rows: 78
Columns: dataset, epsilon, n_cal, n_test, coverage, lower_ci, upper_ci, violation, structural_violation, reject_rate, accepted_accuracy_empirical

### Scenario 2 — Multiclass correctness classifier — RQ2: Multiclass correctness classifier

- **Status**: empirical

- Accepted top-1 accuracy is reported empirically; the formal guarantee remains a proof obligation.
- This scenario evaluates CE multiclass reject as a correctness classifier, not a K-class prediction-set method.
Outcome snapshot:
- **datasets**: 20
- **mean_accepted_top1_accuracy**: nan
- **mean_reject_rate**: 1.0000
- **hinge_collapse_events**: 0

Rows: 80
Columns: dataset, epsilon, ncf, n_cal, n_test, n_classes, accepted_top1_accuracy, reject_rate, ambiguity_rate, expected_collapse, guarantee_status

### Scenario 3 — Threshold regression heuristic baseline — RQ3: Threshold regression heuristic baseline

- **Status**: empirical

- Headline finding: threshold reject does NOT select by uncertainty — accepted-subset interval width equals full-set interval width (~0 delta).
- Mean interval_width_delta across all rows: -0.0000 (near zero confirms the null result).
- Threshold-based regression reject remains explicitly heuristic in this suite.
- Both interval width and MSE are tracked on the accepted subset to capture the trade-off.
- The difficulty-normalised approach (C3) is deferred to a standalone scenario post-RT2.
Outcome snapshot:
- **datasets**: 22
- **mean_reject_rate**: 0.2083
- **mean_accepted_mse_empirical**: 0.0092
- **mean_interval_width_delta**: -0.0000

Rows: 220
Columns: dataset, confidence, effective_confidence, threshold_quantile, effective_threshold, threshold_source, n_cal, n_test, interval_coverage_all, accepted_coverage_empirical, interval_width_all, accepted_interval_width_empirical, interval_width_delta, mse_all, accepted_mse_empirical, reject_rate

### Scenario 4 — NCF and blend weight grid — RQ4: NCF selection and precision-coverage tradeoff

- **Status**: empirical

- NCFs tested: default, ensured. Entropy and explicit hinge/margin modes are excluded from this suite.
- accept_rate is the fraction of accepted instances — NOT ICP label-set coverage.
- w >= 0.7 converges NCF behavior; w=0.3 amplifies differences between NCFs where present.
- Accepted accuracy delta is always empirical and benchmarked against the non-reject baseline.
Outcome snapshot:
- **rows**: 368
- **datasets**: 46
- **best_accuracy_delta**: 0.3810
- **ncfs_tested**: ['default', 'ensured']
- **w_values_tested**: [0.3, 0.5, 0.7, 1.0]

Rows: 368
Columns: task_type, dataset, ncf, w, accept_rate, accepted_accuracy, accepted_accuracy_delta

### Scenario 5 — Explanation quality on accepted instances — RQ5: Explanation quality on accepted instances

- **Status**: empirical

- Explanation quality is evaluated only empirically; no conformal claim is attached.
- Regime boundaries: low (<=15%), moderate (15%–40%), high (>40%) reject rate.
- Paper finding: accuracy_delta is most reliable in the low regime.
- mean_feature_weight_variance is not included — it is not a paper metric.
Outcome snapshot:
- **datasets**: 46
- **mean_accuracy_delta**: 0.0893
- **mean_ece_delta**: -0.0876
- **regime_summary**: (see json artifact)

Rows: 46
Columns: dataset, task_type, n_test, confidence, reject_rate, regime, baseline_accuracy, accepted_accuracy, accuracy_delta, baseline_ece, accepted_ece, ece_delta

### Scenario 6 — Finite-sample stress tests — RQ6: Finite-sample stress tests

- **Status**: empirical

- Stress tests focus on finite-sample behavior and extreme confidence settings.
- The suite reports coverage violations empirically rather than claiming new guarantees.
- violation is computed from actual coverage, not hard-coded.
- extreme_confidence probe uses the same violation logic as small_calibration.
Outcome snapshot:
- **rows**: 51
- **violations**: 28
- **max_reject_rate**: 1.0000
- **small_cal_violations**: 27
- **extreme_conf_violations**: 1

Rows: 51
Columns: dataset, probe, n_cal, epsilon, coverage, reject_rate, error_rate, violation, matched_count

## Supplementary Scenarios

### Scenario 7 — NCF coverage validity sweep (supplementary) — C1: NCF coverage validity sweep (supplementary)

- **Status**: empirical

- SUPPLEMENTARY — may yield misleading results before the RT-2 fix.
- Empirical companion to Proposition 1: coverage >= 1-epsilon across (NCF, w, epsilon) grid.
- structural_violation: CI upper bound < 1-epsilon; cannot be attributed to finite-sample noise.
Outcome snapshot:
- **datasets**: 26
- **total_violations**: 0
- **structural_violations**: 0
- **violations_by_ncf_w**: (see json artifact)

Rows: 416
Columns: dataset, ncf, w, epsilon, n_cal, n_test, coverage, lower_ci, upper_ci, violation, structural_violation, accept_rate
