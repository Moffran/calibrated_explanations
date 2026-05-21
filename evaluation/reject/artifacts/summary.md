# Reject Evaluation Summary

This report aggregates the outcome of the real `WrapCalibratedExplainer` reject evaluation suite.

## Core Scenarios

### Scenario 1 — Binary marginal coverage sweep — RQ1: Binary marginal coverage preservation

- **Status**: formal_target

- Coverage is reported as standard label-set coverage from the conformal prediction sets.
- Accepted accuracy is included as a separate empirical metric and not treated as the conformal guarantee.
- Observed coverage violations: 5/9 (0.5556).
- Structural violations (CI upper bound < 1-epsilon, cannot be attributed to finite-sample noise): 0/9.
Outcome snapshot:
- **datasets**: 3
- **rows**: 9
- **violation_rate**: 0.5556
- **structural_violations**: 0
- **mean_coverage**: 0.9354

Rows: 9
Columns: dataset, epsilon, n_cal, n_test, coverage, lower_ci, upper_ci, violation, structural_violation, reject_rate, accepted_accuracy_empirical

### Scenario 2 — Multiclass correctness classifier — RQ2: Multiclass correctness classifier

- **Status**: empirical

- Accepted top-1 accuracy is reported empirically; the formal guarantee remains a proof obligation.
- This scenario evaluates CE multiclass reject as a correctness classifier, not a K-class prediction-set method.
Outcome snapshot:
- **datasets**: 2
- **mean_accepted_top1_accuracy**: nan
- **mean_reject_rate**: 1.0000
- **hinge_collapse_events**: 0

Rows: 8
Columns: dataset, epsilon, ncf, n_cal, n_test, n_classes, accepted_top1_accuracy, reject_rate, ambiguity_rate, expected_collapse, guarantee_status

### Scenario 3 — Threshold regression heuristic baseline — RQ3: Threshold regression heuristic baseline

- **Status**: empirical

- Headline finding: threshold reject does NOT select by uncertainty — accepted-subset interval width equals full-set interval width (~0 delta).
- Mean interval_width_delta across all rows: -0.0000 (near zero confirms the null result).
- Threshold-based regression reject remains explicitly heuristic in this suite.
- Both interval width and MSE are tracked on the accepted subset to capture the trade-off.
- The difficulty-normalised approach (C3) is deferred to a standalone scenario post-RT2.
Outcome snapshot:
- **datasets**: 2
- **mean_reject_rate**: 0.3275
- **mean_accepted_mse_empirical**: 0.0217
- **mean_interval_width_delta**: -0.0000

Rows: 12
Columns: dataset, confidence, effective_confidence, threshold_quantile, effective_threshold, threshold_source, n_cal, n_test, interval_coverage_all, accepted_coverage_empirical, interval_width_all, accepted_interval_width_empirical, interval_width_delta, mse_all, accepted_mse_empirical, reject_rate

### Scenario 4 — NCF and blend weight grid — RQ4: NCF selection and precision-coverage tradeoff

- **Status**: empirical

- NCFs tested: default, ensured. Entropy and explicit hinge/margin modes are excluded from this suite.
- accept_rate is the fraction of accepted instances — NOT ICP label-set coverage.
- w >= 0.7 converges NCF behavior; w=0.3 amplifies differences between NCFs where present.
- Accepted accuracy delta is always empirical and benchmarked against the non-reject baseline.
Outcome snapshot:
- **rows**: 40
- **datasets**: 5
- **best_accuracy_delta**: 0.2255
- **ncfs_tested**: ['default', 'ensured']
- **w_values_tested**: [0.3, 0.5, 0.7, 1.0]

Rows: 40
Columns: task_type, dataset, ncf, w, accept_rate, accepted_accuracy, accepted_accuracy_delta

### Scenario 5 — Explanation quality on accepted instances — RQ5: Explanation quality on accepted instances

- **Status**: empirical

- Explanation quality is evaluated only empirically; no conformal claim is attached.
- Regime boundaries: low (<=15%), moderate (15%–40%), high (>40%) reject rate.
- Paper finding: accuracy_delta is most reliable in the low regime.
- mean_feature_weight_variance is not included — it is not a paper metric.
Outcome snapshot:
- **datasets**: 5
- **mean_accuracy_delta**: 0.0798
- **mean_ece_delta**: -0.0029
- **regime_summary**: (see json artifact)

Rows: 5
Columns: dataset, task_type, n_test, confidence, reject_rate, regime, baseline_accuracy, accepted_accuracy, accuracy_delta, baseline_ece, accepted_ece, ece_delta

### Scenario 6 — Finite-sample stress tests — RQ6: Finite-sample stress tests

- **Status**: empirical

- Stress tests focus on finite-sample behavior and extreme confidence settings.
- The suite reports coverage violations empirically rather than claiming new guarantees.
- violation is computed from actual coverage, not hard-coded.
- extreme_confidence probe uses the same violation logic as small_calibration.
Outcome snapshot:
- **rows**: 22
- **violations**: 11
- **max_reject_rate**: 1.0000
- **small_cal_violations**: 10
- **extreme_conf_violations**: 1

Rows: 22
Columns: dataset, probe, n_cal, epsilon, coverage, reject_rate, error_rate, violation, matched_count

### Scenario 8 — Difficulty estimator reject ablation — Ablation: Difficulty estimator reject ablation

- **Status**: empirical

- Measures the current indirect difficulty effect through Venn-Abers scaling only; reject scoring itself is unchanged.
- Arms compare use_difficulty in {False, True} crossed with reject NCF in {default, ensured}.
- Difficulty summary columns use the same deterministic reference estimator in all arms so selection differences are comparable.
- This scenario does not test difficulty-normalized reject NCFs; it quantifies the baseline before that experiment.
- With `default`, enabling difficulty changed accept_rate by -40.6 pp, rejected_error_capture_rate by +24.8 pp, and accepted_accuracy by -44.4 pp.
- With `default`, mean empirical coverage shifted by +8.1 pp across the swept confidence grid.
- With `default` and difficulty enabled, rejected instances were harder than accepted ones by 0.260 mean difficulty units.
- For `default`, the current difficulty path acts mainly as a stricter reject gate: it captures more errors, but at the cost of accepting far fewer instances and lowering accepted accuracy.
- With `ensured`, enabling difficulty changed accept_rate by -26.6 pp, rejected_error_capture_rate by +18.5 pp, and accepted_accuracy by -44.8 pp.
- With `ensured`, mean empirical coverage shifted by +7.4 pp across the swept confidence grid.
- With `ensured` and difficulty enabled, rejected instances were harder than accepted ones by 0.252 mean difficulty units.
- For `ensured`, the current difficulty path acts mainly as a stricter reject gate: it captures more errors, but at the cost of accepting far fewer instances and lowering accepted accuracy.
- The markdown now includes a by-confidence table so the headline summary is no longer averaged over hidden epsilon values.
- Integrity checks verify reject_rate = ambiguity_rate + novelty_rate, accepted instances match singleton prediction sets, and no positive ambiguity appears without prediction sets.
- Empirical coverage is reported only for rows whose prediction-set columns are label-index aligned; unsupported rows stay `nan` instead of inventing a value.
Outcome snapshot:
- **rows**: 160
- **datasets**: 5
- **seeds**: 2
- **mean_accept_rate**: 0.1887
- **mean_accuracy_delta**: -0.1357
- **default_accept_rate_no_difficulty**: 0.4280
- **default_accept_rate_with_difficulty**: 0.0217
- **default_accept_rate_delta**: -0.4063
- **default_accepted_accuracy_delta**: -0.4441
- **default_accuracy_delta_delta**: -0.4242
- **default_rejected_error_capture_rate_delta**: 0.2484
- **default_singleton_error_rate_delta**: 0.7870
- **default_difficulty_gap_with_difficulty**: 0.2595
- **default_empirical_coverage_no_difficulty**: 0.9025
- **default_empirical_coverage_with_difficulty**: 0.9839
- **default_coverage_gap_delta**: 0.0813
- **ensured_accept_rate_no_difficulty**: 0.2852
- **ensured_accept_rate_with_difficulty**: 0.0197
- **ensured_accept_rate_delta**: -0.2656
- **ensured_accepted_accuracy_delta**: -0.4481
- **ensured_accuracy_delta_delta**: -0.4281
- **ensured_rejected_error_capture_rate_delta**: 0.1854
- **ensured_singleton_error_rate_delta**: 0.7633
- **ensured_difficulty_gap_with_difficulty**: 0.2521
- **ensured_empirical_coverage_no_difficulty**: 0.9101
- **ensured_empirical_coverage_with_difficulty**: 0.9839
- **ensured_coverage_gap_delta**: 0.0738
- **mean_difficulty_gap_with_difficulty**: 0.2610
- **unique_confidences**: 4
- **min_epsilon**: 0.0100
- **max_epsilon**: 0.2000
- **max_abs_reject_partition_residual**: 0.0000
- **max_abs_accept_singleton_residual**: 0.0000
- **positive_ambiguity_without_prediction_set_rows**: 0
- **equal_positive_ambiguity_novelty_rows**: 0
- **coverage_defined_rows**: 96
- **min_empirical_coverage_gap**: -0.1550
- **max_empirical_coverage_gap**: 0.2000

Rows: 160
Columns: task_type, dataset, seed, confidence, epsilon, n_train, n_cal, n_test, ncf, use_difficulty, arm, accept_rate, reject_rate, ambiguity_rate, novelty_rate, accepted_accuracy, full_accuracy, accuracy_delta, singleton_error_rate, error_rate_defined, rejected_error_capture_rate, mean_difficulty_all, mean_difficulty_accepted, mean_difficulty_rejected, empty_rate, singleton_rate, multilabel_rate, empirical_coverage, coverage_gap, coverage_defined, has_prediction_set, reject_partition_residual, accept_singleton_residual, ambiguity_multilabel_residual, novelty_empty_residual, ambiguity_equals_novelty, ambiguity_equals_novelty_positive, positive_ambiguity_without_prediction_set

### Scenario 9 - Difficulty-normalized reject NCF strategy ablation — Ablation: Difficulty-normalized reject NCF strategy ablation

- **Status**: empirical

- Compares indirect VA-difficulty support against direct experimental difficulty-normalized reject scoring.
- Primary scientific contrast is A vs C (default NCF, no VA difficulty in either arm).
- Arms D and F are diagnostic for potential difficulty double-counting when VA and score normalization are both enabled.
- Includes strategy metadata and difficulty_reject_auc for reject-selectivity diagnostics.
- Includes accepted-accuracy comparison at matched reject-rate bins for A vs C.
- Direct normalization (C vs A) changed reject_rate by +0.2310, difficulty-gap by +0.3912, and difficulty_reject_auc by +0.2474.
- At matched reject-rate bins, C minus A mean accepted_accuracy is +0.0689.
- For C vs A, ambiguity_rate changed by +0.2790 and novelty_rate by -0.0480.
- Double-count diagnostics: D-B reject_rate delta +0.1130, F-E reject_rate delta +0.0287; difficulty-gap deltas are +0.5649 and +0.1382.
- Recommended arm for next iteration: C (primary A-vs-C contrast with direct normalization and no VA double-count risk).
Outcome snapshot:
- **rows**: 12420
- **datasets**: 46
- **seeds**: 5
- **mean_accept_rate**: 0.1197
- **mean_accuracy_delta**: -0.0047
- **A_vs_C_reject_rate_delta**: 0.2310
- **A_vs_C_difficulty_gap_delta**: 0.3912
- **A_vs_C_difficulty_reject_auc_delta**: 0.2474
- **A_vs_C_ambiguity_rate_delta**: 0.2790
- **A_vs_C_novelty_rate_delta**: -0.0480
- **A_vs_C_matched_bin_accepted_accuracy_delta**: 0.0689
- **D_minus_B_reject_rate_delta**: 0.1130
- **F_minus_E_reject_rate_delta**: 0.0287
- **D_minus_B_difficulty_gap_delta**: 0.5649
- **F_minus_E_difficulty_gap_delta**: 0.1382
- **recommended_arm**: C
- **recommendation_reason**: primary A-vs-C contrast with direct normalization and no VA double-count risk

Rows: 12420
Columns: task_type, dataset, seed, confidence, epsilon, n_train, n_cal, n_test, arm_code, arm_label, ncf, strategy, use_va_difficulty, difficulty_normalized, double_count_difficulty, accept_rate, reject_rate, ambiguity_rate, novelty_rate, accepted_accuracy, full_accuracy, accuracy_delta, singleton_error_rate, error_rate_defined, rejected_error_capture_rate, mean_difficulty_all, mean_difficulty_accepted, mean_difficulty_rejected, difficulty_gap_rejected_minus_accepted, difficulty_reject_auc, empty_rate, singleton_rate, multilabel_rate, empirical_coverage, coverage_gap, coverage_defined

## Supplementary Scenarios

### Scenario 7 — NCF coverage validity sweep (supplementary) — C1: NCF coverage validity sweep (supplementary)

- **Status**: empirical

- SUPPLEMENTARY — may yield misleading results before the RT-2 fix.
- Empirical companion to Proposition 1: coverage >= 1-epsilon across (NCF, w, epsilon) grid.
- structural_violation: CI upper bound < 1-epsilon; cannot be attributed to finite-sample noise.
Outcome snapshot:
- **datasets**: 3
- **total_violations**: 0
- **structural_violations**: 0
- **violations_by_ncf_w**: (see json artifact)

Rows: 48
Columns: dataset, ncf, w, epsilon, n_cal, n_test, coverage, lower_ci, upper_ci, violation, structural_violation, accept_rate
