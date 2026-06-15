# Reject Evaluation Summary

This report aggregates the outcome of the real `WrapCalibratedExplainer` reject evaluation suite.

## Core Scenarios

### Scenario 1 — Binary marginal coverage sweep — RQ1: Binary marginal coverage preservation

- **Status**: formal_target

- Coverage is reported as standard label-set coverage from the conformal prediction sets.
- Accepted accuracy is included as a separate empirical metric and not treated as the conformal guarantee.
- Observed coverage violations: 129/390 (0.3308).
- Structural violations (CI upper bound < 1-epsilon, cannot be attributed to finite-sample noise): 15/390.
- Uses 5-seed grid matching Scenarios 8-12 for stable violation counts.
Outcome snapshot:
- **datasets**: 26
- **seeds**: 5
- **rows**: 390
- **violation_rate**: 0.3308
- **structural_violations**: 15
- **mean_coverage**: 0.9482

Rows: 390
Columns: dataset, seed, epsilon, n_cal, n_test, coverage, lower_ci, upper_ci, violation, structural_violation, reject_rate, singleton_precision, singleton_recall, singleton_correct_count, singleton_count, singleton_precision_recall_defined, accepted_accuracy_empirical

### Scenario 2 - Multiclass correctness proxy — RQ2: Multiclass correctness proxy

- **Status**: empirical

- Primary empirical accuracy is computed in the binary proxy space: singleton {1}/{0} is compared with 1[top-1 prediction is correct].
- Accepted top-1 accuracy remains a precision-style diagnostic on {1} rows only; it is not the proxy classifier accuracy.
- This scenario opts into the multiclass-only experimental.multiclass_top1_correctness strategy.
- It evaluates CE multiclass reject as a binary correctness proxy, not a default rule and not a K-class prediction-set method.
- Accepted instances are restricted to {1} positive correctness-proxy singletons.
- {0} singletons are proxy-negative singletons: the aggregate non-top1 event is conforming, but no specific alternative class is selected.
- reject_rate is retained as a compatibility alias for non_accepted_rate in this proxy scenario.
- Hinge NCF is used for both 'default' and 'ensured' paths. Margin NCF was removed (it produced identical scores for both columns, making singletons impossible).
Outcome snapshot:
- **datasets**: 20
- **mean_proxy_singleton_accuracy**: 0.8841
- **mean_singleton_precision**: 0.8841
- **mean_singleton_recall**: 0.5031
- **mean_accepted_top1_accuracy**: 0.9006
- **mean_proxy_negative_singleton_accuracy**: 0.5289
- **mean_non_accepted_rate**: 0.4711
- **mean_reject_rate**: 0.4711
- **mean_positive_singleton_rate**: 0.5289
- **mean_correct_singleton_rate**: 0.5289
- **mean_proxy_negative_singleton_rate**: 0.0200
- **mean_error_singleton_rate**: 0.0200
- **collapse_events**: 8

Rows: 80
Columns: dataset, epsilon, ncf, n_cal, n_test, n_classes, proxy_singleton_accuracy, proxy_singleton_accuracy_defined, proxy_singleton_count, singleton_precision, singleton_recall, singleton_correct_count, singleton_count, singleton_precision_recall_defined, accepted_top1_accuracy, proxy_negative_singleton_accuracy, non_accepted_rate, reject_rate, positive_singleton_rate, correct_singleton_rate, proxy_negative_singleton_rate, error_singleton_rate, ambiguity_rate, novelty_rate, expected_collapse, guarantee_status

### Scenario 3 - Thresholded regression binary-event reject validity — RQ3: Threshold regression heuristic baseline

- **Status**: empirical

- Thresholded regression reject is evaluated as binary conformal classification over event labels.
- Scalar event: y <= threshold. Interval event: low < y <= high.
- Coverage is empirical event-label coverage from conformal prediction sets over {0, 1}.
- Singleton precision, recall, and empirical singleton error are derived from those same event labels.
- No interval-width selection, interval coverage, or accepted-interval-width diagnostic is part of Scenario 3.
- Observed event-coverage violations: 69/264.
- Structural violations (CI upper bound < confidence): 8/264.
Outcome snapshot:
- **datasets**: 22
- **rows**: 264
- **mean_event_coverage**: 0.9384
- **mean_reject_rate**: 0.2189
- **mean_empirical_singleton_error**: 0.0811
- **coverage_violations**: 69
- **structural_violations**: 8

Rows: 264
Columns: dataset, confidence, epsilon, effective_confidence, threshold_type, threshold_id, threshold_quantile, threshold_lower_quantile, threshold_upper_quantile, effective_threshold, threshold_value, threshold_low, threshold_high, threshold_source, n_cal, n_test, event_prevalence, empirical_event_coverage, coverage_defined, lower_ci, upper_ci, violation, structural_violation, empirical_singleton_error, reject_rate, n_total, n_empty, n_singleton, n_ambiguity, novelty_rate, singleton_rate, ambiguity_rate, reject_rate_from_sets, singleton_precision, singleton_recall, singleton_correct_count, singleton_count, singleton_precision_recall_defined

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
Columns: task_type, dataset, ncf, w, accept_rate, accepted_accuracy, accepted_accuracy_delta, singleton_precision, singleton_recall, singleton_correct_count, singleton_count, singleton_precision_recall_defined

### Scenario 5 — Explanation quality on accepted instances — RQ5: Explanation quality on accepted instances

- **Status**: empirical

- Explanation quality is evaluated only empirically; no conformal claim is attached.
- Regime boundaries: low (<=15%), moderate (15%–40%), high (>40%) reject rate.
- Paper finding: accuracy_delta is most reliable in the low regime.
- mean_feature_weight_variance is not included — it is not a paper metric.
Outcome snapshot:
- **datasets**: 46
- **mean_accuracy_delta**: 0.0934
- **mean_ece_delta**: -0.0593
- **regime_summary**: (see json artifact)

Rows: 46
Columns: dataset, task_type, n_test, confidence, reject_rate, regime, baseline_accuracy, accepted_accuracy, accuracy_delta, baseline_ece, accepted_ece, ece_delta, singleton_precision, singleton_recall, singleton_correct_count, singleton_count, singleton_precision_recall_defined

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
Columns: dataset, probe, n_cal, epsilon, coverage, reject_rate, error_rate, violation, matched_count, singleton_precision, singleton_recall, singleton_correct_count, singleton_count, singleton_precision_recall_defined

### Scenario 8 — Difficulty estimator reject ablation — Ablation: Difficulty estimator reject ablation

- **Status**: empirical

- Measures the current indirect difficulty effect through Venn-Abers scaling only; reject scoring itself is unchanged.
- Arms compare use_difficulty in {False, True} crossed with reject NCF in {default, ensured}.
- Difficulty summary columns use the same deterministic reference estimator in all arms so selection differences are comparable.
- This scenario does not test difficulty-normalized reject NCFs; it quantifies the baseline before that experiment.
- With `default`, enabling difficulty changed accept_rate by -42.0 pp, rejected_error_capture_rate by +18.8 pp, and accepted_accuracy by -10.1 pp.
- With `default`, mean empirical coverage shifted by +4.4 pp across the swept confidence grid.
- With `default` and difficulty enabled, rejected instances were harder than accepted ones by 0.039 mean difficulty units.
- For `default`, the current difficulty path acts mainly as a stricter reject gate: it captures more errors, but at the cost of accepting far fewer instances and lowering accepted accuracy.
- With `ensured`, enabling difficulty changed accept_rate by -24.7 pp, rejected_error_capture_rate by +7.6 pp, and accepted_accuracy by -10.5 pp.
- With `ensured`, mean empirical coverage shifted by +2.8 pp across the swept confidence grid.
- With `ensured` and difficulty enabled, rejected instances were harder than accepted ones by 0.076 mean difficulty units.
- For `ensured`, the current difficulty path acts mainly as a stricter reject gate: it captures more errors, but at the cost of accepting far fewer instances and lowering accepted accuracy.
- The markdown now includes a by-confidence table so the headline summary is no longer averaged over hidden epsilon values.
- Integrity checks verify reject_rate = ambiguity_rate + novelty_rate, accepted instances match singleton prediction sets, and no positive ambiguity appears without prediction sets.
- Empirical coverage is reported only for rows whose prediction-set columns are label-index aligned; unsupported rows stay `nan` instead of inventing a value.
Outcome snapshot:
- **rows**: 8280
- **datasets**: 46
- **seeds**: 5
- **mean_accept_rate**: 0.3344
- **mean_accuracy_delta**: 0.0007
- **default_accept_rate_no_difficulty**: 0.6332
- **default_accept_rate_with_difficulty**: 0.2132
- **default_accept_rate_delta**: -0.4200
- **default_accepted_accuracy_delta**: -0.1006
- **default_accuracy_delta_delta**: -0.0956
- **default_rejected_error_capture_rate_delta**: 0.1883
- **default_singleton_error_rate_delta**: 0.4576
- **default_difficulty_gap_with_difficulty**: 0.0395
- **default_empirical_coverage_no_difficulty**: 0.8965
- **default_empirical_coverage_with_difficulty**: 0.9403
- **default_coverage_gap_delta**: 0.0438
- **ensured_accept_rate_no_difficulty**: 0.3693
- **ensured_accept_rate_with_difficulty**: 0.1220
- **ensured_accept_rate_delta**: -0.2474
- **ensured_accepted_accuracy_delta**: -0.1051
- **ensured_accuracy_delta_delta**: -0.0923
- **ensured_rejected_error_capture_rate_delta**: 0.0762
- **ensured_singleton_error_rate_delta**: 0.3665
- **ensured_difficulty_gap_with_difficulty**: 0.0758
- **ensured_empirical_coverage_no_difficulty**: 0.9130
- **ensured_empirical_coverage_with_difficulty**: 0.9406
- **ensured_coverage_gap_delta**: 0.0275
- **mean_difficulty_gap_with_difficulty**: 0.0501
- **unique_confidences**: 9
- **min_epsilon**: 0.0100
- **max_epsilon**: 0.2000
- **max_abs_reject_partition_residual**: 0.0000
- **max_abs_accept_singleton_residual**: 0.0000
- **positive_ambiguity_without_prediction_set_rows**: 0
- **equal_positive_ambiguity_novelty_rows**: 4
- **coverage_defined_rows**: 4680
- **min_empirical_coverage_gap**: -0.2179
- **max_empirical_coverage_gap**: 0.2000

Rows: 8280
Columns: task_type, dataset, seed, confidence, epsilon, n_train, n_cal, n_test, ncf, use_difficulty, arm, accept_rate, reject_rate, ambiguity_rate, novelty_rate, accepted_accuracy, full_accuracy, accuracy_delta, singleton_error_rate, singleton_precision, singleton_recall, singleton_correct_count, singleton_count, singleton_precision_recall_defined, error_rate_defined, rejected_error_capture_rate, mean_difficulty_all, mean_difficulty_accepted, mean_difficulty_rejected, empty_rate, singleton_rate, multilabel_rate, empirical_coverage, coverage_gap, coverage_defined, has_prediction_set, reject_partition_residual, accept_singleton_residual, ambiguity_multilabel_residual, novelty_empty_residual, ambiguity_equals_novelty, ambiguity_equals_novelty_positive, positive_ambiguity_without_prediction_set

### Scenario 9 - Difficulty-normalized reject NCF strategy ablation — Ablation: Difficulty-normalized reject NCF strategy ablation

- **Status**: empirical

- Compares indirect VA-difficulty support against direct experimental difficulty-normalized reject scoring.
- Primary scientific contrast is A vs C (default NCF, no VA difficulty in either arm).
- Arms D and F are diagnostic for potential difficulty double-counting when VA and score normalization are both enabled.
- Includes strategy metadata and difficulty_reject_auc for reject-selectivity diagnostics.
- Includes accepted-accuracy comparison at matched reject-rate bins for A vs C.
- Direct normalization (C vs A) changed reject_rate by +0.0313, difficulty-gap by +0.2950, and difficulty_reject_auc by +0.1985.
- At matched reject-rate bins, C minus A mean accepted_accuracy is -0.0053.
- For C vs A, ambiguity_rate changed by +0.0217 and novelty_rate by +0.0096.
- Double-count diagnostics: D-B reject_rate delta -0.1473, F-E reject_rate delta +0.1093; difficulty-gap deltas are +0.3909 and +0.2846.
- Recommended arm for next iteration: C (primary A-vs-C contrast with direct normalization and no VA double-count risk). NOTE: Scenario 12 shows arm C has more structural coverage violations than arm A. This recommendation is for selectivity/accuracy only; promotion requires Scenario 13 clearance.
Outcome snapshot:
- **rows**: 12420
- **datasets**: 46
- **seeds**: 5
- **mean_accept_rate**: 0.3959
- **mean_accuracy_delta**: 0.0136
- **A_vs_C_reject_rate_delta**: 0.0313
- **A_vs_C_difficulty_gap_delta**: 0.2950
- **A_vs_C_difficulty_reject_auc_delta**: 0.1985
- **A_vs_C_ambiguity_rate_delta**: 0.0217
- **A_vs_C_novelty_rate_delta**: 0.0096
- **A_vs_C_matched_bin_accepted_accuracy_delta**: -0.0053
- **D_minus_B_reject_rate_delta**: -0.1473
- **F_minus_E_reject_rate_delta**: 0.1093
- **D_minus_B_difficulty_gap_delta**: 0.3909
- **F_minus_E_difficulty_gap_delta**: 0.2846
- **recommended_arm**: C
- **recommendation_reason**: primary A-vs-C contrast with direct normalization and no VA double-count risk
- **coverage_validity_caveat**: NOTE: Scenario 12 shows arm C has more structural coverage violations than arm A. This recommendation is for selectivity/accuracy only; promotion requires Scenario 13 clearance.
- **metric_consistency_note**: (see json artifact)

Rows: 12420
Columns: task_type, dataset, seed, confidence, epsilon, n_train, n_cal, n_test, arm_code, arm_label, estimator_type, ncf, strategy, use_va_difficulty, difficulty_normalized, double_count_difficulty, accept_rate, reject_rate, ambiguity_rate, novelty_rate, accepted_accuracy, full_accuracy, accuracy_delta, singleton_error_rate, singleton_precision, singleton_recall, singleton_correct_count, singleton_count, singleton_precision_recall_defined, error_rate_defined, rejected_error_capture_rate, mean_difficulty_all, mean_difficulty_accepted, mean_difficulty_rejected, difficulty_gap_rejected_minus_accepted, difficulty_reject_auc, empty_rate, singleton_rate, multilabel_rate, empirical_coverage, coverage_gap, coverage_defined

## Supplementary Scenarios

### Scenario 7 - NCF coverage validity sweep (supplementary) — C1: NCF coverage validity sweep (supplementary)

- **Status**: empirical

- SUPPLEMENTARY empirical diagnostic; not a standalone proof of conformal validity.
- Coverage is measured from prediction sets stored in result.metadata['prediction_set'].
- Observed row-level coverage violations: 841/2080.
- Observed row-level structural violations: 100/2080.
- The dominant tendency is singleton collapse on harder datasets: when accept_rate/singleton_rate is high, prediction-set coverage tracks ordinary baseline accuracy rather than gaining much from ambiguity sets.
- High-accept structural rows (accept_rate >= 0.95): 28/100.
- Collapsed by (dataset, seed, ncf, epsilon), structural violations are 38/520; this avoids over-reading repeated w rows for default NCF.
- structural_violation means the Clopper-Pearson upper bound is below 1-epsilon in this finite test batch; it is strong diagnostic evidence, not a separate theorem.
Outcome snapshot:
- **datasets**: 26
- **seeds**: 5
- **rows**: 2080
- **coverage_defined_count**: 2080
- **coverage_undefined_count**: 0
- **total_violations**: 841
- **structural_violations**: 100
- **independent_condition_groups**: 520
- **independent_total_violations**: 277
- **independent_structural_violations**: 38
- **high_accept_structural_violations**: 28
- **mean_by_ncf_epsilon**: [{'ncf': 'default', 'epsilon': 0.05, 'mean_coverage': 0.9496592720489462, 'mean_baseline_accuracy': 0.8057468961796157, 'mean_accept_rate': 0.5476097607730878, 'mean_singleton_rate': 0.5476097607730878, 'structural_violations': 24}, {'ncf': 'default', 'epsilon': 0.1, 'mean_coverage': 0.8980092196086135, 'mean_baseline_accuracy': 0.8057468961796157, 'mean_accept_rate': 0.7126330193726184, 'mean_singleton_rate': 0.7126330193726184, 'structural_violations': 28}, {'ncf': 'ensured', 'epsilon': 0.05, 'mean_coverage': 0.9554412237323779, 'mean_baseline_accuracy': 0.8057468961796157, 'mean_accept_rate': 0.35266214164592086, 'mean_singleton_rate': 0.35266214164592086, 'structural_violations': 23}, {'ncf': 'ensured', 'epsilon': 0.1, 'mean_coverage': 0.9088091546268486, 'mean_baseline_accuracy': 0.8057468961796157, 'mean_accept_rate': 0.4566215221436756, 'mean_singleton_rate': 0.4566215221436756, 'structural_violations': 25}]
- **top_structural_datasets**: [{'dataset': 'je4243', 'structural': 19, 'mean_coverage': 0.884931506849315, 'mean_accept_rate': 0.3184931506849315, 'mean_singleton_rate': 0.3184931506849315}, {'dataset': 'heartS', 'structural': 18, 'mean_coverage': 0.8935185185185185, 'mean_accept_rate': 0.5474537037037037, 'mean_singleton_rate': 0.5474537037037037}, {'dataset': 'creditA', 'structural': 18, 'mean_coverage': 0.911322463768116, 'mean_accept_rate': 0.7323369565217391, 'mean_singleton_rate': 0.7323369565217391}, {'dataset': 'liver', 'structural': 15, 'mean_coverage': 0.9233695652173914, 'mean_accept_rate': 0.32753623188405795, 'mean_singleton_rate': 0.32753623188405795}, {'dataset': 'kc3', 'structural': 11, 'mean_coverage': 0.9225, 'mean_accept_rate': 0.5630769230769231, 'mean_singleton_rate': 0.5630769230769231}, {'dataset': 'colic', 'structural': 8, 'mean_coverage': 0.9069444444444444, 'mean_accept_rate': 0.6546875, 'mean_singleton_rate': 0.6546875}, {'dataset': 'pc1req', 'structural': 7, 'mean_coverage': 0.9, 'mean_accept_rate': 0.24761904761904763, 'mean_singleton_rate': 0.24761904761904763}, {'dataset': 'spectf', 'structural': 1, 'mean_coverage': 0.924074074074074, 'mean_accept_rate': 0.4810185185185185, 'mean_singleton_rate': 0.4810185185185185}]
- **structural_violations_by_ncf_w**: (see json artifact)
- **violations_by_ncf_w**: (see json artifact)

Rows: 2080
Columns: dataset, seed, ncf, w, epsilon, n_cal, n_test, coverage_defined, coverage, baseline_accuracy, coverage_lift_over_baseline_accuracy, lower_ci, upper_ci, violation, structural_violation, accept_rate, mean_prediction_set_size, singleton_rate, ambiguity_rate, novelty_rate, singleton_precision, singleton_recall, singleton_correct_count, singleton_count, singleton_precision_recall_defined
