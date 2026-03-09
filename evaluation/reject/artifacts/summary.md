# Reject Evaluation Summary

This report aggregates the outcome of the real `WrapCalibratedExplainer` reject evaluation suite.

## Integration Validation (A–D)

## Research Evaluation (E–K)

### Scenario F — Multiclass correctness evaluation — RQ2: Multiclass correctness evaluation

- **Status**: empirical

- Accepted top-1 accuracy is reported empirically while the formal guarantee remains a proof obligation.
- The artifact explicitly marks `guarantee_status=empirical` to avoid over-claiming.
- This scenario evaluates CE multiclass reject as a correctness classifier, not as a K-class prediction-set method.
Outcome snapshot:
- **datasets**: 20
- **mean_accepted_top1_accuracy**: 0.7620
- **mean_reject_rate**: 0.6034

Rows: 80
Columns: dataset, epsilon, ncf, n_cal, n_test, accepted_top1_accuracy, reject_rate, ambiguity_rate, guarantee_status

### Scenario G — Threshold regression empirical coverage — RQ3: Threshold regression accepted-subset behaviour

- **Status**: empirical

- Accepted-subset coverage is reported as an empirical quantity only.
- Threshold-based regression reject remains explicitly heuristic in this suite.
- Both interval width and MSE are tracked on the accepted subset to capture the trade-off.
Outcome snapshot:
- **datasets**: 22
- **mean_reject_rate**: 0.2083
- **mean_accepted_mse_empirical**: 0.0092

Rows: 220
Columns: dataset, confidence, threshold_quantile, n_cal, n_test, interval_coverage_all, accepted_coverage_empirical, interval_width_all, accepted_interval_width_empirical, mse_all, accepted_mse_empirical, reject_rate

### Scenario H — NCF grid — RQ4: NCF selection and precision-coverage tradeoff

- **Status**: empirical

- This grid compares hinge, margin, entropy, and ensured NCFs across binary and multiclass settings.
- Accepted accuracy delta is always empirical and benchmarked against the non-reject baseline.
Outcome snapshot:
- **rows**: 736
- **datasets**: 46
- **best_accuracy_delta**: 0.5806

Rows: 736
Columns: task_type, dataset, ncf, w, accept_rate, accepted_accuracy, accepted_accuracy_delta

### Scenario I — Explanation quality — RQ5: Explanation quality on accepted instances

- **Status**: empirical

- Explanation quality is evaluated only empirically; no conformal claim is attached to these metrics.
- Feature-weight stability is computed from per-instance explanation weight vectors.
Outcome snapshot:
- **datasets**: 46
- **mean_accuracy_delta**: 0.0933
- **mean_ece_delta**: -0.0594

Rows: 46
Columns: dataset, n_test, confidence, baseline_accuracy, accepted_accuracy, accuracy_delta, baseline_ece, accepted_ece, ece_delta, reject_rate, weight_variance_all, weight_variance_accepted

### Scenario J — Stress tests — RQ6: Finite-sample stress tests

- **Status**: empirical

- Stress tests focus on finite-sample behavior and extreme confidence settings.
- The suite reports coverage violations empirically rather than claiming new guarantees.
Outcome snapshot:
- **rows**: 51
- **violations**: 27
- **max_reject_rate**: 1.0000

Rows: 51
Columns: dataset, probe, n_cal, epsilon, coverage, reject_rate, error_rate, violation, matched_count

### Scenario K — Regression reject comparison — RQ7: Difficulty-normalised regression reject (K1)

- **Status**: target_formal

- The primary method follows the 2024 conformal regression with reject option paper via difficulty-based Mondrian categories.
- Threshold and value-bin methods are retained only as heuristic baselines.
- Accepted-subset metrics are explicitly empirical for the heuristic baselines.
Outcome snapshot:
- **datasets**: 22
- **methods**: paper_difficulty_mondrian, threshold_baseline, value_bin_width_baseline
- **best_accepted_mae**: 0.0000

Rows: 264
Columns: dataset, confidence, method, difficulty_estimator, requested_reject_rate, empirical_reject_rate, accepted_coverage, accepted_interval_width, accepted_mae, accepted_mse, interval_coverage_all, mse_all, guarantee_status
