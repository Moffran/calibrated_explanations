# Scenario 8 — Difficulty estimator reject ablation

Rows: 160

## Key findings

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

## Outcome snapshot

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

## Result table

| task_type | dataset | seed | confidence | epsilon | n_train | n_cal | n_test | ncf | use_difficulty | arm | accept_rate | reject_rate | ambiguity_rate | novelty_rate | accepted_accuracy | full_accuracy | accuracy_delta | singleton_error_rate | error_rate_defined | rejected_error_capture_rate | mean_difficulty_all | mean_difficulty_accepted | mean_difficulty_rejected | empty_rate | singleton_rate | multilabel_rate | empirical_coverage | coverage_gap | coverage_defined | has_prediction_set | reject_partition_residual | accept_singleton_residual | ambiguity_multilabel_residual | novelty_empty_residual | ambiguity_equals_novelty | ambiguity_equals_novelty_positive | positive_ambiguity_without_prediction_set |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| binary | breast_cancer | 42 | 0.8000 | 0.2000 | 341 | 114 | 114 | default | no | default|difficulty=0 | 0.8860 | 0.1140 | 0.0000 | 0.1140 | 0.9901 | 0.9649 | 0.0252 | 0.0970 | yes | 0.7500 | 1.8789 | 1.9055 | 1.6721 | 0.1140 | 0.8860 | 0.0000 | 0.8772 | 0.0772 | yes | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no | no | no |
| binary | breast_cancer | 42 | 0.8633 | 0.1367 | 341 | 114 | 114 | default | no | default|difficulty=0 | 0.8947 | 0.1053 | 0.0000 | 0.1053 | 0.9804 | 0.9649 | 0.0155 | 0.0351 | yes | 0.5000 | 1.8789 | 1.9015 | 1.6863 | 0.1053 | 0.8947 | 0.0000 | 0.8772 | 0.0139 | yes | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no | no | no |
| binary | breast_cancer | 42 | 0.9267 | 0.0733 | 341 | 114 | 114 | default | no | default|difficulty=0 | 0.9649 | 0.0351 | 0.0000 | 0.0351 | 0.9636 | 0.9649 | -0.0013 | 0.0396 | yes | 0.0000 | 1.8789 | 1.8814 | 1.8077 | 0.0351 | 0.9649 | 0.0000 | 0.9298 | 0.0032 | yes | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no | no | no |
| binary | breast_cancer | 42 | 0.9900 | 0.0100 | 341 | 114 | 114 | default | no | default|difficulty=0 | 0.9386 | 0.0614 | 0.0614 | 0.0000 | 0.9720 | 0.9649 | 0.0071 | 0.0107 | yes | 0.2500 | 1.8789 | 1.8927 | 1.6679 | 0.0000 | 0.9386 | 0.0614 | 0.9737 | -0.0163 | yes | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no | no | no |
| binary | breast_cancer | 42 | 0.8000 | 0.2000 | 341 | 114 | 114 | ensured | no | ensured|difficulty=0 | 0.6228 | 0.3772 | 0.3772 | 0.0000 | 0.9718 | 0.9649 | 0.0069 | 0.3211 | yes | 0.5000 | 1.8789 | 1.7432 | 2.1028 | 0.0000 | 0.6228 | 0.3772 | 0.9825 | 0.1825 | yes | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no | no | no |
| binary | breast_cancer | 42 | 0.8633 | 0.1367 | 341 | 114 | 114 | ensured | no | ensured|difficulty=0 | 0.6228 | 0.3772 | 0.3772 | 0.0000 | 0.9718 | 0.9649 | 0.0069 | 0.2194 | yes | 0.5000 | 1.8789 | 1.7432 | 2.1028 | 0.0000 | 0.6228 | 0.3772 | 0.9825 | 0.1191 | yes | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no | no | no |
| binary | breast_cancer | 42 | 0.9267 | 0.0733 | 341 | 114 | 114 | ensured | no | ensured|difficulty=0 | 0.6228 | 0.3772 | 0.3772 | 0.0000 | 0.9718 | 0.9649 | 0.0069 | 0.1177 | yes | 0.5000 | 1.8789 | 1.7432 | 2.1028 | 0.0000 | 0.6228 | 0.3772 | 0.9825 | 0.0558 | yes | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no | no | no |
| binary | breast_cancer | 42 | 0.9900 | 0.0100 | 341 | 114 | 114 | ensured | no | ensured|difficulty=0 | 0.6053 | 0.3947 | 0.3947 | 0.0000 | 0.9710 | 0.9649 | 0.0061 | 0.0165 | yes | 0.5000 | 1.8789 | 1.7354 | 2.0989 | 0.0000 | 0.6053 | 0.3947 | 0.9825 | -0.0075 | yes | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no | no | no |
| binary | breast_cancer | 42 | 0.8000 | 0.2000 | 341 | 114 | 114 | default | yes | default|difficulty=1 | 0.0351 | 0.9649 | 0.9649 | 0.0000 | 1.0000 | 0.9649 | 0.0351 | 1.0000 | yes | 1.0000 | 1.8789 | 1.3681 | 1.8974 | 0.0000 | 0.0351 | 0.9649 | 0.9912 | 0.1912 | yes | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no | no | no |
| binary | breast_cancer | 42 | 0.8633 | 0.1367 | 341 | 114 | 114 | default | yes | default|difficulty=1 | 0.0351 | 0.9649 | 0.9649 | 0.0000 | 1.0000 | 0.9649 | 0.0351 | 1.0000 | yes | 1.0000 | 1.8789 | 1.3681 | 1.8974 | 0.0000 | 0.0351 | 0.9649 | 0.9912 | 0.1279 | yes | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no | no | no |
| binary | breast_cancer | 42 | 0.9267 | 0.0733 | 341 | 114 | 114 | default | yes | default|difficulty=1 | 0.0351 | 0.9649 | 0.9649 | 0.0000 | 1.0000 | 0.9649 | 0.0351 | 1.0000 | yes | 1.0000 | 1.8789 | 1.3681 | 1.8974 | 0.0000 | 0.0351 | 0.9649 | 0.9912 | 0.0646 | yes | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no | no | no |
| binary | breast_cancer | 42 | 0.9900 | 0.0100 | 341 | 114 | 114 | default | yes | default|difficulty=1 | 0.0351 | 0.9649 | 0.9649 | 0.0000 | 1.0000 | 0.9649 | 0.0351 | 0.2850 | yes | 1.0000 | 1.8789 | 1.3681 | 1.8974 | 0.0000 | 0.0351 | 0.9649 | 0.9912 | 0.0012 | yes | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no | no | no |

_Showing first 12 of 160 rows._

## Arm Summary

This table averages each arm across all datasets, seeds, and confidence levels.

| ncf | use_difficulty | accept_rate | accepted_accuracy | accuracy_delta | empirical_coverage | coverage_gap | rejected_error_capture_rate | mean_difficulty_accepted | mean_difficulty_rejected | singleton_rate | empty_rate | multilabel_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default | no | 0.4280 | 0.8997 | 0.0597 | 0.9025 | 0.0075 | 0.6943 | 1.9229 | 1.8816 | 0.4280 | 0.0562 | 0.5158 |
| default | yes | 0.0217 | 0.4556 | -0.3645 | 0.9839 | 0.0889 | 0.9426 | 1.6881 | 1.9477 | 0.0217 | 0.0244 | 0.9538 |
| ensured | no | 0.2852 | 0.8926 | 0.0526 | 0.9101 | 0.0151 | 0.7635 | 1.8564 | 1.9914 | 0.2852 | 0.0234 | 0.6913 |
| ensured | yes | 0.0197 | 0.4444 | -0.3756 | 0.9839 | 0.0889 | 0.9489 | 1.6949 | 1.9470 | 0.0197 | 0.0239 | 0.9565 |

## By Confidence And Arm

This table keeps the reject operating point visible instead of averaging across the whole epsilon sweep.
`empirical_coverage` is computed from the returned prediction sets; `coverage_gap = empirical_coverage - confidence`.

| confidence | epsilon | ncf | use_difficulty | accept_rate | accepted_accuracy | rejected_error_capture_rate | empirical_coverage | coverage_gap | ambiguity_rate | novelty_rate | singleton_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8000 | 0.2000 | default | no | 0.5301 | 0.8705 | 0.6198 | 0.8369 | 0.0369 | 0.3652 | 0.1047 | 0.5301 |
| 0.8000 | 0.2000 | default | yes | 0.0257 | 0.4467 | 0.9288 | 0.9808 | 0.1808 | 0.9169 | 0.0575 | 0.0257 |
| 0.8633 | 0.1367 | default | no | 0.5118 | 0.8803 | 0.6268 | 0.8632 | -0.0002 | 0.4098 | 0.0784 | 0.5118 |
| 0.8633 | 0.1367 | default | yes | 0.0257 | 0.4467 | 0.9288 | 0.9808 | 0.1174 | 0.9441 | 0.0303 | 0.0257 |
| 0.9267 | 0.0733 | default | no | 0.4526 | 0.9103 | 0.6731 | 0.9240 | -0.0027 | 0.5075 | 0.0400 | 0.4526 |
| 0.9267 | 0.0733 | default | yes | 0.0257 | 0.4467 | 0.9288 | 0.9808 | 0.0541 | 0.9643 | 0.0100 | 0.0257 |
| 0.9900 | 0.0100 | default | no | 0.2176 | 0.9566 | 0.8572 | 0.9861 | -0.0039 | 0.7808 | 0.0016 | 0.2176 |
| 0.9900 | 0.0100 | default | yes | 0.0100 | 0.5000 | 0.9841 | 0.9931 | 0.0031 | 0.9900 | 0.0000 | 0.0100 |
| 0.8000 | 0.2000 | ensured | no | 0.3093 | 0.8640 | 0.7173 | 0.8407 | 0.0407 | 0.6345 | 0.0562 | 0.3093 |
| 0.8000 | 0.2000 | ensured | yes | 0.0229 | 0.4333 | 0.9371 | 0.9808 | 0.1808 | 0.9223 | 0.0549 | 0.0229 |
| 0.8633 | 0.1367 | ensured | no | 0.3343 | 0.8754 | 0.7148 | 0.8824 | 0.0190 | 0.6351 | 0.0306 | 0.3343 |
| 0.8633 | 0.1367 | ensured | yes | 0.0229 | 0.4333 | 0.9371 | 0.9808 | 0.1174 | 0.9487 | 0.0285 | 0.0229 |
| 0.9267 | 0.0733 | ensured | no | 0.3374 | 0.8927 | 0.7270 | 0.9282 | 0.0016 | 0.6557 | 0.0069 | 0.3374 |
| 0.9267 | 0.0733 | ensured | yes | 0.0229 | 0.4333 | 0.9371 | 0.9808 | 0.0541 | 0.9649 | 0.0122 | 0.0229 |
| 0.9900 | 0.0100 | ensured | no | 0.1600 | 0.9609 | 0.8947 | 0.9891 | -0.0009 | 0.8400 | 0.0000 | 0.1600 |
| 0.9900 | 0.0100 | ensured | yes | 0.0100 | 0.5000 | 0.9841 | 0.9931 | 0.0031 | 0.9900 | 0.0000 | 0.0100 |

## Difficulty Effect By Confidence And NCF

This table compares `use_difficulty=yes` against `use_difficulty=no` at the same confidence and NCF.
Negative `coverage_gap_delta` means the difficulty-enabled arm covered fewer true labels at that operating point.

| confidence | epsilon | ncf | accept_rate_delta | accepted_accuracy_delta | rejected_error_capture_rate_delta | empirical_coverage_delta | coverage_gap_delta |
|---|---|---|---|---|---|---|---|
| 0.8000 | 0.2000 | default | -0.5045 | -0.4238 | 0.3090 | 0.1439 | 0.1439 |
| 0.8633 | 0.1367 | default | -0.4862 | -0.4336 | 0.3020 | 0.1176 | 0.1176 |
| 0.9267 | 0.0733 | default | -0.4269 | -0.4636 | 0.2557 | 0.0568 | 0.0568 |
| 0.9900 | 0.0100 | default | -0.2076 | -0.4566 | 0.1268 | 0.0070 | 0.0070 |
| 0.8000 | 0.2000 | ensured | -0.2864 | -0.4306 | 0.2198 | 0.1401 | 0.1401 |
| 0.8633 | 0.1367 | ensured | -0.3114 | -0.4421 | 0.2224 | 0.0984 | 0.0984 |
| 0.9267 | 0.0733 | ensured | -0.3145 | -0.4593 | 0.2102 | 0.0525 | 0.0525 |
| 0.9900 | 0.0100 | ensured | -0.1500 | -0.4609 | 0.0893 | 0.0041 | 0.0041 |

## Difficulty Effect By NCF

This table compares `use_difficulty=yes` against `use_difficulty=no` within each public reject NCF.
Negative `accept_rate_delta` means the difficulty-enabled arm rejects more aggressively.
Positive `rejected_error_capture_rate_delta` means the difficulty-enabled arm captures more mistakes in the rejected subset.

| ncf | accept_rate_delta | accepted_accuracy_delta | accuracy_delta_delta | empirical_coverage_delta | coverage_gap_delta | rejected_error_capture_rate_delta | singleton_error_rate_delta | mean_difficulty_rejected_delta |
|---|---|---|---|---|---|---|---|---|
| default | -0.4063 | -0.4441 | -0.4242 | 0.0813 | 0.0813 | 0.2484 | 0.7870 | 0.0660 |
| ensured | -0.2656 | -0.4481 | -0.4281 | 0.0738 | 0.0738 | 0.1854 | 0.7633 | -0.0444 |

## Integrity Audit

These checks flag impossible reject geometry and whether ambiguity could be coming from a fallback path without prediction sets.
All residuals should stay near zero; positive `positive_ambiguity_without_prediction_set_rows` would be suspicious.

| ncf | use_difficulty | rows | max_abs_reject_partition_residual | max_abs_accept_singleton_residual | max_abs_ambiguity_multilabel_residual | max_abs_novelty_empty_residual | equal_ambiguity_novelty_rows | equal_positive_ambiguity_novelty_rows | coverage_defined_rows | positive_ambiguity_without_prediction_set_rows | min_coverage_gap | max_coverage_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default | no | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 | 0 | 24 | 0 | -0.0717 | 0.0831 |
| default | yes | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 24 | 0 | -0.0095 | 0.2000 |
| ensured | no | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 24 | 0 | -0.1550 | 0.1825 |
| ensured | yes | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 24 | 0 | -0.0095 | 0.2000 |
