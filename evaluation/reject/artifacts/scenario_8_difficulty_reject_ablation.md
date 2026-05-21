# Scenario 8 — Difficulty estimator reject ablation

Rows: 8280

## Key findings

- Measures the current indirect difficulty effect through Venn-Abers scaling only; reject scoring itself is unchanged.
- Arms compare use_difficulty in {False, True} crossed with reject NCF in {default, ensured}.
- Difficulty summary columns use the same deterministic reference estimator in all arms so selection differences are comparable.
- This scenario does not test difficulty-normalized reject NCFs; it quantifies the baseline before that experiment.
- With `default`, enabling difficulty changed accept_rate by -22.2 pp, rejected_error_capture_rate by +11.5 pp, and accepted_accuracy by -10.9 pp.
- With `default`, mean empirical coverage shifted by +4.4 pp across the swept confidence grid.
- With `default` and difficulty enabled, rejected instances were harder than accepted ones by 0.093 mean difficulty units.
- For `default`, the current difficulty path acts mainly as a stricter reject gate: it captures more errors, but at the cost of accepting far fewer instances and lowering accepted accuracy.
- With `ensured`, enabling difficulty changed accept_rate by -9.7 pp, rejected_error_capture_rate by +4.9 pp, and accepted_accuracy by -12.2 pp.
- With `ensured`, mean empirical coverage shifted by +2.8 pp across the swept confidence grid.
- With `ensured` and difficulty enabled, rejected instances were harder than accepted ones by 0.173 mean difficulty units.
- For `ensured`, the current difficulty path acts mainly as a stricter reject gate: it captures more errors, but at the cost of accepting far fewer instances and lowering accepted accuracy.
- The markdown now includes a by-confidence table so the headline summary is no longer averaged over hidden epsilon values.
- Integrity checks verify reject_rate = ambiguity_rate + novelty_rate, accepted instances match singleton prediction sets, and no positive ambiguity appears without prediction sets.
- Empirical coverage is reported only for rows whose prediction-set columns are label-index aligned; unsupported rows stay `nan` instead of inventing a value.

## Outcome snapshot

- **rows**: 8280
- **datasets**: 46
- **seeds**: 5
- **mean_accept_rate**: 0.1736
- **mean_accuracy_delta**: -0.0321
- **default_accept_rate_no_difficulty**: 0.3612
- **default_accept_rate_with_difficulty**: 0.1387
- **default_accept_rate_delta**: -0.2224
- **default_accepted_accuracy_delta**: -0.1086
- **default_accuracy_delta_delta**: -0.1082
- **default_rejected_error_capture_rate_delta**: 0.1145
- **default_singleton_error_rate_delta**: 0.4451
- **default_difficulty_gap_with_difficulty**: 0.0925
- **default_empirical_coverage_no_difficulty**: 0.8965
- **default_empirical_coverage_with_difficulty**: 0.9403
- **default_coverage_gap_delta**: 0.0438
- **ensured_accept_rate_no_difficulty**: 0.1459
- **ensured_accept_rate_with_difficulty**: 0.0488
- **ensured_accept_rate_delta**: -0.0971
- **ensured_accepted_accuracy_delta**: -0.1223
- **ensured_accuracy_delta_delta**: -0.1183
- **ensured_rejected_error_capture_rate_delta**: 0.0488
- **ensured_singleton_error_rate_delta**: 0.3579
- **ensured_difficulty_gap_with_difficulty**: 0.1729
- **ensured_empirical_coverage_no_difficulty**: 0.9130
- **ensured_empirical_coverage_with_difficulty**: 0.9406
- **ensured_coverage_gap_delta**: 0.0275
- **mean_difficulty_gap_with_difficulty**: 0.1118
- **unique_confidences**: 9
- **min_epsilon**: 0.0100
- **max_epsilon**: 0.2000
- **max_abs_reject_partition_residual**: 0.0000
- **max_abs_accept_singleton_residual**: 0.0000
- **positive_ambiguity_without_prediction_set_rows**: 0
- **equal_positive_ambiguity_novelty_rows**: 0
- **coverage_defined_rows**: 4680
- **min_empirical_coverage_gap**: -0.2179
- **max_empirical_coverage_gap**: 0.2000

## Result table

| task_type | dataset | seed | confidence | epsilon | n_train | n_cal | n_test | ncf | use_difficulty | arm | accept_rate | reject_rate | ambiguity_rate | novelty_rate | accepted_accuracy | full_accuracy | accuracy_delta | singleton_error_rate | error_rate_defined | rejected_error_capture_rate | mean_difficulty_all | mean_difficulty_accepted | mean_difficulty_rejected | empty_rate | singleton_rate | multilabel_rate | empirical_coverage | coverage_gap | coverage_defined | has_prediction_set | reject_partition_residual | accept_singleton_residual | ambiguity_multilabel_residual | novelty_empty_residual | ambiguity_equals_novelty | ambiguity_equals_novelty_positive | positive_ambiguity_without_prediction_set |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| binary | breast_cancer | 42 | 0.8000 | 0.2000 | 341 | 114 | 114 | default | no | default|difficulty=0 | 0.8333 | 0.1667 | 0.0000 | 0.1667 | 1.0000 | 0.9561 | 0.0439 | 0.0400 | yes | 1.0000 | 1.8789 | 1.9130 | 1.7080 | 0.1667 | 0.8333 | 0.0000 | 0.8333 | 0.0333 | yes | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no | no | no |
| binary | breast_cancer | 42 | 0.8237 | 0.1763 | 341 | 114 | 114 | default | no | default|difficulty=0 | 0.8684 | 0.1316 | 0.0000 | 0.1316 | 0.9899 | 0.9561 | 0.0338 | 0.0514 | yes | 0.8000 | 1.8789 | 1.9139 | 1.6473 | 0.1316 | 0.8684 | 0.0000 | 0.8596 | 0.0359 | yes | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no | no | no |
| binary | breast_cancer | 42 | 0.8475 | 0.1525 | 341 | 114 | 114 | default | no | default|difficulty=0 | 0.9035 | 0.0965 | 0.0000 | 0.0965 | 0.9903 | 0.9561 | 0.0342 | 0.0620 | yes | 0.8000 | 1.8789 | 1.9035 | 1.6480 | 0.0965 | 0.9035 | 0.0000 | 0.8947 | 0.0472 | yes | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no | no | no |
| binary | breast_cancer | 42 | 0.8713 | 0.1287 | 341 | 114 | 114 | default | no | default|difficulty=0 | 0.9211 | 0.0789 | 0.0000 | 0.0789 | 0.9810 | 0.9561 | 0.0248 | 0.0541 | yes | 0.6000 | 1.8789 | 1.8990 | 1.6439 | 0.0789 | 0.9211 | 0.0000 | 0.9035 | 0.0323 | yes | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no | no | no |
| binary | breast_cancer | 42 | 0.8950 | 0.1050 | 341 | 114 | 114 | default | no | default|difficulty=0 | 0.9298 | 0.0702 | 0.0000 | 0.0702 | 0.9811 | 0.9561 | 0.0250 | 0.0375 | yes | 0.6000 | 1.8789 | 1.8996 | 1.6038 | 0.0702 | 0.9298 | 0.0000 | 0.9123 | 0.0173 | yes | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no | no | no |
| binary | breast_cancer | 42 | 0.9187 | 0.0813 | 341 | 114 | 114 | default | no | default|difficulty=0 | 0.9474 | 0.0526 | 0.0000 | 0.0526 | 0.9722 | 0.9561 | 0.0161 | 0.0302 | yes | 0.4000 | 1.8789 | 1.8934 | 1.6172 | 0.0526 | 0.9474 | 0.0000 | 0.9211 | 0.0023 | yes | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no | no | no |
| binary | breast_cancer | 42 | 0.9425 | 0.0575 | 341 | 114 | 114 | default | no | default|difficulty=0 | 0.9474 | 0.0526 | 0.0000 | 0.0526 | 0.9722 | 0.9561 | 0.0161 | 0.0051 | yes | 0.4000 | 1.8789 | 1.8934 | 1.6172 | 0.0526 | 0.9474 | 0.0000 | 0.9211 | -0.0214 | yes | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no | no | no |
| binary | breast_cancer | 42 | 0.9663 | 0.0337 | 341 | 114 | 114 | default | no | default|difficulty=0 | 0.9912 | 0.0088 | 0.0000 | 0.0088 | 0.9646 | 0.9561 | 0.0085 | 0.0252 | yes | 0.2000 | 1.8789 | 1.8820 | 1.5238 | 0.0088 | 0.9912 | 0.0000 | 0.9561 | -0.0101 | yes | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no | no | no |
| binary | breast_cancer | 42 | 0.9900 | 0.0100 | 341 | 114 | 114 | default | no | default|difficulty=0 | 0.9474 | 0.0526 | 0.0526 | 0.0000 | 0.9722 | 0.9561 | 0.0161 | 0.0106 | yes | 0.4000 | 1.8789 | 1.8934 | 1.6172 | 0.0000 | 0.9474 | 0.0526 | 0.9737 | -0.0163 | yes | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no | no | no |
| binary | breast_cancer | 42 | 0.8000 | 0.2000 | 341 | 114 | 114 | ensured | no | ensured|difficulty=0 | 0.6316 | 0.3684 | 0.3684 | 0.0000 | 0.9722 | 0.9561 | 0.0161 | 0.3167 | yes | 0.6000 | 1.8789 | 1.7423 | 2.1129 | 0.0000 | 0.6316 | 0.3684 | 0.9825 | 0.1825 | yes | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no | no | no |
| binary | breast_cancer | 42 | 0.8237 | 0.1763 | 341 | 114 | 114 | ensured | no | ensured|difficulty=0 | 0.6316 | 0.3684 | 0.3684 | 0.0000 | 0.9722 | 0.9561 | 0.0161 | 0.2791 | yes | 0.6000 | 1.8789 | 1.7423 | 2.1129 | 0.0000 | 0.6316 | 0.3684 | 0.9825 | 0.1587 | yes | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no | no | no |
| binary | breast_cancer | 42 | 0.8475 | 0.1525 | 341 | 114 | 114 | ensured | no | ensured|difficulty=0 | 0.6316 | 0.3684 | 0.3684 | 0.0000 | 0.9722 | 0.9561 | 0.0161 | 0.2415 | yes | 0.6000 | 1.8789 | 1.7423 | 2.1129 | 0.0000 | 0.6316 | 0.3684 | 0.9825 | 0.1350 | yes | yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no | no | no |

_Showing first 12 of 8280 rows._

## Arm Summary

This table averages each arm across all datasets, seeds, and confidence levels.

| ncf | use_difficulty | accept_rate | accepted_accuracy | accuracy_delta | empirical_coverage | coverage_gap | rejected_error_capture_rate | mean_difficulty_accepted | mean_difficulty_rejected | singleton_rate | empty_rate | multilabel_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default | no | 0.3612 | 0.8653 | 0.0578 | 0.8965 | 0.0015 | 0.7464 | 1.8934 | 1.9243 | 0.3612 | 0.0538 | 0.5850 |
| default | yes | 0.1387 | 0.7567 | -0.0505 | 0.9403 | 0.0453 | 0.8610 | 1.8140 | 1.9065 | 0.1387 | 0.0224 | 0.8388 |
| ensured | no | 0.1459 | 0.7828 | -0.0263 | 0.9130 | 0.0180 | 0.8653 | 1.9815 | 1.9235 | 0.1459 | 0.0260 | 0.8281 |
| ensured | yes | 0.0488 | 0.6605 | -0.1447 | 0.9406 | 0.0456 | 0.9141 | 1.7542 | 1.9271 | 0.0488 | 0.0293 | 0.9219 |

## By Confidence And Arm

This table keeps the reject operating point visible instead of averaging across the whole epsilon sweep.
`empirical_coverage` is computed from the returned prediction sets; `coverage_gap = empirical_coverage - confidence`.

| confidence | epsilon | ncf | use_difficulty | accept_rate | accepted_accuracy | rejected_error_capture_rate | empirical_coverage | coverage_gap | ambiguity_rate | novelty_rate | singleton_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8000 | 0.2000 | default | no | 0.4775 | 0.8361 | 0.6242 | 0.8031 | 0.0031 | 0.4051 | 0.1175 | 0.4775 |
| 0.8000 | 0.2000 | default | yes | 0.2222 | 0.7566 | 0.7819 | 0.8837 | 0.0837 | 0.7255 | 0.0523 | 0.2222 |
| 0.8237 | 0.1763 | default | no | 0.4686 | 0.8374 | 0.6354 | 0.8206 | -0.0031 | 0.4327 | 0.0987 | 0.4686 |
| 0.8237 | 0.1763 | default | yes | 0.2072 | 0.7554 | 0.7881 | 0.9003 | 0.0765 | 0.7497 | 0.0431 | 0.2072 |
| 0.8475 | 0.1525 | default | no | 0.4563 | 0.8413 | 0.6450 | 0.8431 | -0.0044 | 0.4638 | 0.0799 | 0.4563 |
| 0.8475 | 0.1525 | default | yes | 0.1909 | 0.7507 | 0.7982 | 0.9127 | 0.0652 | 0.7710 | 0.0381 | 0.1909 |
| 0.8713 | 0.1287 | default | no | 0.4285 | 0.8452 | 0.6799 | 0.8714 | 0.0001 | 0.5074 | 0.0642 | 0.4285 |
| 0.8713 | 0.1287 | default | yes | 0.1706 | 0.7470 | 0.8168 | 0.9280 | 0.0567 | 0.8056 | 0.0239 | 0.1706 |
| 0.8950 | 0.1050 | default | no | 0.4079 | 0.8523 | 0.7048 | 0.8949 | -0.0001 | 0.5436 | 0.0485 | 0.4079 |
| 0.8950 | 0.1050 | default | yes | 0.1495 | 0.7545 | 0.8507 | 0.9427 | 0.0477 | 0.8322 | 0.0183 | 0.1495 |
| 0.9187 | 0.0813 | default | no | 0.3666 | 0.8686 | 0.7599 | 0.9236 | 0.0049 | 0.5980 | 0.0354 | 0.3666 |
| 0.9187 | 0.0813 | default | yes | 0.1331 | 0.7708 | 0.8724 | 0.9509 | 0.0321 | 0.8533 | 0.0137 | 0.1331 |
| 0.9425 | 0.0575 | default | no | 0.3232 | 0.8834 | 0.8053 | 0.9415 | -0.0010 | 0.6533 | 0.0236 | 0.3232 |
| 0.9425 | 0.0575 | default | yes | 0.1001 | 0.7593 | 0.9065 | 0.9647 | 0.0222 | 0.8930 | 0.0070 | 0.1001 |
| 0.9663 | 0.0337 | default | no | 0.2474 | 0.9311 | 0.8849 | 0.9729 | 0.0066 | 0.7387 | 0.0139 | 0.2474 |
| 0.9663 | 0.0337 | default | yes | 0.0645 | 0.7898 | 0.9474 | 0.9820 | 0.0157 | 0.9309 | 0.0046 | 0.0645 |
| 0.9900 | 0.0100 | default | no | 0.0747 | 0.9621 | 0.9785 | 0.9971 | 0.0071 | 0.9228 | 0.0025 | 0.0747 |
| 0.9900 | 0.0100 | default | yes | 0.0108 | 0.6679 | 0.9869 | 0.9977 | 0.0077 | 0.9885 | 0.0007 | 0.0108 |
| 0.8000 | 0.2000 | ensured | no | 0.2000 | 0.7703 | 0.7978 | 0.8274 | 0.0274 | 0.7386 | 0.0614 | 0.2000 |
| 0.8000 | 0.2000 | ensured | yes | 0.1083 | 0.7096 | 0.8597 | 0.8918 | 0.0918 | 0.8262 | 0.0655 | 0.1083 |
| 0.8237 | 0.1763 | ensured | no | 0.1872 | 0.7737 | 0.8130 | 0.8506 | 0.0269 | 0.7621 | 0.0507 | 0.1872 |
| 0.8237 | 0.1763 | ensured | yes | 0.0863 | 0.6868 | 0.8670 | 0.9028 | 0.0791 | 0.8581 | 0.0556 | 0.0863 |
| 0.8475 | 0.1525 | ensured | no | 0.1741 | 0.7787 | 0.8265 | 0.8731 | 0.0256 | 0.7855 | 0.0403 | 0.1741 |
| 0.8475 | 0.1525 | ensured | yes | 0.0645 | 0.6764 | 0.8928 | 0.9142 | 0.0667 | 0.8852 | 0.0503 | 0.0645 |
| 0.8713 | 0.1287 | ensured | no | 0.1584 | 0.7678 | 0.8473 | 0.8939 | 0.0226 | 0.8084 | 0.0332 | 0.1584 |
| 0.8713 | 0.1287 | ensured | yes | 0.0511 | 0.6768 | 0.9029 | 0.9271 | 0.0558 | 0.9131 | 0.0357 | 0.0511 |
| 0.8950 | 0.1050 | ensured | no | 0.1593 | 0.7817 | 0.8517 | 0.9125 | 0.0175 | 0.8174 | 0.0233 | 0.1593 |
| 0.8950 | 0.1050 | ensured | yes | 0.0439 | 0.6836 | 0.9126 | 0.9366 | 0.0416 | 0.9300 | 0.0261 | 0.0439 |
| 0.9187 | 0.0813 | ensured | no | 0.1493 | 0.7779 | 0.8659 | 0.9348 | 0.0161 | 0.8363 | 0.0144 | 0.1493 |
| 0.9187 | 0.0813 | ensured | yes | 0.0353 | 0.6668 | 0.9254 | 0.9514 | 0.0326 | 0.9472 | 0.0175 | 0.0353 |
| 0.9425 | 0.0575 | ensured | no | 0.1386 | 0.7843 | 0.8829 | 0.9534 | 0.0109 | 0.8539 | 0.0074 | 0.1386 |
| 0.9425 | 0.0575 | ensured | yes | 0.0271 | 0.6076 | 0.9381 | 0.9645 | 0.0220 | 0.9641 | 0.0088 | 0.0271 |
| 0.9663 | 0.0337 | ensured | no | 0.1095 | 0.8054 | 0.9211 | 0.9746 | 0.0084 | 0.8874 | 0.0031 | 0.1095 |
| 0.9663 | 0.0337 | ensured | yes | 0.0198 | 0.5942 | 0.9423 | 0.9791 | 0.0128 | 0.9764 | 0.0039 | 0.0198 |
| 0.9900 | 0.0100 | ensured | no | 0.0366 | 0.8565 | 0.9816 | 0.9971 | 0.0071 | 0.9634 | 0.0000 | 0.0366 |
| 0.9900 | 0.0100 | ensured | yes | 0.0029 | 0.5657 | 0.9864 | 0.9979 | 0.0079 | 0.9970 | 0.0001 | 0.0029 |

## Difficulty Effect By Confidence And NCF

This table compares `use_difficulty=yes` against `use_difficulty=no` at the same confidence and NCF.
Negative `coverage_gap_delta` means the difficulty-enabled arm covered fewer true labels at that operating point.

| confidence | epsilon | ncf | accept_rate_delta | accepted_accuracy_delta | rejected_error_capture_rate_delta | empirical_coverage_delta | coverage_gap_delta |
|---|---|---|---|---|---|---|---|
| 0.8000 | 0.2000 | default | -0.2553 | -0.0795 | 0.1577 | 0.0806 | 0.0806 |
| 0.8237 | 0.1763 | default | -0.2614 | -0.0820 | 0.1526 | 0.0796 | 0.0796 |
| 0.8475 | 0.1525 | default | -0.2654 | -0.0906 | 0.1532 | 0.0696 | 0.0696 |
| 0.8713 | 0.1287 | default | -0.2579 | -0.0982 | 0.1370 | 0.0566 | 0.0566 |
| 0.8950 | 0.1050 | default | -0.2584 | -0.0978 | 0.1458 | 0.0478 | 0.0478 |
| 0.9187 | 0.0813 | default | -0.2335 | -0.0978 | 0.1124 | 0.0273 | 0.0273 |
| 0.9425 | 0.0575 | default | -0.2231 | -0.1241 | 0.1012 | 0.0232 | 0.0232 |
| 0.9663 | 0.0337 | default | -0.1829 | -0.1412 | 0.0625 | 0.0091 | 0.0091 |
| 0.9900 | 0.0100 | default | -0.0639 | -0.2943 | 0.0084 | 0.0007 | 0.0007 |
| 0.8000 | 0.2000 | ensured | -0.0917 | -0.0607 | 0.0619 | 0.0643 | 0.0643 |
| 0.8237 | 0.1763 | ensured | -0.1009 | -0.0870 | 0.0540 | 0.0522 | 0.0522 |
| 0.8475 | 0.1525 | ensured | -0.1096 | -0.1022 | 0.0664 | 0.0411 | 0.0411 |
| 0.8713 | 0.1287 | ensured | -0.1073 | -0.0909 | 0.0556 | 0.0332 | 0.0332 |
| 0.8950 | 0.1050 | ensured | -0.1154 | -0.0981 | 0.0608 | 0.0241 | 0.0241 |
| 0.9187 | 0.0813 | ensured | -0.1140 | -0.1112 | 0.0594 | 0.0166 | 0.0166 |
| 0.9425 | 0.0575 | ensured | -0.1116 | -0.1767 | 0.0552 | 0.0111 | 0.0111 |
| 0.9663 | 0.0337 | ensured | -0.0897 | -0.2112 | 0.0212 | 0.0044 | 0.0044 |
| 0.9900 | 0.0100 | ensured | -0.0337 | -0.2907 | 0.0048 | 0.0009 | 0.0009 |

## Difficulty Effect By NCF

This table compares `use_difficulty=yes` against `use_difficulty=no` within each public reject NCF.
Negative `accept_rate_delta` means the difficulty-enabled arm rejects more aggressively.
Positive `rejected_error_capture_rate_delta` means the difficulty-enabled arm captures more mistakes in the rejected subset.

| ncf | accept_rate_delta | accepted_accuracy_delta | accuracy_delta_delta | empirical_coverage_delta | coverage_gap_delta | rejected_error_capture_rate_delta | singleton_error_rate_delta | mean_difficulty_rejected_delta |
|---|---|---|---|---|---|---|---|---|
| default | -0.2224 | -0.1086 | -0.1082 | 0.0438 | 0.0438 | 0.1145 | 0.4451 | -0.0178 |
| ensured | -0.0971 | -0.1223 | -0.1183 | 0.0275 | 0.0275 | 0.0488 | 0.3579 | 0.0036 |

## Integrity Audit

These checks flag impossible reject geometry and whether ambiguity could be coming from a fallback path without prediction sets.
All residuals should stay near zero; positive `positive_ambiguity_without_prediction_set_rows` would be suspicious.

| ncf | use_difficulty | rows | max_abs_reject_partition_residual | max_abs_accept_singleton_residual | max_abs_ambiguity_multilabel_residual | max_abs_novelty_empty_residual | equal_ambiguity_novelty_rows | equal_positive_ambiguity_novelty_rows | coverage_defined_rows | positive_ambiguity_without_prediction_set_rows | min_coverage_gap | max_coverage_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default | no | 2070 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 54 | 0 | 1170 | 0 | -0.1936 | 0.1524 |
| default | yes | 2070 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 20 | 0 | 1170 | 0 | -0.2179 | 0.2000 |
| ensured | no | 2070 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 | 0 | 1170 | 0 | -0.1579 | 0.1912 |
| ensured | yes | 2070 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 1170 | 0 | -0.1357 | 0.2000 |
