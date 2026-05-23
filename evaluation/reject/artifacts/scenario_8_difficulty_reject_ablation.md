# Scenario 8 — Difficulty estimator reject ablation

Rows: 8280

## Key findings

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

## Outcome snapshot

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
| default | no | 0.6332 | 0.8614 | 0.0551 | 0.8965 | 0.0015 | 0.5878 | 1.9148 | 1.9225 | 0.6332 | 0.0237 | 0.3431 |
| default | yes | 0.2132 | 0.7608 | -0.0405 | 0.9403 | 0.0453 | 0.7760 | 1.8627 | 1.9021 | 0.2132 | 0.0021 | 0.7848 |
| ensured | no | 0.3693 | 0.8344 | 0.0274 | 0.9130 | 0.0180 | 0.7806 | 1.9611 | 1.9223 | 0.3693 | 0.0440 | 0.5866 |
| ensured | yes | 0.1220 | 0.7293 | -0.0649 | 0.9406 | 0.0456 | 0.8568 | 1.8463 | 1.9221 | 0.1220 | 0.0236 | 0.8545 |

## By Confidence And Arm

This table keeps the reject operating point visible instead of averaging across the whole epsilon sweep.
`empirical_coverage` is computed from the returned prediction sets; `coverage_gap = empirical_coverage - confidence`.

| confidence | epsilon | ncf | use_difficulty | accept_rate | accepted_accuracy | rejected_error_capture_rate | empirical_coverage | coverage_gap | ambiguity_rate | novelty_rate | singleton_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8000 | 0.2000 | default | no | 0.8018 | 0.8349 | 0.4219 | 0.8031 | 0.0031 | 0.1341 | 0.0641 | 0.8018 |
| 0.8000 | 0.2000 | default | yes | 0.3759 | 0.7614 | 0.6132 | 0.8837 | 0.0837 | 0.6159 | 0.0082 | 0.3759 |
| 0.8237 | 0.1763 | default | no | 0.7871 | 0.8323 | 0.4279 | 0.8206 | -0.0031 | 0.1625 | 0.0504 | 0.7871 |
| 0.8237 | 0.1763 | default | yes | 0.3385 | 0.7617 | 0.6408 | 0.9003 | 0.0765 | 0.6558 | 0.0056 | 0.3385 |
| 0.8475 | 0.1525 | default | no | 0.7688 | 0.8349 | 0.4358 | 0.8431 | -0.0044 | 0.1936 | 0.0375 | 0.7688 |
| 0.8475 | 0.1525 | default | yes | 0.2980 | 0.7567 | 0.6785 | 0.9127 | 0.0652 | 0.6987 | 0.0033 | 0.2980 |
| 0.8713 | 0.1287 | default | no | 0.7284 | 0.8413 | 0.4957 | 0.8714 | 0.0001 | 0.2449 | 0.0267 | 0.7284 |
| 0.8713 | 0.1287 | default | yes | 0.2486 | 0.7558 | 0.7261 | 0.9280 | 0.0567 | 0.7503 | 0.0011 | 0.2486 |
| 0.8950 | 0.1050 | default | no | 0.6977 | 0.8473 | 0.5276 | 0.8949 | -0.0001 | 0.2859 | 0.0164 | 0.6977 |
| 0.8950 | 0.1050 | default | yes | 0.2175 | 0.7602 | 0.7695 | 0.9427 | 0.0477 | 0.7823 | 0.0003 | 0.2175 |
| 0.9187 | 0.0813 | default | no | 0.6422 | 0.8652 | 0.6000 | 0.9236 | 0.0049 | 0.3477 | 0.0101 | 0.6422 |
| 0.9187 | 0.0813 | default | yes | 0.1857 | 0.7703 | 0.8093 | 0.9509 | 0.0321 | 0.8142 | 0.0001 | 0.1857 |
| 0.9425 | 0.0575 | default | no | 0.5776 | 0.8758 | 0.6634 | 0.9415 | -0.0010 | 0.4165 | 0.0059 | 0.5776 |
| 0.9425 | 0.0575 | default | yes | 0.1423 | 0.7663 | 0.8565 | 0.9647 | 0.0222 | 0.8577 | 0.0000 | 0.1423 |
| 0.9663 | 0.0337 | default | no | 0.4812 | 0.9130 | 0.7638 | 0.9729 | 0.0066 | 0.5166 | 0.0022 | 0.4812 |
| 0.9663 | 0.0337 | default | yes | 0.0946 | 0.7789 | 0.9104 | 0.9820 | 0.0157 | 0.9054 | 0.0000 | 0.0946 |
| 0.9900 | 0.0100 | default | no | 0.2141 | 0.9681 | 0.9538 | 0.9971 | 0.0071 | 0.7859 | 0.0000 | 0.2141 |
| 0.9900 | 0.0100 | default | yes | 0.0173 | 0.6974 | 0.9800 | 0.9977 | 0.0077 | 0.9827 | 0.0000 | 0.0173 |
| 0.8000 | 0.2000 | ensured | no | 0.4269 | 0.8165 | 0.6949 | 0.8274 | 0.0274 | 0.4663 | 0.1067 | 0.4269 |
| 0.8000 | 0.2000 | ensured | yes | 0.2411 | 0.7612 | 0.7519 | 0.8918 | 0.0918 | 0.7090 | 0.0499 | 0.2411 |
| 0.8237 | 0.1763 | ensured | no | 0.4185 | 0.8226 | 0.7116 | 0.8506 | 0.0269 | 0.4934 | 0.0881 | 0.4185 |
| 0.8237 | 0.1763 | ensured | yes | 0.2083 | 0.7502 | 0.7679 | 0.9028 | 0.0791 | 0.7493 | 0.0424 | 0.2083 |
| 0.8475 | 0.1525 | ensured | no | 0.4078 | 0.8246 | 0.7277 | 0.8731 | 0.0256 | 0.5220 | 0.0701 | 0.4078 |
| 0.8475 | 0.1525 | ensured | yes | 0.1572 | 0.7387 | 0.8132 | 0.9142 | 0.0667 | 0.8046 | 0.0382 | 0.1572 |
| 0.8713 | 0.1287 | ensured | no | 0.3945 | 0.8203 | 0.7513 | 0.8939 | 0.0226 | 0.5506 | 0.0549 | 0.3945 |
| 0.8713 | 0.1287 | ensured | yes | 0.1361 | 0.7410 | 0.8333 | 0.9271 | 0.0558 | 0.8334 | 0.0305 | 0.1361 |
| 0.8950 | 0.1050 | ensured | no | 0.3991 | 0.8282 | 0.7564 | 0.9125 | 0.0175 | 0.5622 | 0.0387 | 0.3991 |
| 0.8950 | 0.1050 | ensured | yes | 0.1214 | 0.7503 | 0.8503 | 0.9366 | 0.0416 | 0.8559 | 0.0227 | 0.1214 |
| 0.9187 | 0.0813 | ensured | no | 0.3920 | 0.8284 | 0.7721 | 0.9348 | 0.0161 | 0.5855 | 0.0225 | 0.3920 |
| 0.9187 | 0.0813 | ensured | yes | 0.0947 | 0.7300 | 0.8810 | 0.9514 | 0.0326 | 0.8903 | 0.0150 | 0.0947 |
| 0.9425 | 0.0575 | ensured | no | 0.3746 | 0.8364 | 0.7983 | 0.9534 | 0.0109 | 0.6145 | 0.0109 | 0.3746 |
| 0.9425 | 0.0575 | ensured | yes | 0.0720 | 0.6878 | 0.9083 | 0.9645 | 0.0220 | 0.9184 | 0.0095 | 0.0720 |
| 0.9663 | 0.0337 | ensured | no | 0.3338 | 0.8545 | 0.8493 | 0.9746 | 0.0084 | 0.6620 | 0.0042 | 0.3338 |
| 0.9663 | 0.0337 | ensured | yes | 0.0563 | 0.6734 | 0.9215 | 0.9791 | 0.0128 | 0.9397 | 0.0040 | 0.0563 |
| 0.9900 | 0.0100 | ensured | no | 0.1768 | 0.9284 | 0.9642 | 0.9971 | 0.0071 | 0.8231 | 0.0001 | 0.1768 |
| 0.9900 | 0.0100 | ensured | yes | 0.0104 | 0.6727 | 0.9836 | 0.9979 | 0.0079 | 0.9895 | 0.0001 | 0.0104 |

## Difficulty Effect By Confidence And NCF

This table compares `use_difficulty=yes` against `use_difficulty=no` at the same confidence and NCF.
Negative `coverage_gap_delta` means the difficulty-enabled arm covered fewer true labels at that operating point.

| confidence | epsilon | ncf | accept_rate_delta | accepted_accuracy_delta | rejected_error_capture_rate_delta | empirical_coverage_delta | coverage_gap_delta |
|---|---|---|---|---|---|---|---|
| 0.8000 | 0.2000 | default | -0.4259 | -0.0736 | 0.1914 | 0.0806 | 0.0806 |
| 0.8237 | 0.1763 | default | -0.4486 | -0.0706 | 0.2130 | 0.0796 | 0.0796 |
| 0.8475 | 0.1525 | default | -0.4708 | -0.0782 | 0.2427 | 0.0696 | 0.0696 |
| 0.8713 | 0.1287 | default | -0.4799 | -0.0855 | 0.2304 | 0.0566 | 0.0566 |
| 0.8950 | 0.1050 | default | -0.4802 | -0.0871 | 0.2419 | 0.0478 | 0.0478 |
| 0.9187 | 0.0813 | default | -0.4564 | -0.0949 | 0.2092 | 0.0273 | 0.0273 |
| 0.9425 | 0.0575 | default | -0.4353 | -0.1096 | 0.1931 | 0.0232 | 0.0232 |
| 0.9663 | 0.0337 | default | -0.3866 | -0.1341 | 0.1465 | 0.0091 | 0.0091 |
| 0.9900 | 0.0100 | default | -0.1968 | -0.2707 | 0.0262 | 0.0007 | 0.0007 |
| 0.8000 | 0.2000 | ensured | -0.1859 | -0.0554 | 0.0570 | 0.0643 | 0.0643 |
| 0.8237 | 0.1763 | ensured | -0.2102 | -0.0724 | 0.0563 | 0.0522 | 0.0522 |
| 0.8475 | 0.1525 | ensured | -0.2506 | -0.0859 | 0.0855 | 0.0411 | 0.0411 |
| 0.8713 | 0.1287 | ensured | -0.2584 | -0.0794 | 0.0821 | 0.0332 | 0.0332 |
| 0.8950 | 0.1050 | ensured | -0.2777 | -0.0779 | 0.0939 | 0.0241 | 0.0241 |
| 0.9187 | 0.0813 | ensured | -0.2973 | -0.0984 | 0.1089 | 0.0166 | 0.0166 |
| 0.9425 | 0.0575 | ensured | -0.3026 | -0.1486 | 0.1100 | 0.0111 | 0.0111 |
| 0.9663 | 0.0337 | ensured | -0.2775 | -0.1811 | 0.0722 | 0.0044 | 0.0044 |
| 0.9900 | 0.0100 | ensured | -0.1664 | -0.2558 | 0.0194 | 0.0009 | 0.0009 |

## Difficulty Effect By NCF

This table compares `use_difficulty=yes` against `use_difficulty=no` within each public reject NCF.
Negative `accept_rate_delta` means the difficulty-enabled arm rejects more aggressively.
Positive `rejected_error_capture_rate_delta` means the difficulty-enabled arm captures more mistakes in the rejected subset.

| ncf | accept_rate_delta | accepted_accuracy_delta | accuracy_delta_delta | empirical_coverage_delta | coverage_gap_delta | rejected_error_capture_rate_delta | singleton_error_rate_delta | mean_difficulty_rejected_delta |
|---|---|---|---|---|---|---|---|---|
| default | -0.4200 | -0.1006 | -0.0956 | 0.0438 | 0.0438 | 0.1883 | 0.4576 | -0.0203 |
| ensured | -0.2474 | -0.1051 | -0.0923 | 0.0275 | 0.0275 | 0.0762 | 0.3665 | -0.0003 |

## Integrity Audit

These checks flag impossible reject geometry and whether ambiguity could be coming from a fallback path without prediction sets.
All residuals should stay near zero; positive `positive_ambiguity_without_prediction_set_rows` would be suspicious.

| ncf | use_difficulty | rows | max_abs_reject_partition_residual | max_abs_accept_singleton_residual | max_abs_ambiguity_multilabel_residual | max_abs_novelty_empty_residual | equal_ambiguity_novelty_rows | equal_positive_ambiguity_novelty_rows | coverage_defined_rows | positive_ambiguity_without_prediction_set_rows | min_coverage_gap | max_coverage_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default | no | 2070 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 72 | 0 | 1170 | 0 | -0.1936 | 0.1524 |
| default | yes | 2070 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 41 | 0 | 1170 | 0 | -0.2179 | 0.2000 |
| ensured | no | 2070 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5 | 4 | 1170 | 0 | -0.1579 | 0.1912 |
| ensured | yes | 2070 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 19 | 0 | 1170 | 0 | -0.1357 | 0.2000 |
