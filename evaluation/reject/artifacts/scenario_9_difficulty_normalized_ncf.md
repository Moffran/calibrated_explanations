# Scenario 9 - Difficulty-normalized reject NCF strategy ablation

Rows: 12420

## Key findings

- Compares indirect VA-difficulty support against direct experimental difficulty-normalized reject scoring.
- Primary scientific contrast is A vs C (default NCF, no VA difficulty in either arm).
- Arms D and F are diagnostic for potential difficulty double-counting when VA and score normalization are both enabled.
- Includes strategy metadata and difficulty_reject_auc for reject-selectivity diagnostics.
- Includes accepted-accuracy comparison at matched reject-rate bins for A vs C.
- Direct normalization (C vs A) changed reject_rate by +0.0313, difficulty-gap by +0.2950, and difficulty_reject_auc by +0.1985.
- At matched reject-rate bins, C minus A mean accepted_accuracy is -0.0053.
- For C vs A, ambiguity_rate changed by +0.0217 and novelty_rate by +0.0096.
- Double-count diagnostics: D-B reject_rate delta -0.1473, F-E reject_rate delta +0.1093; difficulty-gap deltas are +0.3909 and +0.2846.
- Recommended arm for next iteration: C (primary A-vs-C contrast with direct normalization and no VA double-count risk).

## Outcome snapshot

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
- **metric_consistency_note**: {'full_grid_auc_delta': 0.19847119235003774, 'hi_conf_auc_delta': 0.24921106666866505, 'lo_conf_auc_delta': 0.16366233970904076, 'scenario_11_matched_delta': 0.0155, 'note': 'Full-grid positive delta is strongest in high-confidence rows. Scenario 11 matched operating-point selection reduces the observed difficulty-AUC effect and remains the promotion decision gate.'}

## Result table

| task_type | dataset | seed | confidence | epsilon | n_train | n_cal | n_test | arm_code | arm_label | estimator_type | ncf | strategy | use_va_difficulty | difficulty_normalized | double_count_difficulty | accept_rate | reject_rate | ambiguity_rate | novelty_rate | accepted_accuracy | full_accuracy | accuracy_delta | singleton_error_rate | singleton_precision | singleton_recall | singleton_correct_count | singleton_count | singleton_precision_recall_defined | error_rate_defined | rejected_error_capture_rate | mean_difficulty_all | mean_difficulty_accepted | mean_difficulty_rejected | difficulty_gap_rejected_minus_accepted | difficulty_reject_auc | empty_rate | singleton_rate | multilabel_rate | empirical_coverage | coverage_gap | coverage_defined |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| binary | breast_cancer | 42 | 0.8000 | 0.2000 | 341 | 114 | 114 | A | A|va=0|strategy=builtin.default|ncf=default | deterministic | default | builtin.default | no | no | no | 0.8333 | 0.1667 | 0.0000 | 0.1667 | 1.0000 | 0.9561 | 0.0439 | 0.0400 | 1.0000 | 0.8333 | 95 | 95 | yes | yes | 1.0000 | 1.8789 | 1.9130 | 1.7080 | -0.2050 | 0.3501 | 0.1667 | 0.8333 | 0.0000 | 0.8333 | 0.0333 | yes |
| binary | breast_cancer | 42 | 0.8237 | 0.1763 | 341 | 114 | 114 | A | A|va=0|strategy=builtin.default|ncf=default | deterministic | default | builtin.default | no | no | no | 0.8684 | 0.1316 | 0.0000 | 0.1316 | 0.9899 | 0.9561 | 0.0338 | 0.0514 | 0.9899 | 0.8596 | 98 | 99 | yes | yes | 0.8000 | 1.8789 | 1.9139 | 1.6473 | -0.2666 | 0.2700 | 0.1316 | 0.8684 | 0.0000 | 0.8596 | 0.0359 | yes |
| binary | breast_cancer | 42 | 0.8475 | 0.1525 | 341 | 114 | 114 | A | A|va=0|strategy=builtin.default|ncf=default | deterministic | default | builtin.default | no | no | no | 0.9035 | 0.0965 | 0.0000 | 0.0965 | 0.9903 | 0.9561 | 0.0342 | 0.0620 | 0.9903 | 0.8947 | 102 | 103 | yes | yes | 0.8000 | 1.8789 | 1.9035 | 1.6480 | -0.2555 | 0.2701 | 0.0965 | 0.9035 | 0.0000 | 0.8947 | 0.0472 | yes |
| binary | breast_cancer | 42 | 0.8713 | 0.1287 | 341 | 114 | 114 | A | A|va=0|strategy=builtin.default|ncf=default | deterministic | default | builtin.default | no | no | no | 0.9211 | 0.0789 | 0.0000 | 0.0789 | 0.9810 | 0.9561 | 0.0248 | 0.0541 | 0.9810 | 0.9035 | 103 | 105 | yes | yes | 0.6000 | 1.8789 | 1.8990 | 1.6439 | -0.2551 | 0.2656 | 0.0789 | 0.9211 | 0.0000 | 0.9035 | 0.0323 | yes |
| binary | breast_cancer | 42 | 0.8950 | 0.1050 | 341 | 114 | 114 | A | A|va=0|strategy=builtin.default|ncf=default | deterministic | default | builtin.default | no | no | no | 0.9298 | 0.0702 | 0.0000 | 0.0702 | 0.9811 | 0.9561 | 0.0250 | 0.0375 | 0.9811 | 0.9123 | 104 | 106 | yes | yes | 0.6000 | 1.8789 | 1.8996 | 1.6038 | -0.2958 | 0.2146 | 0.0702 | 0.9298 | 0.0000 | 0.9123 | 0.0173 | yes |
| binary | breast_cancer | 42 | 0.9187 | 0.0813 | 341 | 114 | 114 | A | A|va=0|strategy=builtin.default|ncf=default | deterministic | default | builtin.default | no | no | no | 0.9474 | 0.0526 | 0.0000 | 0.0526 | 0.9722 | 0.9561 | 0.0161 | 0.0302 | 0.9722 | 0.9211 | 105 | 108 | yes | yes | 0.4000 | 1.8789 | 1.8934 | 1.6172 | -0.2762 | 0.2407 | 0.0526 | 0.9474 | 0.0000 | 0.9211 | 0.0023 | yes |
| binary | breast_cancer | 42 | 0.9425 | 0.0575 | 341 | 114 | 114 | A | A|va=0|strategy=builtin.default|ncf=default | deterministic | default | builtin.default | no | no | no | 0.9474 | 0.0526 | 0.0000 | 0.0526 | 0.9722 | 0.9561 | 0.0161 | 0.0051 | 0.9722 | 0.9211 | 105 | 108 | yes | yes | 0.4000 | 1.8789 | 1.8934 | 1.6172 | -0.2762 | 0.2407 | 0.0526 | 0.9474 | 0.0000 | 0.9211 | -0.0214 | yes |
| binary | breast_cancer | 42 | 0.9663 | 0.0337 | 341 | 114 | 114 | A | A|va=0|strategy=builtin.default|ncf=default | deterministic | default | builtin.default | no | no | no | 0.9912 | 0.0088 | 0.0000 | 0.0088 | 0.9646 | 0.9561 | 0.0085 | 0.0252 | 0.9646 | 0.9561 | 109 | 113 | yes | yes | 0.2000 | 1.8789 | 1.8820 | 1.5238 | -0.3582 | 0.0796 | 0.0088 | 0.9912 | 0.0000 | 0.9561 | -0.0101 | yes |
| binary | breast_cancer | 42 | 0.9900 | 0.0100 | 341 | 114 | 114 | A | A|va=0|strategy=builtin.default|ncf=default | deterministic | default | builtin.default | no | no | no | 0.9474 | 0.0526 | 0.0526 | 0.0000 | 0.9722 | 0.9561 | 0.0161 | 0.0106 | 0.9722 | 0.9211 | 105 | 108 | yes | yes | 0.4000 | 1.8789 | 1.8934 | 1.6172 | -0.2762 | 0.2407 | 0.0000 | 0.9474 | 0.0526 | 0.9737 | -0.0163 | yes |
| binary | breast_cancer | 42 | 0.8000 | 0.2000 | 341 | 114 | 114 | B | B|va=1|strategy=builtin.default|ncf=default | deterministic | default | builtin.default | yes | no | no | 0.0000 | 1.0000 | 1.0000 | 0.0000 | nan | 0.9561 | nan | nan | nan | 0.0000 | 0 | 0 | no | no | 1.0000 | 1.8789 | nan | 1.8789 | nan | nan | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.2000 | yes |
| binary | breast_cancer | 42 | 0.8237 | 0.1763 | 341 | 114 | 114 | B | B|va=1|strategy=builtin.default|ncf=default | deterministic | default | builtin.default | yes | no | no | 0.0000 | 1.0000 | 1.0000 | 0.0000 | nan | 0.9561 | nan | nan | nan | 0.0000 | 0 | 0 | no | no | 1.0000 | 1.8789 | nan | 1.8789 | nan | nan | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.1763 | yes |
| binary | breast_cancer | 42 | 0.8475 | 0.1525 | 341 | 114 | 114 | B | B|va=1|strategy=builtin.default|ncf=default | deterministic | default | builtin.default | yes | no | no | 0.0000 | 1.0000 | 1.0000 | 0.0000 | nan | 0.9561 | nan | nan | nan | 0.0000 | 0 | 0 | no | no | 1.0000 | 1.8789 | nan | 1.8789 | nan | nan | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.1525 | yes |

_Showing first 12 of 12420 rows._

## Arm Summary

| arm_code | strategy | ncf | use_va_difficulty | difficulty_normalized | double_count_difficulty | accept_rate | reject_rate | accepted_accuracy | accuracy_delta | ambiguity_rate | novelty_rate | rejected_error_capture_rate | mean_difficulty_accepted | mean_difficulty_rejected | difficulty_gap_rejected_minus_accepted | difficulty_reject_auc | empirical_coverage | coverage_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | builtin.default | default | no | no | no | 0.6332 | 0.3668 | 0.8614 | 0.0551 | 0.3431 | 0.0237 | 0.5878 | 1.9148 | 1.9225 | 0.0100 | 0.4986 | 0.8965 | 0.0015 |
| B | builtin.default | default | yes | no | no | 0.2132 | 0.7868 | 0.7608 | -0.0405 | 0.7848 | 0.0021 | 0.7760 | 1.8627 | 1.9021 | 0.0298 | 0.5318 | 0.9403 | 0.0453 |
| C | experimental.difficulty_normalized | default | no | yes | no | 0.6019 | 0.3981 | 0.8548 | 0.0478 | 0.3648 | 0.0333 | 0.6123 | 1.7927 | 2.0849 | 0.3049 | 0.6971 | 0.8840 | -0.0110 |
| D | experimental.difficulty_normalized | default | yes | yes | yes | 0.3604 | 0.6396 | 0.8013 | -0.0056 | 0.6302 | 0.0094 | 0.6351 | 1.6704 | 2.0774 | 0.4206 | 0.8764 | 0.8879 | -0.0071 |
| E | experimental.difficulty_normalized | ensured | no | yes | no | 0.3379 | 0.6621 | 0.8398 | 0.0327 | 0.5951 | 0.0670 | 0.8009 | 1.8973 | 1.9278 | 0.0308 | 0.5015 | 0.8917 | -0.0033 |
| F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.2286 | 0.7714 | 0.7873 | -0.0185 | 0.7356 | 0.0358 | 0.7541 | 1.7053 | 2.0082 | 0.3154 | 0.8021 | 0.8968 | 0.0018 |

## By Confidence And Arm

| confidence | epsilon | arm_code | strategy | ncf | use_va_difficulty | difficulty_normalized | double_count_difficulty | reject_rate | accepted_accuracy | ambiguity_rate | novelty_rate | difficulty_gap_rejected_minus_accepted | difficulty_reject_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8000 | 0.2000 | A | builtin.default | default | no | no | no | 0.1982 | 0.8349 | 0.1341 | 0.0641 | 0.0167 | 0.5000 |
| 0.8237 | 0.1763 | A | builtin.default | default | no | no | no | 0.2129 | 0.8323 | 0.1625 | 0.0504 | 0.0099 | 0.4973 |
| 0.8475 | 0.1525 | A | builtin.default | default | no | no | no | 0.2312 | 0.8349 | 0.1936 | 0.0375 | 0.0006 | 0.4975 |
| 0.8713 | 0.1287 | A | builtin.default | default | no | no | no | 0.2716 | 0.8413 | 0.2449 | 0.0267 | 0.0096 | 0.5037 |
| 0.8950 | 0.1050 | A | builtin.default | default | no | no | no | 0.3023 | 0.8473 | 0.2859 | 0.0164 | 0.0094 | 0.5024 |
| 0.9187 | 0.0813 | A | builtin.default | default | no | no | no | 0.3578 | 0.8652 | 0.3477 | 0.0101 | 0.0125 | 0.4984 |
| 0.9425 | 0.0575 | A | builtin.default | default | no | no | no | 0.4224 | 0.8758 | 0.4165 | 0.0059 | 0.0165 | 0.4994 |
| 0.9663 | 0.0337 | A | builtin.default | default | no | no | no | 0.5188 | 0.9130 | 0.5166 | 0.0022 | 0.0159 | 0.4998 |
| 0.9900 | 0.0100 | A | builtin.default | default | no | no | no | 0.7859 | 0.9681 | 0.7859 | 0.0000 | -0.0139 | 0.4790 |
| 0.8000 | 0.2000 | B | builtin.default | default | yes | no | no | 0.6241 | 0.7614 | 0.6159 | 0.0082 | 0.0040 | 0.5002 |
| 0.8237 | 0.1763 | B | builtin.default | default | yes | no | no | 0.6615 | 0.7617 | 0.6558 | 0.0056 | 0.0121 | 0.5121 |
| 0.8475 | 0.1525 | B | builtin.default | default | yes | no | no | 0.7020 | 0.7567 | 0.6987 | 0.0033 | 0.0169 | 0.5198 |
| 0.8713 | 0.1287 | B | builtin.default | default | yes | no | no | 0.7514 | 0.7558 | 0.7503 | 0.0011 | 0.0411 | 0.5418 |
| 0.8950 | 0.1050 | B | builtin.default | default | yes | no | no | 0.7825 | 0.7602 | 0.7823 | 0.0003 | 0.0294 | 0.5339 |
| 0.9187 | 0.0813 | B | builtin.default | default | yes | no | no | 0.8143 | 0.7703 | 0.8142 | 0.0001 | 0.0284 | 0.5309 |
| 0.9425 | 0.0575 | B | builtin.default | default | yes | no | no | 0.8577 | 0.7663 | 0.8577 | 0.0000 | 0.0462 | 0.5400 |
| 0.9663 | 0.0337 | B | builtin.default | default | yes | no | no | 0.9054 | 0.7789 | 0.9054 | 0.0000 | 0.0421 | 0.5509 |
| 0.9900 | 0.0100 | B | builtin.default | default | yes | no | no | 0.9827 | 0.6974 | 0.9827 | 0.0000 | 0.0926 | 0.6167 |
| 0.8000 | 0.2000 | C | experimental.difficulty_normalized | default | no | yes | no | 0.2415 | 0.8365 | 0.1529 | 0.0886 | 0.2748 | 0.6242 |
| 0.8237 | 0.1763 | C | experimental.difficulty_normalized | default | no | yes | no | 0.2507 | 0.8352 | 0.1801 | 0.0706 | 0.2839 | 0.6416 |
| 0.8475 | 0.1525 | C | experimental.difficulty_normalized | default | no | yes | no | 0.2636 | 0.8387 | 0.2116 | 0.0521 | 0.2943 | 0.6601 |
| 0.8713 | 0.1287 | C | experimental.difficulty_normalized | default | no | yes | no | 0.2922 | 0.8428 | 0.2558 | 0.0364 | 0.3051 | 0.6886 |
| 0.8950 | 0.1050 | C | experimental.difficulty_normalized | default | no | yes | no | 0.3253 | 0.8471 | 0.3004 | 0.0249 | 0.3174 | 0.7053 |
| 0.9187 | 0.0813 | C | experimental.difficulty_normalized | default | no | yes | no | 0.3779 | 0.8572 | 0.3628 | 0.0151 | 0.3316 | 0.7225 |
| 0.9425 | 0.0575 | C | experimental.difficulty_normalized | default | no | yes | no | 0.4536 | 0.8663 | 0.4451 | 0.0086 | 0.3031 | 0.7316 |
| 0.9663 | 0.0337 | C | experimental.difficulty_normalized | default | no | yes | no | 0.5607 | 0.8842 | 0.5571 | 0.0035 | 0.3291 | 0.7654 |
| 0.9900 | 0.0100 | C | experimental.difficulty_normalized | default | no | yes | no | 0.8170 | 0.9218 | 0.8170 | 0.0000 | 0.3075 | 0.7836 |
| 0.8000 | 0.2000 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.4130 | 0.8058 | 0.3835 | 0.0295 | 0.3837 | 0.7950 |
| 0.8237 | 0.1763 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.4504 | 0.8030 | 0.4296 | 0.0208 | 0.3952 | 0.8170 |
| 0.8475 | 0.1525 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.4959 | 0.8034 | 0.4802 | 0.0157 | 0.4951 | 0.8663 |
| 0.8713 | 0.1287 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.5450 | 0.8013 | 0.5353 | 0.0098 | 0.4362 | 0.8804 |
| 0.8950 | 0.1050 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.6072 | 0.8039 | 0.6027 | 0.0046 | 0.4334 | 0.9018 |
| 0.9187 | 0.0813 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.6814 | 0.7990 | 0.6790 | 0.0024 | 0.4175 | 0.9041 |
| 0.9425 | 0.0575 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.7541 | 0.8042 | 0.7530 | 0.0011 | 0.4026 | 0.9090 |
| 0.9663 | 0.0337 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.8387 | 0.8029 | 0.8381 | 0.0006 | 0.4022 | 0.9136 |
| 0.9900 | 0.0100 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.9704 | 0.7699 | 0.9704 | 0.0000 | 0.4160 | 0.9397 |
| 0.8000 | 0.2000 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.6072 | 0.8365 | 0.4656 | 0.1417 | -0.0043 | 0.4339 |
| 0.8237 | 0.1763 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.6179 | 0.8429 | 0.4973 | 0.1206 | 0.0026 | 0.4519 |
| 0.8475 | 0.1525 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.6306 | 0.8349 | 0.5281 | 0.1026 | 0.0048 | 0.4602 |
| 0.8713 | 0.1287 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.6380 | 0.8279 | 0.5548 | 0.0833 | 0.0052 | 0.4828 |
| 0.8950 | 0.1050 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.6406 | 0.8318 | 0.5761 | 0.0645 | 0.0094 | 0.4921 |
| 0.9187 | 0.0813 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.6431 | 0.8322 | 0.5968 | 0.0463 | 0.0385 | 0.5184 |
| 0.9425 | 0.0575 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.6557 | 0.8432 | 0.6254 | 0.0303 | 0.0430 | 0.5283 |
| 0.9663 | 0.0337 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.6921 | 0.8417 | 0.6790 | 0.0131 | 0.0834 | 0.5750 |
| 0.9900 | 0.0100 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.8337 | 0.8982 | 0.8325 | 0.0012 | 0.1728 | 0.6594 |
| 0.8000 | 0.2000 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.5820 | 0.8014 | 0.5059 | 0.0762 | 0.3277 | 0.7764 |
| 0.8237 | 0.1763 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.6320 | 0.7953 | 0.5697 | 0.0622 | 0.3382 | 0.7929 |
| 0.8475 | 0.1525 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.6746 | 0.7929 | 0.6211 | 0.0535 | 0.3702 | 0.8019 |
| 0.8713 | 0.1287 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.7245 | 0.7954 | 0.6796 | 0.0449 | 0.2945 | 0.7940 |
| 0.8950 | 0.1050 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.7725 | 0.7968 | 0.7367 | 0.0358 | 0.2872 | 0.7935 |
| 0.9187 | 0.0813 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.8207 | 0.7934 | 0.7951 | 0.0256 | 0.2938 | 0.8049 |
| 0.9425 | 0.0575 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.8608 | 0.7702 | 0.8439 | 0.0168 | 0.2949 | 0.8087 |
| 0.9663 | 0.0337 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.9019 | 0.7531 | 0.8948 | 0.0070 | 0.3158 | 0.8293 |
| 0.9900 | 0.0100 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.9735 | 0.7734 | 0.9732 | 0.0003 | 0.3127 | 0.8484 |

## Accepted Accuracy At Matched Reject-Rate Bins

A-vs-C matched-bin comparison (primary contrast).

| reject_rate_bin | A_mean_accepted_accuracy | C_mean_accepted_accuracy | accepted_accuracy_delta_C_minus_A | A_mean_reject_rate | C_mean_reject_rate | A_rows | C_rows |
|---|---|---|---|---|---|---|---|
| [0.0, 0.1) | 0.9015 | 0.9160 | 0.0145 | 0.0466 | 0.0611 | 517 | 366 |
| [0.1, 0.2) | 0.9155 | 0.9079 | -0.0076 | 0.1481 | 0.1479 | 362 | 440 |
| [0.2, 0.3) | 0.8572 | 0.8778 | 0.0206 | 0.2468 | 0.2453 | 222 | 209 |
| [0.3, 0.4) | 0.8272 | 0.8263 | -0.0008 | 0.3530 | 0.3520 | 192 | 197 |
| [0.4, 0.5) | 0.8084 | 0.8015 | -0.0069 | 0.4515 | 0.4529 | 145 | 175 |
| [0.5, 0.6) | 0.7872 | 0.7952 | 0.0080 | 0.5495 | 0.5485 | 128 | 136 |
| [0.6, 0.7) | 0.8228 | 0.7915 | -0.0313 | 0.6456 | 0.6460 | 113 | 101 |
| [0.7, 0.8) | 0.8068 | 0.7438 | -0.0630 | 0.7475 | 0.7480 | 96 | 116 |
| [0.8, 0.9) | 0.8408 | 0.7546 | -0.0862 | 0.8447 | 0.8519 | 87 | 108 |
| [0.9, 1.0) | 0.7845 | 0.8840 | 0.0995 | 0.9855 | 0.9821 | 208 | 222 |

## Per-Arm Reject-Rate Bin Aggregates

| reject_rate_bin | arm_code | n_rows | mean_reject_rate | mean_accepted_accuracy |
|---|---|---|---|---|
| [0.0, 0.1) | A | 517 | 0.0466 | 0.9015 |
| [0.0, 0.1) | B | 119 | 0.0346 | 0.9150 |
| [0.0, 0.1) | C | 366 | 0.0611 | 0.9160 |
| [0.0, 0.1) | D | 105 | 0.0502 | 0.9132 |
| [0.0, 0.1) | E | 21 | 0.0742 | 0.9551 |
| [0.0, 0.1) | F | 37 | 0.0487 | 0.9408 |
| [0.1, 0.2) | A | 362 | 0.1481 | 0.9155 |
| [0.1, 0.2) | B | 49 | 0.1466 | 0.9144 |
| [0.1, 0.2) | C | 440 | 0.1479 | 0.9079 |
| [0.1, 0.2) | D | 137 | 0.1461 | 0.8925 |
| [0.1, 0.2) | E | 108 | 0.1500 | 0.9851 |
| [0.1, 0.2) | F | 45 | 0.1484 | 0.9188 |
| [0.2, 0.3) | A | 222 | 0.2468 | 0.8572 |
| [0.2, 0.3) | B | 50 | 0.2617 | 0.8929 |
| [0.2, 0.3) | C | 209 | 0.2453 | 0.8778 |
| [0.2, 0.3) | D | 107 | 0.2517 | 0.8489 |
| [0.2, 0.3) | E | 109 | 0.2558 | 0.9656 |
| [0.2, 0.3) | F | 67 | 0.2521 | 0.8705 |
| [0.3, 0.4) | A | 192 | 0.3530 | 0.8272 |
| [0.3, 0.4) | B | 45 | 0.3500 | 0.8571 |
| [0.3, 0.4) | C | 197 | 0.3520 | 0.8263 |
| [0.3, 0.4) | D | 125 | 0.3606 | 0.8546 |
| [0.3, 0.4) | E | 150 | 0.3505 | 0.9200 |
| [0.3, 0.4) | F | 75 | 0.3541 | 0.8846 |
| [0.4, 0.5) | A | 145 | 0.4515 | 0.8084 |
| [0.4, 0.5) | B | 86 | 0.4572 | 0.8502 |
| [0.4, 0.5) | C | 175 | 0.4529 | 0.8015 |
| [0.4, 0.5) | D | 129 | 0.4580 | 0.8067 |
| [0.4, 0.5) | E | 164 | 0.4561 | 0.8985 |
| [0.4, 0.5) | F | 75 | 0.4557 | 0.8048 |
| [0.5, 0.6) | A | 128 | 0.5495 | 0.7872 |
| [0.5, 0.6) | B | 82 | 0.5461 | 0.8152 |
| [0.5, 0.6) | C | 136 | 0.5485 | 0.7952 |
| [0.5, 0.6) | D | 185 | 0.5553 | 0.8036 |
| [0.5, 0.6) | E | 202 | 0.5544 | 0.8655 |
| [0.5, 0.6) | F | 101 | 0.5531 | 0.7850 |
| [0.6, 0.7) | A | 113 | 0.6456 | 0.8228 |
| [0.6, 0.7) | B | 117 | 0.6484 | 0.7885 |
| [0.6, 0.7) | C | 101 | 0.6460 | 0.7915 |
| [0.6, 0.7) | D | 247 | 0.6524 | 0.7992 |
| [0.6, 0.7) | E | 304 | 0.6533 | 0.8627 |
| [0.6, 0.7) | F | 177 | 0.6523 | 0.7869 |
| [0.7, 0.8) | A | 96 | 0.7475 | 0.8068 |
| [0.7, 0.8) | B | 124 | 0.7503 | 0.7343 |
| [0.7, 0.8) | C | 116 | 0.7480 | 0.7438 |
| [0.7, 0.8) | D | 302 | 0.7528 | 0.7712 |
| [0.7, 0.8) | E | 278 | 0.7500 | 0.7911 |
| [0.7, 0.8) | F | 255 | 0.7537 | 0.7573 |
| [0.8, 0.9) | A | 87 | 0.8447 | 0.8408 |
| [0.8, 0.9) | B | 223 | 0.8552 | 0.7368 |
| [0.8, 0.9) | C | 108 | 0.8519 | 0.7546 |
| [0.8, 0.9) | D | 265 | 0.8492 | 0.7928 |
| [0.8, 0.9) | E | 255 | 0.8464 | 0.7612 |
| [0.8, 0.9) | F | 420 | 0.8534 | 0.7879 |
| [0.9, 1.0) | A | 208 | 0.9855 | 0.7845 |
| [0.9, 1.0) | B | 1175 | 0.9744 | 0.6979 |
| [0.9, 1.0) | C | 222 | 0.9821 | 0.8840 |
| [0.9, 1.0) | D | 468 | 0.9643 | 0.7208 |
| [0.9, 1.0) | E | 479 | 0.9658 | 0.7349 |
| [0.9, 1.0) | F | 818 | 0.9642 | 0.7564 |

## Required Analyses

1. Does direct normalization increase rejection among high-difficulty instances?
Direct normalization (C vs A) changed reject_rate by +0.0313, difficulty-gap by +0.2950, and difficulty_reject_auc by +0.1985.
2. Does it improve accepted accuracy at comparable reject rates?
At matched reject-rate bins, C minus A mean accepted_accuracy is -0.0053.
3. Does it increase ambiguity rate, novelty rate, or both?
For C vs A, ambiguity_rate changed by +0.0217 and novelty_rate by +0.0096.
4. Does using VA difficulty and score normalization together appear to double-count difficulty?
Double-count diagnostics: D-B reject_rate delta -0.1473, F-E reject_rate delta +0.1093; difficulty-gap deltas are +0.3909 and +0.2846.
5. Which arm is recommended for further development?
Recommended arm for next iteration: C (primary A-vs-C contrast with direct normalization and no VA double-count risk).


## Metric Consistency Note (RT-5)

Scenario 9 reports a full-grid A-vs-C difficulty_reject_auc delta of +0.1985, while Scenario 11 reports +0.0155 at matched operating points. This reduction is a selection effect, not a contradiction:

- Scenario 9 averages over all confidence values. The positive delta is strongest in high-confidence rows (conf >= 0.91), where the AUC delta is +0.2492.
- At moderate confidence (conf < 0.91), the Scenario 9 AUC delta is +0.1637: smaller than the high-confidence tail, but still positive.
- Scenario 11 targets reject rates of 10-40%, uses matched operating-point selection, and reduces the observed difficulty-AUC effect to +0.0155.

Conclusion: the full-grid Scenario 9 AUC advantage is not sufficient evidence for public promotion. At matched operating points, accepted-accuracy gains are tiny or negative and the difficulty-selection advantage is much smaller.
