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
- Recommended arm for next iteration: C (primary A-vs-C contrast with direct normalization and no VA double-count risk). NOTE: Scenario 12 shows arm C has more structural coverage violations than arm A. This recommendation is for selectivity/accuracy only; promotion requires Scenario 13 clearance.

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
- **coverage_validity_caveat**: NOTE: Scenario 12 shows arm C has more structural coverage violations than arm A. This recommendation is for selectivity/accuracy only; promotion requires Scenario 13 clearance.
- **metric_consistency_note**: {'full_grid_auc_delta': 0.19847119235003774, 'hi_conf_auc_delta': 0.24921106666866505, 'lo_conf_auc_delta': 0.16366233970904076, 'scenario_11_matched_delta': 0.0155, 'note': 'Full-grid positive delta is strongest in high-confidence rows. Scenario 11 matched operating-point selection reduces the observed difficulty-AUC effect and remains the promotion decision gate.'}

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
Recommended arm for next iteration: C (primary A-vs-C contrast with direct normalization and no VA double-count risk). NOTE: Scenario 12 shows arm C has more structural coverage violations than arm A. This recommendation is for selectivity/accuracy only; promotion requires Scenario 13 clearance.

## Per-Dataset Arm Comparison (all datasets)

Mean over seeds and confidence levels. Rows sorted by arm_code, dataset.

| dataset | arm_code | strategy | accept_rate | accepted_accuracy | accuracy_delta | rejected_error_capture_rate | difficulty_reject_auc | empirical_coverage |
|---|---|---|---|---|---|---|---|---|
| balance | A | builtin.default | 0.8939 | 0.8946 | 0.0578 | 0.3891 | 0.5005 | nan |
| breast_cancer | A | builtin.default | 0.9181 | 0.9779 | 0.0218 | 0.5326 | 0.2263 | 0.9078 |
| cars | A | builtin.default | 0.8920 | 0.9912 | 0.0253 | 0.7552 | 0.3818 | nan |
| cmc | A | builtin.default | 0.3333 | 0.6612 | 0.1521 | 0.7374 | 0.4110 | nan |
| colic | A | builtin.default | 0.7136 | 0.8680 | 0.0485 | 0.4294 | 0.4878 | 0.8796 |
| cool | A | builtin.default | 0.8935 | 0.9836 | 0.0342 | 0.6982 | 0.6053 | nan |
| creditA | A | builtin.default | 0.7862 | 0.8968 | 0.0388 | 0.3823 | 0.5605 | 0.8720 |
| diabetes | A | builtin.default | 0.5541 | 0.8723 | 0.1164 | 0.6490 | 0.5858 | 0.9140 |
| ecoli | A | builtin.default | 0.6147 | 0.9116 | 0.0528 | 0.6046 | 0.6046 | nan |
| german | A | builtin.default | 0.3732 | 0.7873 | 0.1307 | 0.7383 | 0.6937 | 0.9118 |
| glass | A | builtin.default | 0.4610 | 0.8120 | 0.0492 | 0.6043 | 0.5127 | nan |
| haberman | A | builtin.default | 0.4499 | 0.8003 | 0.1547 | 0.7027 | 0.5913 | 0.8967 |
| heartC | A | builtin.default | 0.6350 | 0.8779 | 0.0779 | 0.5656 | 0.5465 | 0.9100 |
| heartH | A | builtin.default | 0.6919 | 0.8716 | 0.0410 | 0.4645 | 0.4488 | 0.9055 |
| heartS | A | builtin.default | 0.6720 | 0.8148 | 0.0318 | 0.4070 | 0.5681 | 0.8642 |
| heat | A | builtin.default | 0.8831 | 0.9986 | 0.0051 | 0.8611 | 0.6610 | nan |
| hepati | A | builtin.default | 0.6717 | 0.9271 | 0.0755 | 0.5994 | 0.8139 | 0.9226 |
| image | A | builtin.default | 0.8969 | 0.9939 | 0.0186 | 0.7848 | 0.3795 | nan |
| iono | A | builtin.default | 0.7635 | 0.9634 | 0.0348 | 0.6207 | 0.4988 | 0.8933 |
| iris | A | builtin.default | 0.6511 | 0.9682 | 0.0182 | 0.6593 | 0.3132 | nan |
| je4042 | A | builtin.default | 0.4613 | 0.8261 | 0.0817 | 0.6371 | 0.5669 | 0.9123 |
| je4243 | A | builtin.default | 0.4618 | 0.6814 | 0.0622 | 0.5969 | 0.4481 | 0.8426 |
| kc1 | A | builtin.default | 0.4187 | 0.8399 | 0.0919 | 0.6958 | 0.5594 | 0.9234 |
| kc2 | A | builtin.default | 0.6168 | 0.8820 | 0.0874 | 0.6001 | 0.4650 | 0.9126 |
| kc3 | A | builtin.default | 0.7303 | 0.9053 | 0.0438 | 0.4576 | 0.6603 | 0.9026 |
| liver | A | builtin.default | 0.4870 | 0.7631 | 0.0994 | 0.6414 | 0.5971 | 0.8770 |
| pc1req | A | builtin.default | 0.3429 | 0.5539 | -0.1127 | 0.6025 | 0.3061 | 0.8688 |
| pc4 | A | builtin.default | 0.8777 | 0.9378 | 0.0419 | 0.4506 | 0.5161 | 0.8989 |
| sonar | A | builtin.default | 0.7153 | 0.9114 | 0.0305 | 0.4425 | 0.4592 | 0.9233 |
| spect | A | builtin.default | 0.6429 | 0.9012 | 0.0194 | 0.4000 | 0.3450 | 0.9056 |
| spectf | A | builtin.default | 0.6432 | 0.8752 | 0.0641 | 0.5230 | 0.3451 | 0.8984 |
| steel | A | builtin.default | 0.7556 | 0.8427 | 0.0807 | 0.4494 | 0.4063 | nan |
| tae | A | builtin.default | 0.2667 | 0.3863 | -0.1456 | 0.7018 | 0.4604 | nan |
| transfusion | A | builtin.default | 0.5285 | 0.8194 | 0.0961 | 0.6039 | 0.4921 | 0.8920 |
| ttt | A | builtin.default | 0.9010 | 0.9946 | 0.0186 | 0.8055 | 0.2956 | 0.9079 |
| user | A | builtin.default | 0.7951 | 0.9151 | 0.0263 | 0.3743 | 0.6382 | nan |
| vehicle | A | builtin.default | 0.7292 | 0.8443 | 0.0984 | 0.5060 | 0.5192 | nan |
| vote | A | builtin.default | 0.7746 | 0.9020 | 0.0673 | 0.4969 | 0.7003 | 0.8795 |
| vowel | A | builtin.default | 0.7384 | 0.9735 | 0.0311 | 0.5485 | 0.4107 | nan |
| wave | A | builtin.default | 0.8292 | 0.9058 | 0.0494 | 0.4148 | 0.6473 | nan |
| wbc | A | builtin.default | 0.7830 | 0.9706 | 0.0093 | 0.3926 | 0.2706 | 0.8858 |
| whole | A | builtin.default | 0.3227 | 0.7393 | 0.0393 | 0.7195 | 0.4854 | nan |
| wine | A | builtin.default | 0.1056 | 1.0000 | 0.0403 | 1.0000 | 0.4061 | nan |
| wineR | A | builtin.default | 0.4863 | 0.8042 | 0.1311 | 0.6628 | 0.4910 | nan |
| wineW | A | builtin.default | 0.5191 | 0.8089 | 0.1436 | 0.6619 | 0.5212 | nan |
| yeast | A | builtin.default | 0.4460 | 0.6830 | 0.0749 | 0.6035 | 0.4335 | nan |
| balance | B | builtin.default | 0.0476 | 0.9419 | 0.1081 | 0.9394 | 0.8476 | nan |
| breast_cancer | B | builtin.default | 0.0507 | 0.7752 | -0.1812 | 0.7418 | 0.8922 | 0.9756 |
| cars | B | builtin.default | 0.8067 | 0.9798 | 0.0139 | 0.4861 | 0.3860 | nan |
| cmc | B | builtin.default | 0.2365 | 0.4808 | -0.0294 | 0.7552 | 0.5716 | nan |
| colic | B | builtin.default | 0.0790 | 0.7562 | -0.0772 | 0.8678 | 0.7468 | 0.9571 |
| cool | B | builtin.default | 0.0550 | 0.7580 | -0.1914 | 0.7622 | 0.2504 | nan |
| creditA | B | builtin.default | 0.0386 | 0.7293 | -0.1323 | 0.9056 | 0.5971 | 0.9747 |
| diabetes | B | builtin.default | 0.0560 | 0.5442 | -0.2079 | 0.8989 | 0.5830 | 0.9690 |
| ecoli | B | builtin.default | 0.2304 | 0.7845 | -0.0704 | 0.6862 | 0.4087 | nan |
| german | B | builtin.default | 0.3317 | 0.7527 | 0.0941 | 0.7329 | 0.5150 | 0.9173 |
| glass | B | builtin.default | 0.1726 | 0.8160 | 0.0534 | 0.8257 | 0.5183 | nan |
| haberman | B | builtin.default | 0.3563 | 0.7616 | 0.1151 | 0.7220 | 0.3358 | 0.9099 |
| heartC | B | builtin.default | 0.0769 | 0.6920 | -0.1045 | 0.8962 | 0.6520 | 0.9508 |
| heartH | B | builtin.default | 0.1250 | 0.8271 | 0.0063 | 0.9124 | 0.5497 | 0.9556 |
| heartS | B | builtin.default | 0.1037 | 0.5896 | -0.1934 | 0.8285 | 0.6291 | 0.9399 |
| heat | B | builtin.default | 0.0511 | 0.8303 | -0.1615 | 0.2083 | 0.5472 | nan |
| hepati | B | builtin.default | 0.5125 | 0.9577 | 0.1061 | 0.7822 | 0.7140 | 0.9477 |
| image | B | builtin.default | 0.3236 | 0.9912 | 0.0180 | 0.7010 | 0.8511 | nan |
| iono | B | builtin.default | 0.0559 | 1.0000 | 0.0786 | 1.0000 | 0.7567 | 0.9730 |
| iris | B | builtin.default | 0.0178 | 0.6667 | -0.2333 | 0.9407 | 0.4321 | nan |
| je4042 | B | builtin.default | 0.1004 | 0.6009 | -0.1441 | 0.8422 | 0.7282 | 0.9617 |
| je4243 | B | builtin.default | 0.1409 | 0.7623 | 0.1326 | 0.9029 | 0.5708 | 0.9266 |
| kc1 | B | builtin.default | 0.2346 | 0.6494 | -0.0978 | 0.8106 | 0.4945 | 0.9398 |
| kc2 | B | builtin.default | 0.3222 | 0.8965 | 0.1024 | 0.8109 | 0.1604 | 0.9267 |
| kc3 | B | builtin.default | 0.6742 | 0.8950 | 0.0338 | 0.4565 | 0.3259 | 0.9166 |
| liver | B | builtin.default | 0.0953 | 0.5871 | -0.0767 | 0.8566 | 0.7647 | 0.9504 |
| pc1req | B | builtin.default | 0.0995 | 0.5485 | -0.1261 | 0.8922 | 0.7703 | 0.9344 |
| pc4 | B | builtin.default | 0.8226 | 0.9128 | 0.0169 | 0.4308 | 0.2747 | 0.9005 |
| sonar | B | builtin.default | 0.1032 | 0.8229 | -0.0595 | 0.8843 | 0.6231 | 0.9323 |
| spect | B | builtin.default | 0.6404 | 0.9066 | 0.0238 | 0.4244 | 0.2052 | 0.9111 |
| spectf | B | builtin.default | 0.6572 | 0.8874 | 0.0763 | 0.5504 | 0.2734 | 0.8704 |
| steel | B | builtin.default | 0.0490 | 0.8741 | 0.1081 | 0.9736 | 0.4029 | nan |
| tae | B | builtin.default | 0.1333 | 0.4061 | -0.1193 | 0.8381 | 0.5007 | nan |
| transfusion | B | builtin.default | 0.0517 | 0.7386 | 0.0161 | 0.9424 | 0.9129 | 0.9875 |
| ttt | B | builtin.default | 0.4870 | 0.9062 | -0.0699 | 0.2485 | 0.4819 | 0.9035 |
| user | B | builtin.default | 0.0508 | 0.9032 | 0.0513 | 0.9667 | 0.3878 | nan |
| vehicle | B | builtin.default | 0.0648 | 0.6456 | -0.0954 | 0.9293 | 0.6619 | nan |
| vote | B | builtin.default | 0.0363 | 0.4444 | -0.3902 | 0.8442 | 0.3955 | 0.9808 |
| vowel | B | builtin.default | 0.3338 | 0.9281 | -0.0137 | 0.6492 | 0.3574 | nan |
| wave | B | builtin.default | 0.0318 | 0.6702 | -0.1867 | 0.9302 | 0.5157 | nan |
| wbc | B | builtin.default | 0.1305 | 0.7361 | -0.2249 | 0.4759 | 0.8683 | 0.9348 |
| whole | B | builtin.default | 0.2722 | 0.7170 | 0.0170 | 0.7679 | 0.1378 | nan |
| wine | B | builtin.default | 0.1123 | 0.9866 | 0.0074 | 0.9167 | 0.6316 | nan |
| wineR | B | builtin.default | 0.3067 | 0.7645 | 0.0912 | 0.7805 | 0.4792 | nan |
| wineW | B | builtin.default | 0.0809 | 0.8061 | 0.1451 | 0.9334 | 0.4739 | nan |
| yeast | B | builtin.default | 0.0462 | 0.7179 | 0.1099 | 0.9603 | 0.4167 | nan |
| balance | C | experimental.difficulty_normalized | 0.8233 | 0.9172 | 0.0804 | 0.5645 | 0.5168 | nan |
| breast_cancer | C | experimental.difficulty_normalized | 0.8869 | 0.9794 | 0.0232 | 0.5811 | 0.3267 | 0.9002 |
| cars | C | experimental.difficulty_normalized | 0.8893 | 0.9918 | 0.0259 | 0.7734 | 0.3647 | nan |
| cmc | C | experimental.difficulty_normalized | 0.2900 | 0.6188 | 0.1096 | 0.7564 | 0.7822 | nan |
| colic | C | experimental.difficulty_normalized | 0.7253 | 0.8579 | 0.0385 | 0.3887 | 0.6041 | 0.8664 |
| cool | C | experimental.difficulty_normalized | 0.8811 | 0.9842 | 0.0348 | 0.7144 | 0.5819 | nan |
| creditA | C | experimental.difficulty_normalized | 0.7844 | 0.8898 | 0.0318 | 0.3689 | 0.7085 | 0.8602 |
| diabetes | C | experimental.difficulty_normalized | 0.5472 | 0.8409 | 0.0851 | 0.6011 | 0.8594 | 0.8928 |
| ecoli | C | experimental.difficulty_normalized | 0.5546 | 0.9160 | 0.0572 | 0.6594 | 0.7827 | nan |
| german | C | experimental.difficulty_normalized | 0.4383 | 0.7667 | 0.1102 | 0.6835 | 0.9123 | 0.8951 |
| glass | C | experimental.difficulty_normalized | 0.4145 | 0.7668 | 0.0040 | 0.5912 | 0.9093 | nan |
| haberman | C | experimental.difficulty_normalized | 0.4550 | 0.7631 | 0.1166 | 0.6819 | 0.8767 | 0.8784 |
| heartC | C | experimental.difficulty_normalized | 0.6492 | 0.8552 | 0.0552 | 0.4942 | 0.7656 | 0.8929 |
| heartH | C | experimental.difficulty_normalized | 0.6226 | 0.8645 | 0.0340 | 0.4829 | 0.7556 | 0.8825 |
| heartS | C | experimental.difficulty_normalized | 0.6687 | 0.8244 | 0.0392 | 0.4354 | 0.6825 | 0.8609 |
| heat | C | experimental.difficulty_normalized | 0.8674 | 0.9987 | 0.0052 | 0.8750 | 0.6648 | nan |
| hepati | C | experimental.difficulty_normalized | 0.6953 | 0.9102 | 0.0586 | 0.5337 | 0.8923 | 0.9104 |
| image | C | experimental.difficulty_normalized | 0.8889 | 0.9947 | 0.0194 | 0.8229 | 0.3991 | nan |
| iono | C | experimental.difficulty_normalized | 0.7213 | 0.9675 | 0.0389 | 0.6844 | 0.5254 | 0.8822 |
| iris | C | experimental.difficulty_normalized | 0.5622 | 0.9856 | 0.0390 | 0.8370 | 0.6085 | nan |
| je4042 | C | experimental.difficulty_normalized | 0.4868 | 0.8019 | 0.0574 | 0.6061 | 0.8421 | 0.8835 |
| je4243 | C | experimental.difficulty_normalized | 0.4113 | 0.6675 | 0.0483 | 0.6226 | 0.8187 | 0.8557 |
| kc1 | C | experimental.difficulty_normalized | 0.4717 | 0.7974 | 0.0502 | 0.6136 | 0.9057 | 0.8995 |
| kc2 | C | experimental.difficulty_normalized | 0.5955 | 0.8475 | 0.0529 | 0.5228 | 0.7556 | 0.8889 |
| kc3 | C | experimental.difficulty_normalized | 0.6684 | 0.9164 | 0.0548 | 0.5898 | 0.7858 | 0.8995 |
| liver | C | experimental.difficulty_normalized | 0.4254 | 0.7868 | 0.1231 | 0.7106 | 0.8882 | 0.8940 |
| pc1req | C | experimental.difficulty_normalized | 0.3503 | 0.6285 | -0.0382 | 0.6006 | 0.7669 | 0.8593 |
| pc4 | C | experimental.difficulty_normalized | 0.8083 | 0.9379 | 0.0420 | 0.4951 | 0.6173 | 0.8867 |
| sonar | C | experimental.difficulty_normalized | 0.6566 | 0.9258 | 0.0448 | 0.5605 | 0.6395 | 0.9153 |
| spect | C | experimental.difficulty_normalized | 0.5944 | 0.8743 | -0.0074 | 0.3541 | 0.5591 | 0.8919 |
| spectf | C | experimental.difficulty_normalized | 0.5609 | 0.8498 | 0.0387 | 0.5097 | 0.6833 | 0.8728 |
| steel | C | experimental.difficulty_normalized | 0.6920 | 0.8443 | 0.0823 | 0.5021 | 0.6185 | nan |
| tae | C | experimental.difficulty_normalized | 0.1570 | 0.3496 | -0.1822 | 0.7940 | 0.8950 | nan |
| transfusion | C | experimental.difficulty_normalized | 0.4970 | 0.7695 | 0.0467 | 0.5705 | 0.8406 | 0.8759 |
| ttt | C | experimental.difficulty_normalized | 0.8900 | 0.9959 | 0.0199 | 0.8493 | 0.2863 | 0.8992 |
| user | C | experimental.difficulty_normalized | 0.7805 | 0.9196 | 0.0307 | 0.4204 | 0.6409 | nan |
| vehicle | C | experimental.difficulty_normalized | 0.5788 | 0.8805 | 0.1346 | 0.6901 | 0.7422 | nan |
| vote | C | experimental.difficulty_normalized | 0.7722 | 0.9031 | 0.0685 | 0.4989 | 0.7099 | 0.8692 |
| vowel | C | experimental.difficulty_normalized | 0.6891 | 0.9765 | 0.0341 | 0.6305 | 0.5395 | nan |
| wave | C | experimental.difficulty_normalized | 0.8259 | 0.9064 | 0.0500 | 0.4241 | 0.6845 | nan |
| wbc | C | experimental.difficulty_normalized | 0.7331 | 0.9715 | 0.0102 | 0.4741 | 0.2985 | 0.8698 |
| whole | C | experimental.difficulty_normalized | 0.3742 | 0.7358 | 0.0358 | 0.6691 | 0.8882 | nan |
| wine | C | experimental.difficulty_normalized | 0.0562 | 1.0000 | 0.0357 | 1.0000 | 0.9342 | nan |
| wineR | C | experimental.difficulty_normalized | 0.4118 | 0.7834 | 0.1103 | 0.6952 | 0.8021 | nan |
| wineW | C | experimental.difficulty_normalized | 0.4661 | 0.8100 | 0.1447 | 0.7067 | 0.7478 | nan |
| yeast | C | experimental.difficulty_normalized | 0.3695 | 0.6464 | 0.0383 | 0.6534 | 0.8192 | nan |
| balance | D | experimental.difficulty_normalized | 0.1317 | 0.7304 | -0.1056 | 0.8048 | 0.9359 | nan |
| breast_cancer | D | experimental.difficulty_normalized | 0.4135 | 0.9203 | -0.0359 | 0.3232 | 0.9948 | 0.9029 |
| cars | D | experimental.difficulty_normalized | 0.7806 | 0.9771 | 0.0112 | 0.4903 | 0.4142 | nan |
| cmc | D | experimental.difficulty_normalized | 0.2519 | 0.4973 | -0.0119 | 0.7359 | 0.9673 | nan |
| colic | D | experimental.difficulty_normalized | 0.2559 | 0.8046 | -0.0149 | 0.7227 | 0.9800 | 0.8981 |
| cool | D | experimental.difficulty_normalized | 0.0860 | 0.9439 | -0.0056 | 0.8993 | 0.9135 | nan |
| creditA | D | experimental.difficulty_normalized | 0.2393 | 0.8153 | -0.0421 | 0.6885 | 0.9862 | 0.8878 |
| diabetes | D | experimental.difficulty_normalized | 0.4449 | 0.7560 | 0.0003 | 0.5720 | 0.9946 | 0.8658 |
| ecoli | D | experimental.difficulty_normalized | 0.1575 | 0.8047 | -0.0536 | 0.7713 | 0.8118 | nan |
| german | D | experimental.difficulty_normalized | 0.4539 | 0.7523 | 0.0946 | 0.6345 | 0.8757 | 0.8856 |
| glass | D | experimental.difficulty_normalized | 0.1907 | 0.6964 | -0.0661 | 0.7856 | 0.9513 | nan |
| haberman | D | experimental.difficulty_normalized | 0.4651 | 0.7630 | 0.1174 | 0.6689 | 0.8479 | 0.8713 |
| heartC | D | experimental.difficulty_normalized | 0.3013 | 0.8074 | 0.0074 | 0.7096 | 0.8452 | 0.8557 |
| heartH | D | experimental.difficulty_normalized | 0.3409 | 0.8875 | 0.0569 | 0.7494 | 0.8833 | 0.8934 |
| heartS | D | experimental.difficulty_normalized | 0.1757 | 0.5974 | -0.1890 | 0.7622 | 0.8966 | 0.9058 |
| heat | D | experimental.difficulty_normalized | 0.2186 | 0.9842 | -0.0093 | 0.3333 | 0.9649 | nan |
| hepati | D | experimental.difficulty_normalized | 0.6753 | 0.9209 | 0.0693 | 0.6089 | 0.8962 | 0.9104 |
| image | D | experimental.difficulty_normalized | 0.5466 | 0.9669 | -0.0085 | 0.3957 | 0.7865 | nan |
| iono | D | experimental.difficulty_normalized | 0.3971 | 0.9455 | 0.0169 | 0.6658 | 0.9171 | 0.8924 |
| iris | D | experimental.difficulty_normalized | 0.4244 | 0.9409 | -0.0068 | 0.3556 | 0.9834 | nan |
| je4042 | D | experimental.difficulty_normalized | 0.2412 | 0.7047 | -0.0397 | 0.7400 | 0.9213 | 0.9000 |
| je4243 | D | experimental.difficulty_normalized | 0.2332 | 0.6704 | 0.0512 | 0.7989 | 0.9497 | 0.8819 |
| kc1 | D | experimental.difficulty_normalized | 0.4776 | 0.7742 | 0.0269 | 0.5873 | 0.9400 | 0.8874 |
| kc2 | D | experimental.difficulty_normalized | 0.5859 | 0.8105 | 0.0159 | 0.4362 | 0.7600 | 0.8739 |
| kc3 | D | experimental.difficulty_normalized | 0.6923 | 0.9159 | 0.0544 | 0.5450 | 0.6511 | 0.9080 |
| liver | D | experimental.difficulty_normalized | 0.1797 | 0.6806 | 0.0169 | 0.8438 | 0.9864 | 0.8995 |
| pc1req | D | experimental.difficulty_normalized | 0.1926 | 0.6011 | -0.0656 | 0.7967 | 0.9558 | 0.8878 |
| pc4 | D | experimental.difficulty_normalized | 0.7951 | 0.9364 | 0.0405 | 0.4870 | 0.5474 | 0.8916 |
| sonar | D | experimental.difficulty_normalized | 0.2116 | 0.8817 | 0.0008 | 0.7911 | 0.9547 | 0.8852 |
| spect | D | experimental.difficulty_normalized | 0.5970 | 0.8587 | -0.0230 | 0.3844 | 0.4607 | 0.8965 |
| spectf | D | experimental.difficulty_normalized | 0.6025 | 0.8572 | 0.0461 | 0.5030 | 0.5759 | 0.8366 |
| steel | D | experimental.difficulty_normalized | 0.2944 | 0.6891 | -0.0728 | 0.6237 | 0.9997 | nan |
| tae | D | experimental.difficulty_normalized | 0.1599 | 0.4913 | -0.0503 | 0.8260 | 0.9848 | nan |
| transfusion | D | experimental.difficulty_normalized | 0.3677 | 0.6880 | -0.0348 | 0.5826 | 0.9939 | 0.8935 |
| ttt | D | experimental.difficulty_normalized | 0.5541 | 0.9464 | -0.0297 | 0.1653 | 0.6101 | 0.8883 |
| user | D | experimental.difficulty_normalized | 0.1942 | 0.9414 | 0.0525 | 0.8844 | 0.9999 | nan |
| vehicle | D | experimental.difficulty_normalized | 0.2902 | 0.7165 | -0.0290 | 0.7242 | 0.9976 | nan |
| vote | D | experimental.difficulty_normalized | 0.5474 | 0.8219 | -0.0127 | 0.4952 | 0.9216 | 0.8936 |
| vowel | D | experimental.difficulty_normalized | 0.7737 | 0.9284 | -0.0141 | 0.1340 | 0.8144 | nan |
| wave | D | experimental.difficulty_normalized | 0.2452 | 0.9182 | 0.0618 | 0.8535 | 0.9464 | nan |
| wbc | D | experimental.difficulty_normalized | 0.1747 | 0.7721 | -0.1892 | 0.3611 | 0.9302 | 0.8913 |
| whole | D | experimental.difficulty_normalized | 0.3482 | 0.7274 | 0.0274 | 0.6954 | 0.8307 | nan |
| wine | D | experimental.difficulty_normalized | 0.1840 | 0.9731 | 0.0101 | 0.9583 | 0.9676 | nan |
| wineR | D | experimental.difficulty_normalized | 0.1797 | 0.6966 | 0.0235 | 0.8409 | 0.8875 | nan |
| wineW | D | experimental.difficulty_normalized | 0.2720 | 0.6979 | 0.0326 | 0.7487 | 0.8832 | nan |
| yeast | D | experimental.difficulty_normalized | 0.2344 | 0.5997 | -0.0091 | 0.7348 | 0.9962 | nan |
| balance | E | experimental.difficulty_normalized | 0.6667 | 0.9695 | 0.1327 | 0.8730 | 0.4271 | nan |
| breast_cancer | E | experimental.difficulty_normalized | 0.5045 | 0.9662 | 0.0101 | 0.5919 | 0.5565 | 0.8819 |
| cars | E | experimental.difficulty_normalized | 0.8441 | 0.9837 | 0.0178 | 0.6242 | 0.1982 | nan |
| cmc | E | experimental.difficulty_normalized | 0.1578 | 0.7169 | 0.2078 | 0.9026 | 0.4642 | nan |
| colic | E | experimental.difficulty_normalized | 0.4630 | 0.8458 | 0.0263 | 0.5856 | 0.4235 | 0.8623 |
| cool | E | experimental.difficulty_normalized | 0.7874 | 0.9815 | 0.0322 | 0.6973 | 0.3618 | nan |
| creditA | E | experimental.difficulty_normalized | 0.3480 | 0.9017 | 0.0438 | 0.7652 | 0.4138 | 0.8676 |
| diabetes | E | experimental.difficulty_normalized | 0.3558 | 0.8801 | 0.1242 | 0.8169 | 0.7233 | 0.9022 |
| ecoli | E | experimental.difficulty_normalized | 0.3598 | 0.9799 | 0.1210 | 0.9590 | 0.5688 | nan |
| german | E | experimental.difficulty_normalized | 0.2093 | 0.5829 | -0.0748 | 0.7738 | 0.6629 | 0.8869 |
| glass | E | experimental.difficulty_normalized | 0.2000 | 0.9133 | 0.1505 | 0.9193 | 0.5497 | nan |
| haberman | E | experimental.difficulty_normalized | 0.0791 | 0.5097 | -0.1431 | 0.9066 | 0.6717 | 0.9166 |
| heartC | E | experimental.difficulty_normalized | 0.1774 | 0.9026 | 0.1026 | 0.8997 | 0.3772 | 0.8991 |
| heartH | E | experimental.difficulty_normalized | 0.1092 | 0.7589 | -0.0716 | 0.8527 | 0.5076 | 0.9156 |
| heartS | E | experimental.difficulty_normalized | 0.3062 | 0.8358 | 0.0506 | 0.7328 | 0.6383 | 0.9000 |
| heat | E | experimental.difficulty_normalized | 0.8156 | 0.9974 | 0.0039 | 0.7083 | 0.3147 | nan |
| hepati | E | experimental.difficulty_normalized | 0.0473 | 0.7478 | -0.1071 | 0.8843 | 0.2207 | 0.9176 |
| image | E | experimental.difficulty_normalized | 0.8077 | 0.9968 | 0.0215 | 0.9188 | 0.2594 | nan |
| iono | E | experimental.difficulty_normalized | 0.1638 | 0.9503 | 0.0217 | 0.8899 | 0.0821 | 0.8838 |
| iris | E | experimental.difficulty_normalized | 0.4044 | 0.9901 | 0.0434 | 0.9556 | 0.4829 | nan |
| je4042 | E | experimental.difficulty_normalized | 0.2683 | 0.7119 | -0.0325 | 0.6917 | 0.4605 | 0.8683 |
| je4243 | E | experimental.difficulty_normalized | 0.1799 | 0.5394 | -0.0798 | 0.7790 | 0.4843 | 0.8664 |
| kc1 | E | experimental.difficulty_normalized | 0.1225 | 0.6771 | -0.0702 | 0.8497 | 0.5538 | 0.8919 |
| kc2 | E | experimental.difficulty_normalized | 0.0925 | 0.6720 | -0.1235 | 0.8368 | 0.4051 | 0.8796 |
| kc3 | E | experimental.difficulty_normalized | 0.2533 | 0.7000 | -0.1604 | 0.7183 | 0.5314 | 0.8855 |
| liver | E | experimental.difficulty_normalized | 0.2944 | 0.7893 | 0.1256 | 0.8038 | 0.6609 | 0.8937 |
| pc1req | E | experimental.difficulty_normalized | 0.2074 | 0.7419 | 0.0752 | 0.8011 | 0.5192 | 0.9153 |
| pc4 | E | experimental.difficulty_normalized | 0.1485 | 0.6947 | -0.2012 | 0.7736 | 0.5326 | 0.9040 |
| sonar | E | experimental.difficulty_normalized | 0.2667 | 0.8824 | 0.0014 | 0.7621 | 0.4175 | 0.8910 |
| spect | E | experimental.difficulty_normalized | 0.2864 | 0.8043 | -0.0775 | 0.5867 | 0.6764 | 0.9045 |
| spectf | E | experimental.difficulty_normalized | 0.0556 | 0.7123 | -0.0989 | 0.9076 | 0.6073 | 0.8650 |
| steel | E | experimental.difficulty_normalized | 0.4907 | 0.9096 | 0.1476 | 0.8111 | 0.3929 | nan |
| tae | E | experimental.difficulty_normalized | 0.1584 | 0.4950 | -0.0408 | 0.8408 | 0.5502 | nan |
| transfusion | E | experimental.difficulty_normalized | 0.4108 | 0.8092 | 0.0864 | 0.6999 | 0.6822 | 0.8911 |
| ttt | E | experimental.difficulty_normalized | 0.2368 | 0.9622 | -0.0139 | 0.7187 | 0.4218 | 0.8986 |
| user | E | experimental.difficulty_normalized | 0.5520 | 0.9480 | 0.0591 | 0.7146 | 0.5491 | nan |
| vehicle | E | experimental.difficulty_normalized | 0.3658 | 0.9611 | 0.2152 | 0.9428 | 0.5351 | nan |
| vote | E | experimental.difficulty_normalized | 0.5103 | 0.9215 | 0.0869 | 0.7413 | 0.6802 | 0.8767 |
| vowel | E | experimental.difficulty_normalized | 0.5308 | 0.9966 | 0.0542 | 0.9625 | 0.4545 | nan |
| wave | E | experimental.difficulty_normalized | 0.6778 | 0.9297 | 0.0733 | 0.6627 | 0.5867 | nan |
| wbc | E | experimental.difficulty_normalized | 0.3491 | 0.9760 | 0.0147 | 0.7981 | 0.1720 | 0.9202 |
| whole | E | experimental.difficulty_normalized | 0.3735 | 0.7422 | 0.0422 | 0.6763 | 0.8134 | nan |
| wine | E | experimental.difficulty_normalized | 0.0593 | 1.0000 | 0.0332 | 1.0000 | 0.8666 | nan |
| wineR | E | experimental.difficulty_normalized | 0.2876 | 0.8290 | 0.1559 | 0.8415 | 0.6303 | nan |
| wineW | E | experimental.difficulty_normalized | 0.3102 | 0.8560 | 0.1907 | 0.8617 | 0.5516 | nan |
| yeast | E | experimental.difficulty_normalized | 0.2476 | 0.7292 | 0.1211 | 0.8236 | 0.5323 | nan |
| balance | F | experimental.difficulty_normalized | 0.0612 | 0.6970 | -0.1395 | 0.8936 | 0.9228 | nan |
| breast_cancer | F | experimental.difficulty_normalized | 0.3739 | 0.9209 | -0.0353 | 0.3823 | 0.9341 | 0.8963 |
| cars | F | experimental.difficulty_normalized | 0.6863 | 0.9837 | 0.0178 | 0.6877 | 0.4019 | nan |
| cmc | F | experimental.difficulty_normalized | 0.1147 | 0.6343 | 0.1251 | 0.9138 | 0.5296 | nan |
| colic | F | experimental.difficulty_normalized | 0.2179 | 0.8272 | 0.0077 | 0.7808 | 0.9302 | 0.9034 |
| cool | F | experimental.difficulty_normalized | 0.0795 | 0.9061 | -0.0426 | 0.8847 | 0.8627 | nan |
| creditA | F | experimental.difficulty_normalized | 0.2345 | 0.7875 | -0.0706 | 0.6743 | 0.9510 | 0.8902 |
| diabetes | F | experimental.difficulty_normalized | 0.4238 | 0.7524 | -0.0034 | 0.5923 | 0.9325 | 0.8794 |
| ecoli | F | experimental.difficulty_normalized | 0.0582 | 0.9744 | 0.1158 | 0.9902 | 0.8939 | nan |
| german | F | experimental.difficulty_normalized | 0.1328 | 0.7023 | 0.0460 | 0.8749 | 0.7456 | 0.9039 |
| glass | F | experimental.difficulty_normalized | 0.1054 | 0.8686 | 0.1044 | 0.9248 | 0.8738 | nan |
| haberman | F | experimental.difficulty_normalized | 0.1805 | 0.7411 | 0.0944 | 0.8284 | 0.8970 | 0.9177 |
| heartC | F | experimental.difficulty_normalized | 0.0933 | 0.5156 | -0.2838 | 0.8525 | 0.6413 | 0.8798 |
| heartH | F | experimental.difficulty_normalized | 0.1119 | 0.7586 | -0.0719 | 0.8241 | 0.7159 | 0.8976 |
| heartS | F | experimental.difficulty_normalized | 0.1218 | 0.5882 | -0.1974 | 0.8285 | 0.6543 | 0.9226 |
| heat | F | experimental.difficulty_normalized | 0.1089 | 1.0000 | 0.0065 | 1.0000 | 0.6132 | nan |
| hepati | F | experimental.difficulty_normalized | 0.1670 | 0.7631 | -0.0876 | 0.7137 | 0.5268 | 0.9061 |
| image | F | experimental.difficulty_normalized | 0.5489 | 0.9664 | -0.0089 | 0.3683 | 0.9991 | nan |
| iono | F | experimental.difficulty_normalized | 0.2632 | 0.8635 | -0.0650 | 0.6366 | 0.8028 | 0.8790 |
| iris | F | experimental.difficulty_normalized | 0.4126 | 0.9466 | -0.0034 | 0.4000 | 0.9925 | nan |
| je4042 | F | experimental.difficulty_normalized | 0.0848 | 0.7533 | 0.0115 | 0.9357 | 0.6729 | 0.9004 |
| je4243 | F | experimental.difficulty_normalized | 0.0974 | 0.6397 | 0.0205 | 0.9071 | 0.8086 | 0.8980 |
| kc1 | F | experimental.difficulty_normalized | 0.3258 | 0.7543 | 0.0070 | 0.6868 | 0.9441 | 0.9028 |
| kc2 | F | experimental.difficulty_normalized | 0.1733 | 0.6859 | -0.1087 | 0.7622 | 0.9132 | 0.8985 |
| kc3 | F | experimental.difficulty_normalized | 0.3819 | 0.8522 | -0.0131 | 0.6728 | 0.6755 | 0.9108 |
| liver | F | experimental.difficulty_normalized | 0.1340 | 0.6089 | -0.0544 | 0.8694 | 0.8335 | 0.9130 |
| pc1req | F | experimental.difficulty_normalized | 0.1228 | 0.5929 | -0.0737 | 0.8517 | 0.7873 | 0.8952 |
| pc4 | F | experimental.difficulty_normalized | 0.1302 | 0.7222 | -0.1739 | 0.8288 | 0.7004 | 0.8962 |
| sonar | F | experimental.difficulty_normalized | 0.1704 | 0.7843 | -0.0992 | 0.7653 | 0.8469 | 0.8931 |
| spect | F | experimental.difficulty_normalized | 0.2747 | 0.8011 | -0.0810 | 0.6356 | 0.7033 | 0.9056 |
| spectf | F | experimental.difficulty_normalized | 0.0922 | 0.6675 | -0.1514 | 0.8475 | 0.6528 | 0.8580 |
| steel | F | experimental.difficulty_normalized | 0.2925 | 0.6841 | -0.0779 | 0.6208 | 0.9988 | nan |
| tae | F | experimental.difficulty_normalized | 0.1771 | 0.5533 | 0.0185 | 0.8197 | 0.8903 | nan |
| transfusion | F | experimental.difficulty_normalized | 0.3540 | 0.6906 | -0.0318 | 0.5921 | 0.9664 | 0.8944 |
| ttt | F | experimental.difficulty_normalized | 0.0880 | 0.8939 | -0.0821 | 0.6646 | 0.6530 | 0.8978 |
| user | F | experimental.difficulty_normalized | 0.1888 | 0.9400 | 0.0511 | 0.8863 | 0.9982 | nan |
| vehicle | F | experimental.difficulty_normalized | 0.2843 | 0.7277 | -0.0181 | 0.7171 | 0.9834 | nan |
| vote | F | experimental.difficulty_normalized | 0.5472 | 0.8219 | -0.0127 | 0.4952 | 0.9145 | 0.8936 |
| vowel | F | experimental.difficulty_normalized | 0.7545 | 0.9364 | -0.0060 | 0.1735 | 0.8533 | nan |
| wave | F | experimental.difficulty_normalized | 0.2380 | 0.9207 | 0.0643 | 0.8508 | 0.8943 | nan |
| wbc | F | experimental.difficulty_normalized | 0.0734 | 0.9011 | -0.0647 | 0.9093 | 0.8024 | 0.8834 |
| whole | F | experimental.difficulty_normalized | 0.3548 | 0.7639 | 0.0639 | 0.7207 | 0.6726 | nan |
| wine | F | experimental.difficulty_normalized | 0.1494 | 0.9686 | 0.0099 | 0.9444 | 0.8404 | nan |
| wineR | F | experimental.difficulty_normalized | 0.1785 | 0.8219 | 0.1488 | 0.8934 | 0.5809 | nan |
| wineW | F | experimental.difficulty_normalized | 0.1342 | 0.7804 | 0.1151 | 0.9101 | 0.4667 | nan |
| yeast | F | experimental.difficulty_normalized | 0.2186 | 0.6145 | 0.0065 | 0.7617 | 0.9592 | nan |


## Metric Consistency Note (RT-5)

Scenario 9 reports a full-grid A-vs-C difficulty_reject_auc delta of +0.1985, while Scenario 11 reports +0.0155 at matched operating points. This reduction is a selection effect, not a contradiction:

- Scenario 9 averages over all confidence values. The positive delta is strongest in high-confidence rows (conf >= 0.91), where the AUC delta is +0.2492.
- At moderate confidence (conf < 0.91), the Scenario 9 AUC delta is +0.1637: smaller than the high-confidence tail, but still positive.
- Scenario 11 targets reject rates of 10-40%, uses matched operating-point selection, and reduces the observed difficulty-AUC effect to +0.0155.

Conclusion: the full-grid Scenario 9 AUC advantage is not sufficient evidence for public promotion. At matched operating points, accepted-accuracy gains are tiny or negative and the difficulty-selection advantage is much smaller.
