# Scenario 10 - Ambiguity-normalized novelty-penalized reject strategy

Rows: 6210

## Key findings

- Compares built-in default, direct difficulty-normalized, and novelty-aware experimental reject strategies.
- Primary contrast is C vs G: direct difficulty normalization with and without novelty penalty.
- Novelty estimator is deterministic, fitted on proper-training features only, and uses no calibration labels/residuals.
- G vs C changed novelty_rate by +0.0036, empty_rate by +0.0036, and novelty_reject_auc by -0.0302.
- G vs C changed ambiguity_rate by +0.0065 and multilabel_rate by +0.0065.
- G vs C changed accepted_accuracy by +0.0041.
- Recommended arm for next iteration: C (difficulty-normalized remains the simpler experimental baseline).

## Outcome snapshot

- **rows**: 6210
- **datasets**: 46
- **seeds**: 5
- **novelty_weight**: 0.1000
- **mean_accept_rate**: 0.6090
- **mean_accuracy_delta**: 0.0514
- **G_minus_C_novelty_rate_delta**: 0.0036
- **G_minus_C_empty_rate_delta**: 0.0036
- **G_minus_C_ambiguity_rate_delta**: 0.0065
- **G_minus_C_multilabel_rate_delta**: 0.0065
- **G_minus_C_accepted_accuracy_delta**: 0.0041
- **G_minus_C_novelty_reject_auc_delta**: -0.0302
- **recommended_arm**: C
- **recommendation_reason**: difficulty-normalized remains the simpler experimental baseline

## Result table

| task_type | dataset | confidence | epsilon | n_train | n_cal | n_test | arm_code | arm_label | ncf | strategy | difficulty_normalized | novelty_penalized | novelty_weight | accept_rate | reject_rate | ambiguity_rate | novelty_rate | accepted_accuracy | full_accuracy | accuracy_delta | singleton_error_rate | error_rate_defined | rejected_error_capture_rate | mean_difficulty_all | mean_difficulty_accepted | mean_difficulty_rejected | difficulty_gap_rejected_minus_accepted | difficulty_reject_auc | mean_novelty_all | mean_novelty_accepted | mean_novelty_rejected | novelty_gap_rejected_minus_accepted | novelty_reject_auc | empty_rate | singleton_rate | multilabel_rate | empirical_coverage | coverage_gap | coverage_defined | seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| binary | breast_cancer | 0.8000 | 0.2000 | 341 | 114 | 114 | A | A|strategy=builtin.default|ncf=default|novelty_weight=0.0 | default | builtin.default | no | no | 0.0000 | 0.8333 | 0.1667 | 0.0000 | 0.1667 | 1.0000 | 0.9561 | 0.0439 | 0.0400 | yes | 1.0000 | 1.8789 | 1.9130 | 1.7080 | -0.2050 | 0.3501 | 0.1343 | 0.1592 | 0.0099 | -0.1494 | 0.4116 | 0.1667 | 0.8333 | 0.0000 | 0.8333 | 0.0333 | yes | 42 |
| binary | breast_cancer | 0.8237 | 0.1763 | 341 | 114 | 114 | A | A|strategy=builtin.default|ncf=default|novelty_weight=0.0 | default | builtin.default | no | no | 0.0000 | 0.8684 | 0.1316 | 0.0000 | 0.1316 | 0.9899 | 0.9561 | 0.0338 | 0.0514 | yes | 0.8000 | 1.8789 | 1.9139 | 1.6473 | -0.2666 | 0.2700 | 0.1343 | 0.1542 | 0.0030 | -0.1513 | 0.3946 | 0.1316 | 0.8684 | 0.0000 | 0.8596 | 0.0359 | yes | 42 |
| binary | breast_cancer | 0.8475 | 0.1525 | 341 | 114 | 114 | A | A|strategy=builtin.default|ncf=default|novelty_weight=0.0 | default | builtin.default | no | no | 0.0000 | 0.9035 | 0.0965 | 0.0000 | 0.0965 | 0.9903 | 0.9561 | 0.0342 | 0.0620 | yes | 0.8000 | 1.8789 | 1.9035 | 1.6480 | -0.2555 | 0.2701 | 0.1343 | 0.1482 | 0.0041 | -0.1442 | 0.4095 | 0.0965 | 0.9035 | 0.0000 | 0.8947 | 0.0472 | yes | 42 |
| binary | breast_cancer | 0.8713 | 0.1287 | 341 | 114 | 114 | A | A|strategy=builtin.default|ncf=default|novelty_weight=0.0 | default | builtin.default | no | no | 0.0000 | 0.9211 | 0.0789 | 0.0000 | 0.0789 | 0.9810 | 0.9561 | 0.0248 | 0.0541 | yes | 0.6000 | 1.8789 | 1.8990 | 1.6439 | -0.2551 | 0.2656 | 0.1343 | 0.1454 | 0.0050 | -0.1405 | 0.4201 | 0.0789 | 0.9211 | 0.0000 | 0.9035 | 0.0323 | yes | 42 |
| binary | breast_cancer | 0.8950 | 0.1050 | 341 | 114 | 114 | A | A|strategy=builtin.default|ncf=default|novelty_weight=0.0 | default | builtin.default | no | no | 0.0000 | 0.9298 | 0.0702 | 0.0000 | 0.0702 | 0.9811 | 0.9561 | 0.0250 | 0.0375 | yes | 0.6000 | 1.8789 | 1.8996 | 1.6038 | -0.2958 | 0.2146 | 0.1343 | 0.1441 | 0.0056 | -0.1385 | 0.4269 | 0.0702 | 0.9298 | 0.0000 | 0.9123 | 0.0173 | yes | 42 |
| binary | breast_cancer | 0.9187 | 0.0813 | 341 | 114 | 114 | A | A|strategy=builtin.default|ncf=default|novelty_weight=0.0 | default | builtin.default | no | no | 0.0000 | 0.9474 | 0.0526 | 0.0000 | 0.0526 | 0.9722 | 0.9561 | 0.0161 | 0.0302 | yes | 0.4000 | 1.8789 | 1.8934 | 1.6172 | -0.2762 | 0.2407 | 0.1343 | 0.1414 | 0.0075 | -0.1339 | 0.4460 | 0.0526 | 0.9474 | 0.0000 | 0.9211 | 0.0023 | yes | 42 |
| binary | breast_cancer | 0.9425 | 0.0575 | 341 | 114 | 114 | A | A|strategy=builtin.default|ncf=default|novelty_weight=0.0 | default | builtin.default | no | no | 0.0000 | 0.9474 | 0.0526 | 0.0000 | 0.0526 | 0.9722 | 0.9561 | 0.0161 | 0.0051 | yes | 0.4000 | 1.8789 | 1.8934 | 1.6172 | -0.2762 | 0.2407 | 0.1343 | 0.1414 | 0.0075 | -0.1339 | 0.4460 | 0.0526 | 0.9474 | 0.0000 | 0.9211 | -0.0214 | yes | 42 |
| binary | breast_cancer | 0.9663 | 0.0337 | 341 | 114 | 114 | A | A|strategy=builtin.default|ncf=default|novelty_weight=0.0 | default | builtin.default | no | no | 0.0000 | 0.9912 | 0.0088 | 0.0000 | 0.0088 | 0.9646 | 0.9561 | 0.0085 | 0.0252 | yes | 0.2000 | 1.8789 | 1.8820 | 1.5238 | -0.3582 | 0.0796 | 0.1343 | 0.1355 | 0.0000 | -0.1355 | 0.3805 | 0.0088 | 0.9912 | 0.0000 | 0.9561 | -0.0101 | yes | 42 |
| binary | breast_cancer | 0.9900 | 0.0100 | 341 | 114 | 114 | A | A|strategy=builtin.default|ncf=default|novelty_weight=0.0 | default | builtin.default | no | no | 0.0000 | 0.9474 | 0.0526 | 0.0526 | 0.0000 | 0.9722 | 0.9561 | 0.0161 | 0.0106 | yes | 0.4000 | 1.8789 | 1.8934 | 1.6172 | -0.2762 | 0.2407 | 0.1343 | 0.1414 | 0.0075 | -0.1339 | 0.4460 | 0.0000 | 0.9474 | 0.0526 | 0.9737 | -0.0163 | yes | 42 |
| binary | breast_cancer | 0.8000 | 0.2000 | 341 | 114 | 114 | C | C|strategy=experimental.difficulty_normalized|ncf=default|novelty_weight=0.0 | default | experimental.difficulty_normalized | yes | no | 0.0000 | 0.7895 | 0.2105 | 0.0000 | 0.2105 | 1.0000 | 0.9561 | 0.0439 | 0.0000 | yes | 1.0000 | 1.8789 | 1.9294 | 1.6893 | -0.2401 | 0.3162 | 0.1343 | 0.1681 | 0.0078 | -0.1603 | 0.3949 | 0.2105 | 0.7895 | 0.0000 | 0.7895 | -0.0105 | yes | 42 |
| binary | breast_cancer | 0.8237 | 0.1763 | 341 | 114 | 114 | C | C|strategy=experimental.difficulty_normalized|ncf=default|novelty_weight=0.0 | default | experimental.difficulty_normalized | yes | no | 0.0000 | 0.8158 | 0.1842 | 0.0000 | 0.1842 | 1.0000 | 0.9561 | 0.0439 | 0.0000 | yes | 1.0000 | 1.8789 | 1.9220 | 1.6879 | -0.2341 | 0.3221 | 0.1343 | 0.1627 | 0.0089 | -0.1537 | 0.4045 | 0.1842 | 0.8158 | 0.0000 | 0.8158 | -0.0080 | yes | 42 |
| binary | breast_cancer | 0.8475 | 0.1525 | 341 | 114 | 114 | C | C|strategy=experimental.difficulty_normalized|ncf=default|novelty_weight=0.0 | default | experimental.difficulty_normalized | yes | no | 0.0000 | 0.8684 | 0.1316 | 0.0000 | 0.1316 | 0.9899 | 0.9561 | 0.0338 | 0.0241 | yes | 0.8000 | 1.8789 | 1.9139 | 1.6473 | -0.2666 | 0.2700 | 0.1343 | 0.1542 | 0.0030 | -0.1513 | 0.3946 | 0.1316 | 0.8684 | 0.0000 | 0.8596 | 0.0121 | yes | 42 |

_Showing first 12 of 6210 rows._

## Arm Summary

| arm_code | strategy | ncf | difficulty_normalized | novelty_penalized | novelty_weight | accept_rate | reject_rate | accepted_accuracy | accuracy_delta | ambiguity_rate | novelty_rate | empty_rate | multilabel_rate | rejected_error_capture_rate | difficulty_gap_rejected_minus_accepted | difficulty_reject_auc | novelty_gap_rejected_minus_accepted | novelty_reject_auc | empirical_coverage | coverage_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | builtin.default | default | no | no | 0.0000 | 0.6332 | 0.3668 | 0.8614 | 0.0551 | 0.3431 | 0.0237 | 0.0237 | 0.3431 | 0.5878 | 0.0100 | 0.4986 | 0.0255 | 0.5076 | 0.8965 | 0.0015 |
| C | experimental.difficulty_normalized | default | yes | no | 0.0000 | 0.6019 | 0.3981 | 0.8548 | 0.0478 | 0.3648 | 0.0333 | 0.0333 | 0.3648 | 0.6123 | 0.3049 | 0.6971 | 0.3412 | 0.6369 | 0.8840 | -0.0110 |
| G | experimental.ambiguity_normalized_novelty_penalized | default | yes | yes | 0.1000 | 0.5918 | 0.4082 | 0.8589 | 0.0512 | 0.3713 | 0.0369 | 0.0369 | 0.3713 | 0.6093 | 0.2466 | 0.6824 | 0.2623 | 0.6067 | 0.8845 | -0.0105 |

## By Confidence And Arm

| confidence | epsilon | arm_code | strategy | reject_rate | accepted_accuracy | ambiguity_rate | novelty_rate | novelty_gap_rejected_minus_accepted | novelty_reject_auc |
|---|---|---|---|---|---|---|---|---|---|
| 0.8000 | 0.2000 | A | builtin.default | 0.1982 | 0.8349 | 0.1341 | 0.0641 | 0.0242 | 0.5056 |
| 0.8237 | 0.1763 | A | builtin.default | 0.2129 | 0.8323 | 0.1625 | 0.0504 | 0.0241 | 0.5064 |
| 0.8475 | 0.1525 | A | builtin.default | 0.2312 | 0.8349 | 0.1936 | 0.0375 | 0.0166 | 0.5063 |
| 0.8713 | 0.1287 | A | builtin.default | 0.2716 | 0.8413 | 0.2449 | 0.0267 | 0.0233 | 0.5097 |
| 0.8950 | 0.1050 | A | builtin.default | 0.3023 | 0.8473 | 0.2859 | 0.0164 | 0.0316 | 0.5134 |
| 0.9187 | 0.0813 | A | builtin.default | 0.3578 | 0.8652 | 0.3477 | 0.0101 | 0.0309 | 0.5073 |
| 0.9425 | 0.0575 | A | builtin.default | 0.4224 | 0.8758 | 0.4165 | 0.0059 | 0.0266 | 0.5066 |
| 0.9663 | 0.0337 | A | builtin.default | 0.5188 | 0.9130 | 0.5166 | 0.0022 | 0.0311 | 0.5090 |
| 0.9900 | 0.0100 | A | builtin.default | 0.7859 | 0.9681 | 0.7859 | 0.0000 | 0.0170 | 0.4991 |
| 0.8000 | 0.2000 | C | experimental.difficulty_normalized | 0.2415 | 0.8365 | 0.1529 | 0.0886 | 0.3477 | 0.6148 |
| 0.8237 | 0.1763 | C | experimental.difficulty_normalized | 0.2507 | 0.8352 | 0.1801 | 0.0706 | 0.3498 | 0.6186 |
| 0.8475 | 0.1525 | C | experimental.difficulty_normalized | 0.2636 | 0.8387 | 0.2116 | 0.0521 | 0.3605 | 0.6266 |
| 0.8713 | 0.1287 | C | experimental.difficulty_normalized | 0.2922 | 0.8428 | 0.2558 | 0.0364 | 0.3670 | 0.6407 |
| 0.8950 | 0.1050 | C | experimental.difficulty_normalized | 0.3253 | 0.8471 | 0.3004 | 0.0249 | 0.3575 | 0.6462 |
| 0.9187 | 0.0813 | C | experimental.difficulty_normalized | 0.3779 | 0.8572 | 0.3628 | 0.0151 | 0.3579 | 0.6481 |
| 0.9425 | 0.0575 | C | experimental.difficulty_normalized | 0.4536 | 0.8663 | 0.4451 | 0.0086 | 0.3013 | 0.6427 |
| 0.9663 | 0.0337 | C | experimental.difficulty_normalized | 0.5607 | 0.8842 | 0.5571 | 0.0035 | 0.3140 | 0.6528 |
| 0.9900 | 0.0100 | C | experimental.difficulty_normalized | 0.8170 | 0.9218 | 0.8170 | 0.0000 | 0.2828 | 0.6481 |
| 0.8000 | 0.2000 | G | experimental.ambiguity_normalized_novelty_penalized | 0.2369 | 0.8357 | 0.1494 | 0.0875 | 0.4939 | 0.6525 |
| 0.8237 | 0.1763 | G | experimental.ambiguity_normalized_novelty_penalized | 0.2545 | 0.8398 | 0.1816 | 0.0729 | 0.4565 | 0.6492 |
| 0.8475 | 0.1525 | G | experimental.ambiguity_normalized_novelty_penalized | 0.2681 | 0.8426 | 0.2103 | 0.0578 | 0.4139 | 0.6323 |
| 0.8713 | 0.1287 | G | experimental.ambiguity_normalized_novelty_penalized | 0.2986 | 0.8461 | 0.2559 | 0.0428 | 0.3407 | 0.6162 |
| 0.8950 | 0.1050 | G | experimental.ambiguity_normalized_novelty_penalized | 0.3359 | 0.8532 | 0.3053 | 0.0307 | 0.2851 | 0.6040 |
| 0.9187 | 0.0813 | G | experimental.ambiguity_normalized_novelty_penalized | 0.3950 | 0.8667 | 0.3739 | 0.0210 | 0.2404 | 0.5928 |
| 0.9425 | 0.0575 | G | experimental.ambiguity_normalized_novelty_penalized | 0.4663 | 0.8702 | 0.4536 | 0.0127 | 0.1687 | 0.5831 |
| 0.9663 | 0.0337 | G | experimental.ambiguity_normalized_novelty_penalized | 0.5847 | 0.8927 | 0.5784 | 0.0063 | -0.0148 | 0.5631 |
| 0.9900 | 0.0100 | G | experimental.ambiguity_normalized_novelty_penalized | 0.8338 | 0.9186 | 0.8332 | 0.0006 | -0.4244 | 0.5095 |

## Required Analyses

1. Does novelty penalization increase novelty/empty-set rejection relative to C?
G vs C changed novelty_rate by +0.0036, empty_rate by +0.0036, and novelty_reject_auc by -0.0302.
2. Does it reduce ambiguity/multi-label rejection relative to C?
G vs C changed ambiguity_rate by +0.0065 and multilabel_rate by +0.0065.
3. Does it preserve accepted accuracy relative to C?
G vs C changed accepted_accuracy by +0.0041.
4. Which arm is recommended for further development?
Recommended arm for next iteration: C (difficulty-normalized remains the simpler experimental baseline).
