# Scenario 10 - Ambiguity-normalized novelty-penalized reject strategy

Rows: 6210

## Key findings

- Compares built-in default, direct difficulty-normalized, and novelty-aware experimental reject strategies.
- Primary contrast is C vs G: direct difficulty normalization with and without novelty penalty.
- Novelty estimator is deterministic, fitted on proper-training features only, and uses no calibration labels/residuals.
- G vs C changed novelty_rate by +0.0019, empty_rate by +0.0019, and novelty_reject_auc by -0.0371.
- G vs C changed ambiguity_rate by +0.0047 and multilabel_rate by +0.0047.
- G vs C changed accepted_accuracy by +0.0037.
- Recommended arm for next iteration: C (difficulty-normalized remains the simpler experimental baseline).

## Outcome snapshot

- **rows**: 6210
- **datasets**: 46
- **seeds**: 5
- **novelty_weight**: 0.1000
- **mean_accept_rate**: 0.3518
- **mean_accuracy_delta**: 0.0523
- **G_minus_C_novelty_rate_delta**: 0.0019
- **G_minus_C_empty_rate_delta**: 0.0019
- **G_minus_C_ambiguity_rate_delta**: 0.0047
- **G_minus_C_multilabel_rate_delta**: 0.0047
- **G_minus_C_accepted_accuracy_delta**: 0.0037
- **G_minus_C_novelty_reject_auc_delta**: -0.0371
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
| A | builtin.default | default | no | no | 0.0000 | 0.3612 | 0.6388 | 0.8653 | 0.0578 | 0.5850 | 0.0538 | 0.0538 | 0.5850 | 0.7464 | 0.0250 | 0.5020 | 0.0442 | 0.5140 | 0.8965 | 0.0015 |
| C | experimental.difficulty_normalized | default | yes | no | 0.0000 | 0.3504 | 0.6496 | 0.8556 | 0.0484 | 0.5902 | 0.0595 | 0.0595 | 0.5902 | 0.7481 | 0.3667 | 0.7031 | 0.4191 | 0.6540 | 0.8840 | -0.0110 |
| G | experimental.ambiguity_normalized_novelty_penalized | default | yes | yes | 0.1000 | 0.3437 | 0.6563 | 0.8593 | 0.0508 | 0.5949 | 0.0614 | 0.0614 | 0.5949 | 0.7475 | 0.3087 | 0.6850 | 0.3266 | 0.6170 | 0.8845 | -0.0105 |

## By Confidence And Arm

| confidence | epsilon | arm_code | strategy | reject_rate | accepted_accuracy | ambiguity_rate | novelty_rate | novelty_gap_rejected_minus_accepted | novelty_reject_auc |
|---|---|---|---|---|---|---|---|---|---|
| 0.8000 | 0.2000 | A | builtin.default | 0.5225 | 0.8361 | 0.4051 | 0.1175 | 0.0384 | 0.5085 |
| 0.8237 | 0.1763 | A | builtin.default | 0.5314 | 0.8374 | 0.4327 | 0.0987 | 0.0416 | 0.5111 |
| 0.8475 | 0.1525 | A | builtin.default | 0.5437 | 0.8413 | 0.4638 | 0.0799 | 0.0259 | 0.5114 |
| 0.8713 | 0.1287 | A | builtin.default | 0.5715 | 0.8452 | 0.5074 | 0.0642 | 0.0451 | 0.5216 |
| 0.8950 | 0.1050 | A | builtin.default | 0.5921 | 0.8523 | 0.5436 | 0.0485 | 0.0598 | 0.5219 |
| 0.9187 | 0.0813 | A | builtin.default | 0.6334 | 0.8686 | 0.5980 | 0.0354 | 0.0511 | 0.5132 |
| 0.9425 | 0.0575 | A | builtin.default | 0.6768 | 0.8834 | 0.6533 | 0.0236 | 0.0469 | 0.5111 |
| 0.9663 | 0.0337 | A | builtin.default | 0.7526 | 0.9311 | 0.7387 | 0.0139 | 0.0447 | 0.5123 |
| 0.9900 | 0.0100 | A | builtin.default | 0.9253 | 0.9621 | 0.9228 | 0.0025 | 0.0379 | 0.5146 |
| 0.8000 | 0.2000 | C | experimental.difficulty_normalized | 0.5393 | 0.8383 | 0.4074 | 0.1319 | 0.4312 | 0.6209 |
| 0.8237 | 0.1763 | C | experimental.difficulty_normalized | 0.5474 | 0.8383 | 0.4365 | 0.1110 | 0.4404 | 0.6287 |
| 0.8475 | 0.1525 | C | experimental.difficulty_normalized | 0.5552 | 0.8410 | 0.4655 | 0.0897 | 0.4555 | 0.6369 |
| 0.8713 | 0.1287 | C | experimental.difficulty_normalized | 0.5751 | 0.8438 | 0.5049 | 0.0702 | 0.4618 | 0.6584 |
| 0.8950 | 0.1050 | C | experimental.difficulty_normalized | 0.5979 | 0.8478 | 0.5450 | 0.0529 | 0.4512 | 0.6691 |
| 0.9187 | 0.0813 | C | experimental.difficulty_normalized | 0.6361 | 0.8562 | 0.5981 | 0.0380 | 0.4543 | 0.6736 |
| 0.9425 | 0.0575 | C | experimental.difficulty_normalized | 0.6886 | 0.8691 | 0.6636 | 0.0249 | 0.3566 | 0.6649 |
| 0.9663 | 0.0337 | C | experimental.difficulty_normalized | 0.7718 | 0.8972 | 0.7577 | 0.0141 | 0.3533 | 0.6809 |
| 0.9900 | 0.0100 | C | experimental.difficulty_normalized | 0.9355 | 0.9005 | 0.9329 | 0.0025 | 0.2642 | 0.6542 |
| 0.8000 | 0.2000 | G | experimental.ambiguity_normalized_novelty_penalized | 0.5379 | 0.8350 | 0.4085 | 0.1295 | 0.6098 | 0.6643 |
| 0.8237 | 0.1763 | G | experimental.ambiguity_normalized_novelty_penalized | 0.5494 | 0.8393 | 0.4387 | 0.1107 | 0.5659 | 0.6628 |
| 0.8475 | 0.1525 | G | experimental.ambiguity_normalized_novelty_penalized | 0.5574 | 0.8423 | 0.4653 | 0.0921 | 0.5142 | 0.6428 |
| 0.8713 | 0.1287 | G | experimental.ambiguity_normalized_novelty_penalized | 0.5782 | 0.8457 | 0.5039 | 0.0743 | 0.4111 | 0.6225 |
| 0.8950 | 0.1050 | G | experimental.ambiguity_normalized_novelty_penalized | 0.6062 | 0.8540 | 0.5486 | 0.0576 | 0.3249 | 0.6064 |
| 0.9187 | 0.0813 | G | experimental.ambiguity_normalized_novelty_penalized | 0.6481 | 0.8664 | 0.6065 | 0.0416 | 0.2778 | 0.5982 |
| 0.9425 | 0.0575 | G | experimental.ambiguity_normalized_novelty_penalized | 0.6963 | 0.8789 | 0.6686 | 0.0278 | 0.1632 | 0.5850 |
| 0.9663 | 0.0337 | G | experimental.ambiguity_normalized_novelty_penalized | 0.7899 | 0.9110 | 0.7739 | 0.0160 | -0.1006 | 0.5670 |
| 0.9900 | 0.0100 | G | experimental.ambiguity_normalized_novelty_penalized | 0.9429 | 0.8815 | 0.9401 | 0.0028 | -0.2981 | 0.5572 |

## Required Analyses

1. Does novelty penalization increase novelty/empty-set rejection relative to C?
G vs C changed novelty_rate by +0.0019, empty_rate by +0.0019, and novelty_reject_auc by -0.0371.
2. Does it reduce ambiguity/multi-label rejection relative to C?
G vs C changed ambiguity_rate by +0.0047 and multilabel_rate by +0.0047.
3. Does it preserve accepted accuracy relative to C?
G vs C changed accepted_accuracy by +0.0037.
4. Which arm is recommended for further development?
Recommended arm for next iteration: C (difficulty-normalized remains the simpler experimental baseline).
