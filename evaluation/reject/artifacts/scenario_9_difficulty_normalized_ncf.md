# Scenario 9 - Difficulty-normalized reject NCF strategy ablation

Rows: 12420

## Key findings

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

## Outcome snapshot

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

## Result table

| task_type | dataset | seed | confidence | epsilon | n_train | n_cal | n_test | arm_code | arm_label | ncf | strategy | use_va_difficulty | difficulty_normalized | double_count_difficulty | accept_rate | reject_rate | ambiguity_rate | novelty_rate | accepted_accuracy | full_accuracy | accuracy_delta | singleton_error_rate | error_rate_defined | rejected_error_capture_rate | mean_difficulty_all | mean_difficulty_accepted | mean_difficulty_rejected | difficulty_gap_rejected_minus_accepted | difficulty_reject_auc | empty_rate | singleton_rate | multilabel_rate | empirical_coverage | coverage_gap | coverage_defined |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| binary | breast_cancer | 42 | 0.8000 | 0.2000 | 341 | 114 | 114 | A | A|va=0|strategy=builtin.default|ncf=default | default | builtin.default | no | no | no | 0.8333 | 0.1667 | 0.0000 | 0.1667 | 1.0000 | 0.9561 | 0.0439 | 0.0400 | yes | 1.0000 | 1.8789 | 1.9130 | 1.7080 | -0.2050 | 0.3501 | 0.1667 | 0.8333 | 0.0000 | 0.8333 | 0.0333 | yes |
| binary | breast_cancer | 42 | 0.8237 | 0.1763 | 341 | 114 | 114 | A | A|va=0|strategy=builtin.default|ncf=default | default | builtin.default | no | no | no | 0.8684 | 0.1316 | 0.0000 | 0.1316 | 0.9899 | 0.9561 | 0.0338 | 0.0514 | yes | 0.8000 | 1.8789 | 1.9139 | 1.6473 | -0.2666 | 0.2700 | 0.1316 | 0.8684 | 0.0000 | 0.8596 | 0.0359 | yes |
| binary | breast_cancer | 42 | 0.8475 | 0.1525 | 341 | 114 | 114 | A | A|va=0|strategy=builtin.default|ncf=default | default | builtin.default | no | no | no | 0.9035 | 0.0965 | 0.0000 | 0.0965 | 0.9903 | 0.9561 | 0.0342 | 0.0620 | yes | 0.8000 | 1.8789 | 1.9035 | 1.6480 | -0.2555 | 0.2701 | 0.0965 | 0.9035 | 0.0000 | 0.8947 | 0.0472 | yes |
| binary | breast_cancer | 42 | 0.8713 | 0.1287 | 341 | 114 | 114 | A | A|va=0|strategy=builtin.default|ncf=default | default | builtin.default | no | no | no | 0.9211 | 0.0789 | 0.0000 | 0.0789 | 0.9810 | 0.9561 | 0.0248 | 0.0541 | yes | 0.6000 | 1.8789 | 1.8990 | 1.6439 | -0.2551 | 0.2656 | 0.0789 | 0.9211 | 0.0000 | 0.9035 | 0.0323 | yes |
| binary | breast_cancer | 42 | 0.8950 | 0.1050 | 341 | 114 | 114 | A | A|va=0|strategy=builtin.default|ncf=default | default | builtin.default | no | no | no | 0.9298 | 0.0702 | 0.0000 | 0.0702 | 0.9811 | 0.9561 | 0.0250 | 0.0375 | yes | 0.6000 | 1.8789 | 1.8996 | 1.6038 | -0.2958 | 0.2146 | 0.0702 | 0.9298 | 0.0000 | 0.9123 | 0.0173 | yes |
| binary | breast_cancer | 42 | 0.9187 | 0.0813 | 341 | 114 | 114 | A | A|va=0|strategy=builtin.default|ncf=default | default | builtin.default | no | no | no | 0.9474 | 0.0526 | 0.0000 | 0.0526 | 0.9722 | 0.9561 | 0.0161 | 0.0302 | yes | 0.4000 | 1.8789 | 1.8934 | 1.6172 | -0.2762 | 0.2407 | 0.0526 | 0.9474 | 0.0000 | 0.9211 | 0.0023 | yes |
| binary | breast_cancer | 42 | 0.9425 | 0.0575 | 341 | 114 | 114 | A | A|va=0|strategy=builtin.default|ncf=default | default | builtin.default | no | no | no | 0.9474 | 0.0526 | 0.0000 | 0.0526 | 0.9722 | 0.9561 | 0.0161 | 0.0051 | yes | 0.4000 | 1.8789 | 1.8934 | 1.6172 | -0.2762 | 0.2407 | 0.0526 | 0.9474 | 0.0000 | 0.9211 | -0.0214 | yes |
| binary | breast_cancer | 42 | 0.9663 | 0.0337 | 341 | 114 | 114 | A | A|va=0|strategy=builtin.default|ncf=default | default | builtin.default | no | no | no | 0.9912 | 0.0088 | 0.0000 | 0.0088 | 0.9646 | 0.9561 | 0.0085 | 0.0252 | yes | 0.2000 | 1.8789 | 1.8820 | 1.5238 | -0.3582 | 0.0796 | 0.0088 | 0.9912 | 0.0000 | 0.9561 | -0.0101 | yes |
| binary | breast_cancer | 42 | 0.9900 | 0.0100 | 341 | 114 | 114 | A | A|va=0|strategy=builtin.default|ncf=default | default | builtin.default | no | no | no | 0.9474 | 0.0526 | 0.0526 | 0.0000 | 0.9722 | 0.9561 | 0.0161 | 0.0106 | yes | 0.4000 | 1.8789 | 1.8934 | 1.6172 | -0.2762 | 0.2407 | 0.0000 | 0.9474 | 0.0526 | 0.9737 | -0.0163 | yes |
| binary | breast_cancer | 42 | 0.8000 | 0.2000 | 341 | 114 | 114 | B | B|va=1|strategy=builtin.default|ncf=default | default | builtin.default | yes | no | no | 0.0000 | 1.0000 | 1.0000 | 0.0000 | nan | 0.9561 | nan | nan | no | 1.0000 | 1.8789 | nan | 1.8789 | nan | nan | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.2000 | yes |
| binary | breast_cancer | 42 | 0.8237 | 0.1763 | 341 | 114 | 114 | B | B|va=1|strategy=builtin.default|ncf=default | default | builtin.default | yes | no | no | 0.0000 | 1.0000 | 1.0000 | 0.0000 | nan | 0.9561 | nan | nan | no | 1.0000 | 1.8789 | nan | 1.8789 | nan | nan | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.1763 | yes |
| binary | breast_cancer | 42 | 0.8475 | 0.1525 | 341 | 114 | 114 | B | B|va=1|strategy=builtin.default|ncf=default | default | builtin.default | yes | no | no | 0.0000 | 1.0000 | 1.0000 | 0.0000 | nan | 0.9561 | nan | nan | no | 1.0000 | 1.8789 | nan | 1.8789 | nan | nan | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.1525 | yes |

_Showing first 12 of 12420 rows._

## Arm Summary

| arm_code | strategy | ncf | use_va_difficulty | difficulty_normalized | double_count_difficulty | accept_rate | reject_rate | accepted_accuracy | accuracy_delta | ambiguity_rate | novelty_rate | rejected_error_capture_rate | mean_difficulty_accepted | mean_difficulty_rejected | difficulty_gap_rejected_minus_accepted | difficulty_reject_auc | empirical_coverage | coverage_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | builtin.default | default | no | no | no | 0.3612 | 0.6388 | 0.8653 | 0.0578 | 0.5850 | 0.0538 | 0.7464 | 1.8934 | 1.9243 | 0.0250 | 0.5020 | 0.8965 | 0.0015 |
| B | builtin.default | default | yes | no | no | 0.1387 | 0.8613 | 0.7567 | -0.0505 | 0.8388 | 0.0224 | 0.8610 | 1.8140 | 1.9065 | 0.0642 | 0.5664 | 0.9403 | 0.0453 |
| C | experimental.difficulty_normalized | default | no | yes | no | 0.1302 | 0.8698 | 0.9147 | 0.0709 | 0.8640 | 0.0058 | 0.9184 | 1.6847 | 1.9754 | 0.4163 | 0.7494 | 0.9795 | 0.0845 |
| D | experimental.difficulty_normalized | default | yes | yes | yes | 0.0257 | 0.9743 | 0.8129 | -0.0126 | 0.9741 | 0.0002 | 0.9814 | 1.5547 | 1.9406 | 0.6291 | 0.8219 | 0.9951 | 0.1001 |
| E | experimental.difficulty_normalized | ensured | no | yes | no | 0.0457 | 0.9543 | 0.8152 | -0.0032 | 0.9498 | 0.0045 | 0.9662 | 1.7461 | 1.9345 | 0.1949 | 0.7033 | 0.9819 | 0.0869 |
| F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.0170 | 0.9830 | 0.6567 | -0.1483 | 0.9326 | 0.0504 | 0.9720 | 1.5772 | 1.9266 | 0.3330 | 0.7918 | 0.9801 | 0.0851 |

## By Confidence And Arm

| confidence | epsilon | arm_code | strategy | ncf | use_va_difficulty | difficulty_normalized | double_count_difficulty | reject_rate | accepted_accuracy | ambiguity_rate | novelty_rate | difficulty_gap_rejected_minus_accepted | difficulty_reject_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8000 | 0.2000 | A | builtin.default | default | no | no | no | 0.5225 | 0.8361 | 0.4051 | 0.1175 | 0.0290 | 0.4990 |
| 0.8237 | 0.1763 | A | builtin.default | default | no | no | no | 0.5314 | 0.8374 | 0.4327 | 0.0987 | 0.0271 | 0.5017 |
| 0.8475 | 0.1525 | A | builtin.default | default | no | no | no | 0.5437 | 0.8413 | 0.4638 | 0.0799 | 0.0102 | 0.5032 |
| 0.8713 | 0.1287 | A | builtin.default | default | no | no | no | 0.5715 | 0.8452 | 0.5074 | 0.0642 | 0.0304 | 0.5166 |
| 0.8950 | 0.1050 | A | builtin.default | default | no | no | no | 0.5921 | 0.8523 | 0.5436 | 0.0485 | 0.0282 | 0.5077 |
| 0.9187 | 0.0813 | A | builtin.default | default | no | no | no | 0.6334 | 0.8686 | 0.5980 | 0.0354 | 0.0250 | 0.4954 |
| 0.9425 | 0.0575 | A | builtin.default | default | no | no | no | 0.6768 | 0.8834 | 0.6533 | 0.0236 | 0.0323 | 0.4990 |
| 0.9663 | 0.0337 | A | builtin.default | default | no | no | no | 0.7526 | 0.9311 | 0.7387 | 0.0139 | 0.0222 | 0.4953 |
| 0.9900 | 0.0100 | A | builtin.default | default | no | no | no | 0.9253 | 0.9621 | 0.9228 | 0.0025 | 0.0093 | 0.4932 |
| 0.8000 | 0.2000 | B | builtin.default | default | yes | no | no | 0.7778 | 0.7566 | 0.7255 | 0.0523 | 0.0174 | 0.5147 |
| 0.8237 | 0.1763 | B | builtin.default | default | yes | no | no | 0.7928 | 0.7554 | 0.7497 | 0.0431 | 0.0344 | 0.5369 |
| 0.8475 | 0.1525 | B | builtin.default | default | yes | no | no | 0.8091 | 0.7507 | 0.7710 | 0.0381 | 0.0497 | 0.5609 |
| 0.8713 | 0.1287 | B | builtin.default | default | yes | no | no | 0.8294 | 0.7470 | 0.8056 | 0.0239 | 0.0788 | 0.5850 |
| 0.8950 | 0.1050 | B | builtin.default | default | yes | no | no | 0.8505 | 0.7545 | 0.8322 | 0.0183 | 0.0639 | 0.5714 |
| 0.9187 | 0.0813 | B | builtin.default | default | yes | no | no | 0.8669 | 0.7708 | 0.8533 | 0.0137 | 0.0654 | 0.5653 |
| 0.9425 | 0.0575 | B | builtin.default | default | yes | no | no | 0.8999 | 0.7593 | 0.8930 | 0.0070 | 0.0897 | 0.5800 |
| 0.9663 | 0.0337 | B | builtin.default | default | yes | no | no | 0.9355 | 0.7898 | 0.9309 | 0.0046 | 0.1038 | 0.5933 |
| 0.9900 | 0.0100 | B | builtin.default | default | yes | no | no | 0.9892 | 0.6679 | 0.9885 | 0.0007 | 0.1301 | 0.6732 |
| 0.8000 | 0.2000 | C | experimental.difficulty_normalized | default | no | yes | no | 0.7458 | 0.8964 | 0.7285 | 0.0173 | 0.3833 | 0.7240 |
| 0.8237 | 0.1763 | C | experimental.difficulty_normalized | default | no | yes | no | 0.7695 | 0.9022 | 0.7561 | 0.0134 | 0.3876 | 0.7283 |
| 0.8475 | 0.1525 | C | experimental.difficulty_normalized | default | no | yes | no | 0.8016 | 0.9082 | 0.7924 | 0.0092 | 0.4027 | 0.7368 |
| 0.8713 | 0.1287 | C | experimental.difficulty_normalized | default | no | yes | no | 0.8417 | 0.8965 | 0.8356 | 0.0062 | 0.4195 | 0.7636 |
| 0.8950 | 0.1050 | C | experimental.difficulty_normalized | default | no | yes | no | 0.8711 | 0.9269 | 0.8673 | 0.0038 | 0.4189 | 0.7488 |
| 0.9187 | 0.0813 | C | experimental.difficulty_normalized | default | no | yes | no | 0.9067 | 0.9591 | 0.9049 | 0.0018 | 0.5264 | 0.7906 |
| 0.9425 | 0.0575 | C | experimental.difficulty_normalized | default | no | yes | no | 0.9291 | 0.9470 | 0.9284 | 0.0007 | 0.4545 | 0.7690 |
| 0.9663 | 0.0337 | C | experimental.difficulty_normalized | default | no | yes | no | 0.9641 | 0.9719 | 0.9641 | 0.0000 | 0.4407 | 0.8321 |
| 0.9900 | 0.0100 | C | experimental.difficulty_normalized | default | no | yes | no | 0.9991 | 1.0000 | 0.9991 | 0.0000 | 0.5659 | 0.9810 |
| 0.8000 | 0.2000 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.9290 | 0.8420 | 0.9277 | 0.0013 | 0.6607 | 0.7637 |
| 0.8237 | 0.1763 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.9410 | 0.8428 | 0.9406 | 0.0004 | 0.6998 | 0.7975 |
| 0.8475 | 0.1525 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.9493 | 0.8387 | 0.9493 | 0.0000 | 0.7177 | 0.8205 |
| 0.8713 | 0.1287 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.9708 | 0.8068 | 0.9707 | 0.0000 | 0.5902 | 0.8289 |
| 0.8950 | 0.1050 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.9877 | 0.7842 | 0.9877 | 0.0000 | 0.4937 | 0.8624 |
| 0.9187 | 0.0813 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.9914 | 0.7895 | 0.9914 | 0.0000 | 0.4603 | 0.8719 |
| 0.9425 | 0.0575 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.9992 | 0.6875 | 0.9992 | 0.0000 | 0.5075 | 0.9624 |
| 0.9663 | 0.0337 | D | experimental.difficulty_normalized | default | yes | yes | yes | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5757 | 1.0000 |
| 0.9900 | 0.0100 | D | experimental.difficulty_normalized | default | yes | yes | yes | 1.0000 | nan | 1.0000 | 0.0000 | nan | nan |
| 0.8000 | 0.2000 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.9076 | 0.8074 | 0.8878 | 0.0198 | 0.1828 | 0.6686 |
| 0.8237 | 0.1763 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.9246 | 0.8051 | 0.9152 | 0.0094 | 0.1914 | 0.6873 |
| 0.8475 | 0.1525 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.9426 | 0.8108 | 0.9375 | 0.0051 | 0.1814 | 0.6951 |
| 0.8713 | 0.1287 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.9506 | 0.8038 | 0.9471 | 0.0035 | 0.1482 | 0.6835 |
| 0.8950 | 0.1050 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.9563 | 0.8312 | 0.9544 | 0.0019 | 0.1837 | 0.7018 |
| 0.9187 | 0.0813 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.9641 | 0.8129 | 0.9634 | 0.0006 | 0.2298 | 0.7402 |
| 0.9425 | 0.0575 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.9711 | 0.8195 | 0.9707 | 0.0004 | 0.2251 | 0.7381 |
| 0.9663 | 0.0337 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.9795 | 0.8374 | 0.9795 | 0.0000 | 0.2479 | 0.7470 |
| 0.9900 | 0.0100 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.9925 | 0.9138 | 0.9925 | 0.0000 | 0.3258 | 0.7955 |
| 0.8000 | 0.2000 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.9483 | 0.6601 | 0.8819 | 0.0664 | 0.2751 | 0.6965 |
| 0.8237 | 0.1763 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.9687 | 0.6839 | 0.9104 | 0.0583 | 0.2852 | 0.7366 |
| 0.8475 | 0.1525 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.9733 | 0.6693 | 0.9200 | 0.0533 | 0.3277 | 0.7794 |
| 0.8713 | 0.1287 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.9852 | 0.6584 | 0.9357 | 0.0495 | 0.3289 | 0.7974 |
| 0.8950 | 0.1050 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.9887 | 0.6290 | 0.9408 | 0.0479 | 0.3421 | 0.8119 |
| 0.9187 | 0.0813 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.9922 | 0.6439 | 0.9460 | 0.0462 | 0.3469 | 0.8373 |
| 0.9425 | 0.0575 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.9942 | 0.6469 | 0.9494 | 0.0448 | 0.3976 | 0.8643 |
| 0.9663 | 0.0337 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.9967 | 0.6540 | 0.9530 | 0.0437 | 0.4245 | 0.8892 |
| 0.9900 | 0.0100 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.9998 | 0.6000 | 0.9563 | 0.0435 | 0.4751 | 0.9487 |

## Accepted Accuracy At Matched Reject-Rate Bins

A-vs-C matched-bin comparison (primary contrast).

| reject_rate_bin | A_mean_accepted_accuracy | C_mean_accepted_accuracy | accepted_accuracy_delta_C_minus_A | A_mean_reject_rate | C_mean_reject_rate | A_rows | C_rows |
|---|---|---|---|---|---|---|---|
| [0.0, 0.1) | 0.8815 | 0.9410 | 0.0595 | 0.0454 | 0.0467 | 286 | 136 |
| [0.1, 0.2) | 0.9031 | 0.9284 | 0.0253 | 0.1477 | 0.1425 | 208 | 46 |
| [0.2, 0.3) | 0.8552 | 0.9268 | 0.0716 | 0.2459 | 0.2481 | 140 | 36 |
| [0.3, 0.4) | 0.8477 | 0.9480 | 0.1004 | 0.3564 | 0.3505 | 116 | 23 |
| [0.4, 0.5) | 0.8242 | 0.9157 | 0.0916 | 0.4524 | 0.4515 | 81 | 27 |
| [0.5, 0.6) | 0.8250 | 0.9480 | 0.1230 | 0.5495 | 0.5529 | 68 | 21 |
| [0.6, 0.7) | 0.8485 | 0.9559 | 0.1074 | 0.6434 | 0.6637 | 66 | 28 |
| [0.7, 0.8) | 0.8637 | 0.8977 | 0.0340 | 0.7452 | 0.7473 | 45 | 41 |
| [0.8, 0.9) | 0.8479 | 0.8503 | 0.0024 | 0.8407 | 0.8602 | 35 | 55 |
| [0.9, 1.0) | 0.8206 | 0.8944 | 0.0738 | 0.9985 | 0.9960 | 1025 | 1657 |

## Per-Arm Reject-Rate Bin Aggregates

| reject_rate_bin | arm_code | n_rows | mean_reject_rate | mean_accepted_accuracy |
|---|---|---|---|---|
| [0.0, 0.1) | A | 286 | 0.0454 | 0.8815 |
| [0.0, 0.1) | B | 74 | 0.0397 | 0.8826 |
| [0.0, 0.1) | C | 136 | 0.0467 | 0.9410 |
| [0.0, 0.1) | D | 17 | 0.0496 | 0.8868 |
| [0.0, 0.1) | E | 1 | 0.0909 | 0.9000 |
| [0.0, 0.1) | F | 2 | 0.0923 | 0.9153 |
| [0.1, 0.2) | A | 208 | 0.1477 | 0.9031 |
| [0.1, 0.2) | B | 41 | 0.1456 | 0.9007 |
| [0.1, 0.2) | C | 46 | 0.1425 | 0.9284 |
| [0.1, 0.2) | D | 13 | 0.1284 | 0.9087 |
| [0.1, 0.2) | E | 6 | 0.1490 | 0.9112 |
| [0.1, 0.2) | F | 9 | 0.1497 | 0.9153 |
| [0.2, 0.3) | A | 140 | 0.2459 | 0.8552 |
| [0.2, 0.3) | B | 40 | 0.2625 | 0.8773 |
| [0.2, 0.3) | C | 36 | 0.2481 | 0.9268 |
| [0.2, 0.3) | D | 9 | 0.2385 | 0.9445 |
| [0.2, 0.3) | E | 6 | 0.2312 | 0.9243 |
| [0.3, 0.4) | A | 116 | 0.3564 | 0.8477 |
| [0.3, 0.4) | B | 37 | 0.3509 | 0.8662 |
| [0.3, 0.4) | C | 23 | 0.3505 | 0.9480 |
| [0.3, 0.4) | D | 7 | 0.3441 | 0.9038 |
| [0.3, 0.4) | E | 2 | 0.3643 | 0.9942 |
| [0.3, 0.4) | F | 1 | 0.3231 | 0.9318 |
| [0.4, 0.5) | A | 81 | 0.4524 | 0.8242 |
| [0.4, 0.5) | B | 68 | 0.4607 | 0.8815 |
| [0.4, 0.5) | C | 27 | 0.4515 | 0.9157 |
| [0.4, 0.5) | D | 10 | 0.4340 | 0.9334 |
| [0.4, 0.5) | F | 1 | 0.4318 | 0.9200 |
| [0.5, 0.6) | A | 68 | 0.5495 | 0.8250 |
| [0.5, 0.6) | B | 44 | 0.5461 | 0.8650 |
| [0.5, 0.6) | C | 21 | 0.5529 | 0.9480 |
| [0.5, 0.6) | D | 7 | 0.5752 | 0.9279 |
| [0.5, 0.6) | E | 43 | 0.5592 | 0.9805 |
| [0.5, 0.6) | F | 1 | 0.5385 | 0.8667 |
| [0.6, 0.7) | A | 66 | 0.6434 | 0.8485 |
| [0.6, 0.7) | B | 40 | 0.6508 | 0.8356 |
| [0.6, 0.7) | C | 28 | 0.6637 | 0.9559 |
| [0.6, 0.7) | D | 4 | 0.6271 | 0.9437 |
| [0.6, 0.7) | F | 5 | 0.6471 | 0.8886 |
| [0.7, 0.8) | A | 45 | 0.7452 | 0.8637 |
| [0.7, 0.8) | B | 68 | 0.7497 | 0.7665 |
| [0.7, 0.8) | C | 41 | 0.7473 | 0.8977 |
| [0.7, 0.8) | D | 6 | 0.7609 | 0.9141 |
| [0.7, 0.8) | E | 75 | 0.7562 | 0.8955 |
| [0.7, 0.8) | F | 10 | 0.7336 | 0.7784 |
| [0.8, 0.9) | A | 35 | 0.8407 | 0.8479 |
| [0.8, 0.9) | B | 131 | 0.8520 | 0.7068 |
| [0.8, 0.9) | C | 55 | 0.8602 | 0.8503 |
| [0.8, 0.9) | D | 8 | 0.8737 | 0.9531 |
| [0.8, 0.9) | E | 151 | 0.8611 | 0.8050 |
| [0.8, 0.9) | F | 33 | 0.8725 | 0.8019 |
| [0.9, 1.0) | A | 1025 | 0.9985 | 0.8206 |
| [0.9, 1.0) | B | 1527 | 0.9865 | 0.6728 |
| [0.9, 1.0) | C | 1657 | 0.9960 | 0.8944 |
| [0.9, 1.0) | D | 1989 | 0.9991 | 0.6721 |
| [0.9, 1.0) | E | 1786 | 0.9863 | 0.7899 |
| [0.9, 1.0) | F | 2008 | 0.9924 | 0.6294 |

## Required Analyses

1. Does direct normalization increase rejection among high-difficulty instances?
Direct normalization (C vs A) changed reject_rate by +0.2310, difficulty-gap by +0.3912, and difficulty_reject_auc by +0.2474.
2. Does it improve accepted accuracy at comparable reject rates?
At matched reject-rate bins, C minus A mean accepted_accuracy is +0.0689.
3. Does it increase ambiguity rate, novelty rate, or both?
For C vs A, ambiguity_rate changed by +0.2790 and novelty_rate by -0.0480.
4. Does using VA difficulty and score normalization together appear to double-count difficulty?
Double-count diagnostics: D-B reject_rate delta +0.1130, F-E reject_rate delta +0.0287; difficulty-gap deltas are +0.5649 and +0.1382.
5. Which arm is recommended for further development?
Recommended arm for next iteration: C (primary A-vs-C contrast with direct normalization and no VA double-count risk).
