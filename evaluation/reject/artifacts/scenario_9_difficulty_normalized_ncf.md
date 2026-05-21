# Scenario 9 - Difficulty-normalized reject NCF strategy ablation

Rows: 12420

## Key findings

- Compares indirect VA-difficulty support against direct experimental difficulty-normalized reject scoring.
- Primary scientific contrast is A vs C (default NCF, no VA difficulty in either arm).
- Arms D and F are diagnostic for potential difficulty double-counting when VA and score normalization are both enabled.
- Includes strategy metadata and difficulty_reject_auc for reject-selectivity diagnostics.
- Includes accepted-accuracy comparison at matched reject-rate bins for A vs C.
- Direct normalization (C vs A) changed reject_rate by +0.0108, difficulty-gap by +0.3416, and difficulty_reject_auc by +0.2012.
- At matched reject-rate bins, C minus A mean accepted_accuracy is -0.0089.
- For C vs A, ambiguity_rate changed by +0.0051 and novelty_rate by +0.0057.
- Double-count diagnostics: D-B reject_rate delta -0.0920, F-E reject_rate delta +0.0234; difficulty-gap deltas are +0.3770 and +0.2766.
- Recommended arm for next iteration: C (primary A-vs-C contrast with direct normalization and no VA double-count risk).

## Outcome snapshot

- **rows**: 12420
- **datasets**: 46
- **seeds**: 5
- **mean_accept_rate**: 0.2230
- **mean_accuracy_delta**: -0.0014
- **A_vs_C_reject_rate_delta**: 0.0108
- **A_vs_C_difficulty_gap_delta**: 0.3416
- **A_vs_C_difficulty_reject_auc_delta**: 0.2012
- **A_vs_C_ambiguity_rate_delta**: 0.0051
- **A_vs_C_novelty_rate_delta**: 0.0057
- **A_vs_C_matched_bin_accepted_accuracy_delta**: -0.0089
- **D_minus_B_reject_rate_delta**: -0.0920
- **F_minus_E_reject_rate_delta**: 0.0234
- **D_minus_B_difficulty_gap_delta**: 0.3770
- **F_minus_E_difficulty_gap_delta**: 0.2766
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
| C | experimental.difficulty_normalized | default | no | yes | no | 0.3504 | 0.6496 | 0.8556 | 0.0484 | 0.5902 | 0.0595 | 0.7481 | 1.7783 | 2.0430 | 0.3667 | 0.7031 | 0.8840 | -0.0110 |
| D | experimental.difficulty_normalized | default | yes | yes | yes | 0.2308 | 0.7692 | 0.8056 | -0.0016 | 0.7229 | 0.0464 | 0.7724 | 1.6777 | 2.0292 | 0.4412 | 0.8579 | 0.8879 | -0.0071 |
| E | experimental.difficulty_normalized | ensured | no | yes | no | 0.1401 | 0.8599 | 0.7924 | -0.0152 | 0.7800 | 0.0798 | 0.8721 | 1.8992 | 1.9310 | 0.0280 | 0.5064 | 0.8917 | -0.0033 |
| F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.1167 | 0.8833 | 0.7493 | -0.0570 | 0.8173 | 0.0660 | 0.8555 | 1.6892 | 1.9619 | 0.3046 | 0.7991 | 0.8968 | 0.0018 |

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
| 0.8000 | 0.2000 | C | experimental.difficulty_normalized | default | no | yes | no | 0.5393 | 0.8383 | 0.4074 | 0.1319 | 0.3280 | 0.6097 |
| 0.8237 | 0.1763 | C | experimental.difficulty_normalized | default | no | yes | no | 0.5474 | 0.8383 | 0.4365 | 0.1110 | 0.3413 | 0.6331 |
| 0.8475 | 0.1525 | C | experimental.difficulty_normalized | default | no | yes | no | 0.5552 | 0.8410 | 0.4655 | 0.0897 | 0.3535 | 0.6522 |
| 0.8713 | 0.1287 | C | experimental.difficulty_normalized | default | no | yes | no | 0.5751 | 0.8438 | 0.5049 | 0.0702 | 0.3688 | 0.6895 |
| 0.8950 | 0.1050 | C | experimental.difficulty_normalized | default | no | yes | no | 0.5979 | 0.8478 | 0.5450 | 0.0529 | 0.3916 | 0.7139 |
| 0.9187 | 0.0813 | C | experimental.difficulty_normalized | default | no | yes | no | 0.6361 | 0.8562 | 0.5981 | 0.0380 | 0.4089 | 0.7382 |
| 0.9425 | 0.0575 | C | experimental.difficulty_normalized | default | no | yes | no | 0.6886 | 0.8691 | 0.6636 | 0.0249 | 0.3524 | 0.7506 |
| 0.9663 | 0.0337 | C | experimental.difficulty_normalized | default | no | yes | no | 0.7718 | 0.8972 | 0.7577 | 0.0141 | 0.3821 | 0.7963 |
| 0.9900 | 0.0100 | C | experimental.difficulty_normalized | default | no | yes | no | 0.9355 | 0.9005 | 0.9329 | 0.0025 | 0.3884 | 0.8349 |
| 0.8000 | 0.2000 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.6377 | 0.8077 | 0.5406 | 0.0971 | 0.4609 | 0.8069 |
| 0.8237 | 0.1763 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.6555 | 0.8071 | 0.5740 | 0.0814 | 0.4606 | 0.8251 |
| 0.8475 | 0.1525 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.6772 | 0.8051 | 0.6068 | 0.0703 | 0.4556 | 0.8433 |
| 0.8713 | 0.1287 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.7041 | 0.8054 | 0.6486 | 0.0555 | 0.4733 | 0.8573 |
| 0.8950 | 0.1050 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.7411 | 0.8113 | 0.6971 | 0.0440 | 0.4730 | 0.8881 |
| 0.9187 | 0.0813 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.7911 | 0.8097 | 0.7583 | 0.0327 | 0.4358 | 0.8777 |
| 0.9425 | 0.0575 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.8369 | 0.8072 | 0.8152 | 0.0216 | 0.4005 | 0.8795 |
| 0.9663 | 0.0337 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.8970 | 0.7926 | 0.8845 | 0.0124 | 0.3885 | 0.8818 |
| 0.9900 | 0.0100 | D | experimental.difficulty_normalized | default | yes | yes | yes | 0.9827 | 0.7988 | 0.9806 | 0.0020 | 0.3683 | 0.8741 |
| 0.8000 | 0.2000 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.8103 | 0.8002 | 0.6493 | 0.1610 | 0.0216 | 0.4562 |
| 0.8237 | 0.1763 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.8250 | 0.8080 | 0.6863 | 0.1388 | 0.0195 | 0.4713 |
| 0.8475 | 0.1525 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.8390 | 0.7900 | 0.7185 | 0.1205 | 0.0149 | 0.4734 |
| 0.8713 | 0.1287 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.8463 | 0.7842 | 0.7462 | 0.1001 | -0.0004 | 0.4924 |
| 0.8950 | 0.1050 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.8489 | 0.7891 | 0.7696 | 0.0794 | -0.0022 | 0.4963 |
| 0.9187 | 0.0813 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.8518 | 0.7804 | 0.7932 | 0.0587 | 0.0263 | 0.5173 |
| 0.9425 | 0.0575 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.8632 | 0.7928 | 0.8251 | 0.0382 | 0.0154 | 0.5138 |
| 0.9663 | 0.0337 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.8908 | 0.7845 | 0.8714 | 0.0194 | 0.0606 | 0.5643 |
| 0.9900 | 0.0100 | E | experimental.difficulty_normalized | ensured | no | yes | no | 0.9635 | 0.8190 | 0.9608 | 0.0026 | 0.2292 | 0.7082 |
| 0.8000 | 0.2000 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.7652 | 0.7909 | 0.6346 | 0.1307 | 0.3225 | 0.7639 |
| 0.8237 | 0.1763 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.8015 | 0.7775 | 0.6887 | 0.1128 | 0.3244 | 0.7913 |
| 0.8475 | 0.1525 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.8288 | 0.7727 | 0.7308 | 0.0980 | 0.3084 | 0.8059 |
| 0.8713 | 0.1287 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.8573 | 0.7643 | 0.7760 | 0.0814 | 0.2882 | 0.7877 |
| 0.8950 | 0.1050 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.8862 | 0.7608 | 0.8195 | 0.0667 | 0.2800 | 0.7849 |
| 0.9187 | 0.0813 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.9156 | 0.7442 | 0.8658 | 0.0498 | 0.2895 | 0.7987 |
| 0.9425 | 0.0575 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.9377 | 0.7086 | 0.9031 | 0.0346 | 0.2971 | 0.8112 |
| 0.9663 | 0.0337 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.9618 | 0.6745 | 0.9438 | 0.0181 | 0.3253 | 0.8440 |
| 0.9900 | 0.0100 | F | experimental.difficulty_normalized | ensured | yes | yes | yes | 0.9951 | 0.7052 | 0.9932 | 0.0019 | 0.3167 | 0.8430 |

## Accepted Accuracy At Matched Reject-Rate Bins

A-vs-C matched-bin comparison (primary contrast).

| reject_rate_bin | A_mean_accepted_accuracy | C_mean_accepted_accuracy | accepted_accuracy_delta_C_minus_A | A_mean_reject_rate | C_mean_reject_rate | A_rows | C_rows |
|---|---|---|---|---|---|---|---|
| [0.0, 0.1) | 0.8815 | 0.8861 | 0.0046 | 0.0454 | 0.0594 | 286 | 205 |
| [0.1, 0.2) | 0.9031 | 0.8876 | -0.0155 | 0.1477 | 0.1500 | 208 | 263 |
| [0.2, 0.3) | 0.8552 | 0.8582 | 0.0031 | 0.2459 | 0.2463 | 140 | 132 |
| [0.3, 0.4) | 0.8477 | 0.8260 | -0.0217 | 0.3564 | 0.3501 | 116 | 128 |
| [0.4, 0.5) | 0.8242 | 0.8267 | 0.0025 | 0.4524 | 0.4519 | 81 | 97 |
| [0.5, 0.6) | 0.8250 | 0.8207 | -0.0044 | 0.5495 | 0.5478 | 68 | 70 |
| [0.6, 0.7) | 0.8485 | 0.8101 | -0.0384 | 0.6434 | 0.6485 | 66 | 53 |
| [0.7, 0.8) | 0.8637 | 0.8040 | -0.0598 | 0.7452 | 0.7484 | 45 | 61 |
| [0.8, 0.9) | 0.8479 | 0.8307 | -0.0172 | 0.8407 | 0.8496 | 35 | 41 |
| [0.9, 1.0) | 0.8206 | 0.8783 | 0.0576 | 0.9985 | 0.9988 | 1025 | 1020 |

## Per-Arm Reject-Rate Bin Aggregates

| reject_rate_bin | arm_code | n_rows | mean_reject_rate | mean_accepted_accuracy |
|---|---|---|---|---|
| [0.0, 0.1) | A | 286 | 0.0454 | 0.8815 |
| [0.0, 0.1) | B | 74 | 0.0397 | 0.8826 |
| [0.0, 0.1) | C | 205 | 0.0594 | 0.8861 |
| [0.0, 0.1) | D | 51 | 0.0595 | 0.8610 |
| [0.0, 0.1) | E | 7 | 0.0586 | 0.8800 |
| [0.0, 0.1) | F | 8 | 0.0524 | 0.8771 |
| [0.1, 0.2) | A | 208 | 0.1477 | 0.9031 |
| [0.1, 0.2) | B | 41 | 0.1456 | 0.9007 |
| [0.1, 0.2) | C | 263 | 0.1500 | 0.8876 |
| [0.1, 0.2) | D | 110 | 0.1473 | 0.8778 |
| [0.1, 0.2) | E | 8 | 0.1387 | 0.9213 |
| [0.1, 0.2) | F | 19 | 0.1437 | 0.8743 |
| [0.2, 0.3) | A | 140 | 0.2459 | 0.8552 |
| [0.2, 0.3) | B | 40 | 0.2625 | 0.8773 |
| [0.2, 0.3) | C | 132 | 0.2463 | 0.8582 |
| [0.2, 0.3) | D | 91 | 0.2511 | 0.8423 |
| [0.2, 0.3) | E | 6 | 0.2731 | 0.9008 |
| [0.2, 0.3) | F | 37 | 0.2460 | 0.8286 |
| [0.3, 0.4) | A | 116 | 0.3564 | 0.8477 |
| [0.3, 0.4) | B | 37 | 0.3509 | 0.8662 |
| [0.3, 0.4) | C | 128 | 0.3501 | 0.8260 |
| [0.3, 0.4) | D | 93 | 0.3558 | 0.8464 |
| [0.3, 0.4) | E | 34 | 0.3628 | 0.8407 |
| [0.3, 0.4) | F | 31 | 0.3564 | 0.8358 |
| [0.4, 0.5) | A | 81 | 0.4524 | 0.8242 |
| [0.4, 0.5) | B | 68 | 0.4607 | 0.8815 |
| [0.4, 0.5) | C | 97 | 0.4519 | 0.8267 |
| [0.4, 0.5) | D | 89 | 0.4567 | 0.8221 |
| [0.4, 0.5) | E | 85 | 0.4533 | 0.8595 |
| [0.4, 0.5) | F | 46 | 0.4507 | 0.7978 |
| [0.5, 0.6) | A | 68 | 0.5495 | 0.8250 |
| [0.5, 0.6) | B | 44 | 0.5461 | 0.8650 |
| [0.5, 0.6) | C | 70 | 0.5478 | 0.8207 |
| [0.5, 0.6) | D | 117 | 0.5545 | 0.8417 |
| [0.5, 0.6) | E | 142 | 0.5575 | 0.8440 |
| [0.5, 0.6) | F | 57 | 0.5563 | 0.7838 |
| [0.6, 0.7) | A | 66 | 0.6434 | 0.8485 |
| [0.6, 0.7) | B | 40 | 0.6508 | 0.8356 |
| [0.6, 0.7) | C | 53 | 0.6485 | 0.8101 |
| [0.6, 0.7) | D | 141 | 0.6520 | 0.8027 |
| [0.6, 0.7) | E | 157 | 0.6497 | 0.8681 |
| [0.6, 0.7) | F | 107 | 0.6529 | 0.7788 |
| [0.7, 0.8) | A | 45 | 0.7452 | 0.8637 |
| [0.7, 0.8) | B | 68 | 0.7497 | 0.7665 |
| [0.7, 0.8) | C | 61 | 0.7484 | 0.8040 |
| [0.7, 0.8) | D | 139 | 0.7506 | 0.7785 |
| [0.7, 0.8) | E | 194 | 0.7508 | 0.8026 |
| [0.7, 0.8) | F | 157 | 0.7542 | 0.7547 |
| [0.8, 0.9) | A | 35 | 0.8407 | 0.8479 |
| [0.8, 0.9) | B | 131 | 0.8520 | 0.7068 |
| [0.8, 0.9) | C | 41 | 0.8496 | 0.8307 |
| [0.8, 0.9) | D | 105 | 0.8467 | 0.7953 |
| [0.8, 0.9) | E | 164 | 0.8475 | 0.7431 |
| [0.8, 0.9) | F | 184 | 0.8533 | 0.7511 |
| [0.9, 1.0) | A | 1025 | 0.9985 | 0.8206 |
| [0.9, 1.0) | B | 1527 | 0.9865 | 0.6728 |
| [0.9, 1.0) | C | 1020 | 0.9988 | 0.8783 |
| [0.9, 1.0) | D | 1134 | 0.9934 | 0.6678 |
| [0.9, 1.0) | E | 1273 | 0.9898 | 0.7005 |
| [0.9, 1.0) | F | 1424 | 0.9883 | 0.6999 |

## Required Analyses

1. Does direct normalization increase rejection among high-difficulty instances?
Direct normalization (C vs A) changed reject_rate by +0.0108, difficulty-gap by +0.3416, and difficulty_reject_auc by +0.2012.
2. Does it improve accepted accuracy at comparable reject rates?
At matched reject-rate bins, C minus A mean accepted_accuracy is -0.0089.
3. Does it increase ambiguity rate, novelty rate, or both?
For C vs A, ambiguity_rate changed by +0.0051 and novelty_rate by +0.0057.
4. Does using VA difficulty and score normalization together appear to double-count difficulty?
Double-count diagnostics: D-B reject_rate delta -0.0920, F-E reject_rate delta +0.0234; difficulty-gap deltas are +0.3770 and +0.2766.
5. Which arm is recommended for further development?
Recommended arm for next iteration: C (primary A-vs-C contrast with direct normalization and no VA double-count risk).
