# Scenario 10 - Ambiguity-normalized novelty-penalized reject strategy

Rows: 6210

## Key findings

- Compares built-in default, direct difficulty-normalized, and novelty-aware experimental reject strategies.
- Primary contrast is C vs G: direct difficulty normalization with and without novelty penalty.
- Novelty estimator is deterministic, fitted on proper-training features only, and uses no calibration labels/residuals.
- G vs C changed novelty_rate by +0.0283, empty_rate by +0.0283, and novelty_reject_auc by -0.0136.
- G vs C changed ambiguity_rate by -0.0418 and multilabel_rate by -0.0418.
- G vs C changed accepted_accuracy by -0.0008.
- Recommended arm for next iteration: G (novelty penalty adds novelty routing without large accepted-accuracy loss).

## Outcome snapshot

- **rows**: 6210
- **datasets**: 46
- **seeds**: 5
- **novelty_weight**: 0.1000
- **mean_accept_rate**: 0.6168
- **mean_accuracy_delta**: 0.0499
- **G_minus_C_novelty_rate_delta**: 0.0283
- **G_minus_C_empty_rate_delta**: 0.0283
- **G_minus_C_ambiguity_rate_delta**: -0.0418
- **G_minus_C_multilabel_rate_delta**: -0.0418
- **G_minus_C_accepted_accuracy_delta**: -0.0008
- **G_minus_C_novelty_reject_auc_delta**: -0.0136
- **recommended_arm**: G
- **recommendation_reason**: novelty penalty adds novelty routing without large accepted-accuracy loss

## Arm Summary

| arm_code | strategy | ncf | difficulty_normalized | novelty_penalized | novelty_weight | accept_rate | reject_rate | accepted_accuracy | accuracy_delta | ambiguity_rate | novelty_rate | empty_rate | multilabel_rate | rejected_error_capture_rate | difficulty_gap_rejected_minus_accepted | difficulty_reject_auc | novelty_gap_rejected_minus_accepted | novelty_reject_auc | empirical_coverage | coverage_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | builtin.default | default | no | no | 0.0000 | 0.6332 | 0.3668 | 0.8614 | 0.0551 | 0.3431 | 0.0237 | 0.0237 | 0.3431 | 0.5878 | 0.0100 | 0.4986 | 0.0255 | 0.5076 | 0.8965 | 0.0015 |
| C | experimental.difficulty_normalized | default | yes | no | 0.0000 | 0.6019 | 0.3981 | 0.8548 | 0.0478 | 0.3648 | 0.0333 | 0.0333 | 0.3648 | 0.6123 | 0.3049 | 0.6971 | 0.3412 | 0.6369 | 0.8840 | -0.0110 |
| G | experimental.ambiguity_normalized_novelty_penalized | default | yes | yes | 0.1000 | 0.6154 | 0.3846 | 0.8540 | 0.0470 | 0.3230 | 0.0616 | 0.0616 | 0.3230 | 0.5927 | 0.2796 | 0.6886 | 0.3517 | 0.6233 | 0.8506 | -0.0444 |

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
| 0.8000 | 0.2000 | G | experimental.ambiguity_normalized_novelty_penalized | 0.2526 | 0.8377 | 0.1069 | 0.1457 | 0.4977 | 0.6559 |
| 0.8237 | 0.1763 | G | experimental.ambiguity_normalized_novelty_penalized | 0.2558 | 0.8385 | 0.1350 | 0.1208 | 0.4907 | 0.6542 |
| 0.8475 | 0.1525 | G | experimental.ambiguity_normalized_novelty_penalized | 0.2637 | 0.8405 | 0.1676 | 0.0961 | 0.4715 | 0.6534 |
| 0.8713 | 0.1287 | G | experimental.ambiguity_normalized_novelty_penalized | 0.2832 | 0.8414 | 0.2107 | 0.0726 | 0.4309 | 0.6442 |
| 0.8950 | 0.1050 | G | experimental.ambiguity_normalized_novelty_penalized | 0.3043 | 0.8455 | 0.2531 | 0.0512 | 0.3673 | 0.6252 |
| 0.9187 | 0.0813 | G | experimental.ambiguity_normalized_novelty_penalized | 0.3499 | 0.8551 | 0.3159 | 0.0340 | 0.3171 | 0.6084 |
| 0.9425 | 0.0575 | G | experimental.ambiguity_normalized_novelty_penalized | 0.4222 | 0.8649 | 0.4006 | 0.0216 | 0.2192 | 0.5941 |
| 0.9663 | 0.0337 | G | experimental.ambiguity_normalized_novelty_penalized | 0.5278 | 0.8826 | 0.5166 | 0.0112 | 0.1529 | 0.5806 |
| 0.9900 | 0.0100 | G | experimental.ambiguity_normalized_novelty_penalized | 0.8019 | 0.9108 | 0.8004 | 0.0015 | 0.0426 | 0.5549 |

## Required Analyses

1. Does novelty penalization increase novelty/empty-set rejection relative to C?
G vs C changed novelty_rate by +0.0283, empty_rate by +0.0283, and novelty_reject_auc by -0.0136.
2. Does it reduce ambiguity/multi-label rejection relative to C?
G vs C changed ambiguity_rate by -0.0418 and multilabel_rate by -0.0418.
3. Does it preserve accepted accuracy relative to C?
G vs C changed accepted_accuracy by -0.0008.
4. Which arm is recommended for further development?
Recommended arm for next iteration: G (novelty penalty adds novelty routing without large accepted-accuracy loss).

## Per-Dataset Arm Comparison (all datasets)

Mean over seeds and confidence levels. Rows sorted by arm_code, dataset.

| dataset | arm_code | strategy | accept_rate | accepted_accuracy | accuracy_delta | novelty_rate | ambiguity_rate | rejected_error_capture_rate |
|---|---|---|---|---|---|---|---|---|
| balance | A | builtin.default | 0.8939 | 0.8946 | 0.0578 | 0.0311 | 0.0750 | 0.3891 |
| breast_cancer | A | builtin.default | 0.9181 | 0.9779 | 0.0218 | 0.0713 | 0.0105 | 0.5326 |
| cars | A | builtin.default | 0.8920 | 0.9912 | 0.0253 | 0.1013 | 0.0067 | 0.7552 |
| cmc | A | builtin.default | 0.3333 | 0.6612 | 0.1521 | 0.0000 | 0.6667 | 0.7374 |
| colic | A | builtin.default | 0.7136 | 0.8680 | 0.0485 | 0.0164 | 0.2701 | 0.4294 |
| cool | A | builtin.default | 0.8935 | 0.9836 | 0.0342 | 0.0828 | 0.0237 | 0.6982 |
| creditA | A | builtin.default | 0.7862 | 0.8968 | 0.0388 | 0.0374 | 0.1765 | 0.3823 |
| diabetes | A | builtin.default | 0.5541 | 0.8723 | 0.1164 | 0.0000 | 0.4459 | 0.6490 |
| ecoli | A | builtin.default | 0.6147 | 0.9116 | 0.0528 | 0.0069 | 0.3784 | 0.6046 |
| german | A | builtin.default | 0.3732 | 0.7873 | 0.1307 | 0.0000 | 0.6268 | 0.7383 |
| glass | A | builtin.default | 0.4610 | 0.8120 | 0.0492 | 0.0000 | 0.5390 | 0.6043 |
| haberman | A | builtin.default | 0.4499 | 0.8003 | 0.1547 | 0.0000 | 0.5501 | 0.7027 |
| heartC | A | builtin.default | 0.6350 | 0.8779 | 0.0779 | 0.0000 | 0.3650 | 0.5656 |
| heartH | A | builtin.default | 0.6919 | 0.8716 | 0.0410 | 0.0011 | 0.3070 | 0.4645 |
| heartS | A | builtin.default | 0.6720 | 0.8148 | 0.0318 | 0.0066 | 0.3214 | 0.4070 |
| heat | A | builtin.default | 0.8831 | 0.9986 | 0.0051 | 0.0893 | 0.0276 | 0.8611 |
| hepati | A | builtin.default | 0.6717 | 0.9271 | 0.0755 | 0.0194 | 0.3090 | 0.5994 |
| image | A | builtin.default | 0.8969 | 0.9939 | 0.0186 | 0.0983 | 0.0048 | 0.7848 |
| iono | A | builtin.default | 0.7635 | 0.9634 | 0.0348 | 0.0787 | 0.1578 | 0.6207 |
| iris | A | builtin.default | 0.6511 | 0.9682 | 0.0182 | 0.0185 | 0.3304 | 0.6593 |
| je4042 | A | builtin.default | 0.4613 | 0.8261 | 0.0817 | 0.0000 | 0.5387 | 0.6371 |
| je4243 | A | builtin.default | 0.4618 | 0.6814 | 0.0622 | 0.0000 | 0.5382 | 0.5969 |
| kc1 | A | builtin.default | 0.4187 | 0.8399 | 0.0919 | 0.0000 | 0.5813 | 0.6958 |
| kc2 | A | builtin.default | 0.6168 | 0.8820 | 0.0874 | 0.0033 | 0.3799 | 0.6001 |
| kc3 | A | builtin.default | 0.7303 | 0.9053 | 0.0438 | 0.0157 | 0.2540 | 0.4576 |
| liver | A | builtin.default | 0.4870 | 0.7631 | 0.0994 | 0.0000 | 0.5130 | 0.6414 |
| pc1req | A | builtin.default | 0.3429 | 0.5539 | -0.1127 | 0.0000 | 0.6571 | 0.6025 |
| pc4 | A | builtin.default | 0.8777 | 0.9378 | 0.0419 | 0.0462 | 0.0762 | 0.4506 |
| sonar | A | builtin.default | 0.7153 | 0.9114 | 0.0305 | 0.0101 | 0.2746 | 0.4425 |
| spect | A | builtin.default | 0.6429 | 0.9012 | 0.0194 | 0.0172 | 0.3399 | 0.4000 |
| spectf | A | builtin.default | 0.6432 | 0.8752 | 0.0641 | 0.0016 | 0.3551 | 0.5230 |
| steel | A | builtin.default | 0.7556 | 0.8427 | 0.0807 | 0.0014 | 0.2430 | 0.4494 |
| tae | A | builtin.default | 0.2667 | 0.3863 | -0.1456 | 0.0000 | 0.7333 | 0.7018 |
| transfusion | A | builtin.default | 0.5285 | 0.8194 | 0.0961 | 0.0000 | 0.4715 | 0.6039 |
| ttt | A | builtin.default | 0.9010 | 0.9946 | 0.0186 | 0.0884 | 0.0105 | 0.8055 |
| user | A | builtin.default | 0.7951 | 0.9151 | 0.0263 | 0.0560 | 0.1490 | 0.3743 |
| vehicle | A | builtin.default | 0.7292 | 0.8443 | 0.0984 | 0.0009 | 0.2699 | 0.5060 |
| vote | A | builtin.default | 0.7746 | 0.9020 | 0.0673 | 0.0370 | 0.1885 | 0.4969 |
| vowel | A | builtin.default | 0.7384 | 0.9735 | 0.0311 | 0.0451 | 0.2165 | 0.5485 |
| wave | A | builtin.default | 0.8292 | 0.9058 | 0.0494 | 0.0192 | 0.1516 | 0.4148 |
| wbc | A | builtin.default | 0.7830 | 0.9706 | 0.0093 | 0.0894 | 0.1276 | 0.3926 |
| whole | A | builtin.default | 0.3227 | 0.7393 | 0.0393 | 0.0000 | 0.6773 | 0.7195 |
| wine | A | builtin.default | 0.1056 | 1.0000 | 0.0403 | 0.0000 | 0.8944 | 1.0000 |
| wineR | A | builtin.default | 0.4863 | 0.8042 | 0.1311 | 0.0000 | 0.5137 | 0.6628 |
| wineW | A | builtin.default | 0.5191 | 0.8089 | 0.1436 | 0.0000 | 0.4809 | 0.6619 |
| yeast | A | builtin.default | 0.4460 | 0.6830 | 0.0749 | 0.0000 | 0.5540 | 0.6035 |
| balance | C | experimental.difficulty_normalized | 0.8233 | 0.9172 | 0.0804 | 0.0457 | 0.1310 | 0.5645 |
| breast_cancer | C | experimental.difficulty_normalized | 0.8869 | 0.9794 | 0.0232 | 0.0809 | 0.0322 | 0.5811 |
| cars | C | experimental.difficulty_normalized | 0.8893 | 0.9918 | 0.0259 | 0.1024 | 0.0082 | 0.7734 |
| cmc | C | experimental.difficulty_normalized | 0.2900 | 0.6188 | 0.1096 | 0.0037 | 0.7063 | 0.7564 |
| colic | C | experimental.difficulty_normalized | 0.7253 | 0.8579 | 0.0385 | 0.0222 | 0.2525 | 0.3887 |
| cool | C | experimental.difficulty_normalized | 0.8811 | 0.9842 | 0.0348 | 0.0861 | 0.0328 | 0.7144 |
| creditA | C | experimental.difficulty_normalized | 0.7844 | 0.8898 | 0.0318 | 0.0475 | 0.1681 | 0.3689 |
| diabetes | C | experimental.difficulty_normalized | 0.5472 | 0.8409 | 0.0851 | 0.0051 | 0.4478 | 0.6011 |
| ecoli | C | experimental.difficulty_normalized | 0.5546 | 0.9160 | 0.0572 | 0.0248 | 0.4206 | 0.6594 |
| german | C | experimental.difficulty_normalized | 0.4383 | 0.7667 | 0.1102 | 0.0001 | 0.5616 | 0.6835 |
| glass | C | experimental.difficulty_normalized | 0.4145 | 0.7668 | 0.0040 | 0.0098 | 0.5757 | 0.5912 |
| haberman | C | experimental.difficulty_normalized | 0.4550 | 0.7631 | 0.1166 | 0.0094 | 0.5357 | 0.6819 |
| heartC | C | experimental.difficulty_normalized | 0.6492 | 0.8552 | 0.0552 | 0.0047 | 0.3461 | 0.4942 |
| heartH | C | experimental.difficulty_normalized | 0.6226 | 0.8645 | 0.0340 | 0.0301 | 0.3473 | 0.4829 |
| heartS | C | experimental.difficulty_normalized | 0.6687 | 0.8244 | 0.0392 | 0.0136 | 0.3177 | 0.4354 |
| heat | C | experimental.difficulty_normalized | 0.8674 | 0.9987 | 0.0052 | 0.0952 | 0.0374 | 0.8750 |
| hepati | C | experimental.difficulty_normalized | 0.6953 | 0.9102 | 0.0586 | 0.0215 | 0.2832 | 0.5337 |
| image | C | experimental.difficulty_normalized | 0.8889 | 0.9947 | 0.0194 | 0.1007 | 0.0103 | 0.8229 |
| iono | C | experimental.difficulty_normalized | 0.7213 | 0.9675 | 0.0389 | 0.0930 | 0.1857 | 0.6844 |
| iris | C | experimental.difficulty_normalized | 0.5622 | 0.9856 | 0.0390 | 0.0430 | 0.3948 | 0.8370 |
| je4042 | C | experimental.difficulty_normalized | 0.4868 | 0.8019 | 0.0574 | 0.0193 | 0.4938 | 0.6061 |
| je4243 | C | experimental.difficulty_normalized | 0.4113 | 0.6675 | 0.0483 | 0.0033 | 0.5854 | 0.6226 |
| kc1 | C | experimental.difficulty_normalized | 0.4717 | 0.7974 | 0.0502 | 0.0031 | 0.5252 | 0.6136 |
| kc2 | C | experimental.difficulty_normalized | 0.5955 | 0.8475 | 0.0529 | 0.0120 | 0.3925 | 0.5228 |
| kc3 | C | experimental.difficulty_normalized | 0.6684 | 0.9164 | 0.0548 | 0.0373 | 0.2944 | 0.5898 |
| liver | C | experimental.difficulty_normalized | 0.4254 | 0.7868 | 0.1231 | 0.0042 | 0.5704 | 0.7106 |
| pc1req | C | experimental.difficulty_normalized | 0.3503 | 0.6285 | -0.0382 | 0.0021 | 0.6476 | 0.6006 |
| pc4 | C | experimental.difficulty_normalized | 0.8083 | 0.9379 | 0.0420 | 0.0635 | 0.1282 | 0.4951 |
| sonar | C | experimental.difficulty_normalized | 0.6566 | 0.9258 | 0.0448 | 0.0280 | 0.3153 | 0.5605 |
| spect | C | experimental.difficulty_normalized | 0.5944 | 0.8743 | -0.0074 | 0.0263 | 0.3793 | 0.3541 |
| spectf | C | experimental.difficulty_normalized | 0.5609 | 0.8498 | 0.0387 | 0.0222 | 0.4169 | 0.5097 |
| steel | C | experimental.difficulty_normalized | 0.6920 | 0.8443 | 0.0823 | 0.0115 | 0.2964 | 0.5021 |
| tae | C | experimental.difficulty_normalized | 0.1570 | 0.3496 | -0.1822 | 0.0172 | 0.8258 | 0.7940 |
| transfusion | C | experimental.difficulty_normalized | 0.4970 | 0.7695 | 0.0467 | 0.0079 | 0.4950 | 0.5705 |
| ttt | C | experimental.difficulty_normalized | 0.8900 | 0.9959 | 0.0199 | 0.0979 | 0.0120 | 0.8493 |
| user | C | experimental.difficulty_normalized | 0.7805 | 0.9196 | 0.0307 | 0.0573 | 0.1621 | 0.4204 |
| vehicle | C | experimental.difficulty_normalized | 0.5788 | 0.8805 | 0.1346 | 0.0235 | 0.3976 | 0.6901 |
| vote | C | experimental.difficulty_normalized | 0.7722 | 0.9031 | 0.0685 | 0.0476 | 0.1801 | 0.4989 |
| vowel | C | experimental.difficulty_normalized | 0.6891 | 0.9765 | 0.0341 | 0.0492 | 0.2617 | 0.6305 |
| wave | C | experimental.difficulty_normalized | 0.8259 | 0.9064 | 0.0500 | 0.0206 | 0.1535 | 0.4241 |
| wbc | C | experimental.difficulty_normalized | 0.7331 | 0.9715 | 0.0102 | 0.1082 | 0.1587 | 0.4741 |
| whole | C | experimental.difficulty_normalized | 0.3742 | 0.7358 | 0.0358 | 0.0000 | 0.6258 | 0.6691 |
| wine | C | experimental.difficulty_normalized | 0.0562 | 1.0000 | 0.0357 | 0.0000 | 0.9438 | 1.0000 |
| wineR | C | experimental.difficulty_normalized | 0.4118 | 0.7834 | 0.1103 | 0.0101 | 0.5781 | 0.6952 |
| wineW | C | experimental.difficulty_normalized | 0.4661 | 0.8100 | 0.1447 | 0.0095 | 0.5244 | 0.7067 |
| yeast | C | experimental.difficulty_normalized | 0.3695 | 0.6464 | 0.0383 | 0.0104 | 0.6201 | 0.6534 |
| balance | G | experimental.ambiguity_normalized_novelty_penalized | 0.8235 | 0.9186 | 0.0818 | 0.0786 | 0.0980 | 0.5745 |
| breast_cancer | G | experimental.ambiguity_normalized_novelty_penalized | 0.8402 | 0.9792 | 0.0231 | 0.1409 | 0.0189 | 0.5874 |
| cars | G | experimental.ambiguity_normalized_novelty_penalized | 0.7934 | 0.9927 | 0.0268 | 0.2000 | 0.0066 | 0.8073 |
| cmc | G | experimental.ambiguity_normalized_novelty_penalized | 0.3433 | 0.6487 | 0.1396 | 0.0090 | 0.6478 | 0.7284 |
| colic | G | experimental.ambiguity_normalized_novelty_penalized | 0.7250 | 0.8539 | 0.0345 | 0.0608 | 0.2142 | 0.3930 |
| cool | G | experimental.ambiguity_normalized_novelty_penalized | 0.8515 | 0.9849 | 0.0356 | 0.1241 | 0.0244 | 0.7354 |
| creditA | G | experimental.ambiguity_normalized_novelty_penalized | 0.7860 | 0.8892 | 0.0313 | 0.0794 | 0.1346 | 0.3796 |
| diabetes | G | experimental.ambiguity_normalized_novelty_penalized | 0.6074 | 0.8260 | 0.0701 | 0.0136 | 0.3791 | 0.5405 |
| ecoli | G | experimental.ambiguity_normalized_novelty_penalized | 0.5899 | 0.9130 | 0.0542 | 0.0529 | 0.3572 | 0.6225 |
| german | G | experimental.ambiguity_normalized_novelty_penalized | 0.5115 | 0.7529 | 0.0963 | 0.0049 | 0.4837 | 0.6033 |
| glass | G | experimental.ambiguity_normalized_novelty_penalized | 0.4362 | 0.7703 | 0.0075 | 0.0181 | 0.5457 | 0.5723 |
| haberman | G | experimental.ambiguity_normalized_novelty_penalized | 0.4897 | 0.7402 | 0.0938 | 0.0246 | 0.4858 | 0.6276 |
| heartC | G | experimental.ambiguity_normalized_novelty_penalized | 0.6871 | 0.8496 | 0.0496 | 0.0277 | 0.2852 | 0.4475 |
| heartH | G | experimental.ambiguity_normalized_novelty_penalized | 0.6795 | 0.8587 | 0.0282 | 0.0584 | 0.2621 | 0.4169 |
| heartS | G | experimental.ambiguity_normalized_novelty_penalized | 0.7107 | 0.8100 | 0.0248 | 0.0481 | 0.2412 | 0.3612 |
| heat | G | experimental.ambiguity_normalized_novelty_penalized | 0.7812 | 0.9986 | 0.0051 | 0.1882 | 0.0306 | 0.8750 |
| hepati | G | experimental.ambiguity_normalized_novelty_penalized | 0.7032 | 0.9094 | 0.0578 | 0.0789 | 0.2179 | 0.5309 |
| image | G | experimental.ambiguity_normalized_novelty_penalized | 0.8348 | 0.9948 | 0.0194 | 0.1607 | 0.0046 | 0.8280 |
| iono | G | experimental.ambiguity_normalized_novelty_penalized | 0.7200 | 0.9674 | 0.0389 | 0.1162 | 0.1638 | 0.6802 |
| iris | G | experimental.ambiguity_normalized_novelty_penalized | 0.5822 | 0.9856 | 0.0390 | 0.0578 | 0.3600 | 0.8370 |
| je4042 | G | experimental.ambiguity_normalized_novelty_penalized | 0.5160 | 0.7999 | 0.0555 | 0.0391 | 0.4449 | 0.5756 |
| je4243 | G | experimental.ambiguity_normalized_novelty_penalized | 0.4542 | 0.6715 | 0.0523 | 0.0107 | 0.5352 | 0.5900 |
| kc1 | G | experimental.ambiguity_normalized_novelty_penalized | 0.4853 | 0.7963 | 0.0490 | 0.0053 | 0.5094 | 0.5974 |
| kc2 | G | experimental.ambiguity_normalized_novelty_penalized | 0.6117 | 0.8489 | 0.0543 | 0.0285 | 0.3598 | 0.5140 |
| kc3 | G | experimental.ambiguity_normalized_novelty_penalized | 0.6602 | 0.9247 | 0.0632 | 0.0872 | 0.2526 | 0.6326 |
| liver | G | experimental.ambiguity_normalized_novelty_penalized | 0.4702 | 0.7726 | 0.1088 | 0.0090 | 0.5208 | 0.6588 |
| pc1req | G | experimental.ambiguity_normalized_novelty_penalized | 0.3704 | 0.6452 | -0.0215 | 0.0169 | 0.6127 | 0.5934 |
| pc4 | G | experimental.ambiguity_normalized_novelty_penalized | 0.8083 | 0.9397 | 0.0438 | 0.0919 | 0.0999 | 0.5043 |
| sonar | G | experimental.ambiguity_normalized_novelty_penalized | 0.7148 | 0.9094 | 0.0285 | 0.0534 | 0.2317 | 0.4385 |
| spect | G | experimental.ambiguity_normalized_novelty_penalized | 0.6187 | 0.8884 | 0.0066 | 0.0419 | 0.3394 | 0.3541 |
| spectf | G | experimental.ambiguity_normalized_novelty_penalized | 0.6218 | 0.8647 | 0.0535 | 0.0222 | 0.3560 | 0.5097 |
| steel | G | experimental.ambiguity_normalized_novelty_penalized | 0.7373 | 0.8414 | 0.0795 | 0.0225 | 0.2402 | 0.4658 |
| tae | G | experimental.ambiguity_normalized_novelty_penalized | 0.2115 | 0.3814 | -0.1504 | 0.0351 | 0.7534 | 0.7340 |
| transfusion | G | experimental.ambiguity_normalized_novelty_penalized | 0.5314 | 0.7640 | 0.0412 | 0.0218 | 0.4469 | 0.5528 |
| ttt | G | experimental.ambiguity_normalized_novelty_penalized | 0.7936 | 0.9958 | 0.0197 | 0.1954 | 0.0110 | 0.8493 |
| user | G | experimental.ambiguity_normalized_novelty_penalized | 0.7676 | 0.9220 | 0.0332 | 0.0955 | 0.1369 | 0.4463 |
| vehicle | G | experimental.ambiguity_normalized_novelty_penalized | 0.6195 | 0.8706 | 0.1247 | 0.0375 | 0.3430 | 0.6416 |
| vote | G | experimental.ambiguity_normalized_novelty_penalized | 0.7729 | 0.8971 | 0.0625 | 0.0972 | 0.1299 | 0.5002 |
| vowel | G | experimental.ambiguity_normalized_novelty_penalized | 0.6937 | 0.9758 | 0.0335 | 0.0653 | 0.2410 | 0.6184 |
| wave | G | experimental.ambiguity_normalized_novelty_penalized | 0.8316 | 0.9033 | 0.0469 | 0.0566 | 0.1117 | 0.4188 |
| wbc | G | experimental.ambiguity_normalized_novelty_penalized | 0.6934 | 0.9703 | 0.0090 | 0.1596 | 0.1470 | 0.4741 |
| whole | G | experimental.ambiguity_normalized_novelty_penalized | 0.3922 | 0.7392 | 0.0392 | 0.0136 | 0.5942 | 0.6500 |
| wine | G | experimental.ambiguity_normalized_novelty_penalized | 0.0636 | 1.0000 | 0.0357 | 0.0000 | 0.9364 | 1.0000 |
| wineR | G | experimental.ambiguity_normalized_novelty_penalized | 0.4571 | 0.7838 | 0.1107 | 0.0235 | 0.5194 | 0.6636 |
| wineW | G | experimental.ambiguity_normalized_novelty_penalized | 0.5135 | 0.7946 | 0.1293 | 0.0299 | 0.4567 | 0.6503 |
| yeast | G | experimental.ambiguity_normalized_novelty_penalized | 0.4025 | 0.6442 | 0.0361 | 0.0325 | 0.5649 | 0.6187 |
