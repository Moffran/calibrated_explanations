# Scenario 11 - Matched operating-point reject selection

Rows: 2760

## Key findings

- Selects confidence values closest to target reject rates instead of averaging over the confidence grid.
- Primary decision gate is A vs C for direct difficulty-normalized scoring.
- Secondary diagnostic is C vs G for the novelty-aware variant.
- C minus A accepted-accuracy deltas by target reject rate are 0.10: +0.0037, 0.20: +0.0000, 0.30: -0.0050, 0.40: -0.0060.
- The strongest matched operating point is target 0.10, with mean accepted-accuracy delta +0.0037.
- Across targets, C increased ambiguity in mean fraction-positive 0.4217 and increased novelty in fraction-positive 0.4076. This operating-point quick run does not show the same consistent ambiguity-up, novelty-down geometry seen in Scenario 9.
- Across targets, C minus A mean difficulty-reject-AUC delta is +0.0169.
- Step 8 is not justified yet; matched operating-point evidence is mixed and does not support public API promotion.
- G minus C mean novelty-rate delta across targets is +0.0208; empty-set delta is +0.0208.
- G minus C mean accepted-accuracy delta across targets is -0.0009.
- G minus C novelty-reject-AUC delta is +0.0551; ambiguity-rate delta is -0.0341.
- The novelty-aware strategy should remain internal only; it is not promotion-ready.

## Outcome snapshot

- **rows**: 2760
- **sweep_rows**: 21390
- **delta_rows**: 1840
- **datasets**: 46
- **seeds**: 5
- **target_reject_rates**: [0.1, 0.2, 0.3, 0.4]
- **A_vs_C_accepted_accuracy_delta_by_target**: {'0.10': 0.003667105136906213, '0.20': 4.1067336827315746e-05, '0.30': -0.004969011196508569, '0.40': -0.006037448182867154}
- **A_vs_C_best_target_by_accepted_accuracy**: 0.1000
- **A_vs_C_best_accepted_accuracy_delta**: 0.0037
- **A_vs_C_mean_difficulty_reject_auc_delta**: 0.0169
- **C_vs_G_mean_novelty_rate_delta**: 0.0208
- **C_vs_G_mean_accepted_accuracy_delta**: -0.0009
- **C_vs_G_mean_novelty_reject_auc_delta**: 0.0551
- **promotion_recommendation**: do_not_promote
- **novelty_strategy_recommendation**: continue_experimental

## Selected Operating Points

| arm_code | target_reject_rate | observed_reject_rate | reject_rate_target_abs_error | selected_confidence | accepted_accuracy | ambiguity_rate | novelty_rate | difficulty_reject_auc | singleton_precision | singleton_recall |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.1000 | 0.1100 | 0.0334 | 0.7365 | 0.8280 | 0.0360 | 0.0740 | 0.5097 | 0.8299 | 0.7361 |
| A | 0.2000 | 0.2078 | 0.0314 | 0.6909 | 0.8444 | 0.0532 | 0.1546 | 0.5131 | 0.8477 | 0.6698 |
| A | 0.3000 | 0.3040 | 0.0273 | 0.6388 | 0.8617 | 0.0766 | 0.2274 | 0.5118 | 0.8656 | 0.6010 |
| A | 0.4000 | 0.3959 | 0.0304 | 0.6445 | 0.8762 | 0.1426 | 0.2533 | 0.5142 | 0.8794 | 0.5300 |
| C | 0.1000 | 0.1640 | 0.0691 | 0.7780 | 0.8317 | 0.0876 | 0.0764 | 0.5648 | 0.8323 | 0.7012 |
| C | 0.2000 | 0.2331 | 0.0388 | 0.7242 | 0.8445 | 0.0871 | 0.1460 | 0.5157 | 0.8446 | 0.6508 |
| C | 0.3000 | 0.3121 | 0.0213 | 0.6567 | 0.8567 | 0.0778 | 0.2343 | 0.4841 | 0.8564 | 0.5902 |
| C | 0.4000 | 0.4049 | 0.0178 | 0.6611 | 0.8702 | 0.1516 | 0.2533 | 0.5511 | 0.8688 | 0.5173 |
| G | 0.1000 | 0.1394 | 0.0470 | 0.8037 | 0.8276 | 0.0499 | 0.0896 | 0.5922 | 0.8290 | 0.7163 |
| G | 0.2000 | 0.2153 | 0.0239 | 0.7714 | 0.8420 | 0.0610 | 0.1543 | 0.5819 | 0.8442 | 0.6637 |
| G | 0.3000 | 0.3051 | 0.0160 | 0.7250 | 0.8598 | 0.0716 | 0.2334 | 0.5842 | 0.8582 | 0.5966 |
| G | 0.4000 | 0.4010 | 0.0129 | 0.6764 | 0.8703 | 0.0853 | 0.3157 | 0.6076 | 0.8716 | 0.5220 |

## Pairwise Delta Aggregates

| comparison_group | target_reject_rate | paired_groups | base_mean_selected_confidence | candidate_mean_selected_confidence | base_mean_target_abs_error | candidate_mean_target_abs_error | C_minus_A_accepted_accuracy_mean | C_minus_A_accepted_accuracy_median | C_minus_A_accepted_accuracy_std | C_minus_A_accepted_accuracy_fraction_positive | C_minus_A_accepted_accuracy_finite_groups | C_minus_A_empirical_coverage_mean | C_minus_A_empirical_coverage_median | C_minus_A_empirical_coverage_std | C_minus_A_empirical_coverage_fraction_positive | C_minus_A_empirical_coverage_finite_groups | C_minus_A_observed_reject_rate_mean | C_minus_A_observed_reject_rate_median | C_minus_A_observed_reject_rate_std | C_minus_A_observed_reject_rate_fraction_positive | C_minus_A_observed_reject_rate_finite_groups | C_minus_A_ambiguity_rate_mean | C_minus_A_ambiguity_rate_median | C_minus_A_ambiguity_rate_std | C_minus_A_ambiguity_rate_fraction_positive | C_minus_A_ambiguity_rate_finite_groups | C_minus_A_novelty_rate_mean | C_minus_A_novelty_rate_median | C_minus_A_novelty_rate_std | C_minus_A_novelty_rate_fraction_positive | C_minus_A_novelty_rate_finite_groups | C_minus_A_rejected_error_capture_rate_mean | C_minus_A_rejected_error_capture_rate_median | C_minus_A_rejected_error_capture_rate_std | C_minus_A_rejected_error_capture_rate_fraction_positive | C_minus_A_rejected_error_capture_rate_finite_groups | C_minus_A_difficulty_reject_auc_mean | C_minus_A_difficulty_reject_auc_median | C_minus_A_difficulty_reject_auc_std | C_minus_A_difficulty_reject_auc_fraction_positive | C_minus_A_difficulty_reject_auc_finite_groups | C_minus_A_difficulty_gap_rejected_minus_accepted_mean | C_minus_A_difficulty_gap_rejected_minus_accepted_median | C_minus_A_difficulty_gap_rejected_minus_accepted_std | C_minus_A_difficulty_gap_rejected_minus_accepted_fraction_positive | C_minus_A_difficulty_gap_rejected_minus_accepted_finite_groups | C_minus_A_singleton_precision_mean | C_minus_A_singleton_precision_median | C_minus_A_singleton_precision_std | C_minus_A_singleton_precision_fraction_positive | C_minus_A_singleton_precision_finite_groups | C_minus_A_singleton_recall_mean | C_minus_A_singleton_recall_median | C_minus_A_singleton_recall_std | C_minus_A_singleton_recall_fraction_positive | C_minus_A_singleton_recall_finite_groups | G_minus_C_accepted_accuracy_mean | G_minus_C_accepted_accuracy_median | G_minus_C_accepted_accuracy_std | G_minus_C_accepted_accuracy_fraction_positive | G_minus_C_accepted_accuracy_finite_groups | G_minus_C_novelty_rate_mean | G_minus_C_novelty_rate_median | G_minus_C_novelty_rate_std | G_minus_C_novelty_rate_fraction_positive | G_minus_C_novelty_rate_finite_groups | G_minus_C_ambiguity_rate_mean | G_minus_C_ambiguity_rate_median | G_minus_C_ambiguity_rate_std | G_minus_C_ambiguity_rate_fraction_positive | G_minus_C_ambiguity_rate_finite_groups | G_minus_C_novelty_reject_auc_mean | G_minus_C_novelty_reject_auc_median | G_minus_C_novelty_reject_auc_std | G_minus_C_novelty_reject_auc_fraction_positive | G_minus_C_novelty_reject_auc_finite_groups | G_minus_C_rejected_error_capture_rate_mean | G_minus_C_rejected_error_capture_rate_median | G_minus_C_rejected_error_capture_rate_std | G_minus_C_rejected_error_capture_rate_fraction_positive | G_minus_C_rejected_error_capture_rate_finite_groups | G_minus_C_difficulty_reject_auc_mean | G_minus_C_difficulty_reject_auc_median | G_minus_C_difficulty_reject_auc_std | G_minus_C_difficulty_reject_auc_fraction_positive | G_minus_C_difficulty_reject_auc_finite_groups | G_minus_C_empty_set_rate_mean | G_minus_C_empty_set_rate_median | G_minus_C_empty_set_rate_std | G_minus_C_empty_set_rate_fraction_positive | G_minus_C_empty_set_rate_finite_groups | G_minus_C_multilabel_rate_mean | G_minus_C_multilabel_rate_median | G_minus_C_multilabel_rate_std | G_minus_C_multilabel_rate_fraction_positive | G_minus_C_multilabel_rate_finite_groups | G_minus_C_singleton_precision_mean | G_minus_C_singleton_precision_median | G_minus_C_singleton_precision_std | G_minus_C_singleton_precision_fraction_positive | G_minus_C_singleton_precision_finite_groups | G_minus_C_singleton_recall_mean | G_minus_C_singleton_recall_median | G_minus_C_singleton_recall_std | G_minus_C_singleton_recall_fraction_positive | G_minus_C_singleton_recall_finite_groups |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_vs_C | 0.1000 | 230 | 0.7365 | 0.7780 | 0.0334 | 0.0691 | 0.0037 | 0.0000 | 0.0249 | 0.4565 | 230.0000 | 0.0127 | 0.0140 | 0.0672 | 0.5615 | 130.0000 | 0.0540 | 0.0180 | 0.0976 | 0.6609 | 230.0000 | 0.0516 | 0.0458 | 0.0854 | 0.6174 | 230.0000 | 0.0024 | 0.0000 | 0.0721 | 0.3957 | 230.0000 | 0.0555 | 0.0000 | 0.1735 | 0.4649 | 228.0000 | 0.0559 | 0.0060 | 0.1967 | 0.5156 | 225.0000 | 0.2455 | 0.1317 | 0.5137 | 0.6356 | 225.0000 | 0.0024 | 0.0000 | 0.0288 | 0.4522 | 230.0000 | -0.0349 | -0.0130 | 0.0656 | 0.1739 | 230.0000 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| A_vs_C | 0.2000 | 230 | 0.6909 | 0.7242 | 0.0314 | 0.0388 | 0.0000 | 0.0000 | 0.0243 | 0.3739 | 230.0000 | 0.0232 | 0.0103 | 0.1008 | 0.5154 | 130.0000 | 0.0253 | 0.0092 | 0.0693 | 0.5696 | 230.0000 | 0.0339 | 0.0000 | 0.1075 | 0.4478 | 230.0000 | -0.0086 | 0.0000 | 0.1035 | 0.3826 | 230.0000 | 0.0174 | 0.0000 | 0.1324 | 0.3553 | 228.0000 | 0.0026 | -0.0249 | 0.1797 | 0.3826 | 230.0000 | 0.1031 | 0.0000 | 0.3468 | 0.4652 | 230.0000 | -0.0032 | 0.0000 | 0.0306 | 0.3522 | 230.0000 | -0.0190 | -0.0114 | 0.0522 | 0.2478 | 230.0000 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| A_vs_C | 0.3000 | 230 | 0.6388 | 0.6567 | 0.0273 | 0.0213 | -0.0050 | 0.0000 | 0.0278 | 0.3043 | 230.0000 | -0.0129 | 0.0000 | 0.1407 | 0.4000 | 130.0000 | 0.0081 | 0.0000 | 0.0533 | 0.4609 | 230.0000 | 0.0012 | 0.0000 | 0.1401 | 0.3261 | 230.0000 | 0.0068 | 0.0000 | 0.1340 | 0.4000 | 230.0000 | -0.0098 | 0.0000 | 0.1157 | 0.2544 | 228.0000 | -0.0277 | -0.0421 | 0.1843 | 0.2522 | 230.0000 | 0.0141 | -0.0248 | 0.2962 | 0.3261 | 230.0000 | -0.0092 | 0.0000 | 0.0326 | 0.2652 | 230.0000 | -0.0108 | -0.0037 | 0.0466 | 0.3348 | 230.0000 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| A_vs_C | 0.4000 | 230 | 0.6445 | 0.6611 | 0.0304 | 0.0178 | -0.0060 | 0.0000 | 0.0317 | 0.2826 | 230.0000 | 0.0225 | 0.0000 | 0.2195 | 0.4231 | 130.0000 | 0.0090 | 0.0015 | 0.0505 | 0.5000 | 230.0000 | 0.0090 | 0.0000 | 0.2078 | 0.2957 | 230.0000 | -0.0000 | 0.0000 | 0.1948 | 0.4522 | 230.0000 | -0.0123 | 0.0000 | 0.1108 | 0.2588 | 228.0000 | 0.0369 | -0.0154 | 0.1990 | 0.3826 | 230.0000 | 0.0602 | -0.0048 | 0.2688 | 0.4087 | 230.0000 | -0.0106 | 0.0000 | 0.0363 | 0.2348 | 230.0000 | -0.0127 | -0.0070 | 0.0490 | 0.2913 | 230.0000 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| C_vs_G | 0.1000 | 230 | 0.7780 | 0.8037 | 0.0691 | 0.0470 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | -0.0042 | -0.0011 | 0.0169 | 0.2826 | 230.0000 | 0.0132 | 0.0067 | 0.0540 | 0.5261 | 230.0000 | -0.0378 | -0.0292 | 0.0628 | 0.1435 | 230.0000 | 0.0170 | 0.0000 | 0.1858 | 0.4652 | 230.0000 | -0.0551 | -0.0061 | 0.1354 | 0.1316 | 228.0000 | 0.0275 | 0.0000 | 0.2101 | 0.4783 | 230.0000 | 0.0132 | 0.0067 | 0.0540 | 0.5261 | 230.0000 | -0.0378 | -0.0292 | 0.0628 | 0.1435 | 230.0000 | -0.0034 | -0.0000 | 0.0198 | 0.3043 | 230.0000 | 0.0151 | 0.0052 | 0.0348 | 0.5217 | 230.0000 |
| C_vs_G | 0.2000 | 230 | 0.7242 | 0.7714 | 0.0388 | 0.0239 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | -0.0025 | -0.0000 | 0.0227 | 0.2957 | 230.0000 | 0.0083 | 0.0000 | 0.0923 | 0.4696 | 230.0000 | -0.0261 | 0.0000 | 0.0986 | 0.1565 | 230.0000 | 0.0512 | 0.0563 | 0.1854 | 0.6000 | 230.0000 | -0.0374 | 0.0000 | 0.1459 | 0.2061 | 228.0000 | 0.0663 | 0.0710 | 0.2151 | 0.6261 | 230.0000 | 0.0083 | 0.0000 | 0.0923 | 0.4696 | 230.0000 | -0.0261 | 0.0000 | 0.0986 | 0.1565 | 230.0000 | -0.0004 | 0.0000 | 0.0242 | 0.3217 | 230.0000 | 0.0129 | 0.0000 | 0.0345 | 0.4913 | 230.0000 |
| C_vs_G | 0.3000 | 230 | 0.6567 | 0.7250 | 0.0213 | 0.0160 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.0031 | 0.0000 | 0.0267 | 0.3739 | 230.0000 | -0.0009 | 0.0005 | 0.1344 | 0.5000 | 230.0000 | -0.0062 | 0.0000 | 0.1392 | 0.1652 | 230.0000 | 0.0886 | 0.1078 | 0.1754 | 0.7043 | 230.0000 | -0.0062 | 0.0000 | 0.1117 | 0.2719 | 228.0000 | 0.1001 | 0.1186 | 0.2181 | 0.7174 | 230.0000 | -0.0009 | 0.0005 | 0.1344 | 0.5000 | 230.0000 | -0.0062 | 0.0000 | 0.1392 | 0.1652 | 230.0000 | 0.0018 | 0.0000 | 0.0303 | 0.3522 | 230.0000 | 0.0064 | 0.0000 | 0.0349 | 0.4478 | 230.0000 |
| C_vs_G | 0.4000 | 230 | 0.6611 | 0.6764 | 0.0178 | 0.0129 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.0001 | 0.0000 | 0.0266 | 0.3478 | 230.0000 | 0.0624 | 0.0089 | 0.1784 | 0.5609 | 230.0000 | -0.0663 | 0.0000 | 0.1779 | 0.0870 | 230.0000 | 0.0637 | 0.0838 | 0.1699 | 0.5957 | 230.0000 | -0.0088 | 0.0000 | 0.1117 | 0.2588 | 228.0000 | 0.0565 | 0.0920 | 0.2166 | 0.5913 | 230.0000 | 0.0624 | 0.0089 | 0.1784 | 0.5609 | 230.0000 | -0.0663 | 0.0000 | 0.1779 | 0.0870 | 230.0000 | 0.0028 | 0.0000 | 0.0312 | 0.3696 | 230.0000 | 0.0047 | 0.0000 | 0.0342 | 0.4087 | 230.0000 |

## Required Analyses

### A vs C

1. Does direct difficulty normalization improve accepted accuracy at matched reject rates?
C minus A accepted-accuracy deltas by target reject rate are 0.10: +0.0037, 0.20: +0.0000, 0.30: -0.0050, 0.40: -0.0060.
2. At which reject-rate targets is it most useful?
The strongest matched operating point is target 0.10, with mean accepted-accuracy delta +0.0037.
3. Does it consistently increase ambiguity and decrease novelty rejection?
Across targets, C increased ambiguity in mean fraction-positive 0.4217 and increased novelty in fraction-positive 0.4076. This operating-point quick run does not show the same consistent ambiguity-up, novelty-down geometry seen in Scenario 9.
4. Does it select higher-difficulty cases for rejection?
Across targets, C minus A mean difficulty-reject-AUC delta is +0.0169.
5. Is Step 8 justified?
Step 8 is not justified yet; matched operating-point evidence is mixed and does not support public API promotion.

### C vs G

1. Does the novelty-aware variant improve novelty routing at matched reject rates?
G minus C mean novelty-rate delta across targets is +0.0208; empty-set delta is +0.0208.
2. Does it improve or harm accepted accuracy?
G minus C mean accepted-accuracy delta across targets is -0.0009.
3. Does it improve novelty selectivity or merely increase empty sets?
G minus C novelty-reject-AUC delta is +0.0551; ambiguity-rate delta is -0.0341.
4. Should it remain internal only?
The novelty-aware strategy should remain internal only; it is not promotion-ready.

## Per-Dataset Arm Comparison (all datasets)

Mean over seeds and selected operating points. Rows sorted by arm_code, dataset.

| dataset | arm_code | observed_reject_rate | accepted_accuracy | accuracy_delta | difficulty_reject_auc |
|---|---|---|---|---|---|
| balance | A | 0.2456 | 0.9612 | 0.1244 | 0.4844 |
| breast_cancer | A | 0.2434 | 0.9880 | 0.0319 | 0.3568 |
| cars | A | 0.2513 | 0.9982 | 0.0323 | 0.3990 |
| cmc | A | 0.2449 | 0.5267 | 0.0175 | 0.4854 |
| colic | A | 0.2410 | 0.8817 | 0.0623 | 0.4959 |
| cool | A | 0.2448 | 0.9960 | 0.0466 | 0.5803 |
| creditA | A | 0.2496 | 0.9185 | 0.0605 | 0.5525 |
| diabetes | A | 0.2386 | 0.8152 | 0.0594 | 0.5717 |
| ecoli | A | 0.2471 | 0.9064 | 0.0476 | 0.6046 |
| german | A | 0.2194 | 0.6999 | 0.0434 | 0.6106 |
| glass | A | 0.2407 | 0.7656 | 0.0028 | 0.5671 |
| haberman | A | 0.2421 | 0.7129 | 0.0673 | 0.6170 |
| heartC | A | 0.2541 | 0.8738 | 0.0738 | 0.5079 |
| heartH | A | 0.2525 | 0.8841 | 0.0536 | 0.5085 |
| heartS | A | 0.2481 | 0.8251 | 0.0399 | 0.5738 |
| heat | A | 0.2474 | 0.9996 | 0.0061 | 0.6898 |
| hepati | A | 0.2403 | 0.9332 | 0.0816 | 0.8529 |
| image | A | 0.2496 | 0.9994 | 0.0241 | 0.3668 |
| iono | A | 0.2521 | 0.9790 | 0.0504 | 0.5252 |
| iris | A | 0.2950 | 0.9876 | 0.0409 | 0.4041 |
| je4042 | A | 0.2352 | 0.7817 | 0.0372 | 0.5084 |
| je4243 | A | 0.2473 | 0.6370 | 0.0178 | 0.4139 |
| kc1 | A | 0.2345 | 0.7973 | 0.0500 | 0.6875 |
| kc2 | A | 0.2392 | 0.8616 | 0.0670 | 0.5138 |
| kc3 | A | 0.2562 | 0.9374 | 0.0759 | 0.6814 |
| liver | A | 0.2442 | 0.7245 | 0.0608 | 0.5647 |
| pc1req | A | 0.2357 | 0.6429 | -0.0237 | 0.3230 |
| pc4 | A | 0.2559 | 0.9705 | 0.0746 | 0.4905 |
| sonar | A | 0.2488 | 0.9253 | 0.0444 | 0.4885 |
| spect | A | 0.2455 | 0.8926 | 0.0108 | 0.2897 |
| spectf | A | 0.2472 | 0.8650 | 0.0539 | 0.3867 |
| steel | A | 0.2501 | 0.8457 | 0.0838 | 0.4076 |
| tae | A | 0.2548 | 0.5223 | -0.0132 | 0.4649 |
| transfusion | A | 0.2386 | 0.7765 | 0.0537 | 0.4876 |
| ttt | A | 0.2510 | 0.9994 | 0.0234 | 0.3465 |
| user | A | 0.2488 | 0.9442 | 0.0553 | 0.6642 |
| vehicle | A | 0.2468 | 0.8323 | 0.0864 | 0.5309 |
| vote | A | 0.2543 | 0.9179 | 0.0833 | 0.6964 |
| vowel | A | 0.2965 | 0.9914 | 0.0490 | 0.4160 |
| wave | A | 0.2497 | 0.9288 | 0.0724 | 0.6645 |
| wbc | A | 0.2543 | 0.9787 | 0.0174 | 0.3087 |
| whole | A | 0.2403 | 0.7289 | 0.0289 | 0.5446 |
| wine | A | 0.5389 | 1.0000 | 0.0389 | 0.4781 |
| wineR | A | 0.2503 | 0.7183 | 0.0452 | 0.4784 |
| wineW | A | 0.2478 | 0.7193 | 0.0539 | 0.5074 |
| yeast | A | 0.2438 | 0.6276 | 0.0195 | 0.4422 |
| balance | C | 0.2508 | 0.9606 | 0.1238 | 0.4369 |
| breast_cancer | C | 0.2535 | 0.9868 | 0.0307 | 0.3262 |
| cars | C | 0.2506 | 0.9987 | 0.0328 | 0.3825 |
| cmc | C | 0.3544 | 0.5348 | 0.0256 | 0.6557 |
| colic | C | 0.2507 | 0.8682 | 0.0488 | 0.4927 |
| cool | C | 0.2500 | 0.9959 | 0.0466 | 0.5502 |
| creditA | C | 0.2478 | 0.9167 | 0.0587 | 0.5069 |
| diabetes | C | 0.2581 | 0.8067 | 0.0508 | 0.6541 |
| ecoli | C | 0.2647 | 0.8975 | 0.0387 | 0.6036 |
| german | C | 0.2505 | 0.6930 | 0.0364 | 0.6698 |
| glass | C | 0.3349 | 0.8014 | 0.0386 | 0.6768 |
| haberman | C | 0.2719 | 0.7034 | 0.0578 | 0.5223 |
| heartC | C | 0.2516 | 0.8588 | 0.0588 | 0.4809 |
| heartH | C | 0.2754 | 0.8732 | 0.0427 | 0.5752 |
| heartS | C | 0.2435 | 0.8269 | 0.0418 | 0.5450 |
| heat | C | 0.2487 | 0.9996 | 0.0061 | 0.6408 |
| hepati | C | 0.2452 | 0.9282 | 0.0766 | 0.8157 |
| image | C | 0.2492 | 0.9981 | 0.0228 | 0.3402 |
| iono | C | 0.2571 | 0.9784 | 0.0499 | 0.4879 |
| iris | C | 0.3183 | 0.9733 | 0.0266 | 0.4334 |
| je4042 | C | 0.3028 | 0.8027 | 0.0583 | 0.6206 |
| je4243 | C | 0.2932 | 0.6289 | 0.0097 | 0.5588 |
| kc1 | C | 0.2632 | 0.7845 | 0.0372 | 0.5944 |
| kc2 | C | 0.2520 | 0.8463 | 0.0517 | 0.5352 |
| kc3 | C | 0.2508 | 0.9112 | 0.0497 | 0.5685 |
| liver | C | 0.2746 | 0.7271 | 0.0633 | 0.6078 |
| pc1req | C | 0.2619 | 0.6553 | -0.0114 | 0.3536 |
| pc4 | C | 0.2524 | 0.9657 | 0.0698 | 0.4518 |
| sonar | C | 0.2524 | 0.9217 | 0.0407 | 0.5265 |
| spect | C | 0.2545 | 0.8951 | 0.0133 | 0.2748 |
| spectf | C | 0.2648 | 0.8481 | 0.0370 | 0.4728 |
| steel | C | 0.2536 | 0.8423 | 0.0803 | 0.4310 |
| tae | C | 0.4855 | 0.5252 | -0.0103 | 0.6825 |
| transfusion | C | 0.2644 | 0.7773 | 0.0545 | 0.4488 |
| ttt | C | 0.2518 | 0.9997 | 0.0237 | 0.3358 |
| user | C | 0.2519 | 0.9369 | 0.0480 | 0.6178 |
| vehicle | C | 0.2918 | 0.8462 | 0.1004 | 0.4908 |
| vote | C | 0.2505 | 0.9145 | 0.0799 | 0.6770 |
| vowel | C | 0.3008 | 0.9913 | 0.0489 | 0.3995 |
| wave | C | 0.2518 | 0.9299 | 0.0735 | 0.6351 |
| wbc | C | 0.2478 | 0.9812 | 0.0199 | 0.2496 |
| whole | C | 0.2449 | 0.7189 | 0.0189 | 0.5556 |
| wine | C | 0.4889 | 1.0000 | 0.0389 | 0.7443 |
| wineR | C | 0.3144 | 0.7218 | 0.0487 | 0.5901 |
| wineW | C | 0.3137 | 0.7342 | 0.0689 | 0.5786 |
| yeast | C | 0.3508 | 0.6288 | 0.0207 | 0.5317 |
| balance | G | 0.2556 | 0.9580 | 0.1212 | 0.5476 |
| breast_cancer | G | 0.2518 | 0.9835 | 0.0273 | 0.5586 |
| cars | G | 0.2501 | 0.9960 | 0.0301 | 0.6754 |
| cmc | G | 0.3005 | 0.5360 | 0.0268 | 0.5802 |
| colic | G | 0.2542 | 0.8715 | 0.0521 | 0.6662 |
| cool | G | 0.2455 | 0.9931 | 0.0438 | 0.6945 |
| creditA | G | 0.2514 | 0.9176 | 0.0597 | 0.6310 |
| diabetes | G | 0.2568 | 0.8097 | 0.0538 | 0.6398 |
| ecoli | G | 0.2610 | 0.9013 | 0.0425 | 0.6228 |
| german | G | 0.2537 | 0.6901 | 0.0335 | 0.6891 |
| glass | G | 0.2895 | 0.7791 | 0.0163 | 0.5655 |
| haberman | G | 0.2570 | 0.7235 | 0.0779 | 0.5538 |
| heartC | G | 0.2467 | 0.8609 | 0.0609 | 0.6003 |
| heartH | G | 0.2602 | 0.8707 | 0.0402 | 0.5827 |
| heartS | G | 0.2444 | 0.8317 | 0.0465 | 0.6794 |
| heat | G | 0.2487 | 0.9989 | 0.0054 | 0.8418 |
| hepati | G | 0.2484 | 0.9238 | 0.0722 | 0.8763 |
| image | G | 0.2489 | 0.9977 | 0.0224 | 0.5897 |
| iono | G | 0.2521 | 0.9717 | 0.0432 | 0.6015 |
| iris | G | 0.2867 | 0.9706 | 0.0239 | 0.4781 |
| je4042 | G | 0.2926 | 0.8022 | 0.0577 | 0.6127 |
| je4243 | G | 0.2692 | 0.6463 | 0.0271 | 0.4824 |
| kc1 | G | 0.2538 | 0.7897 | 0.0424 | 0.6213 |
| kc2 | G | 0.2541 | 0.8519 | 0.0573 | 0.5423 |
| kc3 | G | 0.2538 | 0.9394 | 0.0779 | 0.7154 |
| liver | G | 0.2667 | 0.7285 | 0.0648 | 0.5992 |
| pc1req | G | 0.2571 | 0.6547 | -0.0120 | 0.3795 |
| pc4 | G | 0.2500 | 0.9609 | 0.0650 | 0.5637 |
| sonar | G | 0.2512 | 0.9167 | 0.0358 | 0.6046 |
| spect | G | 0.2455 | 0.8796 | -0.0022 | 0.5429 |
| spectf | G | 0.2602 | 0.8597 | 0.0485 | 0.4205 |
| steel | G | 0.2490 | 0.8373 | 0.0753 | 0.4596 |
| tae | G | 0.3790 | 0.5318 | -0.0037 | 0.5012 |
| transfusion | G | 0.2510 | 0.7639 | 0.0411 | 0.5271 |
| ttt | G | 0.2487 | 0.9970 | 0.0209 | 0.6458 |
| user | G | 0.2512 | 0.9307 | 0.0418 | 0.7268 |
| vehicle | G | 0.2741 | 0.8316 | 0.0857 | 0.5090 |
| vote | G | 0.2481 | 0.9085 | 0.0739 | 0.7807 |
| vowel | G | 0.2841 | 0.9881 | 0.0456 | 0.4841 |
| wave | G | 0.2500 | 0.9265 | 0.0701 | 0.7481 |
| wbc | G | 0.2495 | 0.9722 | 0.0109 | 0.4631 |
| whole | G | 0.2494 | 0.7246 | 0.0246 | 0.5529 |
| wine | G | 0.3792 | 1.0000 | 0.0389 | 0.5467 |
| wineR | G | 0.2786 | 0.7182 | 0.0450 | 0.4899 |
| wineW | G | 0.2892 | 0.7257 | 0.0604 | 0.5808 |
| yeast | G | 0.3013 | 0.6246 | 0.0165 | 0.4337 |
