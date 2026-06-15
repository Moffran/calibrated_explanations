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

## Per-Dataset Arm Comparison (all datasets)

Mean over seeds and confidence levels. Rows are sorted by ncf, use_difficulty, dataset.

| dataset | ncf | use_difficulty | accept_rate | accepted_accuracy | accuracy_delta | rejected_error_capture_rate | empirical_coverage |
|---|---|---|---|---|---|---|---|
| balance | default | no | 0.8939 | 0.8946 | 0.0578 | 0.3891 | nan |
| breast_cancer | default | no | 0.9181 | 0.9779 | 0.0218 | 0.5326 | 0.9078 |
| cars | default | no | 0.8920 | 0.9912 | 0.0253 | 0.7552 | nan |
| cmc | default | no | 0.3333 | 0.6612 | 0.1521 | 0.7374 | nan |
| colic | default | no | 0.7136 | 0.8680 | 0.0485 | 0.4294 | 0.8796 |
| cool | default | no | 0.8935 | 0.9836 | 0.0342 | 0.6982 | nan |
| creditA | default | no | 0.7862 | 0.8968 | 0.0388 | 0.3823 | 0.8720 |
| diabetes | default | no | 0.5541 | 0.8723 | 0.1164 | 0.6490 | 0.9140 |
| ecoli | default | no | 0.6147 | 0.9116 | 0.0528 | 0.6046 | nan |
| german | default | no | 0.3732 | 0.7873 | 0.1307 | 0.7383 | 0.9118 |
| glass | default | no | 0.4610 | 0.8120 | 0.0492 | 0.6043 | nan |
| haberman | default | no | 0.4499 | 0.8003 | 0.1547 | 0.7027 | 0.8967 |
| heartC | default | no | 0.6350 | 0.8779 | 0.0779 | 0.5656 | 0.9100 |
| heartH | default | no | 0.6919 | 0.8716 | 0.0410 | 0.4645 | 0.9055 |
| heartS | default | no | 0.6720 | 0.8148 | 0.0318 | 0.4070 | 0.8642 |
| heat | default | no | 0.8831 | 0.9986 | 0.0051 | 0.8611 | nan |
| hepati | default | no | 0.6717 | 0.9271 | 0.0755 | 0.5994 | 0.9226 |
| image | default | no | 0.8969 | 0.9939 | 0.0186 | 0.7848 | nan |
| iono | default | no | 0.7635 | 0.9634 | 0.0348 | 0.6207 | 0.8933 |
| iris | default | no | 0.6511 | 0.9682 | 0.0182 | 0.6593 | nan |
| je4042 | default | no | 0.4613 | 0.8261 | 0.0817 | 0.6371 | 0.9123 |
| je4243 | default | no | 0.4618 | 0.6814 | 0.0622 | 0.5969 | 0.8426 |
| kc1 | default | no | 0.4187 | 0.8399 | 0.0919 | 0.6958 | 0.9234 |
| kc2 | default | no | 0.6168 | 0.8820 | 0.0874 | 0.6001 | 0.9126 |
| kc3 | default | no | 0.7303 | 0.9053 | 0.0438 | 0.4576 | 0.9026 |
| liver | default | no | 0.4870 | 0.7631 | 0.0994 | 0.6414 | 0.8770 |
| pc1req | default | no | 0.3429 | 0.5539 | -0.1127 | 0.6025 | 0.8688 |
| pc4 | default | no | 0.8777 | 0.9378 | 0.0419 | 0.4506 | 0.8989 |
| sonar | default | no | 0.7153 | 0.9114 | 0.0305 | 0.4425 | 0.9233 |
| spect | default | no | 0.6429 | 0.9012 | 0.0194 | 0.4000 | 0.9056 |
| spectf | default | no | 0.6432 | 0.8752 | 0.0641 | 0.5230 | 0.8984 |
| steel | default | no | 0.7556 | 0.8427 | 0.0807 | 0.4494 | nan |
| tae | default | no | 0.2667 | 0.3863 | -0.1456 | 0.7018 | nan |
| transfusion | default | no | 0.5285 | 0.8194 | 0.0961 | 0.6039 | 0.8920 |
| ttt | default | no | 0.9010 | 0.9946 | 0.0186 | 0.8055 | 0.9079 |
| user | default | no | 0.7951 | 0.9151 | 0.0263 | 0.3743 | nan |
| vehicle | default | no | 0.7292 | 0.8443 | 0.0984 | 0.5060 | nan |
| vote | default | no | 0.7746 | 0.9020 | 0.0673 | 0.4969 | 0.8795 |
| vowel | default | no | 0.7384 | 0.9735 | 0.0311 | 0.5485 | nan |
| wave | default | no | 0.8292 | 0.9058 | 0.0494 | 0.4148 | nan |
| wbc | default | no | 0.7830 | 0.9706 | 0.0093 | 0.3926 | 0.8858 |
| whole | default | no | 0.3227 | 0.7393 | 0.0393 | 0.7195 | nan |
| wine | default | no | 0.1056 | 1.0000 | 0.0403 | 1.0000 | nan |
| wineR | default | no | 0.4863 | 0.8042 | 0.1311 | 0.6628 | nan |
| wineW | default | no | 0.5191 | 0.8089 | 0.1436 | 0.6619 | nan |
| yeast | default | no | 0.4460 | 0.6830 | 0.0749 | 0.6035 | nan |
| balance | default | yes | 0.0476 | 0.9419 | 0.1081 | 0.9394 | nan |
| breast_cancer | default | yes | 0.0507 | 0.7752 | -0.1812 | 0.7418 | 0.9756 |
| cars | default | yes | 0.8067 | 0.9798 | 0.0139 | 0.4861 | nan |
| cmc | default | yes | 0.2365 | 0.4808 | -0.0294 | 0.7552 | nan |
| colic | default | yes | 0.0790 | 0.7562 | -0.0772 | 0.8678 | 0.9571 |
| cool | default | yes | 0.0550 | 0.7580 | -0.1914 | 0.7622 | nan |
| creditA | default | yes | 0.0386 | 0.7293 | -0.1323 | 0.9056 | 0.9747 |
| diabetes | default | yes | 0.0560 | 0.5442 | -0.2079 | 0.8989 | 0.9690 |
| ecoli | default | yes | 0.2304 | 0.7845 | -0.0704 | 0.6862 | nan |
| german | default | yes | 0.3317 | 0.7527 | 0.0941 | 0.7329 | 0.9173 |
| glass | default | yes | 0.1726 | 0.8160 | 0.0534 | 0.8257 | nan |
| haberman | default | yes | 0.3563 | 0.7616 | 0.1151 | 0.7220 | 0.9099 |
| heartC | default | yes | 0.0769 | 0.6920 | -0.1045 | 0.8962 | 0.9508 |
| heartH | default | yes | 0.1250 | 0.8271 | 0.0063 | 0.9124 | 0.9556 |
| heartS | default | yes | 0.1037 | 0.5896 | -0.1934 | 0.8285 | 0.9399 |
| heat | default | yes | 0.0511 | 0.8303 | -0.1615 | 0.2083 | nan |
| hepati | default | yes | 0.5125 | 0.9577 | 0.1061 | 0.7822 | 0.9477 |
| image | default | yes | 0.3236 | 0.9912 | 0.0180 | 0.7010 | nan |
| iono | default | yes | 0.0559 | 1.0000 | 0.0786 | 1.0000 | 0.9730 |
| iris | default | yes | 0.0178 | 0.6667 | -0.2333 | 0.9407 | nan |
| je4042 | default | yes | 0.1004 | 0.6009 | -0.1441 | 0.8422 | 0.9617 |
| je4243 | default | yes | 0.1409 | 0.7623 | 0.1326 | 0.9029 | 0.9266 |
| kc1 | default | yes | 0.2346 | 0.6494 | -0.0978 | 0.8106 | 0.9398 |
| kc2 | default | yes | 0.3222 | 0.8965 | 0.1024 | 0.8109 | 0.9267 |
| kc3 | default | yes | 0.6742 | 0.8950 | 0.0338 | 0.4565 | 0.9166 |
| liver | default | yes | 0.0953 | 0.5871 | -0.0767 | 0.8566 | 0.9504 |
| pc1req | default | yes | 0.0995 | 0.5485 | -0.1261 | 0.8922 | 0.9344 |
| pc4 | default | yes | 0.8226 | 0.9128 | 0.0169 | 0.4308 | 0.9005 |
| sonar | default | yes | 0.1032 | 0.8229 | -0.0595 | 0.8843 | 0.9323 |
| spect | default | yes | 0.6404 | 0.9066 | 0.0238 | 0.4244 | 0.9111 |
| spectf | default | yes | 0.6572 | 0.8874 | 0.0763 | 0.5504 | 0.8704 |
| steel | default | yes | 0.0490 | 0.8741 | 0.1081 | 0.9736 | nan |
| tae | default | yes | 0.1333 | 0.4061 | -0.1193 | 0.8381 | nan |
| transfusion | default | yes | 0.0517 | 0.7386 | 0.0161 | 0.9424 | 0.9875 |
| ttt | default | yes | 0.4870 | 0.9062 | -0.0699 | 0.2485 | 0.9035 |
| user | default | yes | 0.0508 | 0.9032 | 0.0513 | 0.9667 | nan |
| vehicle | default | yes | 0.0648 | 0.6456 | -0.0954 | 0.9293 | nan |
| vote | default | yes | 0.0363 | 0.4444 | -0.3902 | 0.8442 | 0.9808 |
| vowel | default | yes | 0.3338 | 0.9281 | -0.0137 | 0.6492 | nan |
| wave | default | yes | 0.0318 | 0.6702 | -0.1867 | 0.9302 | nan |
| wbc | default | yes | 0.1305 | 0.7361 | -0.2249 | 0.4759 | 0.9348 |
| whole | default | yes | 0.2722 | 0.7170 | 0.0170 | 0.7679 | nan |
| wine | default | yes | 0.1123 | 0.9866 | 0.0074 | 0.9167 | nan |
| wineR | default | yes | 0.3067 | 0.7645 | 0.0912 | 0.7805 | nan |
| wineW | default | yes | 0.0809 | 0.8061 | 0.1451 | 0.9334 | nan |
| yeast | default | yes | 0.0462 | 0.7179 | 0.1099 | 0.9603 | nan |
| balance | ensured | no | 0.7228 | 0.9589 | 0.1221 | 0.8171 | nan |
| breast_cancer | ensured | no | 0.6131 | 0.9694 | 0.0133 | 0.5670 | 0.9809 |
| cars | ensured | no | 0.9492 | 0.9837 | 0.0178 | 0.5593 | nan |
| cmc | ensured | no | 0.1518 | 0.6899 | 0.1808 | 0.8953 | nan |
| colic | ensured | no | 0.4765 | 0.8566 | 0.0372 | 0.5862 | 0.8907 |
| cool | ensured | no | 0.9141 | 0.9808 | 0.0314 | 0.6287 | nan |
| creditA | ensured | no | 0.4053 | 0.9069 | 0.0489 | 0.7223 | 0.9185 |
| diabetes | ensured | no | 0.3830 | 0.8922 | 0.1364 | 0.8097 | 0.9190 |
| ecoli | ensured | no | 0.4526 | 0.9509 | 0.0921 | 0.8463 | nan |
| german | ensured | no | 0.1088 | 0.4536 | -0.2026 | 0.8499 | 0.9260 |
| glass | ensured | no | 0.2693 | 0.9267 | 0.1639 | 0.9149 | nan |
| haberman | ensured | no | 0.0534 | 0.2677 | -0.3611 | 0.9020 | 0.9224 |
| heartC | ensured | no | 0.1945 | 0.8975 | 0.0975 | 0.8954 | 0.9129 |
| heartH | ensured | no | 0.0968 | 0.8329 | 0.0024 | 0.8897 | 0.9262 |
| heartS | ensured | no | 0.3399 | 0.8153 | 0.0323 | 0.6928 | 0.9008 |
| heat | ensured | no | 0.9130 | 0.9976 | 0.0041 | 0.6667 | nan |
| hepati | ensured | no | 0.0452 | 0.7413 | -0.1160 | 0.8870 | 0.9290 |
| image | ensured | no | 0.8823 | 0.9956 | 0.0203 | 0.8656 | nan |
| iono | ensured | no | 0.2048 | 0.9333 | 0.0047 | 0.8284 | 0.9244 |
| iris | ensured | no | 0.5556 | 0.9886 | 0.0386 | 0.9407 | nan |
| je4042 | ensured | no | 0.2547 | 0.7100 | -0.0386 | 0.6999 | 0.9016 |
| je4243 | ensured | no | 0.1842 | 0.5634 | -0.0557 | 0.7846 | 0.8700 |
| kc1 | ensured | no | 0.0590 | 0.6219 | -0.1257 | 0.9015 | 0.9119 |
| kc2 | ensured | no | 0.0862 | 0.7117 | -0.0868 | 0.8563 | 0.9015 |
| kc3 | ensured | no | 0.2161 | 0.6769 | -0.1837 | 0.7200 | 0.8875 |
| liver | ensured | no | 0.3311 | 0.7776 | 0.1138 | 0.7609 | 0.8979 |
| pc1req | ensured | no | 0.2074 | 0.6852 | 0.0308 | 0.8115 | 0.9228 |
| pc4 | ensured | no | 0.1590 | 0.6580 | -0.2380 | 0.7321 | 0.9103 |
| sonar | ensured | no | 0.3143 | 0.8693 | -0.0116 | 0.6974 | 0.8926 |
| spect | ensured | no | 0.2631 | 0.8202 | -0.0620 | 0.6681 | 0.9202 |
| spectf | ensured | no | 0.0490 | 0.6002 | -0.2084 | 0.9036 | 0.8988 |
| steel | ensured | no | 0.5164 | 0.9159 | 0.1540 | 0.8184 | nan |
| tae | ensured | no | 0.1935 | 0.5319 | -0.0012 | 0.8180 | nan |
| transfusion | ensured | no | 0.4810 | 0.8323 | 0.1090 | 0.6748 | 0.9054 |
| ttt | ensured | no | 0.2410 | 0.9655 | -0.0105 | 0.7321 | 0.9029 |
| user | ensured | no | 0.6414 | 0.9414 | 0.0525 | 0.6391 | nan |
| vehicle | ensured | no | 0.4511 | 0.9558 | 0.2099 | 0.9128 | nan |
| vote | ensured | no | 0.5526 | 0.9265 | 0.0919 | 0.7371 | 0.9214 |
| vowel | ensured | no | 0.5705 | 0.9971 | 0.0547 | 0.9669 | nan |
| wave | ensured | no | 0.7083 | 0.9222 | 0.0658 | 0.6050 | nan |
| wbc | ensured | no | 0.3909 | 0.9553 | -0.0060 | 0.5481 | 0.9434 |
| whole | ensured | no | 0.3227 | 0.7393 | 0.0393 | 0.7195 | nan |
| wine | ensured | no | 0.1056 | 1.0000 | 0.0403 | 1.0000 | nan |
| wineR | ensured | no | 0.3399 | 0.8365 | 0.1634 | 0.8193 | nan |
| wineW | ensured | no | 0.3332 | 0.8462 | 0.1809 | 0.8421 | nan |
| yeast | ensured | no | 0.2851 | 0.7366 | 0.1286 | 0.7964 | nan |
| balance | ensured | yes | 0.1543 | 0.8686 | 0.0326 | 0.9192 | nan |
| breast_cancer | ensured | yes | 0.0454 | 0.7752 | -0.1812 | 0.7418 | 0.9809 |
| cars | ensured | yes | 0.7696 | 0.9828 | 0.0169 | 0.5715 | nan |
| cmc | ensured | yes | 0.0876 | 0.6772 | 0.1667 | 0.9536 | nan |
| colic | ensured | yes | 0.0565 | 0.7826 | -0.0507 | 0.9080 | 0.9586 |
| cool | ensured | yes | 0.0020 | 0.8125 | -0.1274 | 0.9932 | nan |
| creditA | ensured | yes | 0.0291 | 0.6966 | -0.1703 | 0.9167 | 0.9791 |
| diabetes | ensured | yes | 0.0548 | 0.5050 | -0.2471 | 0.8989 | 0.9701 |
| ecoli | ensured | yes | 0.1206 | 0.9205 | 0.0558 | 0.9402 | nan |
| german | ensured | yes | 0.0955 | 0.6264 | -0.0291 | 0.8875 | 0.9173 |
| glass | ensured | yes | 0.0755 | 0.7887 | 0.0235 | 0.9492 | nan |
| haberman | ensured | yes | 0.1088 | 0.6706 | 0.0225 | 0.8804 | 0.9361 |
| heartC | ensured | yes | 0.0270 | 0.5509 | -0.2596 | 0.9308 | 0.9315 |
| heartH | ensured | yes | 0.0467 | 0.5789 | -0.2397 | 0.9037 | 0.9375 |
| heartS | ensured | yes | 0.0399 | 0.4188 | -0.3548 | 0.9171 | 0.9572 |
| heat | ensured | yes | 0.0548 | 1.0000 | 0.0065 | 1.0000 | nan |
| hepati | ensured | yes | 0.1097 | 0.6612 | -0.1846 | 0.7546 | 0.9355 |
| image | ensured | yes | 0.3362 | 0.9547 | -0.0206 | 0.5495 | nan |
| iono | ensured | yes | 0.0857 | 0.8697 | -0.0622 | 0.7892 | 0.9314 |
| iris | ensured | yes | 0.0919 | 1.0000 | 0.0505 | 1.0000 | nan |
| je4042 | ensured | yes | 0.0317 | 0.7160 | -0.0240 | 0.9576 | 0.9263 |
| je4243 | ensured | yes | 0.0399 | 0.4844 | -0.1405 | 0.9524 | 0.9300 |
| kc1 | ensured | yes | 0.1848 | 0.7682 | 0.0207 | 0.7912 | 0.9352 |
| kc2 | ensured | yes | 0.0784 | 0.7036 | -0.0779 | 0.8808 | 0.9447 |
| kc3 | ensured | yes | 0.3350 | 0.8106 | -0.0577 | 0.6375 | 0.9176 |
| liver | ensured | yes | 0.0605 | 0.3607 | -0.2867 | 0.8938 | 0.9607 |
| pc1req | ensured | yes | 0.0561 | 0.4467 | -0.2367 | 0.9110 | 0.9386 |
| pc4 | ensured | yes | 0.1005 | 0.6850 | -0.2103 | 0.8225 | 0.9057 |
| sonar | ensured | yes | 0.0529 | 0.4290 | -0.4099 | 0.8855 | 0.9529 |
| spect | ensured | yes | 0.2530 | 0.7735 | -0.1074 | 0.6881 | 0.9192 |
| spectf | ensured | yes | 0.1086 | 0.6243 | -0.1943 | 0.8039 | 0.8881 |
| steel | ensured | yes | 0.0000 | nan | nan | 1.0000 | nan |
| tae | ensured | yes | 0.1921 | 0.5628 | 0.0258 | 0.8117 | nan |
| transfusion | ensured | yes | 0.0517 | 0.7386 | 0.0161 | 0.9424 | 0.9875 |
| ttt | ensured | yes | 0.0933 | 0.8636 | -0.1119 | 0.6527 | 0.9113 |
| user | ensured | yes | 0.0000 | nan | nan | 1.0000 | nan |
| vehicle | ensured | yes | 0.1576 | 0.5459 | -0.1993 | 0.7709 | nan |
| vote | ensured | yes | 0.0363 | 0.4444 | -0.3902 | 0.8442 | 0.9808 |
| vowel | ensured | yes | 0.4342 | 0.9400 | -0.0017 | 0.5524 | nan |
| wave | ensured | yes | 0.1202 | 0.9385 | 0.0827 | 0.8914 | nan |
| wbc | ensured | yes | 0.0619 | 0.9160 | -0.0497 | 0.8926 | 0.9216 |
| whole | ensured | yes | 0.2722 | 0.7170 | 0.0170 | 0.7679 | nan |
| wine | ensured | yes | 0.0907 | 0.9857 | 0.0260 | 0.9722 | nan |
| wineR | ensured | yes | 0.1834 | 0.8294 | 0.1561 | 0.8983 | nan |
| wineW | ensured | yes | 0.1156 | 0.7473 | 0.0820 | 0.9163 | nan |
| yeast | ensured | yes | 0.1074 | 0.7033 | 0.0931 | 0.9221 | nan |
