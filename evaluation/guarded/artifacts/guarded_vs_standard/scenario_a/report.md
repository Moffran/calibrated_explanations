# Scenario A: Guarded vs Standard Calibrated Explanations

## Setup
- Seeds: 30
- Models: logreg, rf
- Test instances sampled per seed: 200
- Guard grid: significance=[0.01, 0.05, 0.1, 0.2], n_neighbors=[3, 5, 10], merge_adjacent=[False, True]

## Purpose
This report keeps only the paper-facing metrics for Scenario A.
The main comparison is the factual-mode violation rate under the emitted guarded rule format and the known domain constraint.
A secondary tradeoff metric is the factual-mode rule count.

## Factual violation rate
| metric         | model   | mode    |   significance |   guarded_mean |   standard_mean |   median_difference_guarded_minus_standard |   wilcoxon_p_value |   _mean_reduction |
|:---------------|:--------|:--------|---------------:|---------------:|----------------:|-------------------------------------------:|-------------------:|------------------:|
| violation_rate | rf      | factual |           0.2  |    0.000694444 |       0.0811389 |                                 -0.07875   |        1.72677e-06 |         0.0804444 |
| violation_rate | logreg  | factual |           0.2  |    0.000694444 |       0.0797917 |                                 -0.0791667 |        1.73331e-06 |         0.0790972 |
| violation_rate | rf      | factual |           0.1  |    0.00668056  |       0.0811389 |                                 -0.075625  |        1.73113e-06 |         0.0744583 |
| violation_rate | logreg  | factual |           0.1  |    0.00747222  |       0.0797917 |                                 -0.0745833 |        1.86265e-09 |         0.0723194 |
| violation_rate | rf      | factual |           0.05 |    0.0230833   |       0.0811389 |                                 -0.05625   |        1.72895e-06 |         0.0580556 |
| violation_rate | logreg  | factual |           0.05 |    0.0261667   |       0.0797917 |                                 -0.0539583 |        1.73113e-06 |         0.053625  |
| violation_rate | rf      | factual |           0.01 |    0.0636389   |       0.0811389 |                                 -0.0208333 |        0.000231566 |         0.0175    |
| violation_rate | logreg  | factual |           0.01 |    0.064       |       0.0797917 |                                 -0.0222917 |        0.000505488 |         0.0157917 |

## Factual rule count
| metric     | model   | mode    |   significance |   guarded_mean |   standard_mean |   median_difference_guarded_minus_standard |   wilcoxon_p_value |
|:-----------|:--------|:--------|---------------:|---------------:|----------------:|-------------------------------------------:|-------------------:|
| rule_count | logreg  | factual |           0.01 |        1.89208 |         3.08667 |                                   -1.17917 |        1.86265e-09 |
| rule_count | rf      | factual |           0.01 |        1.8715  |         3.1375  |                                   -1.25792 |        1.86265e-09 |
| rule_count | logreg  | factual |           0.05 |        1.73286 |         3.08667 |                                   -1.35875 |        1.86265e-09 |
| rule_count | rf      | factual |           0.05 |        1.69619 |         3.1375  |                                   -1.43375 |        1.86265e-09 |
| rule_count | logreg  | factual |           0.1  |        1.57308 |         3.08667 |                                   -1.52875 |        1.86265e-09 |
| rule_count | rf      | factual |           0.1  |        1.54117 |         3.1375  |                                   -1.5925  |        1.86265e-09 |
| rule_count | logreg  | factual |           0.2  |        1.29819 |         3.08667 |                                   -1.80542 |        1.86265e-09 |
| rule_count | rf      | factual |           0.2  |        1.28683 |         3.1375  |                                   -1.83833 |        1.86265e-09 |

## Notes
The CSV outputs retain additional diagnostics for engineering use.
They are not intended as main paper evidence.
A practical starting point from the factual violation-rate table is significance=0.2.
