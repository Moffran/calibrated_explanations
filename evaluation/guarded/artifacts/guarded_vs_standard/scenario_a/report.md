# Scenario A: Guarded vs Standard Calibrated Explanations

## Setup
- Seeds: 2
- Models: logreg, rf
- Test instances sampled per seed: 24
- Guard grid: significance=[0.01, 0.05, 0.1, 0.2], n_neighbors=[3, 5, 10], merge_adjacent=[False, True]

## Key observations
- Guarded explanations reduce plausibility violations on Scenario A in most settings.
- Rule retention decreases as significance increases (stricter guard).
- Runtime overhead is present for guarded mode due to conformity checks.
- Prediction differences on shared rules are generally small but non-zero.

## Statistical summary (Wilcoxon paired tests)
| metric            | mode        |   significance |   guarded_mean |   standard_mean |   median_difference_guarded_minus_standard |   wilcoxon_p_value |
|:------------------|:------------|---------------:|---------------:|----------------:|-------------------------------------------:|-------------------:|
| rule_count        | alternative |           0.01 |     5.42014    |       3         |                                   2.5      |        4.56869e-18 |
| rule_count        | alternative |           0.05 |     5.26389    |       3         |                                   2.5      |        9.65412e-18 |
| rule_count        | alternative |           0.1  |     5.06771    |       3         |                                   2.25     |        1.65589e-17 |
| rule_count        | alternative |           0.2  |     4.77083    |       3         |                                   2        |        9.92614e-16 |
| rule_count        | factual     |           0.01 |     1.90451    |       3         |                                  -1        |        1.05539e-14 |
| rule_count        | factual     |           0.05 |     1.87326    |       3         |                                  -1        |        3.88904e-15 |
| rule_count        | factual     |           0.1  |     1.81076    |       3         |                                  -1        |        3.38386e-15 |
| rule_count        | factual     |           0.2  |     1.67188    |       3         |                                  -1.33333  |        4.06146e-15 |
| runtime_ms        | alternative |           0.01 |   153.946      |     136.103     |                                  10.2794   |        9.51644e-10 |
| runtime_ms        | alternative |           0.05 |   136.107      |     136.103     |                                   7.64172  |        7.33119e-06 |
| runtime_ms        | alternative |           0.1  |   148.182      |     136.103     |                                   7.823    |        0.000107253 |
| runtime_ms        | alternative |           0.2  |   138.272      |     136.103     |                                   7.39759  |        6.29501e-05 |
| runtime_ms        | factual     |           0.01 |  1390.92       |     177.379     |                                  10.9924   |        1.70136e-07 |
| runtime_ms        | factual     |           0.05 |   198.223      |     177.379     |                                   8.70272  |        0.000524463 |
| runtime_ms        | factual     |           0.1  |   193.192      |     177.379     |                                   8.45798  |        0.000269115 |
| runtime_ms        | factual     |           0.2  |   179.668      |     177.379     |                                   7.03501  |        0.0085108   |
| stability_jaccard | alternative |           0.1  |     0.0212537  |       0.194475  |                                  -0.141667 |        4.62834e-16 |
| stability_jaccard | factual     |           0.1  |     0.0716146  |       0.194475  |                                  -0.102083 |        5.80915e-16 |
| violation_rate    | alternative |           0.01 |     0.0516493  |       0.0902778 |                                   0        |        0.0306724   |
| violation_rate    | alternative |           0.05 |     0.00412326 |       0.0902778 |                                   0        |        2.19603e-05 |
| violation_rate    | alternative |           0.1  |     0          |       0.0902778 |                                   0        |        2.03526e-05 |
| violation_rate    | alternative |           0.2  |     0          |       0.0902778 |                                   0        |        2.03526e-05 |
| violation_rate    | factual     |           0.01 |     0.00347222 |       0.0902778 |                                   0        |        2.17081e-05 |
| violation_rate    | factual     |           0.05 |     0          |       0.0902778 |                                   0        |        2.03526e-05 |
| violation_rate    | factual     |           0.1  |     0          |       0.0902778 |                                   0        |        2.03526e-05 |
| violation_rate    | factual     |           0.2  |     0          |       0.0902778 |                                   0        |        2.03526e-05 |

## Practical recommendation
- Start with `significance=0.1` and adjust upward only if plausibility violations are still too high.
- Prefer `n_neighbors=5` as a stable default and enable `merge_adjacent=True` only when rule compactness is a priority.
