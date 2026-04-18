# Perspective 3: Selectivity Cost and Subgroup Impact

## Setup
- Seeds: 2 (cal-size: 2)
- Models: logreg, rf
- Epsilon sweep: [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
- Calibration sizes: [250, 500, 1000, 2000, 5000]
- Density/cal-size epsilon: 0.1

## Analysis 1: Selective Explanation Curves

Metrics over the full epsilon sweep.  As epsilon increases, the guard
becomes more aggressive: more rules are removed, more instances may
lose all explanations, but fewer constraint violations survive.

| model   |   significance |   nonempty_rate |   feature_retention |   rule_retention |   violation_rate |   mean_rules |
|:--------|---------------:|----------------:|--------------------:|-----------------:|-----------------:|-------------:|
| logreg  |           0.01 |         1       |            1        |        0.138095  |         0.078125 |      2       |
| logreg  |           0.05 |         1       |            0.96875  |        0.133631  |         0.078125 |      1.9375  |
| logreg  |           0.1  |         0.96875 |            0.921875 |        0.127381  |         0.078125 |      1.84375 |
| logreg  |           0.2  |         0.9375  |            0.8125   |        0.1125    |         0.046875 |      1.625   |
| logreg  |           0.3  |         0.9375  |            0.765625 |        0.106101  |         0.046875 |      1.53125 |
| logreg  |           0.5  |         0.75    |            0.515625 |        0.0714286 |         0.015625 |      1.03125 |
| rf      |           0.01 |         1       |            1        |        0.138095  |         0.078125 |      2       |
| rf      |           0.05 |         1       |            0.9375   |        0.129464  |         0.078125 |      1.875   |
| rf      |           0.1  |         0.96875 |            0.90625  |        0.125     |         0.078125 |      1.8125  |
| rf      |           0.2  |         0.90625 |            0.8125   |        0.112054  |         0.0625   |      1.625   |
| rf      |           0.3  |         0.90625 |            0.75     |        0.103423  |         0.0625   |      1.5     |
| rf      |           0.5  |         0.6875  |            0.53125  |        0.0732143 |         0.03125  |      1.0625  |

## Analysis 2: Stratified Retention by Density

Density quartiles computed from mean distance to 5 nearest
calibration neighbours.  Q1 = densest, Q4 = sparsest.  ε = 0.1.

| model   | density_quartile   |   nonempty_rate |   feature_retention |   rule_retention |   violation_rate |   mean_rules |
|:--------|:-------------------|----------------:|--------------------:|-----------------:|-----------------:|-------------:|
| logreg  | Q1_dense           |           1     |              1      |        0.138095  |           0.0625 |        2     |
| logreg  | Q2                 |           1     |              1      |        0.138095  |           0.0625 |        2     |
| logreg  | Q3                 |           1     |              1      |        0.138095  |           0.125  |        2     |
| logreg  | Q4_sparse          |           0.875 |              0.6875 |        0.0952381 |           0.0625 |        1.375 |
| rf      | Q1_dense           |           1     |              1      |        0.138095  |           0.0625 |        2     |
| rf      | Q2                 |           1     |              1      |        0.138095  |           0.0625 |        2     |
| rf      | Q3                 |           1     |              0.9375 |        0.129167  |           0.125  |        1.875 |
| rf      | Q4_sparse          |           0.875 |              0.6875 |        0.0946429 |           0.0625 |        1.375 |

## Analysis 3: Calibration-Size Sensitivity

Fixed ε = 0.1, varying calibration set size.

| model   |   cal_size |   nonempty_rate |   feature_retention |   rule_retention |   violation_rate |   mean_rules |
|:--------|-----------:|----------------:|--------------------:|-----------------:|-----------------:|-------------:|
| logreg  |        250 |         0.96875 |            0.890625 |         0.143029 |         0.078125 |      1.78125 |
| logreg  |        500 |         0.96875 |            0.921875 |         0.137019 |         0.078125 |      1.84375 |
| logreg  |       1000 |         0.96875 |            0.921875 |         0.128756 |         0.078125 |      1.84375 |
| logreg  |       2000 |         1       |            0.921875 |         0.115234 |         0.078125 |      1.84375 |
| logreg  |       5000 |         0.96875 |            0.921875 |         0.115234 |         0.078125 |      1.84375 |
| rf      |        250 |         0.96875 |            0.90625  |         0.139423 |         0.046875 |      1.8125  |
| rf      |        500 |         0.96875 |            0.9375   |         0.129613 |         0.078125 |      1.875   |
| rf      |       1000 |         0.96875 |            0.890625 |         0.122768 |         0.078125 |      1.78125 |
| rf      |       2000 |         1       |            0.921875 |         0.127232 |         0.078125 |      1.84375 |
| rf      |       5000 |         0.96875 |            0.921875 |         0.115234 |         0.0625   |      1.84375 |
