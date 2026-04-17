# Perspective 4: Component Ablation

## Setup
- Seeds: 2
- Models: logreg, rf
- Significance levels: [0.01, 0.05, 0.1, 0.2]

## Variants

| Label | Description |
|-------|-------------|
| V1_binary | V1: Binary (standard CE) |
| V2_multibin | V2: Multi-bin only |
| V3_multibin_guard | V3: Multi-bin + guard |
| V4_multibin_guard_merge | V4: Multi-bin + guard + merge |

## Results

Wilcoxon p-value is computed against V2 (multi-bin baseline) for the
violation_rate metric, isolating the guard's contribution beyond
discretisation granularity.

| model   |   significance | variant                 |   violation_rate_mean |   rule_count_mean |   wilcoxon_p_vs_v2_violation |
|:--------|---------------:|:------------------------|----------------------:|------------------:|-----------------------------:|
| logreg  |           0.01 | V1_binary               |             0         |           2       |                          nan |
| logreg  |           0.01 | V2_multibin             |             0.0416667 |           3.1875  |                          nan |
| logreg  |           0.01 | V3_multibin_guard       |             0.078125  |           2       |                          nan |
| logreg  |           0.01 | V4_multibin_guard_merge |             0.03125   |           1.71875 |                          nan |
| logreg  |           0.05 | V1_binary               |             0         |           2       |                          nan |
| logreg  |           0.05 | V2_multibin             |             0.0416667 |           3.1875  |                          nan |
| logreg  |           0.05 | V3_multibin_guard       |             0.078125  |           1.9375  |                          nan |
| logreg  |           0.05 | V4_multibin_guard_merge |             0.078125  |           1.71875 |                          nan |
| logreg  |           0.1  | V1_binary               |             0         |           2       |                          nan |
| logreg  |           0.1  | V2_multibin             |             0.0416667 |           3.1875  |                          nan |
| logreg  |           0.1  | V3_multibin_guard       |             0.078125  |           1.84375 |                          nan |
| logreg  |           0.1  | V4_multibin_guard_merge |             0.078125  |           1.625   |                          nan |
| logreg  |           0.2  | V1_binary               |             0         |           2       |                          nan |
| logreg  |           0.2  | V2_multibin             |             0.0416667 |           3.1875  |                          nan |
| logreg  |           0.2  | V3_multibin_guard       |             0.046875  |           1.625   |                          nan |
| logreg  |           0.2  | V4_multibin_guard_merge |             0.078125  |           1.65625 |                          nan |
| rf      |           0.01 | V1_binary               |             0         |           2       |                          nan |
| rf      |           0.01 | V2_multibin             |             0.0416667 |           3.21875 |                          nan |
| rf      |           0.01 | V3_multibin_guard       |             0.078125  |           2       |                          nan |
| rf      |           0.01 | V4_multibin_guard_merge |             0.03125   |           1.75    |                          nan |
| rf      |           0.05 | V1_binary               |             0         |           2       |                          nan |
| rf      |           0.05 | V2_multibin             |             0.0416667 |           3.21875 |                          nan |
| rf      |           0.05 | V3_multibin_guard       |             0.078125  |           1.875   |                          nan |
| rf      |           0.05 | V4_multibin_guard_merge |             0.078125  |           1.65625 |                          nan |
| rf      |           0.1  | V1_binary               |             0         |           2       |                          nan |
| rf      |           0.1  | V2_multibin             |             0.0416667 |           3.21875 |                          nan |
| rf      |           0.1  | V3_multibin_guard       |             0.078125  |           1.8125  |                          nan |
| rf      |           0.1  | V4_multibin_guard_merge |             0.078125  |           1.625   |                          nan |
| rf      |           0.2  | V1_binary               |             0         |           2       |                          nan |
| rf      |           0.2  | V2_multibin             |             0.0416667 |           3.21875 |                          nan |
| rf      |           0.2  | V3_multibin_guard       |             0.0625    |           1.625   |                          nan |
| rf      |           0.2  | V4_multibin_guard_merge |             0.09375   |           1.53125 |                          nan |

## Interpretation

If V2 already reduces violations compared to V1, part of the benefit
is attributable to finer binning rather than the guard.  V3 vs V2
isolates the guard's marginal contribution.  V4 vs V3 tests whether
bin merging adds value beyond the per-bin guard.
