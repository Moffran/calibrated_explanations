# Scenario C — Real-Dataset Guard Retention Benchmark

## Scientific Question

How does conformal guard filtering affect the number of emitted factual rules across the full dataset universe?  Are guard retention rates stable across diverse real-world datasets at ε=0.10?

**Coverage preservation is not a metric here**: it is structurally invariant under guard filtering (§2.3 of the paper).

## Task-level summary (ε=0.1)

| Task | Std rules | Guarded rules | Retention | Fully filtered |
|---|---|---|---|---|
| binary | 16.000 | 14.668 | 0.178 | 0.087 |
| multiclass | 16.000 | 12.986 | 0.166 | 0.062 |
| regression | 16.000 | 6.816 | 0.118 | 0.051 |

## Per-dataset results (ε=0.1)

| Task | Dataset | d | Std rules | Guarded rules | Retention | Fully filtered |
|---|---|---|---|---|---|---|
| binary | diabetes | 8 | 16.000 | 6.853 | 0.122 | 0.049 |
| binary | german | 27 | 16.000 | 12.796 | 0.222 | 0.078 |
| binary | kc1 | 21 | 16.000 | 12.823 | 0.129 | 0.113 |
| binary | pc4 | 37 | 16.000 | 25.316 | 0.126 | 0.064 |
| binary | ttt | 27 | 16.000 | 15.554 | 0.288 | 0.132 |
| multiclass | cars | 6 | 16.000 | 5.999 | 0.309 | 0.000 |
| multiclass | cmc | 9 | 16.000 | 7.427 | 0.217 | 0.114 |
| multiclass | cool | 8 | 16.000 | 7.228 | 0.258 | 0.060 |
| multiclass | heat | 8 | 16.000 | 7.410 | 0.208 | 0.046 |
| multiclass | image | 19 | 16.000 | 17.031 | 0.135 | 0.083 |
| multiclass | steel | 27 | 16.000 | 24.768 | 0.132 | 0.050 |
| multiclass | vehicle | 18 | 16.000 | 15.402 | 0.114 | 0.079 |
| multiclass | vowel | 11 | 16.000 | 9.797 | 0.120 | 0.071 |
| multiclass | wave | 40 | 16.000 | 34.008 | 0.116 | 0.058 |
| multiclass | wineR | 11 | 16.000 | 9.578 | 0.118 | 0.079 |
| multiclass | wineW | 11 | 16.000 | 9.838 | 0.117 | 0.051 |
| multiclass | yeast | 8 | 16.000 | 7.342 | 0.145 | 0.052 |
| regression | abalone | 8 | 16.000 | 7.088 | 0.121 | 0.070 |
| regression | anacalt | 7 | 16.000 | 2.476 | 0.109 | 0.052 |
| regression | bank8fh | 8 | 16.000 | 7.127 | 0.127 | 0.046 |
| regression | bank8fm | 8 | 16.000 | 6.253 | 0.113 | 0.058 |
| regression | bank8nh | 8 | 16.000 | 7.198 | 0.130 | 0.050 |
| regression | bank8nm | 8 | 16.000 | 6.572 | 0.120 | 0.060 |
| regression | comp | 12 | 16.000 | 10.448 | 0.116 | 0.059 |
| regression | concreate | 8 | 16.000 | 7.008 | 0.121 | 0.039 |
| regression | cooling | 8 | 16.000 | 5.990 | 0.147 | 0.050 |
| regression | deltaA | 5 | 16.000 | 4.357 | 0.112 | 0.069 |
| regression | deltaE | 6 | 16.000 | 5.030 | 0.119 | 0.094 |
| regression | friedm | 5 | 16.000 | 4.577 | 0.118 | 0.028 |
| regression | heating | 8 | 16.000 | 6.002 | 0.147 | 0.050 |
| regression | housing | 9 | 16.000 | 8.070 | 0.131 | 0.040 |
| regression | kin8fh | 8 | 16.000 | 7.052 | 0.114 | 0.042 |
| regression | kin8fm | 8 | 16.000 | 7.166 | 0.116 | 0.036 |
| regression | kin8nh | 8 | 16.000 | 7.317 | 0.118 | 0.028 |
| regression | kin8nm | 8 | 16.000 | 7.386 | 0.121 | 0.022 |
| regression | laser | 4 | 16.000 | 3.543 | 0.112 | 0.067 |
| regression | mg | 6 | 16.000 | 5.304 | 0.111 | 0.048 |
| regression | mortage | 15 | 16.000 | 12.731 | 0.106 | 0.078 |
| regression | plastic | 2 | 16.000 | 1.714 | 0.124 | 0.041 |
| regression | puma8fh | 8 | 16.000 | 7.303 | 0.120 | 0.030 |
| regression | puma8fm | 8 | 16.000 | 7.296 | 0.120 | 0.019 |
| regression | puma8nh | 8 | 16.000 | 7.007 | 0.112 | 0.042 |
| regression | puma8nm | 8 | 16.000 | 6.132 | 0.098 | 0.027 |
| regression | quakes | 3 | 16.000 | 2.696 | 0.125 | 0.052 |
| regression | stock | 9 | 16.000 | 7.874 | 0.110 | 0.046 |
| regression | treasury | 15 | 16.000 | 12.042 | 0.100 | 0.094 |
| regression | wineRed | 11 | 16.000 | 10.040 | 0.122 | 0.057 |
| regression | wineWhite | 11 | 16.000 | 9.871 | 0.115 | 0.066 |
| regression | wizmir | 9 | 16.000 | 7.432 | 0.111 | 0.073 |

## Execution summary

- Total datasets evaluated: 79
- Skipped (too small): 30
- Errors during execution: 1

**Seeds:** 3  **Cal sizes:** [100, 300, 500]  **k:** 5  **normalize_guard:** True  **merge_adjacent:** False

## Metric definitions

| Metric | Definition |
|---|---|
| `mean_standard_rules_per_instance` | Mean rule count from `explain_factual` (max_depth=1) |
| `mean_guarded_rules_per_instance` | Mean `intervals_emitted` per test instance |
| `guard_retention_rate` | `intervals_emitted / intervals_tested` across all test instances |
| `fraction_instances_fully_filtered` | Fraction of test instances with 0 guarded rules |
