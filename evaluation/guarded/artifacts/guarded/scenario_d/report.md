# Scenario D: Real Dataset Correctness

## What this scenario tests

Scenario D asks: does the guard's API remain correct across the full variety of real-world task types — multiclass classification, high-dimensional data, and small calibration sets?

Datasets: breast_cancer (30 features, 2-class), iris (4 features, 3-class).
Model: RandomForestClassifier.
Grid includes use_bonferroni=True, which is untested in Scenario A.

## Primary metric 1: audit_field_completeness

Total interval records with missing fields: **0**

PASS: all required fields present in every audit interval record.

## Primary metric 2: fraction_instances_fully_filtered

At significance=0.10, fraction of instances with 0 emitted rules:

| dataset       |   fraction_instances_fully_filtered |   audit_field_completeness |
|:--------------|------------------------------------:|---------------------------:|
| breast_cancer |                                0.08 |                          1 |
| iris          |                                0    |                          1 |

Values > 0.10 mean the guard is more aggressive than the significance level implies — likely due to small calibration sets or high dimensionality. Values > 0.10 should be flagged in documentation.

## Crashes and API errors

API exception count: **0**

PASS: no exceptions raised across any dataset or configuration.

## Known blind spots exposed by this scenario

1. **Multiclass payload shape**: iris and wine exercise the multiclass code path. Any bug in how prediction["classes"] is stored in audit interval records will appear as a missing field.
2. **use_bonferroni=True**: with max_depth=3 and many bins per feature, Bonferroni divides significance by n_bins, making the guard dramatically stricter. The bonferroni_comparison.png plot quantifies this effect.
3. **Tiny calibration sets**: iris with ~30 calibration instances has p-value granularity of 1/30 ≈ 0.033. At significance=0.05 only 1-2 discrete p-value levels separate filtered from kept intervals.
4. **High-dimensional data**: digits_01 with 64 features tests KNN guard behavior in high dimensions with a small calibration set.
