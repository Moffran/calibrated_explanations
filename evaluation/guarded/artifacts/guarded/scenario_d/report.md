# Scenario D: Real Dataset Correctness

## Setup

- Seeds: 2
- Dataset universe: 2 binary, 2 multiclass, 2 regression datasets
- Ensure-style split policy: test_size=100, calibration_sizes=[100, 300, 500]
- Models: RandomForestClassifier (binary/multiclass), RandomForestRegressor (regression)
- Guard grid: significance=[0.1], n_neighbors=[5], use_bonferroni=[False]

## Purpose

Scenario D asks: does the guard's API remain correct across the full variety of real-world task types — binary classification, multiclass classification, regression, high-dimensional inputs, and small calibration sets?

This run draws from the same dataset registry used by the ensured evaluation suite: 2 binary, 2 multiclass, 2 regression datasets.
The configuration grid can include use_bonferroni=True, which is untested in Scenario A and is retained here as an engineering stress setting.

## Metric contract

This scenario is about API correctness and usability, not about proving that guarded explanations improve scientific quality on real datasets. The two key questions are whether the audit payload is structurally complete and whether the guard remains usable rather than filtering away nearly every explanation.

Accordingly, missing audit fields are correctness failures, while high fully-filtered rates are practicality warnings. Neither metric should be overstated as a standalone scientific result.

## Audit field completeness

Total interval records with missing fields: **0**

PASS: all required fields present in every audit interval record.

## Dataset coverage

Unique datasets skipped by the ensured-style safety checks: **5**

| task       | dataset     | skip_reason                                                         |
|:-----------|:------------|:--------------------------------------------------------------------|
| binary     | pc1req      | not enough remaining samples after test (4 < 500 cal + 100 train)   |
| binary     | haberman    | not enough remaining samples after test (183 < 500 cal + 100 train) |
| multiclass | iris        | not enough remaining samples after test (50 < 500 cal + 100 train)  |
| multiclass | tae         | not enough remaining samples after test (51 < 500 cal + 100 train)  |
| regression | HousingData | not enough remaining samples after test (406 < 500 cal + 100 train) |

## Fraction of instances fully filtered

At significance=0.10, fraction of instances with 0 emitted rules:

|                                |   fraction_instances_fully_filtered |   audit_field_completeness |
|:-------------------------------|------------------------------------:|---------------------------:|
| ('regression', 'abalone', 100) |                               0.05  |                          1 |
| ('regression', 'abalone', 300) |                               0.04  |                          1 |
| ('regression', 'abalone', 500) |                               0.035 |                          1 |

Values > 0.10 mean the guard is more aggressive than the significance level implies — likely due to small calibration sets or high dimensionality. Values > 0.10 should be flagged in documentation.

## Crashes and API errors

API exception count: **0**

PASS: no exceptions raised across any dataset or configuration.

## Interpretation

Binary and multiclass datasets stress the payload shape, class handling, and empty-rule behavior under guarded filtering. Regression datasets stress the separate guarded regression path and confirm the audit contract holds outside classification.

use_bonferroni=True is intentionally included because it makes the guard much stricter and therefore exposes edge cases in the audit and empty-rule paths. That sensitivity is valuable for engineering hardening, but it should not become a headline comparison against the default configuration.

Scenario D supports an engineering claim: the guarded API behaves correctly across realistic dataset shapes. It does not by itself justify a broad real-world effectiveness claim.
