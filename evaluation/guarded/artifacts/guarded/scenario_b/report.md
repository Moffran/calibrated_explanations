# Scenario B: OOD Detection Quality

## Setup

- Seeds: 10
- Calibration size: 3000
- Train size: 5000
- ID test size: 1000
- OOD test size per shift level: 1000
- Paper-facing slice: normalize_guard=True, n_neighbors=[5], significance=0.1

## Purpose

Scenario B asks: can the guard's p-values reliably distinguish out-of-distribution perturbations from in-distribution perturbations?

Data: calibration and training from N(0, I_d). In-distribution test instances also from N(0, I_d). OOD instances are N(0, I_d) + shift_vector, with shift magnitude = 1σ (mild), 2σ (moderate), 5σ (extreme).

For each test instance, explain_factual(guarded=True) is called and the interval-level guard p-values are combined into one Fisher p-value per instance. AUROC treats 1 - p_combined as the anomaly score against the ground-truth OOD label. The rejection-rate diagnostic is computed separately from raw interval-level p-values on in-distribution audit rows.

The paper-facing slice of this scenario uses normalize_guard=True and n_neighbors=[5].

## Metric contract

The primary metric is AUROC computed from Fisher-combined per-instance guard p-values. This is the direct detection-quality result because it measures ranking quality without depending on a specific threshold.

The secondary diagnostic is the interval-level rejection rate on in-distribution audit rows at the configured significance. It is not a valid statement about Fisher-combined instance scores and should be read only as a calibration-style sanity check for raw interval decisions.

## AUROC by shift level and dimensionality

Mean AUROC on the paper-facing slice:

```csv
shift_level,n_dim,auroc

extreme,2,0.9982201500000001

extreme,5,0.9746341

extreme,10,0.9408309499999999

extreme,20,0.8996422000000001

mild,2,0.6078785499999999

mild,5,0.58510455

mild,10,0.5526853

mild,20,0.5403107

moderate,2,0.8123440000000001

moderate,5,0.7556834

moderate,10,0.6937752500000001

moderate,20,0.6478198
```

AUROC above 0.80 for moderate or extreme shift indicates useful separation. AUROC near 0.50 indicates that the guard is close to random on this synthetic shift task.

## Interval-level rejection rate on in-distribution rows

Mean rejection rate at significance=0.1 on the paper-facing slice:

```csv
n_dim,fpr_at_significance

2,0.13191875

5,0.13932147435897438

10,0.13744285677929663

20,0.12977743590385993
```

We flag configurations where the empirical interval-level rejection rate exceeds 1.5 times the nominal threshold (0.150). 15 configuration(s) exceeded that bound in the paper-facing slice.

## Interpretation

Higher dimensionality makes KNN distance concentration more severe, so AUROC should be expected to degrade as n_dim grows. Mild shifts are allowed to be difficult; the important question is whether moderate and extreme shifts remain separable.

normalize_guard=False and n_neighbors=1 remain in the CSV and plots as engineering stress tests. They are useful for understanding failure modes, but they should not be promoted to headline evidence because the paper claim is about the default guarded configuration.

p-values are discrete with step 1/n_cal. Extremely small significance levels can therefore make the raw interval-level rejection diagnostic look artificially inactive even when AUROC remains informative.
