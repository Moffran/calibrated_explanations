# Scenario B: OOD Detection Quality

This checked-in report is a quick-run engineering artifact.
For the paper, AUROC should be described as using Fisher-combined per-instance p-values, while any rejection-rate diagnostic should be computed from raw interval-level p-values on in-distribution audit rows.

## Setup

- Seeds: 2
- Calibration size: 300
- Train size: 500
- ID test size: 100
- OOD test size per shift level: 100
- Paper-facing slice: normalize_guard=True, n_neighbors=[5], significance=0.1

## Purpose

Scenario B asks: can the guard's p-values reliably distinguish out-of-distribution perturbations from in-distribution perturbations?

Data: calibration and training from N(0, I_d). In-distribution test instances also from N(0, I_d). OOD instances are N(0, I_d) + shift_vector, with shift magnitude = 1σ (mild), 2σ (moderate), 5σ (extreme).

For each test instance, explain_guarded_factual is called and the interval-level p-values are combined into one Fisher p-value per instance. AUROC treats \(1 - p_{\mathrm{combined}}\) as the anomaly score against the ground-truth OOD label. Any rejection-rate diagnostic should be computed separately from raw interval-level p-values on in-distribution audit rows.

## Metric contract

The primary metric is AUROC computed from Fisher-combined per-instance guard p-values. This is the direct detection-quality result because it measures ranking quality without depending on a specific threshold.

The secondary diagnostic is the interval-level rejection rate on in-distribution audit rows at the configured significance. It is not a valid statement about Fisher-combined instance scores and should be read only as a calibration-style sanity check for raw interval decisions.

## AUROC by shift level and dimensionality

Mean AUROC (averaged over seeds and n_neighbors, normalize_guard=True):

|                  |    auroc |   fpr_at_significance |
|:-----------------|---------:|----------------------:|
| ('extreme', 2)   | 0.99865  |                 0     |
| ('extreme', 10)  | 0.9894   |                 0.095 |
| ('moderate', 2)  | 0.811625 |                 0     |
| ('moderate', 10) | 0.707175 |                 0.095 |

AUROC > 0.80: guard is reliably detecting OOD perturbations for this config.
AUROC < 0.60: guard is near-random on this synthetic shift task.

## FPR at significance

Mean FPR@0.1 across all configurations:

|   n_dim |   fpr_at_significance |
|--------:|----------------------:|
|       2 |                 0     |
|      10 |                 0.095 |

This quantity should be interpreted as an empirical interval-level rejection rate. It is not a conformal guarantee for Fisher-combined instance scores.

## Interpretation

Higher dimensionality makes KNN distance concentration more severe, so AUROC should be expected to degrade as `n_dim` grows. Mild shifts are allowed to be difficult; the important question is whether moderate and extreme shifts remain separable.

Non-default settings such as `normalize_guard=False` and `n_neighbors=1` are useful engineering stress tests because they expose failure modes, but they should not be promoted to headline evidence. The paper claim is about the default guarded configuration.

p-values are discrete with step `1 / n_cal`. Extremely small significance levels can therefore make the raw interval-level rejection diagnostic look artificially inactive even when AUROC remains informative.
