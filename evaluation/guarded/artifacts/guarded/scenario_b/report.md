# Scenario B: OOD Detection Quality

## What this scenario tests

Scenario B asks: can the guard's p-values reliably distinguish out-of-distribution perturbations from in-distribution perturbations?

Data: calibration and training from N(0, I_d). In-distribution test instances also from N(0, I_d). OOD instances are N(0, I_d) + shift_vector, with shift magnitude = 1σ (mild), 2σ (moderate), 5σ (extreme).

For each test instance, explain_guarded_factual is called and the mean p-value across all audit intervals is extracted per instance. AUROC treats (1 − mean_p_value) as the anomaly score against the ground-truth OOD label. FPR is measured at the instance level: the fraction of in-distribution instances whose mean interval p-value falls below significance.

## AUROC by shift level and dimensionality

Mean AUROC (averaged over seeds and n_neighbors, normalize_guard=True):

|                  |    auroc |   fpr_at_significance |
|:-----------------|---------:|----------------------:|
| ('extreme', 2)   | 0.99865  |                 0     |
| ('extreme', 10)  | 0.9894   |                 0.095 |
| ('moderate', 2)  | 0.811625 |                 0     |
| ('moderate', 10) | 0.707175 |                 0.095 |

AUROC > 0.80: guard is reliably detecting OOD perturbations for this config.
AUROC < 0.60: guard is near-random — flagging OOD and in-distribution alike.

## Effect of normalize_guard

Mean AUROC by normalize_guard (all dims and shifts):

| normalize_guard   |    auroc |
|:------------------|---------:|
| True              | 0.876713 |

If normalize_guard=False substantially lowers AUROC, features at different scales dominate the KNN distance and destroy detection quality.

## FPR at significance

Mean FPR@0.1 across all configurations:

|   n_dim |   fpr_at_significance |
|--------:|----------------------:|
|       2 |                 0     |
|      10 |                 0.095 |

By conformal validity, FPR should be ≤ 0.1. Values materially higher indicate the guard rejects more in-distribution perturbations than the theory guarantees.

## Known blind spots exposed by this scenario

1. **Curse of dimensionality**: AUROC degrades as n_dim increases because KNN distances concentrate. This defines where the guard's design breaks down.
2. **normalize_guard=False**: If a dominant-scale feature swamps the distance metric, AUROC will be near 0.5 even for large shifts.
3. **n_neighbors=1 instability**: High AUROC variance across seeds signals unreliable guard behavior.
4. **Minimum p-value granularity**: p-values are discrete with step 1/n_cal. With n_cal=300 the minimum possible p-value is ~0.003. Setting significance < 1/n_cal means the guard can never reject anything — FPR will be exactly 0.
