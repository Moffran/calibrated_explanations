# Calibrated Explainer Performance Comparison

This benchmark compares the legacy explanation pipeline with the optimised
implementation introduced in this update. Both paths are executed on synthetic
classification and regression tasks using random forests to capture realistic
perturbation workloads.

## Methodology

1. Build calibrated explainers using 1,500-sample datasets generated via
   `make_classification` and `make_regression`.
2. Fit random forest learners (100 estimators for classification, 120 for
   regression) and initialise `CalibratedExplainer` instances with the new
   discretiser initialisation.
3. Generate 200 test instances per task.
4. Run the optimised `CalibratedExplainer.explain` implementation and the
   preserved legacy path (`calibrated_explanations.core._legacy_explain.explain`) three
   times each, measuring average wall-clock durations.
5. Verify that both paths produce numerically identical explanation payloads
   before reporting timings.

Command:

```bash
PYTHONPATH=./src:. python evaluation/scripts/compare_explain_performance.py
```

## Results

| Task           | Legacy Time (s) | Modern Time (s) | Speedup |
|----------------|----------------:|----------------:|--------:|
| Classification | 3.6649          | 1.1954          | 3.07×   |
| Regression     | 3.4091          | 1.1760          | 2.90×   |

The optimised implementation reduces explanation time by roughly 3× while
preserving output parity with the historical algorithm.【e70f63†L1-L10】
