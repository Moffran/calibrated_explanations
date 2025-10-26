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
4. Execute five strategy variants three times each, recording mean wall-clock
   durations: the historical legacy pipeline, the modern pipeline, cached
   modern (`cache=True`), parallel modern (`use_parallel=True`), and the
   combined cached+parallel path.
5. Verify that every variant produces numerically identical explanation payloads
   before reporting timings.

Command:

```bash
PYTHONPATH=./src:. python evaluation/scripts/compare_explain_performance.py
```

## Results

### Classification

#### `explain_factual`

| Strategy        | Time (s) | Speedup vs Legacy | Speedup vs Modern |
|-----------------|---------:|------------------:|------------------:|
| Legacy          |    0.907 |             1.00× |             0.37× |
| Modern          |    0.337 |             2.69× |             1.00× |
| Cached          |    0.362 |             2.51× |             0.93× |
| Parallel        |    0.334 |             2.72× |             1.01× |
| Cache + Parallel|    0.343 |             2.65× |             0.98× |

#### `explore_alternatives`

| Strategy        | Time (s) | Speedup vs Legacy | Speedup vs Modern |
|-----------------|---------:|------------------:|------------------:|
| Legacy          |    2.616 |             1.00× |             0.27× |
| Modern          |    0.698 |             3.75× |             1.00× |
| Cached          |    0.566 |             4.62× |             1.23× |
| Parallel        |    0.440 |             5.94× |             1.59× |
| Cache + Parallel|    0.469 |             5.57× |             1.49× |

### Regression

#### `explain_factual`

| Strategy        | Time (s) | Speedup vs Legacy | Speedup vs Modern |
|-----------------|---------:|------------------:|------------------:|
| Legacy          |    0.511 |             1.00× |             0.49× |
| Modern          |    0.253 |             2.02× |             1.00× |
| Cached          |    0.188 |             2.73× |             1.35× |
| Parallel        |    0.285 |             1.80× |             0.89× |
| Cache + Parallel|    0.227 |             2.25× |             1.11× |

#### `explore_alternatives`

| Strategy        | Time (s) | Speedup vs Legacy | Speedup vs Modern |
|-----------------|---------:|------------------:|------------------:|
| Legacy          |    1.447 |             1.00× |             0.27× |
| Modern          |    0.391 |             3.70× |             1.00× |
| Cached          |    0.347 |             4.17× |             1.13× |
| Parallel        |    0.451 |             3.21× |             0.87× |
| Cache + Parallel|    0.341 |             4.25× |             1.15× |

## Analysis

- The modern baseline continues to deliver sizeable wins over the legacy path, ranging from 2.0× faster on regression `explain_factual` to 3.8× on classification `explore_alternatives`.
- Classification workloads benefit most from parallelism: turning on `use_parallel=True` cuts `explore_alternatives` latency to 0.44 s (5.9× faster than legacy and 1.6× over the modern default) with only negligible gains for `explain_factual`.
- Regression `explain_factual` prefers caching over pure parallelism; enabling `cache=True` trims latency to 0.19 s (2.7× vs legacy and 1.35× vs modern) while the parallel-only variant incurs extra overhead.
- For regression `explore_alternatives`, caching—alone or combined with parallelism—delivers the best balance, reaching 0.34 s per query and yielding a 4.2× speedup over the legacy pipeline.
- All modern variants continue to produce identical explanation payloads to the legacy implementation while allowing feature flags to target workload-specific performance sweet spots.
