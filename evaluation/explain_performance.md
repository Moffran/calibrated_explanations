# Calibrated Explainer Performance Comparison

This benchmark compares the legacy explanation pipeline with the optimised
implementation introduced in this update. Both paths are executed on synthetic
classification and regression tasks using random forests to capture realistic
perturbation workloads.

## Methodology

1. Build calibrated explainers using 2,000-sample datasets generated via
   `make_classification` and `make_regression` (64 total features with 16
   informative dimensions; classification adds no redundant features and
   regression noise is set to 0.2).
2. Fit random forest learners (10 estimators for both classification and
   regression) and initialise `CalibratedExplainer` instances with the new
   discretiser initialisation.
3. Reserve 500 samples for calibration and 100 held-out test instances per task.
4. Execute five strategy variants with a single warm-up run followed by ten
   timed repeats, recording mean wall-clock durations: the historical legacy
   pipeline, the modern pipeline, cached modern (`cache=True`), parallel modern
   (`use_parallel=True`, thread strategy), and the combined cached+parallel
   path (namespace `benchmark`, version `v1`).
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
| Legacy          |   85.73 |             1.00× |             0.03× |
| Modern          |    2.56 |            33.49× |             1.00× |
| Cached          |    2.60 |            33.00× |             0.99× |
| Parallel        |    2.52 |            33.99× |             1.01× |
| Cache + Parallel|    2.15 |            39.85× |             1.19× |

#### `explore_alternatives`

| Strategy        | Time (s) | Speedup vs Legacy | Speedup vs Modern |
|-----------------|---------:|------------------:|------------------:|
| Legacy          |  158.91 |             1.00× |             0.02× |
| Modern          |    2.58 |            61.61× |             1.00× |
| Cached          |    2.45 |            64.96× |             1.05× |
| Parallel        |    2.72 |            58.47× |             0.95× |
| Cache + Parallel|    2.44 |            65.19× |             1.06× |

### Regression

#### `explain_factual`

| Strategy        | Time (s) | Speedup vs Legacy | Speedup vs Modern |
|-----------------|---------:|------------------:|------------------:|
| Legacy          |   71.22 |             1.00× |             0.03× |
| Modern          |    2.01 |            35.51× |             1.00× |
| Cached          |    1.76 |            40.54× |             1.14× |
| Parallel        |    1.87 |            38.10× |             1.07× |
| Cache + Parallel|    1.58 |            45.18× |             1.27× |

#### `explore_alternatives`

| Strategy        | Time (s) | Speedup vs Legacy | Speedup vs Modern |
|-----------------|---------:|------------------:|------------------:|
| Legacy          |  135.08 |             1.00× |             0.02× |
| Modern          |    3.01 |            44.88× |             1.00× |
| Cached          |    2.71 |            49.89× |             1.11× |
| Parallel        |    2.89 |            46.81× |             1.04× |
| Cache + Parallel|    3.15 |            42.86× |             0.96× |

## Analysis

- Even without cache or parallel enabled, the modern pipeline cuts legacy runtimes of 71–159 s down to roughly 2–3 s (33–62× improvements across all workloads).
- Classification runs gain the most from enabling both optimisations: cache+parallel clocks 2.15 s / 39.85× for `explain_factual` and 2.44 s / 65.19× for `explore_alternatives`, edging out the other variants.
- Regression `explain_factual` also prefers the combined strategy, dropping to 1.58 s (45.18× vs legacy, 1.27× vs modern).
- Regression `explore_alternatives` is cache-bound: caching alone lands at 2.71 s (49.89× vs legacy, 1.11× vs modern) while adding the parallel executor nudges latency up to 3.15 s.
- Every modern variant continues to match the legacy payload bit-for-bit, so teams can toggle cache and parallel strictly based on latency goals.
