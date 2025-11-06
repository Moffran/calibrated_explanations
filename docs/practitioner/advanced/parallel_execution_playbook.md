# Parallel execution playbook (opt-in)

Experienced teams can squeeze additional throughput from calibrated explanations
by tuning the parallel executor. This playbook distils the current heuristics
while ADR-004 improvements are still underway, so you can decide when the extra
complexity is worthwhile. Pair it with the configuration steps in
{doc}`../../foundations/how-to/tune_runtime_performance` and roll back to the
sequential baseline whenever the gains are marginal.

```{tip}
Keep these controls off until you have baseline explanations, calibration
checks, and governance sign-off. Parallel overhead can exceed any speed-up on
small workloads, especially on Windows.
```

## Quick decision matrix

- **Stay sequential** when ``n_instances × n_features < 50,000`` or when
  Windows deployments cannot rely on thread-safe payloads.
- **Use instance-parallel** when instances dominate (``n_instances ≥ 4 × workers × 64``)
  and feature counts stay below ~64; aim for chunks between 256 and 1,024 rows.
- **Use feature-parallel** when features dominate (``n_features ≥ 4 × workers × 4``)
  and the test set stays below ~200 instances; prefer thread-based backends on
  Windows.
- **Avoid process pools on Windows** unless work items take hundreds of
  milliseconds and payloads are shared via memory mapping.

## Strategy checkpoints

### Sequential (`SequentialExplainPlugin`)

Sequential execution remains the reference path. It wins by default on small or
medium datasets and whenever ``CE_PARALLEL`` stays disabled. Use it to validate
correctness before enabling any executor.

### Instance-parallel (`InstanceParallelExplainPlugin`)

Best for wide batches of instances with relatively few features.

- **Chunk size** – ``min_batch_size`` doubles as the chunk size. Target
  ``max(256, ceil(n_instances / (workers × 3)))`` and keep the lower bound at 128
  to avoid thrashing.
- **Workers** – favour threads on Windows. On Linux or macOS, threads work well
  when NumPy dominates; processes only help when Python control flow is the
  bottleneck and payloads serialise cheaply.
- **Fallback** – switch back to sequential when chunk sizes shrink below 128 or
  when feature counts top 128 (feature-parallel usually performs better there).

### Feature-parallel (`FeatureParallelExplainPlugin`)

Ideal for feature-dense explain workloads with modest instance counts.

- **Gating** – raise ``min_batch_size`` so parallelism only kicks in when
  ``work_items = n_instances × n_features`` crosses roughly ``workers × n_instances × 2``
  (practical floor ≈ 10,000). Smaller payloads run sequentially faster.
- **Workers** – cap thread pools to physical cores to avoid contending with the
  model's own inference threads. Prefer threads on Windows; allow joblib/threads
  on Linux when arrays are memmapped.
- **Fallback** – revert to sequential when feature counts fall within a single
  worker group (e.g. ``n_features ≤ workers × 4``) or when test sets exceed 250
  rows.

## Backend recommendations

- **Windows** – use the thread backend. Joblib defaults to processes and inherits
  heavy spawn costs.
- **Linux/macOS** – threads work best when NumPy releases the GIL; processes help
  only when the workload is Python-bound and payloads are light or shared.
- **Joblib** – consider it experimental until ADR-004 rolls out batching. Move
  immutable arrays to module-level globals to let joblib memmap instead of
  pickling per task.

## Parameter tuning checklist

1. Estimate total work (`n_instances × n_features`).
2. Choose the strategy using the decision matrix above.
3. Select worker counts (logical cores for threads, physical cores for
   processes).
4. Derive chunk sizes from the recommended bounds; clamp instance chunks to
   256–1,024 rows.
5. Keep feature-parallel gated behind at least ~10k work items to avoid needless
   overhead.
6. Benchmark against sequential; adopt the executor only when you see ≥1.2×
   speed-up.

## Current limitations (pre-ADR-004)

These constraints are being tracked in the runtime roadmap (Release Plan v0.9.0
item 11 and v0.10.0 item 4) and will ease once ADR-004 lands:

- ``min_batch_size`` still serves as both the gating threshold and the chunk
  size for instance parallelism.
- Feature tasks serialise large payloads, so process-based backends pay heavy
  pickling costs.
- ``auto`` strategy ignores workload hints; manual tuning stays necessary.
- Telemetry lacks per-strategy timings and utilisation metrics, limiting
  observability.

## Related resources

- {doc}`../../foundations/how-to/tune_runtime_performance` – enable/disable the
  executor and cache.
- {doc}`../../foundations/concepts/telemetry` – instrument runtime metrics when
  you opt in to parallel execution.
- `improvement_docs/parallel_execution_improvement_plan.md` – internal task
  breakdown tracking the ADR-004 remediation.
