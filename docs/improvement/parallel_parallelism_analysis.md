# Parallel Execution Analysis (Feature & Instance)

This note summarizes current behavior, identified bottlenecks, and recommended fixes for feature- and instance-parallel explain execution.

## Findings
- **Pool lifecycle**: `parallel_feature` / `parallel_instance` call `executor.map` without entering the executor context, so `_pool` is `None` and each call spins up a new `ThreadPoolExecutor`/`ProcessPoolExecutor`/joblib backend. Process pool creation is especially expensive on Windows.
- **Aggressive serial short-circuiting**: `ParallelExecutor.map` forces serial when `work_items < min_batch_size` (default 8 after the change), and `_auto_strategy` also returns sequential for `work_items < max(2*min_batch_size, 16)`. Instance-parallel inherits `min_instances_for_parallel` from the same threshold, so many workloads never reach parallel execution even when enabled without tuned knobs.
- **Granularity gating**: Granularity defaults to `feature`; instance-parallel only runs when `granularity=instance` is explicitly set. Backend `strategy` (threads/processes/joblib/sequential) does not switch feature vs instance.
- **Tiny chunks**: Default `feature_chunk_size` / `instance_chunk_size` are `None`, so `chunksize=1` is used. This causes many small tasks, high scheduling and pickling overhead, and poor scaling for process-based backends.
- **GIL-bound workload**: Feature tasks are light and often GIL-bound, reducing benefit from threads; processes help only if overhead is amortized.
- **Fallback visibility**: Only pool init failures warn. Serial fallbacks due to tiny workloads, granularity mismatch, or execution errors are silent unless telemetry is wired.

## Recommendations (priority order)
1. **Manage executor lifetime**: Enter `ParallelExecutor` for the duration of explain/calibrate runs and reuse the pool instead of creating a new pool per `map` call. This is critical for process backends.
2. **Relax/adapt thresholds**: Drop `min_batch_size`/tiny-workload guard toward ~8–16 by default (or scale with `n_features`/`n_instances`), and give instance-parallel its own `min_instances_for_parallel` knob (e.g., `max(8, chunk_size)`) so small-but-parallel-worthy workloads are not forced serial. Expose these via env/config so CI vs prod can differ.
3. **Coarsen chunks**: Set auto chunk heuristics (or sensible defaults) for `feature_chunk_size` / `instance_chunk_size` to reduce task granularity and amortize scheduling/pickling overhead.
4. **Make granularity explicit/auto**: Expose `granularity` clearly in the API; consider an auto mode that selects feature vs instance based on `(n_instances, n_features)` to avoid accidental feature-only runs.
5. **Improve fallback visibility**: Emit warnings (or structured telemetry) when parallel is enabled but downgraded to serial, including reason codes: tiny workload, granularity mismatch, executor init failure, or map failure with serial retry.
6. **Backend guidance/tuning**: Default to threads on Windows; allow opt-in processes with clear guidance on pickling cost. Make worker callables top-level to ease process/joblib usage and reduce pickle size.

## Rationale if unchanged
Without the above adjustments, parallel paths often run serially or pay heavy startup/overhead costs, making them slower than the already-fast sequential implementation. Addressing lifecycle, thresholds, chunking, and visibility should unlock measurable gains and make behavior transparent.
