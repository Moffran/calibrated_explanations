# ADR-004: Parallel Backend Abstraction

Status: Deferred (post-1.0 evaluation)
Date: 2025-08-16
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None

## Context

Some calibration and explanation workflows run per-fold, per-bootstrap, or per-feature perturbations. Naive threading risks GIL contention; pure multiprocessing has serialization overhead. Users may run in heterogeneous environments (laptops, CI, HPC clusters). Need a minimal abstraction to swap execution strategy without invasive conditional code.

## Decision

Provide a pluggable parallel execution facade `ParallelExecutor` with strategy implementations:

- `ThreadPoolExecutorStrategy` (wraps `concurrent.futures.ThreadPoolExecutor`) for I/O bound or C-accelerated code paths.
- `ProcessPoolExecutorStrategy` for CPU-bound pure Python loops when data is picklable.
- `JoblibBackendStrategy` (optional) leveraging joblib if installed for advanced backends (loky, multiprocessing, threading) and batching.
- `SerialStrategy` fallback to preserve determinism and simplify debugging.

Selection algorithm:

1. If user supplies explicit strategy object -> use it.
2. Else if env var `CE_PARALLEL=off` -> Serial.
3. Else inspect workload characteristics (estimated task cost, data size heuristics, n_tasks) and choose:
   - Thread if tasks release GIL (detected by whitelist of known functions or user hint) and n_tasks >= 4.
   - Process if CPU-bound and data payload size per task < threshold (configurable) else Serial (avoid fork storm / OOM).
   - Joblib if installed and user sets `CE_PARALLEL=joblib`.

Unified API: `executor.map(func, iterable)` and context manager semantics. Provide cancellation & graceful shutdown.

Configuration surface via `ParallelConfig` (max_workers, preferred, task_size_hint_bytes, force_serial_on_failure=True).

Instrumentation: timing per task, queue wait time, worker utilization (approx), exported via metrics hooks.

## Alternatives Considered

1. Hard-code joblib everywhere (adds dependency, hides heuristics, less explicit control).
2. Always use multiprocessing (high overhead for small tasks, poor on Windows spawn cost).
3. Rely on users to wrap loops externally (inconsistent performance, duplicates logic, harder to test).
4. Use Ray/Dask (too heavy for library core at current scale; may integrate later via plugin).

## Consequences

Positive:

- Central point to optimize heuristics without touching algorithm code.
- Easier debugging with serial fallback.
- Future extension to distributed frameworks via new strategy class.

Negative / Risks:

- Heuristic misclassification (may choose suboptimal strategy) -> mitigate with logging hint & override.
- Maintenance of strategy matrix as code evolves.
- Added light abstraction layer overhead.

## Implementation status (2025-10-07)

- No parallel executor abstraction has been merged; related configuration flags
  remain inert placeholders on `ExplainerConfig`.
- Release plan schedules any parallel backend work after the v0.9.0 documentation
  and packaging milestone, keeping it out of the v1.0.0 critical path.【F:improvement_docs/RELEASE_PLAN_v1.md†L84-L114】
