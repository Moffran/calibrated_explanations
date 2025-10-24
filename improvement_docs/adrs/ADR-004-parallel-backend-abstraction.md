> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Retain indefinitely as architectural record · Implementation window: Per ADR status (see Decision).

# ADR-004: Parallel Backend Abstraction

Status: Proposed (targeting v0.9.0 opt-in release)
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

### Operational clarifications

- **Opt-in default:** parallel execution stays disabled unless `ParallelConfig(enable=True)` is supplied or `CE_PARALLEL` is set to a non-`off` value. Document per-release guidance highlighting when to keep serial mode.
- **API contract preservation:** strategy selection and configuration MUST remain optional
  enhancements. `WrapCalibratedExplainer` public entry points (`fit`,
  `calibrate`, `explain_factual`, `explore_alternatives`, `predict`,
  `predict_proba`, and plotting/uncertainty helpers) keep their existing
  signatures without deprecation warnings or behavioural breaking changes.
- **Graceful degradation:** if a strategy raises during executor creation or task execution, automatically fall back to `SerialStrategy` and emit a structured warning; failures must not abort explanation flows.
- **Compatibility with caching:** after `fork`/`spawn`, call the `forksafe_reset()` cache hook defined in ADR-003 so per-process caches do not leak stale references.
- **Resource limits:** honour `max_workers` caps and expose heuristics for CPU count detection, respecting container cgroup quotas. Provide guardrails to avoid oversubscription in CI (detect via env flags, default to serial).
- **Telemetry contract:** record aggregated metrics (`tasks_submitted`, `tasks_completed`, `worker_utilisation_pct`, `fallbacks_to_serial`) and ensure they flow through the same telemetry API used for caching metrics. Logging must be suppressible in production by honoring existing verbosity settings.
- **Testing requirements:** add unit tests for each strategy path, fork/spawn lifecycle, telemetry emission, and failure fallback. Provide an integration benchmark demonstrating improved throughput on the Python fallback explain path.

### Documentation & rollout requirements

- Extend configuration docs and release notes with parallel strategy descriptions, environment variable matrix, and troubleshooting tips for common platforms (macOS spawn, Windows spawn).
- Ship an upgrade guide snippet covering interaction with plugin-provided executors and guidance for opting out when running within user-managed pools.
- Capture heuristics and defaults in an appendix so future contributors can tune thresholds without re-auditing the ADR history.

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

- Parallel executor abstraction is scheduled for v0.9.0 delivery alongside the
  caching controls, but no code has landed yet; configuration flags remain
  placeholders pending implementation.【F:improvement_docs/RELEASE_PLAN_v1.md†L120-L176】
- Telemetry and fallback wiring must be verified during the v1.0.0-rc staging
  window before enabling broader defaults.【F:improvement_docs/RELEASE_PLAN_v1.md†L178-L197】
