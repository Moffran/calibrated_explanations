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

The original ADR proposed a full pluggable `ParallelExecutor` with multiple strategy implementations (thread, process, joblib, serial) and an automatic selection algorithm. After review and a decision to pursue a pragmatic, low-risk rollout for v0.9.1, the ADR is amended to the following two-stage approach:

1) v0.9.1 Scoped Deliverable — ParallelFacade (conservative, low-risk)

- Implement a small `ParallelFacade` (also referenced as "ParallelFacade" or "ParallelChooser") that centralizes conservative selection heuristics and telemetry hooks. The facade will:
  - Accept the existing executor-like objects used by callers and expose a lightweight decision API (e.g., decide_use_parallel(...)).
  - Apply conservative heuristics to prefer serial execution unless clear benefits exist (checks include executor enabled flag, granularity, explicit min thresholds for instances/features, platform heuristics for spawn cost, and an optional task_size_hint_bytes).
  - Honor explicit operator overrides (env var `CE_PARALLEL=off|on|joblib`) and per-call hints.
  - Emit a compact, structured telemetry decision record (decision, reason, n_instances, n_features, bytes_hint, platform, executor_type) via a pluggable telemetry hook so the team can measure real-world benefits before broader rollout.
  - Be intentionally small and conservative: it does not implement new strategy classes, cgroup detection, or fork-safety mechanics in v0.9.1.

  Rationale: this incremental approach reduces immediate engineering risk, provides measurable telemetry to inform further investment, and preserves the current, tested plugin execution paths while enabling safer opt-in parallelism.

2) v0.10 (deferred) — Full ParallelExecutor Strategy Matrix (optional)

- The full `ParallelExecutor` with multiple strategy implementations, workload-aware `_auto_strategy` selection, deep cgroup/container awareness, fork-safety hooks, cooperative cancellation, and expanded telemetry (timings, worker utilisation, fallback counters) is deferred to a later decision point targeting v0.10. The v0.9.1 facade will collect the field evidence needed to determine whether the full ADR should be implemented, postponed, or called off entirely.

Selection algorithm and unified API for the full strategy (v0.10) remain as originally described in this ADR, but are explicitly out of scope for v0.9.1.

### Operational clarifications

The following clarifications capture the phased approach and guardrails for the v0.9.1 facade and the deferred v0.10 work:

- Opt-in default (unchanged): parallel execution stays disabled unless an executor or an enabling flag is supplied. The v0.9.1 facade will respect `executor.config.enabled`, `ParallelConfig` flags, and `CE_PARALLEL` overrides.

- Domain-specific orchestration wraps the facade: per-domain runtimes (e.g., explain) should provide their own wrappers around the shared `ParallelExecutor` so that heuristics, chunk sizing, and nesting guards remain co-located with domain logic while the low-level executor stays in `calibrated_explanations.parallel`.

- API contract preservation: strategy selection and configuration MUST remain optional enhancements. `WrapCalibratedExplainer` public entry points (`fit`, `calibrate`, `explain_factual`, `explore_alternatives`, `predict`, `predict_proba`, and plotting/uncertainty helpers) keep their existing signatures without deprecation warnings or behavioural breaking changes.

- Graceful degradation (v0.9.1 minimum): the facade will detect basic executor errors (missing executor, disabled executor, platform hints indicating high spawn cost) and prefer serial; it will also surface a telemetry record when it forces a fallback to serial. Full automatic fallback-on-exception semantics for strategy construction remain part of the v0.10 scope.

- Compatibility with caching (deferred): fork/spawn cache hygiene is a known issue and remains part of the v0.10 scope. The v0.9.1 facade will document the requirement and provide guidance but will not implement automated fork-safety hooks.

- Resource limits: honour `max_workers` caps and expose heuristics for CPU count detection, respecting container cgroup quotas. Provide guardrails to avoid oversubscription in CI (detect via env flags, default to serial). The facade will include conservative defaults and expose overrides for CI/staging.

- Telemetry contract (phased): v0.9.1 will require the facade to emit compact decision telemetry (as described above). The fuller set of runtime metrics (`tasks_submitted`, `tasks_completed`, `worker_utilisation_pct`, etc.) will be part of the v0.10 effort once the facade's field telemetry indicates sufficient benefit to justify the instrumentation cost.

- Testing requirements (v0.9.1): add unit tests for the facade decision logic (env var overrides, thresholds, platform heuristics), and a micro-benchmark harness for evidence collection. More exhaustive lifecycle tests (fork/spawn, joblib backends) are deferred to v0.10.

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

### Implementation status (2025-11-05)

- v0.9.1 decision: implement the conservative "ParallelFacade" scoped deliverable (see Decision above). The facade will centralize conservative selection heuristics, emit decision telemetry, and provide a safe opt-in path for the existing executor interface. This work is small, testable, and intended to ship in the v0.9.1 governance & observability hardening milestone.

- v0.10: the full `ParallelExecutor` strategy matrix and deeper runtime instrumentation remain a candidate for v0.10; the team will decide to proceed, postpone, or cancel after analyzing the telemetry collected from the facade in v0.9.1.

### Testing and rollout guidance (v0.9.1)

- Add unit tests for the facade decision logic (respect env var overrides, minimum thresholds, platform heuristics).
- Add a small micro-benchmark harness (evaluation/parallel_ablation.py) that records wall-clock improvements for canonical workloads and stores results in JSON so the v0.10 decision can be evidence-driven.
- Ship the facade behind the existing executor opt-in flags; do not change the default behaviour for users.

### Notes

- This amendment aims to provide an incremental, low-risk path to gather data about when parallelism improves performance in realistic usage. If the telemetry shows limited benefit, the full ADR-004 work may be postponed or scoped smaller during v0.10 planning.
