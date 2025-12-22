> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Retain indefinitely as architectural record · Implementation window: Per ADR status (see Decision).

# ADR-004: Parallel Backend Abstraction

Status: Accepted
Date: 2025-08-16
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None

## Context

Some calibration and explanation workflows run per-fold, per-bootstrap, or per-feature perturbations. Naive threading risks GIL contention; pure multiprocessing has serialization overhead. Users may run in heterogeneous environments (laptops, CI, HPC clusters). Need a minimal abstraction to swap execution strategy without invasive conditional code.

## Decision

Adopt an explicit, opt-in parallelism surface with serial-by-default behaviour and no automatic strategy selection:

- Provide a minimal `ParallelFacade` that accepts user-supplied executors and standardizes how core routines invoke them.
- Do not auto-select strategies or infer parallelism based on workload size. Callers must explicitly pass an executor or enable parallelism through configuration.
- Keep supported strategies limited to what callers provide (serial, threads, processes, joblib). The core library does not implement a strategy matrix.
- Ensure parallel execution remains deterministic with respect to input ordering and output aggregation (parallelism may affect timing but not results).
- Defer any heuristic selection, cgroup awareness, or advanced telemetry to a separate ADR if evidence warrants it.

### Operational clarifications

The following clarifications capture the guardrails for explicit, opt-in parallelism:

- Opt-in default (unchanged): parallel execution stays disabled unless an executor or an enabling flag is supplied. The facade will respect `executor.config.enabled`, `ParallelConfig` flags, and `CE_PARALLEL` overrides.

- Domain-specific orchestration wraps the facade: per-domain runtimes (e.g., explain) should provide their own wrappers around the shared facade so that chunk sizing and nesting guards remain co-located with domain logic while the low-level executor stays in `calibrated_explanations.parallel`.

- API contract preservation: strategy selection and configuration MUST remain optional enhancements. `WrapCalibratedExplainer` public entry points (`fit`, `calibrate`, `explain_factual`, `explore_alternatives`, `predict`, `predict_proba`, and plotting/uncertainty helpers) keep their existing signatures without deprecation warnings or behavioural breaking changes.

- Graceful degradation: if the supplied executor is missing or disabled, fall back to serial with a warning. No automated strategy construction or fallback matrix is required.

- Compatibility with caching: fork/spawn cache hygiene is out of scope; document that cache is process-local and must be reset by caller if needed.

- Resource limits: honor `max_workers` caps from the supplied executor and avoid adding new auto-detection heuristics in core.

- Telemetry contract: no mandatory telemetry emission for decisions. Optional debug logs are acceptable.

- Testing requirements: add unit tests verifying opt-in behavior, explicit overrides, and serial fallback when executor is disabled.

### Documentation & rollout requirements

- Extend configuration docs and release notes with the explicit opt-in model, environment variable controls, and troubleshooting tips for common platforms (macOS spawn, Windows spawn).
- Ship an upgrade guide snippet covering interaction with plugin-provided executors and guidance for opting out when running within user-managed pools.

## Alternatives Considered

1. Hard-code joblib everywhere (adds dependency, hides heuristics, less explicit control).
2. Always use multiprocessing (high overhead for small tasks, poor on Windows spawn cost).
3. Rely on users to wrap loops externally (inconsistent performance, duplicates logic, harder to test).
4. Use Ray/Dask (too heavy for library core at current scale; may integrate later via plugin).

## Consequences

Positive:

- Central point to invoke user-supplied executors without touching algorithm code.
- Easier debugging with serial fallback.
- Future extension to distributed frameworks via new strategy class.

Negative / Risks:

- Some users may expect automatic parallelism and need to enable it explicitly.
- Added light abstraction layer overhead.

### Implementation status (2025-12-13)

- v0.9.1 decision: implement the explicit opt-in "ParallelFacade" scoped deliverable (see Decision above). The facade standardizes executor usage without adding selection heuristics or telemetry requirements.
- **Update (2025-12-13):** Feature-parallel execution strategy has been deprecated and shimmed to fall back to instance-parallel execution. Benchmarking revealed that feature-parallelism introduced significant overhead without providing performance benefits for typical workloads. The `FeatureParallelExplanationPlugin` and `FeatureParallelAlternativeExplanationPlugin` now alias to their instance-parallel counterparts to maintain API compatibility.

### Testing and rollout guidance (v0.9.1)

- Add unit tests for opt-in behavior and serial fallback when the executor is disabled.
- Ship the facade behind the existing executor opt-in flags; do not change the default behaviour for users.

### Notes

- This amendment limits scope to explicit, opt-in parallelism to keep the OSS core predictable. Any broader automation should be revisited only with evidence and a separate ADR.
