> **Status note (2025-11-29):** Last edited 2025-11-29 · Archive after: v1.0.0 GA · Implementation: Fully completed in v0.10.0 · All ADR-003 gates satisfied per `docs/improvement/adr\ mending/ADR-003/COMPLETION_REPORT.md`.

# ADR-003: Caching Key & Eviction Strategy

Status: Accepted (implemented in v0.10.0)
Date: 2025-08-16
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None

## Context

Repeated calibration and explanation generation can recompute identical intermediate artifacts (model predictions, conformal calibration intervals, feature attribution tensors). Current informal caching can grow unbounded, risks memory bloat, and lacks stable key semantics. We need deterministic cache keys, configurable eviction, and safe invalidation on code/config changes.

## Decision

Introduce a unified in-process cache layer with:

- Key structure: tuple(namespace, version_tag, hash(payload_subset)) where:
  - namespace distinguishes domain ("calibration", "explanation" ,"dataset", etc.)
  - version_tag changes when algorithm parameters or code version affecting semantics changes (derived from `__version__` + strategy revision id)
  - payload_subset is a stable hash (blake2) over selected normalized inputs (e.g., model identifier, n_samples bucket, seed, feature schema hash)
- Default backend: LRU (size-bounded) using `cachetools` with both max items and approximate memory budget (est via `Pympler` sizing; fallback to `sys.getsizeof`).
- Eviction policy: item removed if over size or memory budget; optional TTL for time-sensitive artifacts (unused initially).
- Instrumentation: per-namespace hit/miss counters exposed via a lightweight metrics API (python dict or optional `prometheus_client`).
- Config surface: environment variables + programmatic (`CacheConfig`) for max_items, max_mem_mb, enable/disable namespaces.
- Invalidation triggers: bump version_tag automatically when: library minor version increments; strategy implementation changes recorded via a STRATEGY_REV registry; user explicit flush call.

### Operational clarifications

- **Default posture:** cache stays disabled unless users explicitly opt-in via `CacheConfig(enable=True)` or `CE_CACHE=on`. Shipping in v0.9.0 requires documentation to highlight the opt-in behaviour and explain rollback instructions.
- **API contract preservation:** the cache layer MUST NOT deprecate or require callers to
  change any `WrapCalibratedExplainer` public methods (`fit`, `calibrate`,
  `explain_factual`, `explore_alternatives`, `predict`, `predict_proba`,
  plotting helpers, or uncertainty/threshold options). Behaviour stays
  additive and transparent to the published contract.
- **Thread/process safety:** in-process cache must be guarded by a `threading.RLock` and expose a `forksafe_reset()` hook so the parallel executor (ADR-004) can clear per-process state after `fork`/`spawn`.
- **Failure modes:** cache lookup failures should fall back to recomputation with a warning, never crash the explain path. Size/memory limit breaches surface as debug logs with aggregate counters for eviction.
- **Telemetry contract:** emit structured metrics (`cache_hits`, `cache_misses`, `cache_evictions`, `cache_errors`) via the existing telemetry hook so the release plan's staging validation can track effectiveness. Metrics collection must be no-op when telemetry is disabled.
- **Testing expectations:** add regression tests covering deterministic keys, eviction thresholds, telemetry emission, and opt-in/opt-out toggles. Include a performance smoke benchmark demonstrating that cache hits improve explain latency without regressing cache-disabled behaviour.

### Documentation & rollout requirements

- Update README, docs/plugins.md, and release notes with configuration tables, tuning guidance, and the support policy for the cache namespace taxonomy.
- Record STRATEGY_REV identifiers in the ADR appendix and reference them from the release checklist to ensure invalidation discipline.
- Provide migration guidance for enterprise deployments describing how cache directories/logs interact with existing observability tooling.

## Alternatives Considered

1. No caching (status quo): simpler but repeated recomputation adds latency and energy cost.
2. Joblib.Memory per function: easy but scatters cache directories, weak central control, no cross-function coordination.
3. Redis/external store: enables multi-process & distributed reuse but adds dependency + ops burden; premature for current scope.
4. Deterministic file-based artifact store: future extension; higher persistence complexity now.

## Consequences

Positive:

- Predictable memory footprint with guardrails.
- Reproducible results (stable key hash discipline).
- Observability into cache effectiveness (hit ratio guides tuning).
- Foundation for future pluggable backends (Redis, disk) via single abstraction.

Negative / Risks:

- Overhead hashing large inputs (mitigate by hashing schema + lightweight identifiers not raw arrays when possible).
- Approximate memory sizing may mis-estimate (document variance; allow item count cap override).
- Additional dependency (`cachetools`, optional `pympler`).

## Implementation status (2025-10-07)

- Prototype cache scaffolding must land by v0.9.0 with unit/integration tests and
  documentation updates per the release plan.【F:docs/improvement/RELEASE_PLAN_v1.md†L120-L176】
- No cache layer has been introduced in v0.6.0 yet; the implementation work
  tracks the v0.9.0 milestone and remains outstanding.【F:src/calibrated_explanations/api/config.py†L33-L52】
