# ADR-003: Caching Key & Eviction Strategy

Status: Proposed (baseline, feature-flagged)
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
- Default backend: LRU (size-bounded) using `cachetools` with both max items and approximate memory budget (est via `Pympler` fallback to `sys.getsizeof`).
- Eviction policy: item removed if over size or memory budget; optional TTL for time-sensitive artifacts (unused initially).
- Instrumentation: per-namespace hit/miss counters exposed via a lightweight metrics API (python dict or optional `prometheus_client`).
- Config surface: environment variables + programmatic (`CacheConfig`) for max_items, max_mem_mb, enable/disable namespaces.
- Invalidation triggers: bump version_tag automatically when: library minor version increments; strategy implementation changes recorded via a STRATEGY_REV registry; user explicit flush call.

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

## Adoption & Migration

Phase E (v0.7.0): Introduce module `calibrated_explanations.cache` behind a feature flag (disabled by default). Start with pure, high-cost idempotent steps.
Phase later: Expand to explanation generation if stable; add micro-benchmarks and perf guards.

## Open Questions

- Do we need per-model namespace segmentation beyond key hashing? (Likely no initially.)
- Whether to surface a public cache API or keep internal until stable.
- Should TTL be activated for stochastic strategies to avoid stale randomness perception?

## Decision Notes

Revisit after first perf measurements post Phase 2 to adjust defaults.
