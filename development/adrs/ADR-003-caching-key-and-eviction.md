---
id: "ADR-003"
title: "Caching Key & Eviction Strategy"
status: "accepted"
description: "Governing architectural decision for the explanation cache key design and eviction policy. Remains active as long as the cache key contract governs performance-sensitive explain paths; superseded when the caching strategy is revised."
supersedes: []
superseded_by: []
---
> **Status note (2026-07-21):** Last edited 2026-07-21 · Implementation: Fully completed in v0.10.0; caching remains opt-in (`CE_CACHE`) pending a clean `1.0.1` telemetry regression sweep (see `development/current-work/v1.0.1_plan.md`) · Historical ADR-003 gate closure evidence is in `development/finished-work/RELEASE_PLAN_status_appendix.md`.

# ADR-003: Caching Key & Eviction Strategy

Date: 2025-08-16
Deciders: Core maintainers
Reviewers: TBD

## Context

Repeated calibration and explanation generation can recompute identical intermediate artifacts (model predictions, conformal calibration intervals, feature attribution tensors). Current informal caching can grow unbounded, risks memory bloat, and lacks stable key semantics. We need deterministic cache keys, configurable eviction, and safe invalidation on code/config changes.

## Decision

Adopt a lightweight, opt-in in-process cache with simple, deterministic keys and minimal configuration:

- Key structure: tuple(namespace, version_tag, hash(payload_subset)) where:
  - namespace distinguishes domain ("calibration", "explanation", "dataset", etc.)
  - version_tag changes when algorithm parameters or code version affecting semantics changes (derived from `__version__` + strategy revision id)
  - payload_subset is a stable hash (blake2) over normalized inputs (e.g., model identifier, n_samples bucket, seed, feature schema hash)
- Default backend: size-bounded LRU using `cachetools` (max items only; no memory sizing heuristics).
- Optional TTL support may be added later, but is off by default.
- Config surface: environment variables + programmatic (`CacheConfig`) for enable/disable and max_items; namespace allow/deny lists for coarse control.
- Invalidation triggers: bump version_tag when algorithm semantics change or when users explicitly clear the cache.

### Operational clarifications

- **Default posture:** cache stays disabled unless users explicitly opt-in via `CacheConfig(enable=True)` or `CE_CACHE=on`. Documentation must highlight the opt-in behaviour.
- **API contract preservation:** the cache layer MUST NOT deprecate or require callers to
  change any `WrapCalibratedExplainer` public methods (`fit`, `calibrate`,
  `explain_factual`, `explore_alternatives`, `predict`, `predict_proba`,
  plotting helpers, or uncertainty/threshold options). Behaviour stays
  additive and transparent to the published contract.
- **Thread/process safety:** cache entries are process-local; fork/spawn hygiene is not handled automatically in this ADR and remains a caller responsibility.
- **Failure modes:** cache lookup failures should fall back to recomputation with a warning, never crash the explain path.
- **Observability:** no mandatory telemetry contract. Optional debug logging is sufficient for OSS users to validate behaviour.
- **Testing expectations:** add regression tests covering deterministic keys, eviction thresholds, and opt-in/opt-out toggles.

### Documentation & rollout requirements

- Update README and release notes with configuration tables, tuning guidance, and the support policy for the cache namespace taxonomy.
- Record STRATEGY_REV identifiers in the ADR appendix and reference them from the release checklist to ensure invalidation discipline.


## Governed claims

- `CE-CAP-CACHE-001` — Caching is opt-in, transparent to public APIs, and governed by deterministic key and fallback visibility constraints.

## Alternatives Considered

1. No caching (status quo): simpler but repeated recomputation adds latency and energy cost.
2. Joblib.Memory per function: easy but scatters cache directories, weak central control, no cross-function coordination.
3. Redis/external store: enables multi-process & distributed reuse but adds dependency + ops burden; premature for current scope.
4. Deterministic file-based artifact store: future extension; higher persistence complexity now.

## Consequences

Positive:

- Predictable memory footprint with simple guardrails.
- Reproducible results (stable key hash discipline).
- Low dependency surface and reduced operational complexity.
- Foundation for future pluggable backends (Redis, disk) via single abstraction.

Negative / Risks:

- Overhead hashing large inputs (mitigate by hashing schema + lightweight identifiers not raw arrays when possible).
- No built-in cache effectiveness metrics unless users add their own logging.
- Additional dependency (`cachetools`).

## Implementation status (2026-06-11, superseding the 2025-10-07 note below)

**Compliance verification (2026-06-11):** Reviewed code and RTD — no ADR-003
gaps found. Namespaced/versioned blake2b cache keys
(`cache/cache.py:253-285` `make_key(namespace, version, parts)`), a
`cachetools` LRU backend with in-package fallback (`cache/cache.py:39-189`),
telemetry counters (`cache/cache.py:290`), and an opt-in default-off posture
(`CacheConfig.enabled: bool = False`, `cache/cache.py:356`) are all
implemented. ADR-003 is fully compliant; caching remains explicit opt-in
(`CE_CACHE`). On-by-default graduation is deferred until a clean telemetry
regression sweep (`development/current-work/v1.0.1_plan.md`, row T3) and at
least one further maintenance cycle show stable, unsurprising telemetry with
no fallback-rate regressions; reopen via a GitHub issue against ADR-003 when
that evidence exists.

<details>
<summary>Historical note (2025-10-07, pre-implementation)</summary>

- Cache scaffolding should land with unit tests and documentation updates per
  the release plan.
- No cache layer has been introduced in v0.6.0 yet; the implementation work
  tracks the v0.9.0 milestone and remains outstanding.

</details>
