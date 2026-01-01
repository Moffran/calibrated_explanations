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

## Implementation status (2025-10-07)

- Cache scaffolding should land with unit tests and documentation updates per the release plan.【F:docs/improvement/RELEASE_PLAN_v1.md†L120-L176】
- No cache layer has been introduced in v0.6.0 yet; the implementation work
  tracks the v0.9.0 milestone and remains outstanding.【F:src/calibrated_explanations/api/config.py†L33-L52】
