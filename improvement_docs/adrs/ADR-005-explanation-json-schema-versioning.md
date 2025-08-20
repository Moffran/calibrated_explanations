# ADR-005: Explanation JSON Schema Versioning

Status: Proposed
Date: 2025-08-16
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None

## Context

Explanations (feature importances, per-sample contributions, intervals) are serialized to JSON for persistence, interop, and visual tooling. Current outputs are ad-hoc dicts without explicit schema versioning, making downstream consumers brittle as structures evolve.

## Decision

Establish a versioned explanation envelope:

```json
{
  "schema_version": "1.0.0",
  "type": "feature_attribution",
  "generator": {
    "library_version": "0.6.0",
    "strategy": "isotonic_conformal",
    "parameters_hash": "<blake2-short>"
  },
  "meta": {"dataset_hash": "...", "n_features": 42, "created_at": "ISO8601"},
  "payload": { "values": [...], "feature_names": [...], "baseline": [...], "extra": {...} }
}
```

Rules:

- `schema_version` follows semver for schema (independent from library version) starting at 1.0.0.
- Minor increments allow additive, backwards-compatible fields (consumers ignore unknown keys).
- Major increments indicate breaking structural changes.
- `type` enumerated (initial set): feature_attribution, interval, global_importance, calibration_diagnostics.
- `generator.parameters_hash` ensures reproducibility trace.
- `payload` structure type-dependent; each type documented in `docs/schema/`.

Validation:

- Provide `validate_explanation(obj)` that checks required fields, value types, and semantic invariants.
- Ship JSON Schema drafts for each type (`schema/v1/*.json`).
- CI test loads fixtures and validates with `jsonschema` library.

## Alternatives Considered

1. Tie schema to library version only (harder multi-producer compatibility, forces sync upgrades).
2. Omit envelope and rely on dynamic inspection (fragile; poor tooling alignment).
3. Use Protocol Buffers / Avro (stronger typing but higher barrier, JSON sufficient initially).

## Consequences

Positive:

- Stable contract for external tools & plugins.
- Explicit evolution path (semver) reduces upgrade fear.
- Easier debugging via self-describing envelope.

Negative / Risks:

- Maintenance overhead (schema docs + fixtures).
- Potential duplication of metadata (acceptable tradeoff).

## Adoption & Migration

Phase 0: Implement envelope + validator; keep old internal paths but do not promise stability.
Phase 1: Migrate library outputs to envelope; add golden tests.
Phase 2: Deprecate undocumented legacy shapes; announce removal.

## Open Questions

- Embed compression (e.g., base64 zstd) for large arrays? Maybe later.
- Provide streaming generation for very large explanations? Future.
- Need a registry for custom `type` extensions? Possibly with `x-` prefix.

## Decision Notes

Revisit after first external consumer feedback cycle.
