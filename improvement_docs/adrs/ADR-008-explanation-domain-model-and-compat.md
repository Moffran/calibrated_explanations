> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Retain indefinitely as architectural record · Implementation window: Per ADR status (see Decision).

# ADR-008: Explanation Domain Model & Legacy-Compatibility Strategy

Status: Accepted
Date: 2025-08-22
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None

## Context

Issue reported: The current `explanations.py` stores explanation data in a dict with a mix of
singletons (for instance-level values) and lists/arrays (per-feature rule values).
This makes iteration/filtering awkward and hard to reason about.

## Decision

Introduce an internal domain model with first-class rule objects:

- `Explanation`: top-level metadata (task, model info, calibration params, provenance), and `rules: list[FeatureRule]`.
- `FeatureRule`: per-rule payload (feature id/name, predicate/interval, attribution/weight, support, confidence/uncertainty, local details).
- Build `Explanation` internally from existing pipelines, but keep public APIs and serializers returning the legacy dict shape via an adapter until schema v1 is adopted.

Rationale: improves clarity, enables easy filtering/transforms, and aligns with future schema and visualization work.

## Consequences

- Positive: simpler iteration/filtering, safer invariants, clearer ownership of fields, smoother future schema migration.
- Negative: adds an adapter layer; requires adapter parity tests and slight maintenance.

## Alternatives

- Keep dict-only approach (status quo): continues complexity and duplication in consumers.
- Hard break to new dict shape now: would violate deprecation policy and break golden tests.

## Adoption & Migration

- Phase B: implement domain model (`explanations/models.py`) + adapters; add tests ensuring adapter output matches golden fixtures byte-for-byte.
- Phase B/C: round-trip serialization (domain → JSON → domain) using ADR-005 envelope.
- Removal: continue supporting legacy public dicts via adapter until v0.8.0 per deprecation policy (ADR-011).

## Open Questions

- Do we expose the domain model publicly later? Initial stance: internal only; revisit post v0.7.0.
- Naming and minimal required fields for `FeatureRule`—finalize in implementation PR.
