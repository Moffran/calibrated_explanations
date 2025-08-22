# ADR-008: Explanation Domain Model & Legacy-Compatibility Strategy

Status: Proposed
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

- Phase 2: implement domain model (`explanations/models.py`) + adapters; add tests ensuring adapter output matches golden fixtures byte-for-byte.
- Phase 5: align JSON schema v1 with `rules: []`, provide migration tool legacy→v1; continue supporting legacy until v0.8.0.

## Open Questions

- Do we expose the domain model publicly later? Initial stance: internal only; revisit post v0.7.0.
- Naming and minimal required fields for `FeatureRule`—finalize in implementation PR.
