> **Status note (2026-01-12):** Last edited 2026-01-12 · Archive after: Retain indefinitely as architectural record · Implementation window: v0.11.x.

# ADR-031: Calibrator Serialization & Explainer State Persistence

Status: Draft
Date: 2026-01-12
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None
Related: ADR-009-input-preprocessing-and-mapping-policy, ADR-021-calibrated-interval-semantics

## Context

The OSS runtime lacks a stable, versioned serialization contract for
calibrators and explainer state. Users currently reconstruct explainers by
retraining or wiring custom pickling logic, which undermines reproducibility
and makes it difficult to share deterministic artifacts across environments.
ADR-009 describes mapping persistence expectations for preprocessing, and
ADR-021 defines calibrated interval semantics that must remain invariant across
sessions. A dedicated serialization contract is required to guarantee those
semantics while allowing future migrations.

## Decision

1. **Versioned primitive contract for calibrators.**
   - All built-in calibrators must implement `to_primitive()` returning a
     JSON-safe `dict` containing a required `schema_version` and a
     calibrator-specific payload.
   - A corresponding `from_primitive(payload: Mapping[str, object])` must
     reconstruct the calibrator, validating the `schema_version` and raising a
     clear, documented exception on incompatibility.

2. **Explainer state persistence API.**
   - `Explainer.save_state(path_or_fileobj)` and
     `Explainer.load_state(path_or_fileobj)` capture and restore:
       - calibrator primitives (classification/regression as applicable),
       - preprocessing mappings (per ADR-009 export/import contract),
       - plugin identifiers + metadata used for resolution,
       - RNG seeds or deterministic flags used during fit,
       - a manifest containing `schema_version`, timestamps, and checksums.
   - Implementations may provide convenience helpers
     (`save_state_to_bytes`, `load_state_from_bytes`) for in-memory workflows.

3. **Schema version policy.**
   - `schema_version` is mandatory and must be incremented on any incompatible
     change.
   - Loading an unknown or incompatible version must fail fast with an error
     message that lists the supported version range and migration guidance.

4. **Serialization invariants.**
   - Calibrator round-trips must preserve the semantics defined in ADR-021
     (probability bounds, interval ordering, and monotonicity expectations).
   - Mapping primitives must remain JSON-safe and deterministic per ADR-009.

## Consequences

Positive:
- Deterministic, portable artifacts for explainer state and calibrated outputs.
- Clear migration points for future format changes.
- Enables reproducible parity fixtures and benchmarking workflows.

Negative / Risks:
- Additional maintenance for schema evolution and compatibility testing.
- Requires careful handling of third-party calibrator data that may not be
  JSON-safe by default.

## Adoption & Migration

- v0.11.x: introduce the calibrator `to_primitive`/`from_primitive` contract
  and explainer `save_state`/`load_state` API with round-trip tests.
- v0.11.x: document mapping export/import persistence alongside the new state
  persistence API, aligning with ADR-009 guidance.

## Open Questions

- Should we provide an official on-disk artifact format (tarball + manifest) or
  keep the API generic and allow integrators to choose storage backends?
- How should third-party calibrators declare or extend schema versions while
  still participating in the core manifest?
