---
id: "ADR-031"
title: "Calibrator Serialization & Explainer State Persistence"
status: "accepted"
description: "Governing architectural decision for the calibrator save/load serialization contract and the state-persistence trust boundary. Remains active as long as this contract governs calibrator state persistence; superseded when the persistence strategy is revised."
superseded_by: []
---
> **Status note (2026-07-22):** Last edited 2026-07-22 · Archive after: Retain indefinitely as architectural record · Implementation window: v0.11.x (initial), security hardening in this revision.
>
> **Security hardening addendum (2026-07-22):** The original decision below
> allowed calibrators without a `to_primitive()` contract to fall back to
> pickling, and allowed the whole `WrapCalibratedExplainer` wrapper to be
> persisted as `wrapper.pkl`, both checksum-verified with SHA-256. That design
> was unsafe: a SHA-256 checksum recorded *inside the same artifact* proves
> only that the artifact is internally self-consistent -- it says nothing
> about who produced it. An attacker who can write to a state directory can
> replace `wrapper.pkl` (or a `python_pickle` calibrator payload) with a
> malicious pickle and recompute the checksum in the same manifest; the
> artifact remains "valid" and `pickle.loads()` executes arbitrary code
> during `load_state()`. This addendum supersedes the pickle-fallback and
> whole-wrapper-pickle parts of the original decision; see "Trust boundary
> (schema v3)" below for the corrected model. `CE-REQ-SERIAL-GOV-001` records
> the current, revised acceptance criteria.

# ADR-031: Calibrator Serialization & Explainer State Persistence

Date: 2026-02-27 (security addendum: 2026-07-22)
Deciders: Core maintainers
Reviewers: Core maintainers
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

## Trust boundary (schema v3)

A persisted state artifact is **untrusted input** unless the operator
establishes trust externally (e.g. it never left storage they control, or it
is protected by a signing/provenance system outside this library). This ADR
distinguishes three properties that earlier revisions conflated:

- **Integrity** -- detecting *accidental or uncoordinated* modification (disk
  corruption, a partial copy, an interrupted write). A SHA-256 checksum
  recorded in the manifest proves this and only this.
- **Authenticity / provenance** -- establishing *who created* the artifact.
  Nothing in the schema v3 format establishes this. A checksum computed from
  bytes stored alongside those same bytes proves nothing about origin: anyone
  who can write the payload can recompute the checksum. Provenance, if
  required, must come from a mechanism external to the artifact (e.g. a
  signature verified against a key the *loader* already trusts, delivered out
  of band -- not a key or signature stored inside the artifact itself).
- **Safe parsing** -- ensuring that *loading* an artifact's data cannot
  execute code, regardless of whether the artifact is trusted. This is what
  schema v3 adds: `load_state()` only ever parses JSON and constructs
  library-known objects from JSON-safe primitives. It never calls
  `pickle.loads`/`pickle.load`/`joblib.load`/`cloudpickle.loads` (or an
  equivalent arbitrary-object deserializer) on artifact-provided bytes -- not
  even after a successful checksum match, not even for a schema version that
  claims to be legacy-compatible, and not even when the payload's declared
  type matches an object the caller already trusts.

Consequently: **integrity checking does not imply safety, and safety does not
require trust.** Schema v3 provides safe parsing unconditionally; it does not
and cannot provide authenticity, which remains the operator's responsibility.

## Decision

1. **Versioned primitive contract for calibrators.**
   - All built-in calibrators must implement `to_primitive()` returning a
     JSON-safe `dict` containing a required `schema_version` and a
     calibrator-specific payload.
   - A corresponding `from_primitive(payload: Mapping[str, object])` must
     reconstruct the calibrator, validating the `schema_version` and raising a
     clear, documented exception on incompatibility.
   - Calibrators that do not implement this contract are **not** persisted by
     falling back to pickling. `save_state()` fails closed with a
     `SerializationError` naming the unsupported calibrator, rather than
     silently writing an executable payload.
   - Calibrator types are resolved for restoration only through a fixed,
     explicit set of trusted restorers (`venn_abers`, `interval_regressor`,
     `fast_collection` of the former two); an unrecognized `calibrator_type`
     fails closed. Restoration never dynamically imports a class named by the
     artifact.

2. **Explainer state persistence API.**
   - `WrapCalibratedExplainer.save_state(path_or_fileobj)` writes a directory
     artifact containing only JSON-safe declarative data: an
     `explainer_state.json` (wrapper/learner-identity/calibration
     configuration), an optional `calibrator_primitive.json`, an optional
     `preprocessing_mapping.json`, and a `manifest.json` (schema version,
     timestamp, artifact type, and a `files` map of filename -> SHA-256).
     **No pickled wrapper or calibrator bytes are ever written.**
   - `WrapCalibratedExplainer.load_state(path_or_fileobj, *, learner=None,
     preprocessor=None, difficulty_estimator=None, mc=None)` reconstructs the
     wrapper from that declarative data. Because a generic fitted learner (and
     an arbitrary custom preprocessor) cannot be safely reconstructed from
     bytes, the caller must supply those live, already-fitted runtime objects;
     they are validated against persisted identity/shape metadata (task,
     feature count, classes, preprocessor identity) before being used, and
     reconstruction fails clearly on any mismatch. Built-in preprocessing
     (the deterministic `BuiltinEncoder`) is the one case JSON-safe enough to
     reconstruct automatically from its persisted mapping.

3. **Schema version policy.**
   - `schema_version` is mandatory and must be incremented on any incompatible
     change. The current safe schema is **v3**.
   - Loading an unknown or incompatible version must fail fast with an error
     message that lists the supported version range and migration guidance.
   - Schema **v1/v2** persisted the whole wrapper (and, for calibrators
     without a primitive contract, arbitrary Python objects) via pickle. Per
     the trust boundary above, `load_state()` refuses v1/v2 artifacts
     **unconditionally**, before any pickle byte is read -- regardless of
     checksum validity. Migration requires opening the artifact with a
     trusted, older calibrated-explanations environment the operator already
     controls and re-saving it with the current `save_state()`; no in-library
     migration escape hatch is provided; see "Adoption & Migration".

4. **Serialization invariants.**
   - Calibrator round-trips must preserve the semantics defined in ADR-021
     (probability bounds, interval ordering, and monotonicity expectations).
   - Mapping primitives must remain JSON-safe and deterministic per ADR-009.

5. **Artifact-parsing hardening.**
   - Manifest file inventories are validated before any file's contents are
     interpreted as anything other than raw bytes for hashing: absolute
     paths, `..` traversal, duplicate entries (after path normalization),
     files outside a fixed safe-schema allow-list (e.g. a `wrapper.pkl`, even
     with a matching checksum), and on-disk files not declared in the
     manifest are all rejected. Symlinks resolving outside the artifact
     directory are rejected. JSON top-level types and the manifest's
     `schema_version`/`artifact_type` are validated before any nested field
     is read.


## Governed claims

- `CE-CAP-SERIAL-001` — Calibrator and explainer persistence use versioned, non-executable primitive state, fail fast on incompatible schema versions, and reject legacy pickle-based artifacts unconditionally; checksums detect corruption only and are never treated as authentication.

## Consequences

Positive:
- Deterministic, portable artifacts for explainer state and calibrated outputs.
- Clear migration points for future format changes.
- Enables reproducible parity fixtures and benchmarking workflows.
- Normal `load_state()` cannot be turned into an arbitrary-code-execution
  primitive by a hostile or compromised state directory, even one whose
  checksums were recomputed to match.

Negative / Risks:
- Additional maintenance for schema evolution and compatibility testing.
- Requires careful handling of third-party calibrator data that may not be
  JSON-safe by default; unsupported calibrators fail closed at save time
  instead of being silently persisted.
- Legacy (schema v1/v2) artifacts are no longer loadable by normal
  `load_state()`; operators holding such artifacts must migrate them via a
  trusted older environment (see below). This is an intentional, security-
  motivated backward-compatibility break.
- `load_state()` requires the caller to supply the learner (and, for custom
  preprocessors, the preprocessor) again at load time, since these cannot be
  safely reconstructed from persisted bytes. This is a source-level API
  change from the previous single-argument `load_state(path)` signature.

## Adoption & Migration

- v0.11.x: introduce the calibrator `to_primitive`/`from_primitive` contract
  and explainer `save_state`/`load_state` API with round-trip tests.
- v0.11.x: document mapping export/import persistence alongside the new state
  persistence API, aligning with ADR-009 guidance.
- Security hardening (this revision): move to schema v3 (JSON-safe
  declarative artifacts only), require caller-supplied runtime objects at
  load time, and reject schema v1/v2 (pickle-based) artifacts unconditionally.
  To migrate a v1/v2 artifact: open it with a trusted, older
  calibrated-explanations environment the operator already controls (the
  version that produced it, or any version predating this hardening), then
  call `save_state()` again there to produce a schema v3 artifact. Only do
  this for artifacts the operator created themselves and still trusts --
  never for a downloaded or otherwise untrusted artifact. No in-library
  "unsafe legacy load" API is provided: the risk of an easily-invoked
  arbitrary-code-execution method outweighs the convenience, and the
  documented external-environment migration path above is sufficient.

## Open Questions

- Should we provide an official on-disk artifact format (tarball + manifest) or
  keep the API generic and allow integrators to choose storage backends?
- How should third-party calibrators declare or extend schema versions while
  still participating in the core manifest? (Current answer: they must
  implement the same JSON-safe `to_primitive`/`from_primitive` contract as the
  built-ins and be added to the fixed set of trusted restorers; there is no
  artifact-driven dynamic-import extension point, by design.)
- Should a future revision add an external, opt-in signature/provenance
  mechanism (verified against a key the loader supplies, not one stored in
  the artifact) for operators who need artifact authenticity guarantees
  beyond "I control the storage"?
