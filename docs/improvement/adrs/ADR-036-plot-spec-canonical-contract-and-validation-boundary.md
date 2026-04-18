> **Status note (2026-03-20):** Last edited 2026-03-20 · Archive after: Retain indefinitely as architectural record · Implementation window: v0.11.1+.

# ADR-036: PlotSpec Canonical Contract and Validation Boundary

Status: Accepted
Date: 2026-03-20
Deciders: Core maintainers
Reviewers: TBD
Supersedes: ADR-007-visualization-abstraction-layer, ADR-016-plot-spec-separation
Superseded-by: None
Related: ADR-037-visualization-extension-and-rendering-governance, ADR-023-matplotlib-coverage-exemption

## Context

ADR-007 and ADR-016 introduced and refined PlotSpec, but left overlapping authority across canonical representation, validation responsibilities, compatibility logic, and renderer behavior. The overlap created ambiguous ownership and allowed drift between semantic contract and rendering concerns.

This ADR replaces that fragmented surface with one enforceable semantic contract for PlotSpec itself. Rendering governance is intentionally separated into ADR-037.

## Decision

### 1. PlotSpec definition and scope

- PlotSpec is the canonical semantic intermediate representation (IR) for PlotSpec-enabled plotting paths.
- This ADR governs PlotSpec semantics, canonical representation, and validation boundaries.
- This ADR does **not** govern renderer implementation details; renderer governance belongs to ADR-037.

### 2. Canonical in-memory representation

- The canonical in-memory PlotSpec representation **MUST** be a dataclass-based model.
- Canonical PlotSpec objects **MUST** be typed dataclass instances from the maintained PlotSpec model surface.
- Dict/JSON payloads **MUST NOT** be treated as canonical PlotSpec in memory.

### 3. Boundary serialization is non-canonical

- Dict/JSON forms are boundary serialization artifacts for import/export, persistence, transport, or compatibility translation.
- Dict/JSON forms **MAY** exist at boundaries, but they **MUST** be translated into canonical dataclass PlotSpec before canonical validation and before renderer consumption.
- Canonical contract reviews and conformance checks **MUST NOT** accept raw dict/JSON as equivalent to canonical objects.

### 4. Builder output contract

- Plot builders **MUST** return canonical PlotSpec objects only.
- Builders **MUST NOT** return renderer-specific payloads, mixed canonical/legacy envelopes, or raw dict substitutes as final builder output.

### 5. Validation boundary

- Canonical PlotSpec validation **MUST** execute after builder output and before renderer invocation.
- Validation **MUST** enforce:
  - required semantic fields,
  - type/shape constraints for canonical dataclass fields,
  - semantic consistency for the declared plot kind and mode,
  - prohibition of backend-specific formatting leakage.
- A renderer **MUST NOT** be responsible for establishing canonical validity.

### 6. Forbidden backend leakage

- Canonical PlotSpec **MUST NOT** include renderer-owned formatting fields (backend-specific color codes, figure engine directives, matplotlib/plotly-specific knobs, pixel-layout directives, or other backend-tied rendering controls).
- Canonical PlotSpec **MAY** include semantic roles/tokens (for example, semantic style roles) that are backend-neutral.
- Mapping semantic roles/tokens to backend-specific output is renderer behavior and is outside canonical semantics.

### 7. Compatibility boundary rules

- Legacy payload compatibility translation **MUST** live only in explicit serializer/translator boundaries.
- Compatibility logic **MUST NOT** be embedded in canonical PlotSpec dataclasses, canonical validators, or renderer contracts.
- Compatibility translation **MUST** complete before renderer boundary entry.

### 8. Minimum semantic obligations

Each canonical PlotSpec instance **MUST** contain the minimum semantic identity required for deterministic interpretation:

- semantic plot kind,
- task mode,
- canonical data payload required by that kind,
- stable semantic identity metadata required for downstream auditing.

Additional optional fields **MAY** be defined, but required semantics **MUST** remain backend-agnostic and auditable.

### 9. Semantic contract vs rendering behavior

- Canonical PlotSpec defines semantic meaning and data intent.
- Renderers define visual realization of that semantic intent.
- Pixel parity with legacy plotting is **not** a PlotSpec semantic requirement.
- Pixel parity obligations, where required, remain in legacy-renderer territory.

### 10. Current default-path posture (v0.11.1)

- Legacy plotting remains the default plotting path in v0.11.1.
- PlotSpec path remains opt-in and non-default in v0.11.1.
- This ADR does not define a hard default-promotion gate for v0.11.1.

## Non-goals

- This ADR does not define renderer/plugin registry behavior (ADR-037 scope).
- This ADR does not define release sequencing or implementation journaling.
- This ADR does not authorize runtime plot-kind extension.

## Consequences

### Positive

- Establishes one enforceable canonical PlotSpec contract.
- Prevents semantic drift caused by renderer-driven schema expansion.
- Makes validation and compatibility boundaries auditable in code review.

### Negative / Risks

- Existing code paths that treated dicts as equivalent to canonical PlotSpec require strict boundary translation.
- Some convenience shortcuts are disallowed where they bypass canonical validation boundaries.

## Adoption & Migration

- ADR-007 and ADR-016 are superseded by this ADR and ADR-037.
- In v0.11.1, legacy remains default and PlotSpec remains opt-in.
- A v0.11.2 release-planning follow-up will revisit PlotSpec default-path promotion and define a stricter readiness gate at that time.

## Future considerations (narrow)

- A later ADR or amendment **MAY** refine canonical validation strictness and semantic field requirements as more kinds stabilize.
- Any future change that affects default-path promotion gating belongs to release planning and a dedicated ADR update, not implicit interpretation.
