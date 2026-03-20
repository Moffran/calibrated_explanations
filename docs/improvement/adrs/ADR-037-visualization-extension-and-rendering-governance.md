> **Status note (2026-03-20):** Last edited 2026-03-20 · Archive after: Retain indefinitely as architectural record · Implementation window: v0.11.1+.

# ADR-037: Visualization Extension and Rendering Governance

Status: Accepted
Date: 2026-03-20
Deciders: Core maintainers
Reviewers: TBD
Supersedes: ADR-007-visualization-abstraction-layer, ADR-014-plot-plugin-strategy, ADR-016-plot-spec-separation
Superseded-by: None
Related: ADR-036-plot-spec-canonical-contract-and-validation-boundary, ADR-006-plugin-registry-trust-model, ADR-023-matplotlib-coverage-exemption

## Context

ADR-007, ADR-014, and ADR-016 collectively described builder/renderer behavior, extension metadata, and default-path posture, but with overlapping and sometimes weak language. That overlap made governance non-deterministic and difficult to enforce.

This ADR defines strict governance for visualization extension and rendering boundaries. PlotSpec semantic contract authority remains in ADR-036.

## Decision

### 1. Scope

- This ADR governs visualization extension and rendering governance only.
- This ADR **MUST NOT** redefine PlotSpec semantic contract rules established by ADR-036.

### 2. Builder contract

- Builders are components that derive plotting intent from explanation/domain payloads and construct canonical PlotSpec artifacts.
- Builders **MUST** emit canonical PlotSpec objects only.
- Builders **MUST NOT** emit raw backend payloads as builder outputs.

### 3. Renderer contract

- Renderers are components that realize visual output from canonical semantic input.
- Renderers **MUST** consume validated canonical PlotSpec only.
- Renderers **MUST NOT** accept unvalidated canonical objects or boundary dict/JSON artifacts as renderer contract input.
- Compatibility translation **MUST** occur before renderer boundary entry.

### 4. Deterministic extension governance metadata

Visualization extension descriptors/registrations **MUST** provide minimal but strict metadata sufficient for deterministic governance and resolution. At minimum, metadata **MUST** include:

- stable extension identifier,
- extension type (builder and/or renderer),
- supported semantic plot kinds,
- supported modes,
- capability/version markers required for deterministic resolver decisions,
- trust/provenance information required by governance policy.

Metadata **MUST NOT** be so thin that resolver outcomes depend on implicit defaults or non-auditable side effects.

### 5. Default and legacy behavior (v0.11.1)

- Legacy plotting remains the default path in v0.11.1.
- PlotSpec-driven rendering remains opt-in/non-default in v0.11.1.
- This ADR does not define a v0.11.1 hard gate for default promotion.

### 6. Plot kind extension policy (current state)

- Plot kinds are core-defined only in v0.11.1.
- Runtime plot-kind extension is not permitted by this ADR.
- Registries/resolvers **MUST NOT** admit runtime kind registration in the current state.

### 7. Out-of-scope and deferred policy

- A later ADR or amendment **MAY** introduce runtime kind extension only through a strict registry contract with explicit governance constraints.
- Runtime kind extension is explicitly out of scope for this ADR and for v0.11.1.

## Non-goals

- This ADR does not redefine canonical PlotSpec fields or semantic validation rules (ADR-036 scope).
- This ADR does not prescribe pixel-level parity requirements.
- This ADR does not serve as a release plan.

## Consequences

### Positive

- Builder/renderer boundaries become testable and reviewable.
- Extension resolution becomes deterministic under strict minimal metadata.
- Removes ambiguity about runtime kind extension in the current release.

### Negative / Risks

- Existing extension descriptors that rely on implicit behavior require metadata hardening.
- Short-term flexibility is reduced where runtime extension had been informally assumed.

## Adoption & Migration

- ADR-007, ADR-014, and ADR-016 are superseded by ADR-036 and ADR-037.
- ADR-023 remains separate and unchanged.
- In v0.11.1, legacy plotting remains default; PlotSpec remains opt-in.
- A v0.11.2 release-planning follow-up will revisit default-path promotion, tighten the readiness gate, and update PlotSpec defaults only when that gate is defined and met.

## Future considerations (narrow)

- If runtime kind extension is proposed later, it requires a dedicated ADR/amendment defining strict registry validation, governance metadata requirements, and deterministic conflict resolution.
