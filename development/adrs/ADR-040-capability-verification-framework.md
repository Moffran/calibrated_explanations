---
id: "ADR-040"
title: "Capability Verification Framework and Requirements-as-Code Governance"
status: "accepted"
description: "Governing the capability-claim, requirement, TIF, test and evidence chain used for externally visible verification."
superseded_by: []
---
> **Status note (2026-07-07):** Last edited 2026-07-07 · Archive after: Retain indefinitely as architectural record · Implementation window: v0.11.5 ratification and v1.0.0 release governance.

# ADR-040: Capability Verification Framework and Requirements-as-Code Governance

Date: 2026-07-07
Deciders: Core maintainers
Reviewers: Core maintainers
Related: ADR-002-validation-and-exception-design, ADR-005-explanation-payload-schema, ADR-011-deprecation-and-migration-policy, ADR-012-documentation-and-gallery-build-policy, ADR-013-interval-calibrator-plugin-strategy, ADR-021-calibrated-interval-semantics, ADR-030-test-quality-priorities-and-enforcement, ADR-031-calibrator-serialization-and-state-persistence, ADR-039-conditional-calibration-and-explanation-semantics, Standard-004-documentation-standard-audience-hubs, Standard-005-logging-and-observability-standard

## Context

CE exposes scientific and engineering capabilities whose public value depends on more
than importability or isolated unit tests. Prediction, explanation, conditional
calibration, guarded explanations, reject policies, visualization, narratives, and
filtering all create user-visible claims. Some claims are API-contract claims; some
are numerical-behavior claims; some are documentation-boundary or governance claims;
and some are statistical-method claims whose validity depends on assumptions such as
exchangeability, sufficient calibration data, and correct task framing.

Historically, many such statements lived in documentation, examples, release notes,
or tests without a single repository-level chain connecting the claim to the
requirement it decomposes into, the public API surface exercised, the executable
verification stimulus, and the evidence record produced by a concrete run. This made
it easy to overclaim: a smoke test could be mistaken for statistical validity, a
shape check could be mistaken for numerical correctness, or documentation prose could
be treated as verified merely because it existed.

By v0.11.5-dev, the repository already contains an implemented capability
verification structure:

- capability claims in `development/capabilities/claims/`;
- requirements in `development/capabilities/requirements/`;
- TIF specifications and executables in `development/capabilities/verification/tif/`;
- verification scenarios and helpers in `development/capabilities/verification/`;
- pytest capability-contract tests in `tests/capabilities/`;
- raw evidence records in `reports/verification/`;
- curated closure evidence in `development/capabilities/evidence/`;
- structural validation in `scripts/quality/validate_capability_chain.py`;
- raw-evidence generation in `scripts/generate_tif_evidence.py`;
- release-facing make targets `capability-chain-check` and `capability-evidence-refresh`.

The TIF architecture evidence records that the framework was applied across the CE
capability surface, not only to Mondrian conditional calibration. Active TIF areas
include factual and alternative explanations, conjunctions, filtering, guarded
explanations, Mondrian conditional calibration, narratives, prediction, binary and
multiclass classification, probabilistic regression, reject policies, and
visualization. ADR-039 extends one capability chain under this framework; it does not
own or define the framework itself.

A governing ADR is needed because this structure is now architecture and release
governance, not a local documentation convention. Without a binding decision, future
work could bypass the chain, add capability claims without requirements, encode
acceptance criteria only in tests, generate evidence without assumption boundaries,
or use private implementation details in TIF scenarios.

## Decision

### D1. Capability verification chain is canonical

Externally visible CE capability claims MUST be governed through the following chain:

```text
ADR / Standard
    -> constrains
Capability claim
    -> decomposes into
Requirement
    -> is exercised through
TIF verification interface or documented exemption
    -> is executed by
Test / verification gate / review
    -> produces
Evidence record
```

ADRs and Standards constrain the chain. They are not themselves capability claims.
Capability claims describe user-visible product or library behavior. Requirements
state scoped, testable obligations. TIF interfaces define reusable public-API
stimuli and structured observations. Tests and gates execute verification. Evidence
records preserve what was checked, against which version and configuration, and with
what result.

### D2. Canonical repository locations

The repository MUST use the following locations for capability verification material:

| Material | Location |
|---|---|
| Capability claims | `development/capabilities/claims/` |
| Requirements | `development/capabilities/requirements/` |
| TIF specifications and executables | `development/capabilities/verification/tif/` |
| Verification scenarios and helpers | `development/capabilities/verification/` |
| Pytest capability-contract tests | `tests/capabilities/` |
| Raw generated evidence | `reports/verification/` |
| Curated release or closure evidence | `development/capabilities/evidence/` |
| Structural chain validator | `scripts/quality/validate_capability_chain.py` |
| Raw evidence generator | `scripts/generate_tif_evidence.py` |

New locations for the same material MUST NOT be introduced without superseding this
ADR or explicitly amending it.

### D3. Claims are not requirements

A capability claim MUST NOT contain detailed acceptance criteria, test scenarios,
parameter-specific obligations, or implementation mechanics. A claim SHOULD normally
decompose into multiple requirements.

A one-to-one claim-to-requirement mapping is allowed only when the capability is
genuinely atomic and the requirement or claim metadata records an explicit atomic
rationale. Atomic rationale MUST NOT be used to avoid decomposing broad behavioral
claims.

### D4. Requirements are requirements-as-code records

Each requirement MUST state:

- a stable `CE-REQ-...` identifier;
- obligation type;
- claim references;
- ADR or Standard references when applicable;
- status and verification status;
- scope;
- observable behavior;
- acceptance criteria;
- verification method;
- TIF references or a documented TIF exemption;
- verification targets when executable verification exists;
- required evidence fields;
- assumption boundary.

Requirements MUST be specific enough to determine whether an implementation satisfies
them without reading test code as the only source of acceptance criteria.

### D5. TIF is the public-API verification interface layer

TIF means Test Interface Framework. It is the layer between requirements and tests.
A TIF interface observes CE behavior; it does not own the final pytest assertion.

Behavioral CE TIF scenarios MUST:

- stimulate CE through `WrapCalibratedExplainer` or a small public-API helper that
  itself uses `WrapCalibratedExplainer`;
- use the public `fit -> calibrate -> predict/explain` lifecycle unless the
  requirement specifically concerns pre-fit or pre-calibration error behavior;
- return structured observations as a dataclass or dictionary;
- avoid private members and internal implementation modules;
- avoid direct construction of explanation objects such as `FactualExplanation(...)`
  or `AlternativeExplanation(...)`;
- avoid final pytest assertions except local sanity checks needed to prevent invalid
  observations.

Tests call TIF scenarios and assert observations against the requirement acceptance
criteria. TIF scenarios MUST NOT encode acceptance criteria as hidden-only logic.

### D6. TIF exemptions are explicit and narrow

A requirement that cannot or should not be exercised through a behavioral TIF MUST
carry a documented TIF exemption. Recognized exemption classes include documentation
boundary, schema validation, repository policy, metadata linkage, and static
importability checks.

A TIF exemption MUST state why a public-API behavioral TIF is inappropriate. It MUST
NOT be used merely because writing a TIF is inconvenient.

Behavioral-contract requirements MUST NOT use TIF exemptions unless this ADR is
amended or superseded.

### D7. Evidence records have defined semantics

Raw evidence records MUST be generated from concrete verification execution and MUST
include enough metadata to reconstruct the run. At minimum, raw evidence SHOULD
record:

- evidence identifier;
- claim identifiers;
- requirement identifiers;
- TIF identifiers when applicable;
- verification type;
- result;
- timestamp;
- commit SHA;
- package version;
- scenario observations;
- acceptance outcomes.

Curated evidence summaries MAY aggregate raw evidence or record non-TIF reviews, but
MUST distinguish themselves from raw executable evidence.

Tests are not evidence. Passing tests may produce evidence, but the test file itself
is not the durable evidence record.

### D8. Verification strength and assumption boundaries are mandatory

Every claim, requirement, and evidence summary MUST distinguish what is verified from
what is assumed or out of scope.

Verification types include, but are not limited to:

- API contract;
- behavioral contract;
- numerical behavior;
- statistical-method alignment;
- empirical smoke;
- visualization behavior or visualization structure;
- documentation boundary;
- repository policy;
- metadata linkage;
- serialization contract.

Evidence MUST NOT be used to overclaim. In particular:

- empirical tests do not prove finite-sample conformal validity by themselves;
- output-shape tests do not prove calibration quality;
- visualization smoke tests do not prove semantic correctness of the visual story;
- narrative generation tests do not prove explanation quality;
- documentation-boundary review does not prove users understood the documentation;
- API-contract verification does not prove causal correctness, domain feasibility,
  or optimality of user-selected categories.

Statistical-method claims MUST state their assumptions explicitly, especially
exchangeability, calibration-set construction, category sample-size constraints,
task-type constraints, and empirical-versus-theoretical boundaries.

### D9. Release-gate use

`make capability-chain-check` is the non-mutating structural gate for capability
verification metadata. It SHOULD run after changes to claims, requirements, TIF
specifications, TIF executables, capability tests, or evidence files. It MAY run on
every PR because it does not regenerate evidence.

`make capability-evidence-refresh` regenerates raw evidence by executing active TIF
scenarios and checking evidence freshness at `HEAD`. It SHOULD run when TIF behavior,
acceptance logic, public capability behavior, or release-closure evidence changes.
It is a release-boundary gate unless explicitly waived in the release plan with a
reason and follow-up owner.

### D10. Relationship to existing ADRs and Standards

This ADR governs the verification framework. It does not replace capability-specific
ADRs. Capability-specific ADRs, such as ADR-039 for conditional calibration, define
public behavior and implementation contracts for their own domains. ADR-040 defines
how such behavior is expressed as claims, requirements, TIFs, tests, and evidence.

When a capability-specific ADR and ADR-040 interact, both apply:

- the capability-specific ADR defines what behavior is required;
- ADR-040 defines how the capability claim and verification evidence are structured;
- ADR-030 continues to govern test quality and anti-pattern enforcement;
- ADR-012 and Standard-004 continue to govern documentation and audience-facing
  presentation;
- Standard-005 and ADR-028 continue to govern logs and observability when evidence
  involves runtime governance events.

## Alternatives considered

### A1. Keep the framework only in CONTRIBUTOR_INSTRUCTIONS and READMEs

Rejected. Contributor instructions and READMEs are useful operational guidance, but
the framework now controls cross-cutting release confidence and evidence semantics.
Keeping it only in instructions would make it too easy to bypass or reinterpret.

### A2. Treat pytest tests as the sole verification authority

Rejected. Tests are execution mechanisms, not requirements and not evidence. Pytest
alone does not preserve claim scope, assumption boundaries, verification strength, or
release evidence metadata.

### A3. Make each capability-specific ADR define its own verification structure

Rejected. This would duplicate rules across ADRs, create inconsistent evidence
semantics, and make cross-capability release gates brittle. Capability-specific ADRs
should define behavior; ADR-040 defines the verification architecture.

### A4. Generate all claims, requirements, and evidence from code annotations

Rejected for v0.11.5 and v1.0.0. Full generation may become useful later, but the
current framework deliberately keeps claims and requirements reviewable as human
architecture artifacts. Generated-only records would risk hiding assumption
boundaries and acceptance criteria in tooling.

### A5. Limit the framework to Mondrian conditional calibration

Rejected. The implemented TIF registry and evidence records already cover multiple
CE capabilities. Mondrian is one chain under the framework, extended by ADR-039 in
v0.11.5, not the owner or boundary of the framework.

## Consequences

### Positive

- Public capability claims become traceable to scoped requirements and evidence.
- Release confidence can distinguish API contracts, numerical behavior,
  documentation boundaries, empirical smoke checks, and statistical assumptions.
- TIF scenarios provide reusable public-API stimuli instead of duplicating setup
  logic across tests.
- Evidence records become durable and auditable rather than implicit in transient
  test output.
- The framework reduces overclaiming by requiring assumption boundaries.
- Future CE and CEE alignment can reference a shared verification discipline.

### Negative / cost

- New externally visible capabilities require more metadata than ordinary code
  changes.
- Contributors must understand the difference between claims, requirements, TIFs,
  tests, and evidence.
- Evidence refresh can be slower than ordinary structural checks and therefore needs
  explicit release-gate discipline.
- Poorly scoped requirements can still create bureaucracy without improving
  confidence; maintainers must reject vague requirements and duplicate claims.

### Neutral / boundary

- ADR-040 does not make every existing requirement perfect. It ratifies the framework
  and defines the rules for maintaining and extending it.
- ADR-040 does not prove statistical validity of CE methods. It governs how such
  claims are scoped and verified.
- ADR-040 does not require every unit test to map to a capability requirement. It
  applies to externally visible capability claims and capability-facing verification.

## Adoption and migration

ADR-040 is accepted as a ratification of the already implemented capability
verification framework.

Immediate v0.11.5 adoption:

1. Keep the active TIF registry and framework documentation in
   `development/capabilities/verification/README.md` and
   `development/capabilities/verification/tif/README.md` aligned with this ADR.
2. Keep `CONTRIBUTOR_INSTRUCTIONS.md` aligned with this ADR as the agent-facing
   operational summary.
3. Use `make capability-chain-check` after any claim, requirement, TIF, capability
   test, or evidence change.
4. Use `make capability-evidence-refresh` at release closure and whenever TIF
   behavior or acceptance logic changes.
5. Treat the ADR-039 Mondrian capability-chain extension as one v0.11.5 application
   of the framework, not as the framework itself.

## Governed claims

- `CE-CAP-MOND-001`

Future adoption:

1. Add a short ADR-040 summary to the release roadmap at the next release-plan
   grooming point.
2. Add or update curated release evidence summaries when a milestone closes.
3. Consider adding JSON/YAML schemas for claim, requirement, TIF, and evidence
   records if structural drift appears despite the validator.
4. Align CEE capability verification terminology with this ADR where CEE inherits or
   wraps OSS CE capability claims.

## Open questions

1. Should capability claim and requirement files receive formal JSON/YAML schemas,
   or is the current validator sufficient through v1.0.0?
2. Which subset of `capability-evidence-refresh` should become mandatory in CI, if
   any, rather than remaining a release-boundary or explicit maintainer gate?
3. Should curated evidence summaries be required for every release, or only for
   milestones that change capability behavior or verification architecture?
4. Should user-facing documentation expose the capability verification framework, or
   should it remain contributor/release-governance documentation until v1.0.0 GA?
5. How should CEE reference OSS CE evidence when CEE wraps CE behavior without
   changing the underlying mathematical implementation?
