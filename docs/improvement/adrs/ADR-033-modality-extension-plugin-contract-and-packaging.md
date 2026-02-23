> **Status note (2026-02-23):** Accepted ADR.

# ADR-033 - Modality Extension Plugin Contract and Packaging Strategy

Status: Accepted
Date: 2026-02-23
Deciders: Core maintainers
Reviewers: Core maintainers
Supersedes: None
Superseded-by: None

## Context

CE already has mature plugin trust/discovery mechanics and mode-aware plugin architecture for explanations, intervals, and plots. However, CE lacks an explicit modality-level contract and compatibility signal for external modality packages (image/audio/text/timeseries/multimodal). This creates three architecture risks:

1. Core bloat risk if modality-specific runtime code is added directly to `calibrated_explanations`.
2. Plugin compatibility ambiguity when third-party packages evolve independently.
3. Resolver ambiguity when multiple plugins support different modalities but share the same task/mode.

The project therefore needs a modality-aware extension contract and rollout strategy that preserves CE-first core behavior, trust policy, and API stability.

Related ADRs:

1. ADR-001 (core decomposition boundaries)
2. ADR-006 (plugin trust model)
3. ADR-010 (core vs optional distribution split)
4. ADR-011 (deprecation/migration policy)
5. ADR-013 (interval plugin strategy)
6. ADR-014 (plot plugin strategy)
7. ADR-015 (explanation plugin architecture)
8. ADR-020 (legacy user API stability)
9. ADR-026 (explanation plugin semantics)
10. ADR-030 (test quality and enforcement)

## Decision

Adopt a staged architecture for modality extensions split across `v0.11.0` and `v0.11.1`:

1. Extend plugin metadata with:
   1. `plugin_api_version` (default `"1.0"` for backward compatibility).
   2. `data_modalities` (default `("tabular",)` for backward compatibility, strictly enforced for legacy plugins).
   3. `plugin_api_version` format is numeric semver-like `MAJOR.MINOR` or `MAJOR.MINOR.PATCH`.
   4. Compatibility policy is major-hard/minor-soft:
      1. reject major mismatch;
      2. accept higher minor/patch with `UserWarning` + governance log to expose forward-compatibility risk.
   5. `data_modalities` is normalized to lowercase and validated against:
      1. canonical modalities: `tabular`, `image`, `audio`, `text`, `timeseries`, `multimodal`;
      2. aliases: `vision -> image`, `time_series -> timeseries`, `multi-modal -> multimodal`;
      3. custom extension namespace: `x-<vendor>-<name>`.
2. Add plugin API compatibility checks in registry/discovery paths using the major-hard/minor-soft policy.
3. Add modality-aware plugin selection helper in registry:
   1. Selection order: trust -> kind -> modality -> mode/task -> supports(model) -> priority.
   2. `priority` defaults to `0` and sorts descending.
   3. If multiple plugins remain tied after priority, raise `ValidationError` requiring explicit plugin identifier selection.
4. Keep CE core dependency-light and add thin shim modules for modality namespaces:
   1. `calibrated_explanations.vision`
   2. `calibrated_explanations.audio`
   3. These shims will raise a custom `MissingExtensionError` (inheriting from CE's base exception and `ImportError`) if the external package is missing.
5. Update the CE CLI to support `--modality` filtering.
6. Keep legacy behavior additive-only in first release with a version-pinned shim/deprecation timeline:
   1. `v0.11.1`: emit `DeprecationWarning` and migration guidance.
   2. `v0.12.0`: remove legacy shim import paths unless superseded by a newer ADR.
7. Default to monorepo multi-package extension structure first, using independent versioning (e.g., Core `v0.11.0`, Vision `v0.1.0`).
8. Validate the contract in CI using:
   1. a lightweight "dummy" modality package inside `tests/`;
   2. one packaging smoke test that installs a fixture extension package and validates entry-point discovery/import behavior.
9. Promote first-party modality packages from rough draft to active release commitments no earlier than the `v0.12.0` planning gate, contingent on `v0.11.0` contract stabilization.

Release split:

1. `v0.11.0` (breaking-focused):
   1. metadata contract parser/validation and modality taxonomy enforcement;
   2. registry compatibility checks;
   3. resolver tie-break/ambiguity behavior that can break implicit selection flows.
2. `v0.11.1` (additive-focused):
   1. CLI `--modality` UX;
   2. `vision`/`audio` shim modules and deprecation warnings;
   3. packaging smoke-test gate and documentation/migration hardening.

## Rationale

1. Preserves current plugin governance and trust semantics instead of introducing a second extension system.
2. Enables external package ecosystem growth without importing heavy dependencies into core.
3. Keeps migration low-risk by defaulting missing metadata fields and avoiding immediate breaking changes.
4. Creates a clear compatibility contract that can be validated in CI and at runtime.
5. Reduces resolver ambiguity and plugin ecosystem drift with deterministic selection and a controlled modality taxonomy.
6. Aligns release content with a breaking-first / additive-follow-up cadence to reduce release risk.

## Alternatives Considered

1. No metadata extension; use existing `capabilities` only.
   1. Rejected: insufficient compatibility and modality semantics.
2. Put modality implementations directly in CE core.
   1. Rejected: conflicts with ADR-001 and ADR-010 boundary goals.
3. Immediate multi-repo package split.
   1. Rejected for initial rollout: increases operational overhead before contract hardening.
4. Immediate breaking contract (mandatory new metadata keys).
   1. Rejected: conflicts with ADR-020 stability and ADR-011 migration policy.

## Consequences

Positive:

1. External modality packages can integrate through a stable, validated contract.
2. Core package stays lightweight and focused on calibrated explanation fundamentals.
3. Registry can make deterministic, trust-aware, modality-aware plugin choices.

Negative / Risks:

1. Temporary duality (legacy + new metadata paths) increases short-term complexity.
2. Shim surfaces require strict deprecation tracking to avoid long-term drift.
3. Ecosystem adoption may lag until author docs and examples are mature.
4. Minor-version forward compatibility warnings may temporarily increase warning volume for early adopters.

## Adoption and Migration

1. Land contract additions and compatibility checks as staged behavior.
2. Add resolver helper and tests.
3. Land metadata parser/taxonomy enforcement and resolver ambiguity handling in `v0.11.0`.
4. Add shim modules with actionable install errors in `v0.11.1`.
5. Update contributor and practitioner docs in `v0.11.1`.
6. Add a packaging smoke test gate for extension install/discovery behavior in `v0.11.1`.
7. Track shim sunset per ADR-011 policy in release planning artifacts using the `v0.11.1` -> `v0.12.0` schedule.
8. Synchronize `RELEASE_PLAN_v1.md` milestones with ADR-033 rollout tasks.

## Open Questions

No blocking open questions remain for the `v0.11.0` / `v0.11.1` split scope.
