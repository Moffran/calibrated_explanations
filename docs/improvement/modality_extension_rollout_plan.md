> **Status note (2026-02-23):** Plan updated to reflect Option E: split ADR-033 rollout across `v0.11.0` (breaking contract/resolver changes) and `v0.11.1` (additive UX/docs/hardening). This preserves CE core stability while sequencing risk.

# CE Modality Extension Rollout Plan (Right-Sized)

## 1. Why This Iteration Exists

This iteration is a scope and ambition check before implementation. The goal is to avoid two failure modes:

1. Under-scoping: shipping only partial plugin metadata changes that do not unlock real external modality integration.
2. Over-scoping: trying to ship vision/audio runtimes, packaging, CI expansion, and migration in one release cycle.

This plan defines a sufficient CE-core adaptation path from start to finish while deferring non-core package complexity.

## 2. Scope Principles (and ADR Anchors)

1. Keep CE core small and stable.
Motivation: avoid dependency bloat and preserve import/runtime characteristics.
ADRs: `ADR-001`, `ADR-010`, `ADR-020`.
2. Reuse the existing trust-governed plugin model.
Motivation: do not fork a second extension mechanism.
ADRs: `ADR-006`, `ADR-013`, `ADR-037`, `ADR-015`, `ADR-026`.
3. Use additive, backward-compatible contract evolution first.
Motivation: preserve existing plugin ecosystem and user APIs.
ADRs: `ADR-011`, `ADR-020`.
4. Ship in phases with measurable exit criteria.
Motivation: reduce release risk and avoid architecture drift.

## 3. Clarifications

### 3.1 Must Have in CE Core

1. Metadata contract extension for modality-awareness and plugin API compatibility.
2. Registry compatibility checks and modality-aware resolver.
3. Thin core shims (`vision.py`, `audio.py`) with actionable install guidance.
4. Documentation and migration guidance aligned with existing ADRs.

### 3.2 Explicitly Deferred from CE Core

1. In-core image/audio explainers.
2. Heavy dependency onboarding in root `project.dependencies`.
3. Full multi-package CI matrix in the same implementation phase as contract changes.
4. Hard migration/removal of legacy behaviors in the first release.

## 4. Phase-by-Phase Implementation Plan (CE Core)

## Phase 0 - Architecture Gate and Scope Freeze

### CE updates

1. Add a focused architecture plan (this document) under `docs/improvement/`.
2. Keep the accepted ADR (`ADR-033`) and this plan synchronized as implementation details evolve.

### Motivation for updates

1. Locks intent and boundaries before code churn.
2. Creates a reviewable artifact for maintainers before touching release milestones.

### Exit criteria

1. Plan and accepted ADR are aligned for implementation readiness.
2. Any remaining follow-up questions are explicitly tracked (not hidden inside code PRs).

---

## Phase 1 - Plugin Metadata Contract v1.1 (Contract Enforcement)

### Release target

`v0.11.0` (breaking for non-conforming external plugin metadata; backward compatible for legacy plugins via defaults).

### CE updates

1. Extend `validate_plugin_meta()` in `src/calibrated_explanations/plugins/base.py`:
   1. `plugin_api_version` default `"1.0"` when absent.
   2. `data_modalities` default `("tabular",)` when absent. This strictly enforces tabular-first behavior for all legacy plugins.
2. Validate both fields:
   1. `plugin_api_version` must match `MAJOR.MINOR` or `MAJOR.MINOR.PATCH` with numeric components only.
   2. Runtime compatibility policy:
      1. Reject major-version mismatch.
      2. Accept higher minor/patch versions with `UserWarning` + governance log entry so forward-compatibility risk is explicit.
   3. `data_modalities` must be a non-empty sequence of non-empty strings, normalized to lowercase.
   4. Enforce canonical core modalities with aliases and extension namespace:
      1. Canonical: `tabular`, `image`, `audio`, `text`, `timeseries`, `multimodal`.
      2. Aliases: `vision -> image`, `time_series -> timeseries`, `multi-modal -> multimodal`.
      3. Allow custom modalities only through `x-<vendor>-<name>` namespace.
3. Keep all existing required keys and trust/checksum behavior unchanged.

### Motivation for updates

1. Adds modality semantics without breaking existing plugins.
2. Introduces explicit compatibility signaling without forcing immediate ecosystem migration.

### Exit criteria

1. Legacy minimal metadata still passes.
2. Invalid new fields fail with clear `ValidationError`.
3. Semver parser and compatibility warnings are covered with deterministic tests.
4. Modality normalization/alias/custom-namespace behavior is covered with deterministic tests.
5. Unit tests expanded in `tests/plugins/test_base_validation.py`.

### ADR references

`ADR-006`, `ADR-011`, `ADR-020`, `ADR-026`.

---

## Phase 2 - Registry Compatibility and Modality Resolver

### Release target

`v0.11.0` (breaking where prior implicit resolver ambiguity is now rejected deterministically).

### CE updates

1. Add plugin API compatibility check in `src/calibrated_explanations/plugins/registry.py`:
   1. Core-supported major: `1`.
   2. Reject incompatible major versions during registration/discovery.
2. Add modality-aware resolver helper:
   1. `find_plugin_for(modality, *, kind="explanation", mode=None, task=None, model=None, trusted_only=True)`.
3. Enforce filtering order:
   1. Trust.
   2. Kind.
   3. Modality (`data_modalities`).
   4. Mode/task metadata.
   5. `supports(model)` safety check.
   6. `priority` metadata (descending, default `0`).
   7. If multiple plugins remain tied after priority, raise `ValidationError` requiring explicit plugin identifier selection.
4. Define CLI-facing resolver/filter semantics for modality selection; implementation lands in `v0.11.1`.

### Motivation for updates

1. Allows external modality plugins to be discovered and selected safely.
2. Reuses trust and descriptor infrastructure already in CE.
3. Avoids introducing a parallel resolver stack.

### Exit criteria

1. New resolver is covered by unit tests.
2. Existing resolver APIs remain backward-compatible.
3. Incompatible plugin API major is rejected deterministically.
4. Ambiguous plugin selection produces deterministic `ValidationError` guidance.

### ADR references

`ADR-006`, `ADR-013`, `ADR-037`, `ADR-015`, `ADR-026`.

---

## Phase 3 - Core Shim Surface for Modality Packages

### Release target

`v0.11.1` (additive UX and migration support).

### CE updates

1. Add:
   1. `src/calibrated_explanations/vision.py`
   2. `src/calibrated_explanations/audio.py`
2. Implement lazy import/registration helpers.
3. Raise actionable `MissingExtensionError` (inheriting from CE's base exception and `ImportError`) with installation hints.
4. If legacy import paths exist, keep shim-based compatibility with a version-pinned schedule:
   1. `v0.11.1`: emit `DeprecationWarning` with migration guidance.
   2. `v0.12.0` or `v1.0.0-rc`: remove legacy shim import paths unless a replacement ADR supersedes this schedule.

### Motivation for updates

1. Improves developer UX without pulling heavy deps into core.
2. Provides a stable bridge while external packages mature.

### Exit criteria

1. Missing-package path gives clear install guidance.
2. Import graph remains lightweight (no heavy imports from package root).
3. Compatibility warning behavior documented with explicit `v0.11.1` -> `v0.12.0` timeline.

### ADR references

`ADR-010`, `ADR-011`, `ADR-020`.

---

## Phase 4 - Documentation and Migration Readiness

### Release target

`v0.11.1` (non-breaking documentation and migration clarity).

### CE updates

1. Update `docs/contributor/plugin-contract.md`:
   1. Add "Plugin Metadata v1.1" section.
   2. Document `plugin_api_version` and `data_modalities`.
   3. Document parser/compatibility rules and canonical modality taxonomy (including alias normalization and `x-*` namespace).
2. Update `docs/plugins.md`:
   1. Add modality-aware plugin discovery narrative.
   2. Clarify trust model remains unchanged.
   3. Document resolver tie-break/ambiguity behavior and explicit plugin identifier override guidance.
3. Update `docs/practitioner/advanced/use_plugins.md`:
   1. Add usage notes for modality plugins and shims.
4. Update `README.md`:
   1. Short core-vs-extension packaging note.
5. Update `CHANGELOG.md` with additive metadata/resolver/shim entries.

### Motivation for updates

1. Prevents contract drift between code and docs.
2. Reduces plugin-author friction and migration errors.

### Exit criteria

1. Docs describe old + new metadata behavior.
2. Migration guidance is explicit about `v0.11.1` warning and `v0.12.0` removal schedule.

### ADR references

`ADR-011`, `ADR-012`, `ADR-020`, `ADR-026`.

---

## Phase 5 - Hardening and Release-Readiness in CE

### Release target

`v0.11.1` (non-breaking test/CI/documentation hardening).

### CE updates

1. Add compatibility tests:
   1. API-major mismatch rejection.
   2. Modality filtering with trust defaults.
   3. Create a lightweight "dummy" modality package inside `tests/` that registers itself dynamically during test setup to validate the contract end-to-end without bloating CI.
   4. Add one packaging smoke test job that installs a fixture extension package and validates entry-point discovery/import behavior.
2. Implement CE CLI `--modality` filtering for listing/reporting plugins using Phase 2 resolver semantics.
3. Add checklist items in improvement/release docs and synchronize release milestone content.
4. Confirm no heavy dependency additions in core package dependencies.

### Motivation for updates

1. Locks behavior before extension package rollout.
2. Keeps CE release quality gates aligned with existing governance.

### Exit criteria

1. `make local-checks-pr` passes.
2. Targeted plugin/registry tests pass.
3. CE CLI modality filtering is covered by CLI tests.
4. No import-time dependency regressions.
5. Packaging smoke test passes for extension install/discovery flow.

### ADR references

`ADR-010`, `ADR-020`, `ADR-030`.

## 5. Rough Draft for Future External Packages (Not Detailed Plans)

These are intentionally rough to avoid over-planning before CE core is ready.

1. `calibrated_explanations_vision`
   1. Vision adapters/explainers only.
   2. Registers via CE plugin entry points.
2. `calibrated_explanations_audio`
   1. Audio adapters/explainers only.
   2. Registers via CE plugin entry points.
3. Optional `calibrated_explanations_viz`
   1. Additional plot builders/renderers beyond current defaults.

Integration models:

1. Same repository (`packages/...`) first (recommended for faster governance alignment).
2. Split repositories later if ownership/release cadence demands it.
3. **Versioning Strategy:** Independent versioning (e.g., Core is `v0.11.0`, Vision is `v0.1.0`). Rely on `plugin_api_version` to guarantee compatibility across different package versions.

## 6. Mapping Options to `RELEASE_PLAN_v1.md`

**Option E - Two-Release Split Mapping (Selected)**

1. `v0.11.0`:
   1. Phase 1 and Phase 2 shipped (contract/resolver behavior that can be API-breaking for non-conforming integrations).
   2. Include migration notes for resolver ambiguity and metadata validation changes.
2. `v0.11.1`:
   1. Phase 3, Phase 4, and Phase 5 shipped (additive shims, CLI UX, docs, and hardening).
   2. Includes packaging smoke-test gating and explicit shim deprecation timeline publication.

## 7. Decision Checklist for Maintainers

*All initial rollout blocker decisions have been resolved:*
1. **Release mapping:** Option E (`v0.11.0` for breaking contract/resolver changes; `v0.11.1` for additive rollout/hardening).
2. **Missing packages:** Raise `MissingExtensionError`.
3. **CI Testing:** Use a lightweight "dummy" modality package in `tests/`.
4. **CLI:** Update CLI to support `--modality` filtering in `v0.11.1`.
5. **Legacy plugins:** Strictly enforce `("tabular",)` default.
6. **Versioning:** Use independent versioning for external packages.
7. **`plugin_api_version`:** Enforce numeric semver format with major-hard/minor-soft compatibility checks.
8. **`data_modalities`:** Enforce canonical taxonomy + alias normalization + `x-*` extension namespace.
9. **Resolver ambiguity:** Raise deterministic `ValidationError` unless explicit plugin identifier is selected.
10. **Shim timeline:** `v0.11.1` warn, `v0.12.0` remove (unless superseded by ADR).

## 8. Related ADRs

1. `docs/improvement/adrs/ADR-001-core-decomposition-boundaries.md`
2. `docs/improvement/adrs/ADR-006-plugin-registry-trust-model.md`
3. `docs/improvement/adrs/ADR-010-core-vs-evaluation-split-and-distribution.md`
4. `docs/improvement/adrs/ADR-011-deprecation-and-migration-policy.md`
5. `docs/improvement/adrs/ADR-013-interval-calibrator-plugin-strategy.md`
6. `docs/improvement/adrs/ADR-037-visualization-extension-and-rendering-governance.md`
7. `docs/improvement/adrs/ADR-015-explanation-plugin.md`
8. `docs/improvement/adrs/ADR-020-legacy-user-api-stability.md`
9. `docs/improvement/adrs/ADR-026-explanation-plugin-semantics.md`
10. `docs/improvement/adrs/ADR-030-test-quality-priorities-and-enforcement.md`
