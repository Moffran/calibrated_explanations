# RELEASE_PLAN Status Appendix

> Appendix scope: status-heavy tracking only (ADR/Standard status tables, gap inventory, historical compliance notes).

## Purpose

This appendix isolates detailed status material from `docs/improvement/RELEASE_PLAN_v1.md` so the master plan remains a lightweight control document.

## Repository authority (reference)

- `kristinebergs/calibrated_explanations` is an active development mirror.
- `Moffran/calibrated_explanations` is authoritative for versions, tags, releases, PyPI publication, changelog, security advisories, and documentation.

## Update cadence

- Update this appendix at milestone boundaries.
- Do not continuously churn status rows after each PR.

## ADR/Standard status table (boundary snapshot)

| Item | Current status | Target milestone or note |
| --- | --- | --- |
| ADR-004 | Partially complete | v0.11.1+ hardening remaining |
| ADR-005 | Partially complete | Remaining payload-provenance closure tracked in milestone plans |
| ADR-006 | Partially complete | v0.11.1 trust-state + deprecation completion |
| ADR-008 | Partially complete | Domain-authoritative migration in v0.11.1+ |
| ADR-009 | Partially complete | Mapping/export hardening follow-through |
| ADR-010 | Partially complete | Core-only vs extras parity automation follow-through |
| ADR-011 | Partially complete | Legacy deprecation path closure in v0.11.1 |
| ADR-012 | Accepted with open hardening | Notebook/runtime ceiling closure in v0.11.1 |
| ADR-013 | Partially complete | Protocol/fallback strictness follow-through |
| ADR-015 | Partially complete | Invariant and bridge-surface hardening follow-through |
| ADR-020 | Accepted with open parity/documentation follow-through | v0.11.1 close-out |
| ADR-026 | Partially complete | Context immutability + telemetry metadata follow-through |
| ADR-027 | Partially complete | Observability policy/docs/examples follow-through |
| ADR-028 | Accepted with open enforcement hardening | v0.11.1 close-out |
| ADR-030 | Accepted | Zero-tolerance ratification targeted v0.11.3 |
| ADR-033 | Accepted | v0.11.1 additive rollout hardening |
| ADR-034 | Proposed→Accepted path | Promote after v0.11.2 Phase B verification |
| ADR-036 | Accepted | v0.11.2 readiness-gate/default-promotion decision |
| ADR-037 | Accepted | v0.11.2 default-path and kind-extension follow-up |
| STD-001 | Partially complete | Final shim removal in v0.11.3 |
| STD-002 | Partially complete | WrapCalibratedExplainer numpydoc closure in v0.11.3 |
| STD-003 | Completed | Monitor for regressions |
| STD-004 | Completed | Monitor for regressions |
| STD-005 | Accepted with open enforcement hardening | v0.11.1 close-out |

## Detailed gap inventory and historical notes

- The authoritative detailed execution/gap notes remain in milestone execution plans:
  - `docs/improvement/v0.11.1_plan.md`
  - `docs/improvement/v0.11.2_plan.md`
  - `docs/improvement/v0.11.3_plan.md`
- Completed items are retained in those plans for traceability and are not removed for brevity.
- Future milestone detail is preserved for continuity but is not maintained continuously outside milestone-boundary updates.


## Migrated detailed status material from RELEASE_PLAN_v1.md

## ADR status appendix (unified severity tables)

This appendix consolidates per-ADR status into a compact, consistent format.

Format rules applied here:
- ADRs with no outstanding gaps show a single-line compliance verification with review date.
- ADRs with residual work show a concise table of open gaps using the project's severity axes.

Unified severity scales (brief)

- Violation impact: 1 (informational) -> 5 (blocks ADR intent).
- Code scope: 1 (single file) -> 5 (cross-cutting).
- Unified severity = impact x scope.

---

### ADR-001 - Package and Boundary Layout

**Compliance verification (2026-02-27):** Reviewed code and RTD - no ADR-001 gaps found; ADR-001 is fully compliant. No further action required.

### ADR-002 - Exception Taxonomy and Validation Contract

**Compliance verification (2026-02-27):** Reviewed code and RTD - no ADR-002 gaps found; ADR-002 is fully compliant. No further action required.

### ADR-003 - Caching Strategy

**Compliance verification (2026-02-27):** Reviewed code and RTD - no ADR-003 gaps found; ADR-003 is fully compliant. No further action required.

### ADR-004 - Parallel Execution Framework

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Documentation naming drift (legacy facade term vs `ParallelExecutor`) | 2 | 2 | 4 | Remove remaining legacy facade naming and standardize on `ParallelExecutor` across active docs. |
| 2 | Implicit `auto` strategy enables auto-selection contrary to ADR decision | 3 | 3 | 9 | Default `strategy="auto"` allows implicit selection; recommend requiring explicit strategy or document ADR exception. |

### ADR-005 - Explanation Payload Schema

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Provenance propagation in legacy adapters | 1 | 1 | 1 | Schema and validation complete; propagation of provenance fields in legacy adapters is tracked under ADR-008. |

### ADR-006 - Plugin Trust Model

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Dual trust state (descriptor `trusted` flag + `_TRUSTED_*` sets) can diverge | 3 | 3 | 9 | Registry maintains both forms; atomicity fixes targeted for v0.11.1. |
| 2 | Accepted registrations emit no governance audit event | 2 | 3 | 6 | Deny/skip paths emit logs; accepted/trusted registrations currently lack structured audit events. |
| 3 | Legacy `_REGISTRY`/`_TRUSTED` lists lack deprecation path | 3 | 2 | 6 | Legacy list-based registry remains; plan deprecations and shims targeted for v0.11.1. |

### ADR-007 - PlotSpec Abstraction (superseded; see ADR-036/ADR-037)

**Superseded routing note (2026-03-20):** ADR-007 is superseded by ADR-036 and ADR-037. Route canonical PlotSpec contract and validation questions to ADR-036, and visualization extension/rendering governance questions to ADR-037.

### ADR-008 - Explanation Domain Model

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Domain model not authoritative source | 5 | 4 | 20 | Core workflows still operate on legacy dicts; domain objects primarily produced at serialization boundaries. |
| 2 | Legacy->domain round-trip fails for conjunctive rules | 4 | 3 | 12 | `domain_to_legacy` casts features to scalars, breaking conjunction support. |
| 3 | Structured model/calibration metadata missing | 4 | 3 | 12 | Explanation dataclass lacks dedicated calibration/model descriptor fields; implementation work remains. |
| 4 | Golden fixture parity tests missing | 3 | 2 | 6 | Add byte-level/golden fixtures for adapter regression detection. |
| 5 | `_safe_pick` silently duplicates endpoints | 3 | 2 | 6 | Interval helper duplicates endpoints instead of flagging inconsistencies. |

### ADR-009 - Input Preprocessing and Mapping Policy

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Mapping export helpers placed on wrapper not explainer | 2 | 3 | 6 | `WrapCalibratedExplainer` exposes mapping persistence; consider thin `CalibratedExplainer` adapters for discoverability. |
| 2 | Export helper does not enforce JSON-safe conversion | 3 | 2 | 6 | Defensive JSON-safe conversion required to protect third-party preprocessors. |
| 3 | Validation helper location differs from ADR text | 2 | 2 | 4 | Non-numeric detection implemented but located on wrapper; consider centralizing helper or documenting deliberate placement. |

### ADR-010 - Optional Dependency Split

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | No automated parity check between core-only and extras-installed runs | 3 | 3 | 9 | CI should compare canonical outputs between install modes to detect optional-extras regressions. |

### ADR-011 - Deprecation and Migration Policy

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Compatibility shims not consistently emitting `deprecate()` warnings | 2 | 2 | 4 | `validate_payload` and some serializer shims should call `deprecate()` to surface consistent warnings. |
| 2 | Legacy-shaped serializer outputs silent on deprecation | 3 | 2 | 6 | Visual serializer compatibility translations should emit structured deprecation warnings. |
| 3 | Legacy registry lists lack deprecation hooks | 3 | 2 | 6 | Public legacy list accessors should call `deprecate()` and be scheduled for removal. |

### ADR-012 - Documentation & Gallery Build Policy

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Notebooks not executed in docs HTML CI | 5 | 4 | 20 | Docs build currently disables notebook execution; ADR requires executed notebooks (release-gate). Add execution step or dedicated notebook CI. |
| 2 | Runtime ceiling enforcement missing (per-example timing) | 3 | 3 | 9 | No CI-level per-example timing enforcement; implement timing checks or tighter `nbsphinx` timeouts. |
| 3 | Gallery tooling decision undocumented for contributors | 2 | 2 | 4 | Document chosen gallery tool and contributor expectations. |

### ADR-013 - Interval Calibrator Plugin Strategy

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Protocol signature mismatch between protocol and reference impl | 3 | 2 | 6 | Align `RegressionIntervalCalibrator` protocol signatures with concrete `IntervalRegressor` methods or add adapters. |
| 2 | FAST wrapper location mismatch vs ADR text (doc drift) | 1 | 1 | 1 | Update ADR text or mirror implementation location. |
| 3 | FAST calibrator may be implicitly included in fallback chains | 3 | 3 | 9 | Prevent FAST-style ids being automatically used in non-fast fallback chains unless explicitly selected. |
| 4 | Protocol enforcement relies on `isinstance` only | 2 | 2 | 4 | Add deeper signature/runtime harness or integration test for third-party plugins. |

### ADR-014 - Visualization Plugin Architecture (superseded; see ADR-037)

**Superseded routing note (2026-03-20):** ADR-014 is superseded by ADR-037. Route builder/renderer governance and extension metadata requirements to ADR-037; route canonical PlotSpec semantics to ADR-036.

### ADR-015 - Explanation Plugin Integration

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Inconsistent invariant enforcement across bridges/validators | 3 | 2 | 6 | Some layers warn while others raise; align to ADR (prefer raising or document strict-mode). |
| 2 | Explainer handle exposes direct `learner` (bypass bridge) | 4 | 2 | 8 | Restrict direct learner access or document as escape hatch with warnings. |
| 3 | Task-scoped enforcement divergence | 3 | 2 | 6 | Ensure interval invariant enforcement is consistent across task types and codepaths. |

### ADR-016 - PlotSpec Separation and Schema (superseded; see ADR-036/ADR-037)

**Superseded routing note (2026-03-20):** ADR-016 is superseded by ADR-036 and ADR-037. Route new work to ADR-036 (canonical PlotSpec contract) and ADR-037 (visualization extension governance).

### ADR-020 - Legacy User API Stability

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Release checklist omits legacy API gate | 4 | 3 | 12 | Add explicit gate to release checklist to protect legacy contract parity. |
| 2 | Wrapper regression tests missing parity assertions | 4 | 3 | 12 | Add parity tests for `explain_factual`/`explore_alternatives` signatures and output normalization. |
| 3 | Contributor workflow ignores contract doc updates | 3 | 3 | 9 | Update `CONTRIBUTING.md` to require contract doc updates for API changes. |

### ADR-021 - Calibrated Interval Semantics

**Compliance verification (2026-02-27):** Reviewed code and RTD - no ADR-021 gaps found; ADR-021 is fully compliant. No further action required.

### ADR-022 - Documentation Information Architecture

*Superseded by Standard-004; see Standard-004 for status.*

### ADR-023 - Matplotlib Coverage Exemption

**Compliance verification (2026-02-27):** Reviewed code and RTD - no ADR-023 gaps found; ADR-023 is fully compliant. No further action required.

### ADR-024 - Legacy Plot Input Contracts

*Retired / maintenance reference: `docs/maintenance/legacy-plotting-reference.md`.*

### ADR-025 - Legacy Plot Rendering Semantics

*Retired / maintenance reference: `docs/maintenance/legacy-plotting-reference.md`.*

### ADR-026 - Explanation Plugin Semantics

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Explanation context exposes mutable dicts | 4 | 3 | 12 | Contexts should be frozen/immutable; enforcement planned in v0.11.0 follow-ups. |
| 2 | Telemetry omits interval dependency hints | 3 | 2 | 6 | Add `interval_dependencies` to batch telemetry. |
| 3 | Mondrian bin objects left mutable in requests | 2 | 2 | 4 | Copy/freeze bins at request construction to satisfy immutability contract. |

### ADR-027 - FAST-Based Feature Filtering

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Observability policy alignment undocumented | 2 | 2 | 4 | Document logging expectations and examples for feature filtering. |
| 2 | Feature-filter telemetry examples sparse | 2 | 2 | 4 | Add practitioner examples showing emitted metadata. |

### ADR-028 - Logging and Governance Observability

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Enforcement tooling for logger domains missing | 2 | 2 | 4 | Add lint/tests to confirm domain-based logger usage. |
| 2 | Observability examples need alignment with Standard-005 | 2 | 2 | 4 | Update docs/examples to match structured logging guidance. |

### ADR-029 - Reject Integration Strategy

_Last gap analysis: 2026-03-03_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Reject strategy expansion beyond binary conformal rejectors not yet implemented | 3 | 3 | 9 | Uncertainty-based and cost-sensitive strategies targeted for v0.11.1 Task 14. |
| 2 | Strategy lifecycle hooks and configuration surface not finalized | 2 | 2 | 4 | Defer detailed strategy config API to v0.11.2. |
| 3 | `RejectResult` public return type not yet migrated to strict `RejectResultV2`; `reject_result_v2_to_legacy()` downgrade active with no deprecation warning | 3 | 2 | 6 | ADR-011 two-minor-release rule: emit deprecation in v0.11.x; switch public return to `RejectResultV2` at v1.0.0-rc. |

### ADR-030 - Test Quality Priorities and Enforcement

_Last gap analysis: 2026-03-03_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Marker hygiene zero-tolerance ratification not yet formalized | 3 | 3 | 9 | Hybrid taxonomy still advisory on some categories; ratification in v0.11.3 Task 3. |
| 2 | Mutation testing policy documentation not published | 2 | 2 | 4 | v0.11.3; declare mutation testing optional for core modules and document. |

### ADR-031 - Calibrator Serialization and State Persistence

**Compliance verification (2026-03-03):** Delivered in v0.11.0 — versioned `to_primitive`/`from_primitive` contracts plus `WrapCalibratedExplainer` `save_state`/`load_state` present and tested. No open gaps.

### ADR-032 - Guarded Explanation Semantics

**Compliance verification (2026-03-20):** ADR-032 now scopes guarded mode to schema-compatible representative-point guarded interval candidates, hard-fail calibration-feature alignment, and audit-field semantics that no longer overclaim semantic identity. No open appendix gaps.

### ADR-033 - Modality Extension Plugin Contract and Packaging Strategy

_Last gap analysis: 2026-03-03_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | CLI `--modality` filtering not yet implemented | 3 | 3 | 9 | v0.11.1 Task 11. |
| 2 | `vision.py`/`audio.py` shims not yet present (must raise `MissingExtensionError`) | 3 | 3 | 9 | v0.11.1 Task 11; shim removal v1.0.0-rc. |
| 3 | Packaging smoke test (extension install + entry-point discovery) missing | 3 | 3 | 9 | v0.11.1 Task 13. |
| 4 | Plugin contributor contract docs not updated for `plugin_api_version`/`data_modalities` requirements | 2 | 2 | 4 | v0.11.1 Task 12. |

### ADR-034 - Centralized Configuration Management

_Last gap analysis: 2026-03-03_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Phase A remaining: allowlisted modules not fully migrated; status not yet Accepted | 4 | 3 | 12 | v0.11.1 Task 15; gate is CI allowlist reduced to zero or documented boundary. |
| 2 | Phase B: `cache/cache.py`, `parallel/parallel.py`, `_feature_filter.py`, `orchestrator.py` use direct env/pyproject reads | 4 | 3 | 12 | v0.11.2 Task 1; done when all four removed from CI allowlist. |
| 3 | `GovernanceEvent` schema not yet extended to `governance.config` lifecycle events | 3 | 2 | 6 | v0.11.1 Task 20; ADR-034 §4 + ADR-028 requirement. |
| 4 | Phase C: remaining direct env/pyproject readers (allowlist not closed) | 2 | 2 | 4 | v0.11.3. |
| 5 | Sensitive-value redaction for governance logs/exports (Open Item 1) | 3 | 2 | 6 | Deferred to v1.0.0; interim: document that no redaction exists until post-GA. |
| 6 | `export_effective()` payload schema not versioned for external consumers (Open Item 2) | 3 | 2 | 6 | v1.0.0-rc gate; must be versioned before external tooling can rely on export. |

## Standards status appendix (unified severity tables)

### Standard-001 - Nomenclature Standardization

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Double-underscore fields still mutated outside legacy | 5 | 4 | 20 | Remove direct `__` mutations from core helpers. |
| 2 | Naming guardrails lack automated enforcement | 4 | 4 | 16 | Enable Ruff/pre-commit enforcement for naming rules. |

### Standard-002 - Documentation Standardisation

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Wrapper public APIs lack full numpydoc blocks | 4 | 3 | 12 | Add full numpydoc blocks to `WrapCalibratedExplainer` and stable public surfaces. |

### Standard-003 - Test Coverage Standard

**Compliance verification (2026-02-27):** Reviewed code and RTD - no STD-003 gaps found; STD-003 is fully compliant. No further action required.

### Standard-004 - Documentation Standard (Audience Hubs)

**Compliance verification (2026-02-27):** Reviewed code and RTD - no STD-004 gaps found; STD-004 is fully compliant. No further action required.

### Standard-005 - Logging and Observability Standard

_Last gap analysis: 2026-03-03_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Enforcement tooling for domain-logger naming missing from CI | 2 | 2 | 4 | ADR-028 gap 1; v0.11.1 Task 7. Add lint/test to confirm all loggers use approved `calibrated_explanations.*` domain prefixes. |
| 2 | Observability examples not yet aligned with Standard-005 naming and structured-context format | 2 | 2 | 4 | ADR-028 gap 2; v0.11.1 Task 7. Update docs/examples to match Standard-005 guidance. |

