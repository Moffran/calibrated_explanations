# RELEASE_PLAN Status Appendix

> Status note (2026-05-12): reviewed at the v0.11.2 milestone boundary; summary rows below are synchronized to v0.11.2 deliverables and currently open follow-through only.

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
| ADR-004 | Partially complete | Only explicit `strategy="auto"` policy closure remains (v1.0.0-rc) |
| ADR-005 | Partially complete | Remaining legacy-adapter provenance follows ADR-008 domain-authoritative migration (v1.0.0-rc) |
| ADR-006 | Partially complete | Accepted-registration audit events and final legacy-list removal remain (v0.11.3) |
| ADR-008 | Partially complete | Domain-authoritative migration and round-trip hardening remain (v1.0.0-rc) |
| ADR-009 | Partially complete | JSON-safe export and helper-placement docs in v0.11.3; wrapper/core surface decision in v1.0.0-rc |
| ADR-010 | Partially complete | Core-only vs extras parity automation follow-through (v0.11.3) |
| ADR-011 | Partially complete | Remaining serializer/list-path deprecation closure (v0.11.3) |
| ADR-012 | Accepted with open hardening | Notebook execution/runtime ceilings remain v1.0.0-rc; gallery-tooling decision docs in v0.11.3 |
| ADR-013 | Partially complete | Protocol/fallback strictness follow-through (v0.11.3) |
| ADR-015 | Partially complete | Invariant consistency in v0.11.3; direct learner-handle posture in v1.0.0-rc |
| ADR-020 | Accepted with open parity/documentation follow-through | v0.11.3 close-out |
| ADR-026 | Partially complete | Context immutability remains (v1.0.0-rc); telemetry quick wins closed in v0.11.2 |
| ADR-027 | Partially complete | Observability policy/docs/examples follow-through (v0.11.3) |
| ADR-028 | Accepted with reopened v0.11.3 correction | Warning-first fallback/degraded-state visibility conflicts with log-first observability posture; Task 8 in `v0.11.3_plan.md` owns classification and migration |
| ADR-030 | Accepted | Zero-tolerance ratification targeted v0.11.3 |
| ADR-033 | Accepted with open removal follow-through | `data_modalities` enforcement closes in v0.11.3; shim removal remains v1.0.0-rc |
| ADR-034 | Accepted with deferred v1.0 open items | Runtime conformance closure complete in v0.11.2; remaining work is redaction + export schema versioning |
| ADR-035 | Accepted with rollout follow-through | Advisory-to-required branch-protection promotion remains platform-governed; re-evaluate in v0.11.3 |
| ADR-036 | Accepted | v0.11.3 Task 6 promoted PlotSpec as the default user-facing plotting path after v0.11.2 mending evidence; monitor canonical dataclass validation and legacy fallback behavior |
| ADR-037 | Accepted | v0.11.3 Task 6 promoted the governed built-in PlotSpec default while preserving the runtime plot-kind extension prohibition and explicit legacy opt-out |
| STD-001 | Accepted with bounded compatibility bridges | Runtime dunder regression guard is now CI-blocking; bridge removals remain targeted for v0.11.3 |
| STD-002 | Completed | WrapCalibratedExplainer numpydoc gap closed in v0.11.3 Task 2; coverage 96.73%, zero pydocstyle violations (2026-06-02) |
| STD-003 | Completed | Monitor for regressions |
| STD-004 | Completed | Monitor for regressions |
| STD-005 | Accepted with reopened v0.11.3 correction | Task 8 must align fallback/degraded-state visibility with `WARNING` log-first behavior and justify remaining `UserWarning` paths |

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

_Last gap analysis: 2026-04-22_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | ~~Documentation naming drift (legacy facade term vs `ParallelExecutor`)~~ | 2 | 2 | 4 | **Closed v0.11.2:** Legacy "facade" term removed from ADR-001 and RELEASE_PLAN_v1 changelog entry; `ParallelExecutor` naming is now consistent in active docs. |
| 2 | Implicit `auto` strategy enables auto-selection contrary to ADR decision | 3 | 3 | 9 | Default `strategy="auto"` allows implicit selection. Target milestone: v1.0.0-rc. Blocker: coordinated strategy API change + user-facing migration guide; architecture-heavy and above quick-win threshold. |

### ADR-005 - Explanation Payload Schema

_Last gap analysis: 2026-04-22_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Provenance propagation in legacy adapters | 1 | 1 | 1 | Schema and validation complete; propagation of provenance fields in legacy adapters is tracked under ADR-008. Target milestone: v1.0.0-rc (follows ADR-008 domain-model adoption). |

### ADR-006 - Plugin Trust Model

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Dual trust state (descriptor `trusted` flag + `_TRUSTED_*` sets) can diverge | 3 | 3 | 9 | Registry maintains both forms; atomicity fixes delivered in v0.11.1 (Task 2). Target milestone: monitor for regressions; re-evaluate at v0.11.3. |
| 2 | Accepted registrations emit no governance audit event | 2 | 3 | 6 | Deny/skip paths emit logs; accepted/trusted registrations currently lack structured audit events. Target milestone: v0.11.3. |
| 3 | Legacy `_REGISTRY`/`_TRUSTED` lists lack deprecation path | 3 | 2 | 6 | Deprecation warnings added in v0.11.1 (Task 1); full list-path removal targeted v0.11.3. Target milestone: v0.11.3. |

### ADR-007 - PlotSpec Abstraction (superseded; see ADR-036/ADR-037)

**Superseded routing note (2026-03-20):** ADR-007 is superseded by ADR-036 and ADR-037. Route canonical PlotSpec contract and validation questions to ADR-036, and visualization extension/rendering governance questions to ADR-037.

### ADR-008 - Explanation Domain Model

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Domain model not authoritative source | 5 | 4 | 20 | Core workflows still operate on legacy dicts; domain objects primarily produced at serialization boundaries. Target milestone: v1.0.0-rc (major refactor; architecture-heavy). |
| 2 | Legacy->domain round-trip fails for conjunctive rules | 4 | 3 | 12 | `domain_to_legacy` casts features to scalars, breaking conjunction support. Target milestone: v1.0.0-rc (follows domain-model authority work). |
| 3 | Structured model/calibration metadata missing | 4 | 3 | 12 | Explanation dataclass lacks dedicated calibration/model descriptor fields. Target milestone: v1.0.0-rc. |
| 4 | Golden fixture parity tests missing | 3 | 2 | 6 | Add byte-level/golden fixtures for adapter regression detection. Target milestone: v0.11.3. |
| 5 | `_safe_pick` silently duplicates endpoints | 3 | 2 | 6 | Interval helper duplicates endpoints instead of flagging inconsistencies. Target milestone: v0.11.3. |

### ADR-009 - Input Preprocessing and Mapping Policy

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Mapping export helpers placed on wrapper not explainer | 2 | 3 | 6 | `WrapCalibratedExplainer` exposes mapping persistence; consider thin `CalibratedExplainer` adapters for discoverability. Target milestone: v1.0.0-rc (public API change). |
| 2 | Export helper does not enforce JSON-safe conversion | 3 | 2 | 6 | Defensive JSON-safe conversion required to protect third-party preprocessors. Target milestone: v0.11.3. |
| 3 | Validation helper location differs from ADR text | 2 | 2 | 4 | Non-numeric detection implemented but located on wrapper; deliberate placement acceptable, document in ADR. Target milestone: v0.11.3 (doc update). |

### ADR-010 - Optional Dependency Split

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | No automated parity check between core-only and extras-installed runs | 3 | 3 | 9 | CI should compare canonical outputs between install modes to detect optional-extras regressions. Target milestone: v0.11.3. |

### ADR-011 - Deprecation and Migration Policy

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Compatibility shims not consistently emitting `deprecate()` warnings | 2 | 2 | 4 | `validate_payload` and some serializer shims should call `deprecate()`. Target milestone: v0.11.3. |
| 2 | Legacy-shaped serializer outputs silent on deprecation | 3 | 2 | 6 | Visual serializer compatibility translations should emit structured deprecation warnings. Target milestone: v0.11.3. |
| 3 | Legacy registry lists lack deprecation hooks | 3 | 2 | 6 | Deprecation hooks added in v0.11.1 (Task 1); remaining serializer shims targeted v0.11.3. Target milestone: v0.11.3. |

### ADR-012 - Documentation & Gallery Build Policy

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Notebooks not executed in docs HTML CI | 5 | 4 | 20 | Docs build currently disables notebook execution; ADR requires executed notebooks. Target milestone: v1.0.0-rc (requires CI infrastructure investment). |
| 2 | Runtime ceiling enforcement missing (per-example timing) | 3 | 3 | 9 | No CI-level per-example timing enforcement. Target milestone: v1.0.0-rc (follows notebook execution gate). |
| 3 | Gallery tooling decision undocumented for contributors | 2 | 2 | 4 | Document chosen gallery tool and contributor expectations. Target milestone: v0.11.3. |

### ADR-013 - Interval Calibrator Plugin Strategy

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Protocol signature mismatch between protocol and reference impl | 3 | 2 | 6 | Align `RegressionIntervalCalibrator` protocol signatures with concrete `IntervalRegressor` methods or add adapters. Target milestone: v0.11.3. |
| 2 | FAST wrapper location mismatch vs ADR text (doc drift) | 1 | 1 | 1 | Update ADR text or mirror implementation location. Target milestone: v0.11.3 (doc-only change). |
| 3 | FAST calibrator may be implicitly included in fallback chains | 3 | 3 | 9 | Prevent FAST-style ids being automatically used in non-fast fallback chains unless explicitly selected. Target milestone: v0.11.3. |
| 4 | Protocol enforcement relies on `isinstance` only | 2 | 2 | 4 | Add deeper signature/runtime harness or integration test for third-party plugins. Target milestone: v0.11.3. |

### ADR-014 - Visualization Plugin Architecture (superseded; see ADR-037)

**Superseded routing note (2026-03-20):** ADR-014 is superseded by ADR-037. Route builder/renderer governance and extension metadata requirements to ADR-037; route canonical PlotSpec semantics to ADR-036.

### ADR-015 - Explanation Plugin Integration

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Inconsistent invariant enforcement across bridges/validators | 3 | 2 | 6 | Some layers warn while others raise; align to ADR. Target milestone: v0.11.3. |
| 2 | Explainer handle exposes direct `learner` (bypass bridge) | 4 | 2 | 8 | Restrict direct learner access or document as escape hatch with warnings. Target milestone: v1.0.0-rc (public API semantic). |
| 3 | Task-scoped enforcement divergence | 3 | 2 | 6 | Ensure interval invariant enforcement is consistent across task types. Target milestone: v0.11.3. |

### ADR-016 - PlotSpec Separation and Schema (superseded; see ADR-036/ADR-037)

**Superseded routing note (2026-03-20):** ADR-016 is superseded by ADR-036 and ADR-037. Route new work to ADR-036 (canonical PlotSpec contract) and ADR-037 (visualization extension governance).

### ADR-020 - Legacy User API Stability

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Release checklist omits legacy API gate | 4 | 3 | 12 | Add explicit gate to release checklist to protect legacy contract parity. Target milestone: v0.11.3. |
| 2 | Wrapper regression tests missing parity assertions | 4 | 3 | 12 | Add parity tests for `explain_factual`/`explore_alternatives` signatures. Target milestone: v0.11.3. |
| 3 | Contributor workflow ignores contract doc updates | 3 | 3 | 9 | Update `CONTRIBUTING.md` to require contract doc updates for API changes. Target milestone: v0.11.3. |

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

_Last gap analysis: 2026-04-22_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Explanation context exposes mutable dicts | 4 | 3 | 12 | Contexts should be frozen/immutable. Target milestone: v1.0.0-rc (architecture-level change; exceeds quick-win threshold). |
| 2 | ~~Telemetry omits interval dependency hints~~ | 3 | 2 | 6 | **Closed v0.11.2:** `interval_dependencies` is already emitted in `ExplanationOrchestrator.invoke`; test `test_should_include_interval_dependencies_in_batch_telemetry_when_plugin_provides_hints` in `tests/unit/core/explain/test_orchestrator_core.py` verifies the invariant. |
| 3 | ~~Mondrian bin objects left mutable in requests~~ | 2 | 2 | 4 | **Closed v0.11.2:** `ExplanationRequest.__post_init__` freezes `bins` to an immutable tuple; test `test_should_freeze_mondrian_bins_at_request_construction_when_caller_mutates` in `tests/unit/core/prediction/test_orchestrator_extras.py` verifies the contract. |

### ADR-027 - FAST-Based Feature Filtering

_Last gap analysis: 2026-04-22_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Observability policy alignment undocumented | 2 | 2 | 4 | Document logging expectations and examples for feature filtering. Target milestone: v0.11.3. |
| 2 | Feature-filter telemetry examples sparse | 2 | 2 | 4 | Add practitioner examples showing emitted metadata. Target milestone: v0.11.3. |

### ADR-028 - Logging and Governance Observability

_Last gap analysis: 2026-05-16_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Fallback/degraded-state visibility is warning-first in multiple runtime paths | 3 | 4 | 12 | New v0.11.3 red-team finding: source and tests still preserve `UserWarning` plus INFO patterns for fallback/degraded-state events, causing notebook-visible noise and bypassing operator-controlled domain logging. Target milestone: v0.11.3 Task 8. |
| 2 | Enforcement tooling for logger domains missing | 2 | 2 | 4 | Delivered in v0.11.1 Task 7 (logger-domain quality script added). Target milestone: monitor. |
| 3 | Observability examples need alignment with Standard-005 | 2 | 2 | 4 | Delivered in v0.11.1 Task 7 (docs examples updated). Target milestone: monitor. |

### ADR-029 - Reject Integration Strategy

_Last gap analysis: 2026-03-03_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Reject strategy expansion beyond binary conformal rejectors not yet implemented | 3 | 3 | 9 | Delivered in v0.11.1 Task 14 (uncertainty-based and cost-sensitive strategies added). Target milestone: closed; monitor for regressions. |
| 2 | Strategy lifecycle hooks and configuration surface not finalized | 2 | 2 | 4 | Strategy config API deferred. Target milestone: v0.11.3. |
 | 3 | `RejectResult` public return type not yet migrated to strict `RejectResultV2` | 2 | 2 | 4 | Group L reset-path closure landed in v0.11.3: removed active deprecation warning from `reject_result_v2_to_legacy()` and kept `RejectResult` as stable v1.0.0 return type. Optional `RejectResultV2` remains available; full public return-type migration is deferred to a new post-v1.0 ADR-011 deprecation cycle (v1.1+ planning). |

### ADR-030 - Test Quality Priorities and Enforcement

_Last gap analysis: 2026-03-03_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Marker hygiene zero-tolerance ratification not yet formalized | 3 | 3 | 9 | Hybrid taxonomy still advisory on some categories. Target milestone: v0.11.3 Task 3. |
| 2 | Mutation testing policy documentation not published | 2 | 2 | 4 | Declare mutation testing optional for core modules and document. Target milestone: v0.11.3. |

### ADR-031 - Calibrator Serialization and State Persistence

**Compliance verification (2026-03-03):** Delivered in v0.11.0 — versioned `to_primitive`/`from_primitive` contracts plus `WrapCalibratedExplainer` `save_state`/`load_state` present and tested. No open gaps.

### ADR-032 - Guarded Explanation Semantics

**Compliance verification (2026-03-20):** ADR-032 now scopes guarded mode to schema-compatible representative-point guarded interval candidates, hard-fail calibration-feature alignment, and audit-field semantics that no longer overclaim semantic identity. No open appendix gaps.

### ADR-033 - Modality Extension Plugin Contract and Packaging Strategy

_Last gap analysis: 2026-03-03_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | CLI `--modality` filtering not yet implemented | 3 | 3 | 9 | Delivered in v0.11.1 Task 11. Target milestone: closed. |
| 2 | `vision.py`/`audio.py` shims not yet present (must raise `MissingExtensionError`) | 3 | 3 | 9 | Delivered in v0.11.1 Task 11; shim removal target milestone: v1.0.0-rc. |
| 3 | Packaging smoke test (extension install + entry-point discovery) missing | 3 | 3 | 9 | Delivered in v0.11.1 Task 13. Target milestone: closed. |
| 4 | Plugin contributor contract docs not updated for `plugin_api_version`/`data_modalities` requirements | 2 | 2 | 4 | Delivered in v0.11.1 Task 12. Target milestone: closed. |

### ADR-034 - Centralized Configuration Management

_Last gap analysis: 2026-04-20_

Runtime conformance closure for v0.11.2 is complete: Phase A and Phase B migration outputs are synchronized in ADR/release-plan surfaces, checker enforcement is zero-violation, and governance lifecycle schema alignment is closed for v0.11.x scope.

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Sensitive-value redaction for governance logs/exports (Open Item 1) | 3 | 2 | 6 | Interim posture remains documented as non-redacted. Target milestone: v1.0.0-rc. |
| 2 | `export_effective()` payload schema not versioned for external consumers (Open Item 2) | 3 | 2 | 6 | Target milestone: v1.0.0-rc; must be versioned before external tooling can rely on export. |

### ADR-035 - CI Workflow Governance

_Last gap analysis: 2026-04-22_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Advisory-to-required branch-protection flip is not enforceable from repository code alone | 2 | 2 | 4 | v0.11.2 closed text/validator/CODEOWNERS gaps; remaining follow-through is platform-governed branch-protection promotion. Target milestone: v0.11.3 re-evaluation. |
| 2 | Two governance caveats remain structurally non-automatable (template auto-application and two-maintainer per-path enforcement) | 1 | 2 | 2 | These are documented governance caveats rather than code defects. Track as accepted operational constraints unless platform capabilities change. |

## Standards status appendix (unified severity tables)

### Standard-001 - Nomenclature Standardization

_Last gap analysis: 2026-04-22_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Remaining compatibility/transitional bridges are still present | 2 | 2 | 4 | Task 8 closed non-legacy regression risk via `check_std001_nomenclature.py` + inventory report; explicit shim decisions and targeted bridge/parity tests are documented in `Standard-001_nomenclature_remediation.md`; approved bridges remain tracked with v0.11.3 removal horizon. Target milestone: v0.11.3. |

### Standard-002 - Documentation Standardisation

**Compliance verification (2026-06-02):** Task 2 closed in v0.11.3. `python scripts/quality/check_docstring_coverage.py` reports 96.73% overall (Modules 99.17%, Classes 100%, Functions 95.75%, Methods 96.52%); `pydocstyle src/calibrated_explanations/core/wrap_explainer.py` reports zero violations. No open gaps.

_Last gap analysis: 2026-06-02 (post-Task-5 stable surface)_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Wrapper public APIs lack full numpydoc blocks | 0 | 0 | 0 | **Resolved 2026-06-02.** Coverage 96.73% ≥ 90% threshold; zero pydocstyle violations on `wrap_explainer.py`. No code changes required — gap was already satisfied on the post-Task-5 stable surface. |

### Standard-003 - Test Coverage Standard

**Compliance verification (2026-02-27):** Reviewed code and RTD - no STD-003 gaps found; STD-003 is fully compliant. No further action required.

### Standard-004 - Documentation Standard (Audience Hubs)

**Compliance verification (2026-02-27):** Reviewed code and RTD - no STD-004 gaps found; STD-004 is fully compliant. No further action required.

### Standard-005 - Logging and Observability Standard

_Last gap analysis: 2026-05-16_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Fallback/degraded-state events rely on `UserWarning` instead of `WARNING` logs | 3 | 4 | 12 | New v0.11.3 red-team finding: STD-005 says degraded behavior with fallbacks belongs at `WARNING`; Task 8 owns inventory, migration, tests, and enforcement. |
| 2 | Enforcement tooling for domain-logger naming missing from CI | 2 | 2 | 4 | Delivered in v0.11.1 Task 7. Target milestone: closed; monitor for regressions. |
| 3 | Observability examples not yet aligned with Standard-005 naming and structured-context format | 2 | 2 | 4 | Delivered in v0.11.1 Task 7. Target milestone: closed; monitor for regressions. |
