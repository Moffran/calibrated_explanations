# RELEASE_PLAN Status Appendix

> Status note (2026-06-02): updated at v0.11.3 complete closure. All Task 9 workstreams (D: appendix synchronization; B: API parity/serializer/interval/invariant; C: warning-to-logging migration; A: accepted-registration governance) closed 2026-06-02. Task 1 (Standard-001 expired bridges) also closed 2026-06-02. Summary rows are synchronized to full v0.11.3 closure evidence.

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
| ADR-006 | Completed | All v0.11.3 gaps closed: gap 3 superseded (Task 5 Group K); gap 2 closed 2026-06-02 (accepted-registration audit events added to all 4 typed registration functions); gap 1 carry-forward (monitor, no code gap) |
| ADR-008 | Partially complete | v0.11.3 golden fixture tests and `_safe_pick` observability closed (gaps 4/5, 2026-06-02); domain-authoritative migration (gaps 1/2/3) remains v1.0.0-rc |
| ADR-009 | Partially complete | JSON-safe export closed (gap 2, 2026-06-02); helper-placement gap 3 closed (Workstream B, 2026-06-02); wrapper/core surface decision in v1.0.0-rc |
| ADR-010 | Completed | Core-only vs extras parity automation closed v0.11.3 (gap 1 closed 2026-06-02; `scripts/quality/check_core_extras_parity.py` added) |
| ADR-011 | Completed | All gaps closed v0.11.3 (gaps 1/2/3 closed 2026-06-02; validate_payload removed, narrative_format migrated to deprecate(), registry half by Group K, serializer half clean) |
| ADR-012 | Accepted with open hardening | Notebook execution/runtime ceilings remain v1.0.0-rc; gallery-tooling decision closed in v0.11.3 (gap 3 closed 2026-06-02) |
| ADR-013 | Completed | All v0.11.3 gaps closed (gaps 1/2/3/4 closed 2026-06-02; protocol tests, FAST chain separation guard, third-party conformance test, doc path corrected) |
| ADR-015 | Partially complete | v0.11.3 gaps 1/3 closed (2026-06-02; invariant consistency + task-type parity tests); gap 2 (direct learner bypass) deferred to v1.0.0-rc |
| ADR-020 | Completed | All v0.11.3 gaps closed (gap 1: release checklist 2026-06-02; gap 2: wrapper parity tests 2026-06-02; gap 3: CONTRIBUTING.md 2026-06-02) |
| ADR-026 | Partially complete | Context immutability remains (v1.0.0-rc); telemetry quick wins closed in v0.11.2 |
| ADR-027 | Completed | All gaps closed v0.11.3 (gaps 1/2 closed 2026-06-02; `docs/practitioner/performance-tuning.md` covers observability policy and telemetry examples) |
| ADR-028 | Completed | All v0.11.3 gaps closed (gap 1 closed 2026-06-02; 15 dual-emission sites removed, fallback events route to WARNING logs, warning-policy inventory script added) |
| ADR-030 | Completed | Zero-tolerance ratification closed v0.11.3 Task 3 (2026-05-12); marker hygiene taxonomy and mutation policy sections added to ADR-030; gaps 1/2 closed in appendix (2026-06-02) |
| ADR-033 | Accepted with open removal follow-through | `data_modalities` enforcement closes in v0.11.3; shim removal remains v1.0.0-rc |
| ADR-034 | Accepted with deferred v1.0 open items | Runtime conformance closure complete in v0.11.2; remaining work is redaction + export schema versioning |
| ADR-035 | Accepted with accepted operational constraint | v0.11.3 re-evaluation complete (2026-06-02): advisory-to-required branch-protection flip is platform-governed; recorded as accepted operational constraint in ADR-035 §v0.11.3 Re-evaluation Record; no in-repo work remains |
| ADR-036 | Accepted | v0.11.3 Task 6 promoted PlotSpec as the default user-facing plotting path after v0.11.2 mending evidence; monitor canonical dataclass validation and legacy fallback behavior |
| ADR-037 | Accepted | v0.11.3 Task 6 promoted the governed built-in PlotSpec default while preserving the runtime plot-kind extension prohibition and explicit legacy opt-out |
| STD-001 | Completed | All v0.11.3 bridges closed (Task 1, 2026-06-02); 0 expired remove_by_v0.11.3 records; checker passes; internal bridge dunders renamed; APPROVED_COMPATIBILITY_BRIDGES = {} |
| STD-002 | Completed | WrapCalibratedExplainer numpydoc gap closed in v0.11.3 Task 2; coverage 96.73%, zero pydocstyle violations (2026-06-02) |
| STD-003 | Completed | Monitor for regressions |
| STD-004 | Completed | Monitor for regressions |
| STD-005 | Completed | All v0.11.3 gaps closed (gap 1 closed 2026-06-02; fallback events aligned with WARNING log-first; 105 remaining UserWarning call sites inventoried and allowlisted) |

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
| 1 | Dual trust state (descriptor `trusted` flag + `_TRUSTED_*` sets) can diverge | 3 | 3 | 9 | **Carry-forward (2026-06-02):** v0.11.3 re-evaluation complete. Four `_TRUSTED_*` module-level sets (`_TRUSTED_EXPLANATIONS`, `_TRUSTED_INTERVALS`, `_TRUSTED_PLOT_BUILDERS`, `_TRUSTED_PLOT_RENDERERS`) remain in `plugins/registry.py`. These are NOT a dual-state remnant — they are the sole authoritative trust-tracking infrastructure for the current plugin architecture. The dual-state issue identified in v0.11.1 was caused by the old list-path API (`register`, `trust_plugin`, `find_for`, `find_for_trusted`) maintaining separate trust state; those four functions were removed by Task 5 Group K. The remaining sets are the single source of truth. No action required; monitor for regressions at next milestone. |
| 2 | ~~Accepted registrations emit no governance audit event~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 9 Workstream A (2026-06-02).** `register_interval_plugin`, `register_plot_builder`, and `register_plot_renderer` now call `_log_plugin_registration_event` after `mutate_trust_atomic` completes, emitting `accepted_registration` governance events. (explanation plugins already emitted via `_register_legacy_plugin`.) 4 new tests in `tests/observability/test_governance_events.py` verify all four typed registration functions emit schema-valid `accepted_registration` events. |
| 3 | ~~Legacy `_REGISTRY`/`_TRUSTED` lists lack deprecation path~~ | 0 | 0 | 0 | **Superseded (2026-06-02) by Task 5 Group K.** The four list-path API functions (`register(plugin)`, `trust_plugin(plugin)`, `find_for(model)`, `find_for_trusted(model)`) have been removed from `plugins/registry.py`. Fail-closed absence tests confirm removal. Legacy list-path deprecation path is fully eliminated. |

### ADR-007 - PlotSpec Abstraction (superseded; see ADR-036/ADR-037)

**Superseded routing note (2026-03-20):** ADR-007 is superseded by ADR-036 and ADR-037. Route canonical PlotSpec contract and validation questions to ADR-036, and visualization extension/rendering governance questions to ADR-037.

### ADR-008 - Explanation Domain Model

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Domain model not authoritative source | 5 | 4 | 20 | Core workflows still operate on legacy dicts; domain objects primarily produced at serialization boundaries. Target milestone: v1.0.0-rc (major refactor; architecture-heavy). |
| 2 | Legacy->domain round-trip fails for conjunctive rules | 4 | 3 | 12 | `domain_to_legacy` casts features to scalars, breaking conjunction support. Target milestone: v1.0.0-rc (follows domain-model authority work). |
| 3 | Structured model/calibration metadata missing | 4 | 3 | 12 | Explanation dataclass lacks dedicated calibration/model descriptor fields. Target milestone: v1.0.0-rc. |
| 4 | ~~Golden fixture parity tests missing~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 9 Workstream B (2026-06-02).** `tests/unit/explanations/test_adapter_golden_fixtures.py` added: 4 tests covering legacy→domain, domain→legacy, and full bidirectional round-trips for the classification factual fixture. |
| 5 | ~~`_safe_pick` silently duplicates endpoints~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 9 Workstream B (2026-06-02).** `_safe_pick` in `explanations/models.py` now emits a `DEBUG` log when endpoint duplication occurs (`"endpoint duplication detected — ragged feature/weight arrays in legacy payload"`). `tests/unit/explanations/test_models_safe_pick.py` added: 3 tests verifying the debug log, no-log on aligned arrays, and None return on empty arrays. |

### ADR-009 - Input Preprocessing and Mapping Policy

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Mapping export helpers placed on wrapper not explainer | 2 | 3 | 6 | `WrapCalibratedExplainer` exposes mapping persistence; consider thin `CalibratedExplainer` adapters for discoverability. Target milestone: v1.0.0-rc (public API change). |
| 2 | ~~Export helper does not enforce JSON-safe conversion~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 9 Workstream B (2026-06-02).** `WrapCalibratedExplainer.export_preprocessor_mapping()` already calls `_validate_json_safe_mapping()` on both the `get_mapping_snapshot` and `mapping_` paths; `import_preprocessor_mapping()` also validates JSON safety on import. Tests in `tests/unit/core/test_wrap_explainer_helpers.py` (`test_export_preprocessor_mapping_rejects_non_json_serialisable_snapshots`, `test_import_preprocessor_mapping_rejects_non_json_serialisable_payload`) verify the enforcement. No code change required. |
| 3 | ~~Validation helper location differs from ADR text~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 9 Workstream B (2026-06-02).** Non-numeric detection is implemented on the wrapper (`WrapCalibratedExplainer`); deliberate placement — the wrapper is the public API surface for preprocessing and ADR-009 §gap 3 alignment is handled by the Phase 2 note in ADR-011 gap 3. No code change required. |

### ADR-010 - Optional Dependency Split

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | ~~No automated parity check between core-only and extras-installed runs~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 9 Workstream B (2026-06-02).** `scripts/quality/check_core_extras_parity.py` added: probes extras availability, runs CE-first API with synthetic data, asserts structural invariants (n_instances, explanations_type, predictions, rules). Run with `--check` for CI-fail-on-violation, `--report` for JSON output. `python scripts/quality/check_core_extras_parity.py` exits 0. |

### ADR-011 - Deprecation and Migration Policy

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | ~~Compatibility shims not consistently emitting `deprecate()` warnings~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 9 Workstream B (2026-06-02).** `validate_payload` removed by Task 1. `ce_agent_utils.py` `narrative_format` deprecation migrated from raw `warnings.warn(DeprecationWarning)` to central `deprecate(key="ce_agent_utils.narrative_format")` helper. All other deprecation shims removed by Task 5. |
| 2 | ~~Legacy-shaped serializer outputs silent on deprecation~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 9 Workstream B (2026-06-02).** Internal `_legacy_payload` / `legacy_payload` helpers in `explanations.py` are private implementation paths, not user-facing compatibility shims; they do not require deprecation warnings. The public serialization surface (`serialization.py`) has no backward-compat shims. No code change required. |
| 3 | ~~Legacy registry lists lack deprecation hooks~~ | 0 | 0 | 0 | **Both phases closed (2026-06-02).** Registry half closed by Task 5 Group K (four functions removed); serializer half confirmed clean — no legacy-shaped serialization shims remain. Final row removed. |

### ADR-012 - Documentation & Gallery Build Policy

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Notebooks not executed in docs HTML CI | 5 | 4 | 20 | Docs build currently disables notebook execution; ADR requires executed notebooks. Target milestone: v1.0.0-rc (requires CI infrastructure investment). |
| 2 | Runtime ceiling enforcement missing (per-example timing) | 3 | 3 | 9 | No CI-level per-example timing enforcement. Target milestone: v1.0.0-rc (follows notebook execution gate). |
| 3 | ~~Gallery tooling decision undocumented for contributors~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 9 Workstream D (2026-06-02).** ADR-012 Open Questions section closed with formal decision: nbconvert for v1.0.0; sphinx-gallery deferred to post-v1.0. Decision text added to ADR-012 "Resolved Questions" section. Doc-only. |

### ADR-013 - Interval Calibrator Plugin Strategy

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | ~~Protocol signature mismatch between protocol and reference impl~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 9 Workstream B (2026-06-02).** `tests/unit/plugins/test_adr013_interval_gaps.py` confirms `IntervalRegressor` implements all `RegressionIntervalCalibrator` protocol methods (`predict_probability`, `predict_uncertainty`, `pre_fit_for_probabilistic`, `compute_proba_cal`, `insert_calibration`, `predict_proba`, `is_multiclass`, `is_mondrian`) and `VennAbers` implements `ClassificationIntervalCalibrator` protocol. No code change required. |
| 2 | ~~FAST wrapper location mismatch vs ADR text (doc drift)~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 9 Workstream D (2026-06-02).** ADR-013 §4 Lifecycle corrected to cite `src/calibrated_explanations/calibration/interval_wrappers.py` (was incorrectly `plugins/interval_wrappers.py`). Implementation-status note added to ADR-013. Doc-only fix; no code change. |
| 3 | ~~FAST calibrator may be implicitly included in fallback chains~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 9 Workstream B (2026-06-02).** `build_interval_chain(fast=False)` confirmed to exclude `core.interval.fast` from the default chain; `build_interval_chain(fast=True)` keeps them in separate chains. Guarded by `tests/unit/plugins/test_adr013_interval_gaps.py` (`test_should_exclude_fast_identifier_from_default_interval_chain`, `test_should_keep_default_and_fast_chains_separate`). |
| 4 | ~~Protocol enforcement relies on `isinstance` only~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 9 Workstream B (2026-06-02).** `test_should_structurally_conform_when_third_party_implements_protocol` in `test_adr013_interval_gaps.py` verifies structural conformance (required method names, `predict_proba` kwarg surface, zero-arg `is_multiclass`/`is_mondrian`) for a third-party object that does not inherit from any CE class. |

### ADR-014 - Visualization Plugin Architecture (superseded; see ADR-037)

**Superseded routing note (2026-03-20):** ADR-014 is superseded by ADR-037. Route builder/renderer governance and extension metadata requirements to ADR-037; route canonical PlotSpec semantics to ADR-036.

### ADR-015 - Explanation Plugin Integration

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | ~~Inconsistent invariant enforcement across bridges/validators~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 9 Workstream B (2026-06-02).** `tests/unit/plugins/test_invariant_consistency.py` confirms that equivalent invalid payloads fail consistently in both `validate_explanation_batch` and `LegacyPredictBridge.predict` for both "interval invariant violated" and "prediction invariant violated" cases. |
| 2 | Explainer handle exposes direct `learner` (bypass bridge) | 4 | 2 | 8 | Restrict direct learner access or document as escape hatch with warnings. Target milestone: v1.0.0-rc (public API semantic). |
| 3 | ~~Task-scoped enforcement divergence~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 9 Workstream B (2026-06-02).** `tests/unit/plugins/test_invariant_consistency.py` extended with `test_should_enforce_interval_invariant_consistently_across_task_types` and `test_should_enforce_prediction_invariant_consistently_across_task_types` parametrized over `regression` and `classification`; both pass. |

### ADR-016 - PlotSpec Separation and Schema (superseded; see ADR-036/ADR-037)

**Superseded routing note (2026-03-20):** ADR-016 is superseded by ADR-036 and ADR-037. Route new work to ADR-036 (canonical PlotSpec contract) and ADR-037 (visualization extension governance).

### ADR-020 - Legacy User API Stability

_Last gap analysis: 2026-02-27_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | ~~Release checklist omits legacy API gate~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 9 Workstream D (2026-06-02).** Step 3 "Legacy API gate" added to `docs/foundations/governance/release_checklist.md`: run `pytest tests/unit/api/test_legacy_user_api_contract.py -v`, confirm zero failures, require contract doc + parity test updates in the same PR for any scheduled legacy surface change. |
| 2 | ~~Wrapper regression tests missing parity assertions~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 9 Workstream B (2026-06-02).** `tests/unit/api/test_legacy_user_api_contract.py` extended with 5 parity tests: explicit threshold forwarding, config-default injection, `explore_alternatives` delegation, and `**kwargs` acceptance assertion for both explain methods. `make local-checks-pr` passes. |
| 3 | ~~Contributor workflow ignores contract doc updates~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 9 Workstream D (2026-06-02).** Sentence added to the "Roadmap and ADR-driven development" section of `.github/CONTRIBUTING.md` directing contributors to update `docs/improvement/legacy_user_api_contract.md` when changes affect the legacy API. |

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
| 1 | ~~Observability policy alignment undocumented~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 9 Workstream D (2026-06-02).** `docs/practitioner/performance-tuning.md` already contains a "Logging and Observability" section documenting the debug-by-default posture, strict-mode escalation to `WARNING`, and governance logger routing examples aligned with Standard-005. No code change required. |
| 2 | ~~Feature-filter telemetry examples sparse~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 9 Workstream D (2026-06-02).** `docs/practitioner/performance-tuning.md` already contains the "Telemetry Events" section listing `filter_enabled`/`filter_skipped`/`filter_error` and the "Debugging Filtered Explanations" section with per-instance ignore-mask metadata example. No code change required. |

### ADR-028 - Logging and Governance Observability

_Last gap analysis: 2026-05-16_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | ~~Fallback/degraded-state visibility is warning-first in multiple runtime paths~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 9 Workstream C (2026-06-02).** 15 dual-emission sites (logger.warning + warnings.warn for same event) removed across `parallel/parallel.py` (5 sites), `plugins/builtins.py` (3), `cache/cache.py` (2), `core/explain/orchestrator.py` (1), `core/calibrated_explainer.py` (4). Each fallback is now visible only via domain-routed `WARNING` log. `scripts/quality/check_warning_policy.py` added: 110 call sites inventoried, 0 unclassified. `tests/unit/perf/test_parallel.py` updated to `caplog` assertion. `make local-checks-pr` passes. |
| 2 | Enforcement tooling for logger domains missing | 2 | 2 | 4 | Delivered in v0.11.1 Task 7 (logger-domain quality script added). Target milestone: monitor. |
| 3 | Observability examples need alignment with Standard-005 | 2 | 2 | 4 | Delivered in v0.11.1 Task 7 (docs examples updated). Target milestone: monitor. |

### ADR-029 - Reject Integration Strategy

_Last gap analysis: 2026-03-03_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Reject strategy expansion beyond binary conformal rejectors not yet implemented | 3 | 3 | 9 | Delivered in v0.11.1 Task 14 (uncertainty-based and cost-sensitive strategies added). Target milestone: closed; monitor for regressions. |
| 2 | Strategy lifecycle hooks and configuration surface not finalized | 2 | 2 | 4 | **Deferred to post-v1.0 (2026-06-02).** Task 5 Group L (ADR-029 gap 1 / RejectResult active deprecation) is now closed. With that blocker resolved, the strategy lifecycle / config surface deferral is now formally recorded: implementing a full strategy lifecycle and config surface is a design/architecture work item that expands beyond bounded v0.11.3 RC hardening. `RejectResult` remains the stable v1.0.0 return type; full strategy config API is deferred to a new post-v1.0 ADR-011 deprecation cycle (v1.1+ planning). |
 | 3 | `RejectResult` public return type not yet migrated to strict `RejectResultV2` | 2 | 2 | 4 | Group L reset-path closure landed in v0.11.3: removed active deprecation warning from `reject_result_v2_to_legacy()` and kept `RejectResult` as stable v1.0.0 return type. Optional `RejectResultV2` remains available; full public return-type migration is deferred to a new post-v1.0 ADR-011 deprecation cycle (v1.1+ planning). |

### ADR-030 - Test Quality Priorities and Enforcement

_Last gap analysis: 2026-03-03_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | ~~Marker hygiene zero-tolerance ratification not yet formalized~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 3 (2026-05-12).** ADR-030 now contains a formal "Marker hygiene" decision section defining the enforced pytest marker taxonomy (`unit`/`integration` inferred from directory; `slow`/`viz`/`viz_render` explicit) and naming `check_marker_hygiene.py` as the enforcement tool. Ratification record added to ADR-030 status note confirming zero-tolerance CI enforcement. |
| 2 | ~~Mutation testing policy documentation not published~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 3 (2026-05-12).** ADR-030 now contains a formal "Mutation testing policy" section declaring mutation testing optional for core modules (recommended for critical-path logic; not a release gate). Tool: `mutmut run --paths-to-mutate src/calibrated_explanations/`. Ratification record added. |

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
| 1 | Advisory-to-required branch-protection flip is not enforceable from repository code alone | 2 | 2 | 4 | **v0.11.3 re-evaluation complete (2026-06-02).** Promotion of `ci-policy/validate-workflows` to a required branch-protection check is platform-governed (requires GitHub administrator action). Validator logic and CODEOWNERS coverage are complete. Recorded as an accepted operational constraint in ADR-035 §v0.11.3 Re-evaluation Record. Pending administrator action; no in-repo work remains. Retarget: monitor / platform action by administrators. |
| 2 | Two governance caveats remain structurally non-automatable (template auto-application and two-maintainer per-path enforcement) | 1 | 2 | 2 | These are documented governance caveats rather than code defects. Track as accepted operational constraints unless platform capabilities change. |

## Standards status appendix (unified severity tables)

### Standard-001 - Nomenclature Standardization

_Last gap analysis: 2026-04-22_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | ~~Remaining compatibility/transitional bridges are still present~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 1 (2026-06-02).** All internal compatibility-bridge dunders renamed to single-underscore helpers across , , , , ; transitional plotting helpers in  renamed.  and  — both empty.  contains 0 expired  records.  exits 0. |

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
| 1 | ~~Fallback/degraded-state events rely on `UserWarning` instead of `WARNING` logs~~ | 0 | 0 | 0 | **Closed v0.11.3 Task 9 Workstream C (2026-06-02).** Dual-emission pattern eliminated across 5 files (15 total sites); fallback events now route to `WARNING` logs per STD-005 §4.1. Warning-policy inventory script (`scripts/quality/check_warning_policy.py`) classifies all 110 call sites with 0 unclassified. |
| 2 | Enforcement tooling for domain-logger naming missing from CI | 2 | 2 | 4 | Delivered in v0.11.1 Task 7. Target milestone: closed; monitor for regressions. |
| 3 | Observability examples not yet aligned with Standard-005 naming and structured-context format | 2 | 2 | 4 | Delivered in v0.11.1 Task 7. Target milestone: closed; monitor for regressions. |
