# RELEASE_PLAN Status Appendix

> Status note (2026-06-15): post-v0.11.3 gap-closure sweep. Closed: ADR-011 (all 3 gaps — guarded-wrapper finalization exception, ledger rebuild, raw-warn sites including `calibrated_explainer.py` `guarded=True` migrated to `deprecate()`), ADR-023 (stale coverage omit entries), ADR-028/STD-005 (0 unclassified warning sites), ADR-032 (`get_guarded_audit` message corrected), ADR-034 gap 3 (status-source conflict resolved; gaps 1-2 formally deferred post-v1.0 in ADR-034 §Open Items), ADR-036 (pipeline validation boundary), ADR-037 (plot-kinds/modes metadata). Added ADR-038 section (3 open gaps: `**kwargs` graduation gate, plugin §5 enforcement, allowlist comment — all v1.0.0-rc or minor).

> Status note (2026-06-11): full ADR gap-analysis sweep at post-Task-13 state, with per-ADR code-evidence verification for the ADRs whose status previously rested on prior closure claims (ADR-003, 005, 012, 021, 023, 026, 031, 035). New gaps found in ADR-011 (deprecation ledger/helper drift introduced by Task 13 guarded deprecations and pre-existing raw warn sites), ADR-028/STD-005 (one unclassified warning site added by Task 12), ADR-032 (stale guidance string), ADR-036 (plugin-path validation boundary), ADR-037 (extension metadata thinness), ADR-034 (status-source conflict on Open Items), and ADR-023 (trivial stale-omit cleanup). Evidence-driven status corrections: ADR-005 gap closed (provenance propagates through both adapters; `schema.validate_payload` intact); ADR-012 gaps rewritten (docs HTML/linkcheck CI job is unwired — `reusable-build-docs.yml` has no caller; notebook execution exists nightly-advisory with timeouts); ADR-026 gap re-scoped (contexts are now frozen dataclasses; only specific nested fields remain unfrozen). All other previously closed rows re-verified via quality gates (`check_adr002_compliance`, `check_marker_hygiene`, `check_std001_nomenclature`, `check_logging_domains`, `check_import_graph`, `check_governance_event_schema`, `check_trust_mutation_primitive`, `check_docstring_coverage` 96.34%, `check_core_extras_parity`, ADR-020 contract tests 11/11) and cited code evidence (ADR-003/021/031/035). New gaps are scheduled in `docs/improvement/v0.11.3_plan.md` Task 15.

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
| ADR-004 | Completed (2026-06-17) | Gap 1 closed (v0.11.4): `deprecate()` fires at `parallel/parallel.py:448-456` when `strategy="auto"` and `enabled=True`; test in `tests/unit/core/test_parallel_deprecation.py`; removal ETA v1.0.0 |
| ADR-005 | Completed (2026-06-11) | Last gap closed on code evidence: provenance propagates through both adapters (`explanations/adapters.py:36-41`, `:68`); `schema.validate_payload` and serialization invariants intact post-Task-1 |
| ADR-006 | Completed | All v0.11.3 gaps closed: gap 3 superseded (Task 5 Group K); gap 2 closed 2026-06-02 (accepted-registration audit events added to all 4 typed registration functions); gap 1 carry-forward (monitor, no code gap) |
| ADR-008 | Partially complete | v0.11.3 golden fixture tests and `_safe_pick` observability closed (gaps 4/5, 2026-06-02); gap 2 (conjunctive round-trip) closed in code pre-v0.11.4 (`adapters.py:82-89`); open: gap 1 (domain authority, v1.0.0-rc), gap 3 (structured metadata, v0.11.4 Task 7) |
| ADR-009 | Completed (2026-06-15) | JSON-safe export closed (gap 2, 2026-06-02); helper-placement gap 3 closed (Workstream B, 2026-06-02); former gap 1 (wrapper placement) is intentional per ADR-009 text and ADR-001 boundaries — closed as compliance verification in v0.11.4 Task 1 |
| ADR-010 | Completed | Core-only vs extras parity automation closed v0.11.3 (gap 1 closed 2026-06-02; `scripts/quality/check_core_extras_parity.py` added) |
| ADR-011 | Closed (2026-06-13) | All gaps resolved: (1) guarded-wrapper deprecations removed in Task 13 (not merely deferred); 9 Task-17 active surfaces target v1.0.0 — within ADR-011 §2 final-cycle allowance; (2) raw `warnings.warn(DeprecationWarning)` bypass sites fixed — `normalization_strategy.py`, `core/reject.py`, `core/explain/__init__.py` all use `deprecate()` helper; (3) active-deprecations ledger rebuilt with all surfaces; `make deprecation-closure` passes with 9 v1.0.0 rows permitted and 0 blocking. `data_modalities` deprecation closed fail-closed 2026-06-13. |
| ADR-012 | Accepted with open hardening (re-evidenced 2026-06-11) | Notebook execution exists (nightly advisory driver + `nbsphinx_execute="always"` on non-RTD builds); real gap is that the docs HTML/linkcheck CI job is unwired (`reusable-build-docs.yml` has no caller). Runtime ceilings advisory-only. Remains v1.0.0-rc |
| ADR-013 | Completed | All v0.11.3 gaps closed (gaps 1/2/3/4 closed 2026-06-02; protocol tests, FAST chain separation guard, third-party conformance test, doc path corrected) |
| ADR-015 | Completed (2026-06-17) | All gaps closed: gaps 1/3 closed 2026-06-02 (invariant consistency + task-type parity tests); gap 2 (direct learner bypass) closed v0.11.4 — `ExplainerHandle.learner` emits `deprecate()` at `plugins/explanations.py:138-145`; active-deprecations ledger row and test `test_should_emit_deprecation_warning_when_learner_accessed` confirm closure |
| ADR-020 | Completed | All v0.11.3 gaps closed (gap 1: release checklist 2026-06-02; gap 2: wrapper parity tests 2026-06-02; gap 3: CONTRIBUTING.md 2026-06-02) |
| ADR-026 | Completed (2026-06-15) | Context immutability fully landed: `ExplanationContext.__post_init__` freezes all nested fields including `helper_handles`, `feature_names`, `categorical_features` (`plugins/explanations.py:64-73`); `IntervalCalibratorContext` freezes `calibration_splits`, `bins`, `residuals`, `difficulty`, `fast_flags` (`plugins/intervals.py:29-54`). Appendix gap was stale; closed in v0.11.4 Task 1. |
| ADR-027 | Completed | All gaps closed v0.11.3 (gaps 1/2 closed 2026-06-02; `docs/practitioner/performance-tuning.md` covers observability policy and telemetry examples) |
| ADR-028 | Closed (2026-06-15) | Warning-policy regression fixed in Task 15; post-Task-17 fix migrated 2 `calibrated_explainer.py` sites to `deprecate()`. `check_warning_policy.py` now reports 114 sites, 0 unclassified (2026-06-15). |
| ADR-030 | Completed | Zero-tolerance ratification closed v0.11.3 Task 3 (2026-05-12); marker hygiene taxonomy and mutation policy sections added to ADR-030; gaps 1/2 closed in appendix (2026-06-02) |
| ADR-032 | Closed (2026-06-13) | All decisions verified. `get_guarded_audit` error message corrected in Task 15: `explanations/explanations.py:236-238` now recommends `explain_factual(..., guarded_options=GuardedOptions())` / `explore_alternatives(...)` — canonical API per ADR-032 decision 1. Deprecated wrappers removed. |
| ADR-033 | Closed (2026-06-17) | All obligations met. `data_modalities` default-fallback removed in v0.11.4 (Task 3): `validate_plugin_meta` now raises `ValidationError` for plugins missing the key; entry-point path retains `UserWarning`-and-skip pre-check. Breaking change documented in `docs/upgrade/v0.11.4-upgrade-checklist.md`. ADR-033 §6.2 shims are permanent. |
| ADR-034 | Accepted with deferred v1.0 open items (reconciled 2026-06-13) | Runtime conformance closure complete in v0.11.2. Status-source conflict resolved: ADR-034 §Open Items now documents "Status: Declared out of scope for v1.0.0-rc" for both redaction and export schema contract items. No RC deferrals remain for this ADR. |
| ADR-035 | Accepted with accepted operational constraint | v0.11.3 re-evaluation complete (2026-06-02): advisory-to-required branch-protection flip is platform-governed; recorded as accepted operational constraint in ADR-035 §v0.11.3 Re-evaluation Record; no in-repo work remains |
| ADR-036 | Closed (2026-06-13) | §5 validation boundary implemented in Task 15: `validate_plot_artifact()` (public, `plotting.py:308`) called at both build/render boundary points (`plotting.py:387`, `:439`). Artifacts that fail `validate_plotspec` raise `ValidationError` before renderer invocation. 3 dedicated boundary tests in `test_plot_plugin_validation_boundary.py` pass. |
| ADR-037 | Closed (2026-06-13) | §4 extension metadata implemented in Task 15: `validate_plugin_meta` (via `plugins/base.py:381-403`) validates `plot_kinds` against allowed values and `plot_modes` against allowed values; both default when absent. 11 tests in `test_plot_extension_metadata.py` pass. |
| ADR-038 | Accepted with open hardening (2026-06-15) | Task 17 delivered `GuardedOptions`/`reject_confidence`/taxonomy; 3 open items remain: `**kwargs` graduation gate (v1.0.0-rc), plugin §5 enforcement (v1.0.0-rc), allowlist comment narrowing (minor, next gap sweep) |
| STD-001 | Completed | All v0.11.3 bridges closed (Task 1, 2026-06-02); 0 expired remove_by_v0.11.3 records; checker passes; internal bridge dunders renamed; APPROVED_COMPATIBILITY_BRIDGES = {} |
| STD-002 | Completed | WrapCalibratedExplainer numpydoc gap closed in v0.11.3 Task 2; coverage 96.73%, zero pydocstyle violations (2026-06-02) |
| STD-003 | Completed | Monitor for regressions |
| STD-004 | Completed | Monitor for regressions |
| STD-005 | Closed (2026-06-15) | Shares ADR-028 closure: 114 sites, 0 unclassified `warnings.warn` sites (verified 2026-06-15). |

## Detailed gap inventory and historical notes

- The authoritative detailed execution/gap notes remain in milestone execution plans:
  - `docs/improvement/archive/v0.11.1_plan.md`
  - `docs/improvement/archive/v0.11.2_plan.md`
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

**Compliance verification (2026-06-11):** Reviewed code and RTD - no ADR-001 gaps found; `check_import_graph.py` reports no violations (re-run 2026-06-11). ADR-001 is fully compliant. No further action required.

### ADR-002 - Exception Taxonomy and Validation Contract

**Compliance verification (2026-06-11):** Reviewed code and RTD - no ADR-002 gaps found; `check_adr002_compliance.py` scanned 120 files with no violations (re-run 2026-06-11). ADR-002 is fully compliant. No further action required.

### ADR-003 - Caching Strategy

**Compliance verification (2026-06-11):** Reviewed code and RTD - no ADR-003 gaps found. Code evidence: namespaced/versioned blake2b cache keys (`cache/cache.py:253-285` `make_key(namespace, version, parts)`), `cachetools` LRU backend with in-package fallback (`cache/cache.py:39-189`), telemetry counters (`cache/cache.py:290`), and opt-in default-off posture (`CacheConfig.enabled: bool = False`, `cache/cache.py:356`). ADR-003 is fully compliant. No further action required.

### ADR-004 - Parallel Execution Framework

**Compliance verification (2026-06-17):** Gap 1 closed in v0.11.4. `deprecate()` call fires at `parallel/parallel.py:448-456` when `strategy == "auto"` and `enabled=True`; verified by `tests/unit/core/test_parallel_deprecation.py`. Active-deprecations ledger row added to `docs/migration/deprecations.md` with removal ETA v1.0.0. The `auto_strategy()` heuristic is retained but deprecated; explicit strategy is required for v1.0.0. No remaining open gaps. `make deprecation-closure` passes (0 blocking).

### ADR-005 - Explanation Payload Schema

**Compliance verification (2026-06-11):** Reviewed code and RTD - no ADR-005 gaps remain. The former gap 1 (provenance propagation in legacy adapters) is closed on code evidence: `legacy_to_domain` reads and preserves `provenance` (`explanations/adapters.py:36-41`) and `domain_to_legacy` emits it (`explanations/adapters.py:68`). The canonical validator survives the Task-1 alias removal (`schema/validation.py:42` `validate_payload`; the removed symbol was only the deprecated `serialization.validate_payload` alias), `explanation_schema_v1.json` is present, and interval invariants are enforced at serialization (`serialization.py:83` `_validate_invariants`). ADR-005 is fully compliant. No further action required. (Deeper domain-model authority work remains tracked under ADR-008.)

### ADR-006 - Plugin Trust Model

**Compliance verification (2026-06-11):** Reviewed code and RTD - no open ADR-006 gaps. The four `_TRUSTED_*` module-level sets in `plugins/registry.py` are the sole authoritative trust-tracking infrastructure (the v0.11.1 dual-state issue was eliminated when Task 5 Group K removed the list-path API); `check_trust_mutation_primitive.py` passes (2026-06-11). Accepted-registration governance events and list-path removal closed 2026-06-02. Monitor for regressions; no further action required.

### ADR-007 - PlotSpec Abstraction (superseded; see ADR-036/ADR-037)

**Superseded routing note (2026-03-20):** ADR-007 is superseded by ADR-036 and ADR-037. Route canonical PlotSpec contract and validation questions to ADR-036, and visualization extension/rendering governance questions to ADR-037.

### ADR-008 - Explanation Domain Model

_Last gap analysis: 2026-06-15_

v0.11.4 sweep: gap 2 (conjunctive round-trip) is already closed in code — `domain_to_legacy` at `explanations/adapters.py:82-89` coerces `list`/`np.ndarray` feature to `tuple(feat)` before appending; `from_legacy_dict` at lines 55-75 handles tuples correctly. Stale appendix description removed. Two open gaps remain.

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Domain model not authoritative source | 5 | 4 | 20 | Re-verified 2026-06-11: `legacy_to_domain`/`domain_to_legacy` are referenced only inside the `explanations` package (adapters + serialization boundary); core explain/predict workflows still operate on legacy dicts. Target milestone: v1.0.0-rc (major refactor; architecture-heavy). |
| 2 | Structured model/calibration metadata missing | 4 | 3 | 12 | Explanation dataclass lacks dedicated calibration/model descriptor fields (generic `provenance`/`metadata` mappings only). Target milestone: v0.11.4 Task 7 (additive internal change; no public API impact). |

### ADR-009 - Input Preprocessing and Mapping Policy

**Compliance verification (2026-06-15):** Reviewed code and ADR text — no ADR-009 gaps found. JSON-safe mapping export closed in v0.11.3 Workstream B (gap 2); helper-placement gap closed in Workstream B (gap 3). The former gap 1 ("mapping export helpers placed on wrapper not explainer") is recorded as an intentional design decision in the ADR text (2026-01-12, lines 73-76): placement on `WrapCalibratedExplainer` is deliberate per ADR-001 boundary rules, and the ADR states "no code change is required." The `auto_encode='auto'` mapping-persistence path is an explicitly deferred post-v1.0 item documented in ADR-009 §Open Questions and §Post-v1.0 open item. ADR-009 is fully compliant. No further action required.

### ADR-010 - Optional Dependency Split

**Compliance verification (2026-06-11):** Reviewed code and RTD - no ADR-010 gaps found; `scripts/quality/check_core_extras_parity.py` passes (re-run 2026-06-11). ADR-010 is fully compliant. No further action required.

### ADR-011 - Deprecation and Migration Policy

**Compliance verification (2026-06-15):** Reviewed code and `docs/migration/deprecations.md` — no ADR-011 gaps found. Evidence: (1) guarded wrappers introduced and immediately removed in v0.11.3 via finalization exception — correctly filed in Removed (history) at `deprecations.md:258-261`; (2) active-deprecations table now carries 9 rows, all correctly filed; `make deprecation-closure` passes (9 v1.0.0 permitted, 0 blocking); (3) `normalization_strategy.py`, `core/reject.py`, and `core/explain/__init__.py` all use `deprecate()` helper; `core/calibrated_explainer.py` `guarded=True` emissions migrated from raw `warnings.warn` to `deprecate(key="guarded_true_boolean_kwarg")` in v0.11.3 post-Task-17 fix; `utils/deprecation.py` (singular) is the legacy `deprecate_public_api_symbol` module for `__init__.py` lazy access and is now clearly differentiated from `utils/deprecations.py` (plural). ADR-011 is fully compliant. No further action required.

### ADR-012 - Documentation & Gallery Build Policy

_Last gap analysis: 2026-06-17_

Evidence update (2026-06-11): the former gap framing ("docs build disables notebook execution") is outdated. Notebook execution now exists in two forms: the nightly advisory driver (`ci-nightly.yml` `notebook-exec-report` job runs `scripts/docs/run_notebooks.py --mode advisory --cell-timeout 30 --notebook-timeout 300`) and `nbsphinx_execute = "always"` for non-RTD, non-linkcheck Sphinx builds (`docs/conf.py:218`; RTD renders without execution). The advisory posture on this dev fork is recorded in `ci-nightly.yml:82-84` as deliberate, with blocking-mode enforcement assigned to the upstream Moffran repo at release time — consistent with ADR-012's advisory-mainline/blocking-release split. The rewritten gaps below reflect what actually remains.

**Gap 1 closed (v0.11.4, 2026-06-17):** `docs-build` job added to `ci-nightly.yml` calling `reusable-build-docs.yml` with `build-target: linkcheck`. Sphinx linkcheck now runs nightly (advisory). `make check-ci-policy` passes.

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Per-example runtime ceiling (<30s) not enforced | 2 | 2 | 4 | The nightly driver enforces cell (30s) and notebook (300s) timeouts in advisory mode only; the ADR's <30s-per-example contribution rule has no blocking gate anywhere. Acceptable for the dev fork per the recorded advisory posture; blocking enforcement is a release-branch obligation. Target milestone: v1.0.0-rc. |

### ADR-013 - Interval Calibrator Plugin Strategy

**Compliance verification (2026-06-11):** Reviewed code and RTD - no ADR-013 gaps found; protocol conformance, FAST chain separation, third-party structural conformance, and doc-path corrections all closed 2026-06-02 and guarded by `tests/unit/plugins/test_adr013_interval_gaps.py`. ADR-013 is fully compliant. No further action required.

### ADR-014 - Visualization Plugin Architecture (superseded; see ADR-037)

**Superseded routing note (2026-03-20):** ADR-014 is superseded by ADR-037. Route builder/renderer governance and extension metadata requirements to ADR-037; route canonical PlotSpec semantics to ADR-036.

### ADR-015 - Explanation Plugin Integration

**Compliance verification (2026-06-17):** Reviewed code — no ADR-015 gaps remain. `ExplainerHandle.learner` property at `plugins/explanations.py:128-146` emits `deprecate("ExplainerHandle.learner is deprecated...", key="plugin:ExplainerHandle.learner", stacklevel=3, raise_on_error=False)` before returning the raw learner (added no later than v0.11.3; `stacklevel` corrected from 2 to 3 on 2026-06-17 so the warning attributes to plugin caller code, not the internal call-site). Active-deprecations ledger row present (`docs/migration/deprecations.md`: `ExplainerHandle.learner property | handle.predict() | v0.11.3 | v1.0.0`). Test `test_should_emit_deprecation_warning_when_learner_accessed` in `tests/plugins/test_explanation_plugins.py:150-159` asserts the `DeprecationWarning` fires on access. ADR-015 gap 2 is closed by the deprecation path. ADR-015 is fully compliant. No further action required.

### ADR-016 - PlotSpec Separation and Schema (superseded; see ADR-036/ADR-037)

**Superseded routing note (2026-03-20):** ADR-016 is superseded by ADR-036 and ADR-037. Route new work to ADR-036 (canonical PlotSpec contract) and ADR-037 (visualization extension governance).

### ADR-020 - Legacy User API Stability

**Compliance verification (2026-06-11):** Reviewed code and RTD - no ADR-020 gaps found; `tests/unit/api/test_legacy_user_api_contract.py` passes 11/11 after the Task-13 guarded-parameter additions (re-run 2026-06-11). Release-checklist gate, parity tests, and CONTRIBUTING guidance closed 2026-06-02. ADR-020 is fully compliant. No further action required.

### ADR-021 - Calibrated Interval Semantics

**Compliance verification (2026-06-11):** Reviewed code and RTD - no ADR-021 gaps found. Code evidence: the inclusive `low <= predict <= high` invariant is enforced with epsilon tolerance in the predict monitor (`plugins/predict_monitor.py:131-182`, epsilon `1e-9`, hard failure on stark breaches — matching the ADR's soft-coercion non-guarantee), at serialization (`serialization.py:83`), and consistently across validators/bridges/task types (`tests/unit/plugins/test_invariant_consistency.py`). ADR-021 is fully compliant. No further action required.

### ADR-022 - Documentation Information Architecture

*Superseded by Standard-004; see Standard-004 for status.*

### ADR-023 - Matplotlib Coverage Exemption

**Compliance verification (2026-06-15):** Reviewed code and `pyproject.toml` — no ADR-023 gaps found. The exemption is correctly scoped: `pyproject.toml` `[tool.coverage.run] omit` now lists only `src/calibrated_explanations/viz/matplotlib_adapter.py`; the four stale entries (`_interval_regressor.py`, `_plots.py`, `_plots_legacy.py`, `_venn_abers.py`) were removed in v0.11.3 Task 15. ADR-023 is fully compliant. No further action required.

### ADR-024 - Legacy Plot Input Contracts

*Retired / maintenance reference: `docs/maintenance/legacy-plotting-reference.md`.*

### ADR-025 - Legacy Plot Rendering Semantics

*Retired / maintenance reference: `docs/maintenance/legacy-plotting-reference.md`.*

### ADR-026 - Explanation Plugin Semantics

**Compliance verification (2026-06-15):** Reviewed code — no ADR-026 gaps found. The 2026-06-11 appendix entry stating that `helper_handles`, `feature_names`, and `categorical_features` are unfrozen was stale: `ExplanationContext.__post_init__` at `plugins/explanations.py:71-73` freezes all three via `_freeze_value()` with `object.__setattr__`. `IntervalCalibratorContext.__post_init__` at `plugins/intervals.py:29-54` already freezes `calibration_splits` (to tuple), `bins`, `residuals`, `difficulty`, and `fast_flags` (all via `MappingProxyType`). All context fields requiring immutability are frozen. ADR-026 is fully compliant. No further action required.

### ADR-027 - FAST-Based Feature Filtering

**Compliance verification (2026-06-11):** Reviewed code and RTD - no ADR-027 gaps found; observability-policy and telemetry documentation closed 2026-06-02 in `docs/practitioner/performance-tuning.md`. ADR-027 is fully compliant. No further action required.

### ADR-028 - Logging and Governance Observability

**Compliance verification (2026-06-15):** Reviewed code and RTD — no ADR-028 gaps found. `check_warning_policy.py` reports 114 call sites with 0 UNCLASSIFIED (re-run 2026-06-15): the `api/config.py:265` site was classified in v0.11.3 Task 15; two `calibrated_explainer.py` `guarded=True` sites migrated from raw `warnings.warn` to `deprecate()` in post-Task-17 fix, reducing the raw count by 2. ADR-028 is fully compliant. No further action required.

### ADR-029 - Reject Integration Strategy

_Last gap analysis: 2026-06-11_

Re-verified 2026-06-11 against code: `RejectPolicy` canonical members in `core/reject/policy.py`, `RejectResult` envelope with `error_rate_defined` sentinel and per-instance metadata keys, `RejectOrchestrator` registry with `builtin.default` (`core/reject/orchestrator.py:902-925`, `resolve_strategy` at 2079), and the `WrapCalibratedExplainer.__init__` constructor constraint (accepts only `learner`) all satisfied.

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Strategy lifecycle hooks and configuration surface not finalized | 2 | 2 | 4 | **Deferred to post-v1.0 (2026-06-02).** Implementing a full strategy lifecycle and config surface is a design/architecture work item beyond bounded v0.11.3 RC hardening. `RejectResult` remains the stable v1.0.0 return type; full strategy config API is deferred to a new post-v1.0 ADR-011 deprecation cycle (v1.1+ planning). |
| 2 | `RejectResult` public return type not yet migrated to strict `RejectResultV2` | 2 | 2 | 4 | Group L reset-path closure landed in v0.11.3: removed active deprecation warning from `reject_result_v2_to_legacy()` and kept `RejectResult` as stable v1.0.0 return type. Optional `RejectResultV2` remains available; full public return-type migration is deferred to a new post-v1.0 ADR-011 deprecation cycle (v1.1+ planning). |

### ADR-030 - Test Quality Priorities and Enforcement

**Compliance verification (2026-06-11):** Reviewed code and RTD - no ADR-030 gaps found; marker-hygiene and mutation-policy ratifications closed v0.11.3 Task 3 (2026-05-12); `check_marker_hygiene.py` reports 0 findings (re-run 2026-06-11). ADR-030 is fully compliant. No further action required.

### ADR-031 - Calibrator Serialization and State Persistence

**Compliance verification (2026-06-11):** Reviewed code - no ADR-031 gaps found. Code evidence: `to_primitive`/`from_primitive` implemented on both calibrator families (`calibration/interval_regressor.py`, `calibration/venn_abers.py`) and `save_state`/`load_state` plus related persistence surfaces on `core/wrap_explainer.py` (15 references). Delivered in v0.11.0 and unchanged by Tasks 11-13. ADR-031 is fully compliant. No further action required.

### ADR-032 - Guarded Explanation Semantics

**Compliance verification (2026-06-15):** Reviewed code and RTD — no ADR-032 gaps found. Re-verified: `get_guarded_audit` error message (`explanations/explanations.py:236-238`) now correctly recommends `explain_factual(..., guarded_options=GuardedOptions())` / `explore_alternatives(...)` — the canonical ADR-032 decision-1 API. All other decisions verified: parameterized guarded API, `GuardedFactualExplanation`/`GuardedAlternativeExplanation` subclasses, `supports_guarded` resolver hard-fail, fast-explainer prohibition, conjunction joint-probe guarding, and `filter_by_target_confidence`. ADR-032 is fully compliant. No further action required.

### ADR-033 - Modality Extension Plugin Contract and Packaging Strategy

_Last gap analysis: 2026-06-17_

Re-verified 2026-06-11: CLI `--modality` filtering, `vision.py`/`audio.py` shims with `MissingExtensionError`, packaging smoke test, contributor docs, and the v0.11.1 entry-point `DeprecationWarning` for missing `data_modalities` (`plugins/registry.py:1830-1845`) are all delivered. Per ADR-033 §6.2 the shim modules are permanent integration paths (the earlier appendix phrase "shim removal" was incorrect).

**Compliance verification (2026-06-17):** Gap 1 closed in v0.11.4. `validate_plugin_meta` (`plugins/base.py:356`) now raises `ValidationError` when `data_modalities` is absent — the `("tabular",)` default-fallback has been removed. All first-party CE plugins already declare `data_modalities` explicitly. Entry-point path continues to emit `UserWarning`-and-skip before calling `validate_plugin_meta` (registry.py:1830-1840). Breaking-change note in `docs/upgrade/v0.11.4-upgrade-checklist.md`. No remaining open gaps.

### ADR-034 - Centralized Configuration Management

_Last gap analysis: 2026-06-15_

Runtime conformance re-verified 2026-06-15: `check_config_manager_usage.py` reports zero violations; v0.11.3 root-namespace exports and process lifecycle API present. Status-source conflict (former gap 3) resolved: ADR-034 §Open Items updated 2026-06-13 — both items formally declared post-v1.0 with rationale.

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Sensitive-value redaction for wider governance logs/exports (Open Item 1) | 3 | 2 | 6 | **Post-v1.0 deferred (2026-06-13).** Config diagnostics redact secret-like keys per §8; wider governance-log redaction declared out of scope for v1.0.0-rc — CE_ env vars are behavioral flags, not secrets. Documented in ADR-034 §Open Items. Target milestone: post-v1.0. |
| 2 | `export_effective()` full compatibility-frozen schema contract (Open Item 2) | 3 | 2 | 6 | **Post-v1.0 deferred (2026-06-13).** `export_effective()` carries `config_schema_version` markers; full compatibility-frozen contract with formal version-gating and stability guarantee is a post-v1.0 item. Documented in ADR-034 §Open Items. Target milestone: post-v1.0. |

### ADR-035 - CI Workflow Governance

_Last gap analysis: 2026-06-11_

Evidence (2026-06-11): `python scripts/quality/validate_ci_policy.py --base-sha HEAD~1 --head-sha HEAD --advisory` passes (includes full-SHA action-pin enforcement and reusable-workflow checks; wired as `make check-ci-policy`).

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Advisory-to-required branch-protection flip is not enforceable from repository code alone | 2 | 2 | 4 | **v0.11.3 re-evaluation complete (2026-06-02).** Promotion of `ci-policy/validate-workflows` to a required branch-protection check is platform-governed (requires GitHub administrator action). Validator logic and CODEOWNERS coverage are complete. Recorded as an accepted operational constraint in ADR-035 §v0.11.3 Re-evaluation Record. Pending administrator action; no in-repo work remains. Retarget: monitor / platform action by administrators. |
| 2 | Two governance caveats remain structurally non-automatable (template auto-application and two-maintainer per-path enforcement) | 1 | 2 | 2 | These are documented governance caveats rather than code defects. Track as accepted operational constraints unless platform capabilities change. |
| 3 | Dependency-constraint changes (`requirements.txt`/`constraints.txt`) are not validated against the nightly parity-reference harness before merge | 3 | 2 | 6 | **v0.11.4 Task 7 incident (found and fixed 2026-06-16).** Commit `9693c3d8` (2026-06-12) removed the `scikit-learn==1.6.1`/`1.5.2` exact pins from `constraints.txt`, widening to an unbounded `scikit-learn>=1.3` floor. That commit's own validation (`pytest -q` + manual version checks) never exercised `tests/parity_reference/run_parity_reference.py`, which is wired only into `ci-nightly.yml` and not into `pytest -q` or any PR-time gate. The result: fresh installs resolved scikit-learn ≥1.8.0, which changed `DecisionTreeRegressor` split-selection behavior (confirmed by bisection: 1.6.1–1.7.2 pass, 1.8.0+ fail) and broke the `regression`/`probabilistic_regression` parity fixtures for 4 nightly runs before being caught and diagnosed. **Fix landed:** `tests/parity_reference/constraints.txt` overlay pins `scikit-learn<1.8` scoped to the `parity-reference` nightly job only (does not narrow the project-wide floor); see `tests/parity_reference/README.md` for the bisection evidence and update procedure, and `docs/improvement/v0.11.4_plan.md` Task 7 for the full investigation record. **Gap remains open:** no PR-time gate exists to catch a future `requirements.txt`/`constraints.txt` change before it ships to nightly; closing that gap (e.g. a parity-harness run conditional on those paths changing) is left as future work, not bundled into the Task 7 fix. |

### ADR-036 - PlotSpec Canonical Contract and Validation Boundary

**Compliance verification (2026-06-15):** Reviewed code and RTD — no ADR-036 gaps found. §5 validation boundary implemented in v0.11.3 Task 15: `validate_plot_artifact()` (`plotting.py:308-331`, public, ADR-036 §5 documented in docstring) is called at both build/render boundaries — `plotting.py:387` (instance path) and `plotting.py:439` (collection path). Artifacts that fail `validate_plotspec` raise `ValidationError` before renderer invocation; non-PlotSpec artifacts pass through unmodified. ADR-036 is fully compliant. No further action required.

### ADR-037 - Visualization Extension and Rendering Governance

**Compliance verification (2026-06-15):** Reviewed code and RTD — no ADR-037 gaps found. §4 extension metadata implemented in v0.11.3 Task 15: `validate_plugin_meta` (`plugins/base.py:381-403`) validates `plot_kinds` and `plot_modes` against allowed values; all four built-in descriptors declare both fields (`builtins.py:1143-1144`, `1190-1191`, `1232-1233`, `1545-1546`). Runtime plot-kind extension prohibition holds. ADR-037 is fully compliant. No further action required.

### ADR-038 - Call-time Configuration Taxonomy and Naming Conventions

_Last gap analysis: 2026-06-15_

ADR-038 accepted 2026-06-12; v0.11.3 Task 17 delivered `GuardedOptions`, `reject_confidence`, canonical summary table, and `[EXPERIMENTAL]` markers on `**kwargs` surfaces. Core taxonomy surfaces verified: `GuardedOptions` in root namespace (`__init__.py`), `reject_confidence=` qualified kwarg on `predict_reject`/`apply_policy`, `RejectPolicySpec` unchanged, four-tier taxonomy documented in ADR-038 §1. Active deprecations for replaced kwargs (`guarded=True`, `significance=`, `n_neighbors=`, `normalize_guard=`, `merge_adjacent=`, `confidence=` on reject path) all filed in `docs/migration/deprecations.md` Active table with v1.0.0 removal ETA.

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | `**kwargs` graduation gate on `explain_factual` / `explore_alternatives` | 3 | 2 | 6 | `CalibratedExplainer.explain_factual` and `explore_alternatives` (and their `WrapCalibratedExplainer` thin delegators) carry `**kwargs` marked `[EXPERIMENTAL]` per ADR-038 §3. ADR-038 §3 graduation gate: these must be replaced with explicit typed arguments (`multi_labels_enabled`, `interval_summary`, and any future tuning params promoted to `*Options`) before the methods exit experimental status. Noted at `RELEASE_PLAN_v1.md:746-751`. Target milestone: v1.0.0-rc. |
| 2 | Plugin §5 taxonomy compliance not enforced by `validate_plugin_meta` | 2 | 2 | 4 | ADR-038 §5 requires third-party plugin authors to follow `*Spec`/`*Options`/`*Config` taxonomy for any configuration surfaces they expose through the plugin contract. `validate_plugin_meta` (`plugins/base.py:306-403`) does not check for taxonomy conformance — there is no enforcement gate. Non-compliant plugin config surfaces will not be detected at registration time. Target milestone: v1.0.0-rc. |
| 3 | Warning-policy allowlist comment for `calibrated_explainer.py` is narrower than the file's covered categories | 1 | 1 | 1 | The `core/calibrated_explainer.py` entry in `check_warning_policy.py` comment reads "verbose-mode migration warning" but the file now also contains the `guarded=True` `UserWarning` (`_use_plugin` path) and the `deprecate(raise_on_error=False)` call for the `guarded=True` deprecation (post-Task-17 fix 2026-06-15). `raise_on_error=False` is correct for this user-facing kwarg so tests using `pytest.warns(DeprecationWarning)` work correctly under `CE_DEPRECATIONS=error`. Update the allowlist comment to reflect all covered categories. Doc/config only. Target: next gap sweep. |

## Standards status appendix (unified severity tables)

### Standard-001 - Nomenclature Standardization

**Compliance verification (2026-06-11):** `check_std001_nomenclature.py` passes (re-run 2026-06-11); compatibility-bridge closure from v0.11.3 Task 1 (2026-06-02) holds. STD-001 is fully compliant. No further action required. (Parameter-naming hardening is tracked separately as v0.11.3 plan Task 14.)

### Standard-002 - Documentation Standardisation

**Compliance verification (2026-06-11):** `check_docstring_coverage.py` reports 96.34% overall (Modules 99.17%, Classes 100%, Functions 94.82%, Methods 96.36%) — above the 90% threshold after Tasks 11-13 additions. Task 2 closure (2026-06-02) holds. STD-002 is fully compliant. No further action required.

### Standard-003 - Test Coverage Standard

**Compliance verification (2026-02-27):** Reviewed code and RTD - no STD-003 gaps found; STD-003 is fully compliant. No further action required.

### Standard-004 - Documentation Standard (Audience Hubs)

**Compliance verification (2026-02-27):** Reviewed code and RTD - no STD-004 gaps found; STD-004 is fully compliant. No further action required.

### Standard-005 - Logging and Observability Standard

**Compliance verification (2026-06-15):** Reviewed code and RTD — no STD-005 gaps found. `check_logging_domains.py` passes; `check_warning_policy.py` reports 114 sites, 0 unclassified (shares ADR-028 closure). STD-005 is fully compliant. No further action required.
