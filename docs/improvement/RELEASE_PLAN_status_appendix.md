# RELEASE_PLAN Status Appendix

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
| ADR-004 | Partially complete | Only explicit `strategy="auto"` policy closure remains (v1.0.0-rc) |
| ADR-005 | Completed (2026-06-11) | Last gap closed on code evidence: provenance propagates through both adapters (`explanations/adapters.py:36-41`, `:68`); `schema.validate_payload` and serialization invariants intact post-Task-1 |
| ADR-006 | Completed | All v0.11.3 gaps closed: gap 3 superseded (Task 5 Group K); gap 2 closed 2026-06-02 (accepted-registration audit events added to all 4 typed registration functions); gap 1 carry-forward (monitor, no code gap) |
| ADR-008 | Partially complete | v0.11.3 golden fixture tests and `_safe_pick` observability closed (gaps 4/5, 2026-06-02); stale coverage omit entries confirmed clean (pyproject.toml:144-153 checked 2026-06-13; no stale paths); domain-authoritative migration (gaps 1/2/3) remains v1.0.0-rc |
| ADR-009 | Partially complete | JSON-safe export closed (gap 2, 2026-06-02); helper-placement gap 3 closed (Workstream B, 2026-06-02); wrapper/core surface decision in v1.0.0-rc |
| ADR-010 | Completed | Core-only vs extras parity automation closed v0.11.3 (gap 1 closed 2026-06-02; `scripts/quality/check_core_extras_parity.py` added) |
| ADR-011 | Closed (2026-06-13) | All gaps resolved: (1) guarded-wrapper deprecations removed in Task 13 (not merely deferred); 9 Task-17 active surfaces target v1.0.0 — within ADR-011 §2 final-cycle allowance; (2) raw `warnings.warn(DeprecationWarning)` bypass sites fixed — `normalization_strategy.py`, `core/reject.py`, `core/explain/__init__.py` all use `deprecate()` helper; (3) active-deprecations ledger rebuilt with all surfaces; `make deprecation-closure` passes with 9 v1.0.0 rows permitted and 0 blocking. `data_modalities` deprecation closed fail-closed 2026-06-13. |
| ADR-012 | Accepted with open hardening (re-evidenced 2026-06-11) | Notebook execution exists (nightly advisory driver + `nbsphinx_execute="always"` on non-RTD builds); real gap is that the docs HTML/linkcheck CI job is unwired (`reusable-build-docs.yml` has no caller). Runtime ceilings advisory-only. Remains v1.0.0-rc |
| ADR-013 | Completed | All v0.11.3 gaps closed (gaps 1/2/3/4 closed 2026-06-02; protocol tests, FAST chain separation guard, third-party conformance test, doc path corrected) |
| ADR-015 | Partially complete | v0.11.3 gaps 1/3 closed (2026-06-02; invariant consistency + task-type parity tests); gap 2 (direct learner bypass) deferred to v1.0.0-rc |
| ADR-020 | Completed | All v0.11.3 gaps closed (gap 1: release checklist 2026-06-02; gap 2: wrapper parity tests 2026-06-02; gap 3: CONTRIBUTING.md 2026-06-02) |
| ADR-026 | Partially complete (re-scoped 2026-06-11) | Context immutability substantially landed (frozen dataclasses + nested freezing in `plugins/explanations.py:35-69`, `plugins/intervals.py:12-43`); residual unfrozen nested fields remain (v1.0.0-rc); telemetry quick wins closed in v0.11.2 |
| ADR-027 | Completed | All gaps closed v0.11.3 (gaps 1/2 closed 2026-06-02; `docs/practitioner/performance-tuning.md` covers observability policy and telemetry examples) |
| ADR-028 | Closed (2026-06-13) | Warning-policy regression fixed in Task 15; `make warning-policy` now reports 0 unclassified sites (verified 2026-06-13: 116 sites — 5 deprecation-helper, 111 allowlisted, 0 unclassified). |
| ADR-030 | Completed | Zero-tolerance ratification closed v0.11.3 Task 3 (2026-05-12); marker hygiene taxonomy and mutation policy sections added to ADR-030; gaps 1/2 closed in appendix (2026-06-02) |
| ADR-032 | Closed (2026-06-13) | All decisions verified. `get_guarded_audit` error message corrected in Task 15: `explanations/explanations.py:236-238` now recommends `explain_factual(..., guarded_options=GuardedOptions())` / `explore_alternatives(...)` — canonical API per ADR-032 decision 1. Deprecated wrappers removed. |
| ADR-033 | Closed (2026-06-13) | All obligations met. `data_modalities` enforcement closed early in v0.11.3: plugin missing the key now emits `UserWarning` and is skipped (fail-closed). `DeprecationWarning`+default-fallback path removed from `plugins/registry.py`. ADR-033 §6.2 shims are permanent. |
| ADR-034 | Accepted with deferred v1.0 open items (reconciled 2026-06-13) | Runtime conformance closure complete in v0.11.2. Status-source conflict resolved: ADR-034 §Open Items now documents "Status: Declared out of scope for v1.0.0-rc" for both redaction and export schema contract items. No RC deferrals remain for this ADR. |
| ADR-035 | Accepted with accepted operational constraint | v0.11.3 re-evaluation complete (2026-06-02): advisory-to-required branch-protection flip is platform-governed; recorded as accepted operational constraint in ADR-035 §v0.11.3 Re-evaluation Record; no in-repo work remains |
| ADR-036 | Closed (2026-06-13) | §5 validation boundary implemented in Task 15: `validate_plot_artifact()` (public, `plotting.py:308`) called at both build/render boundary points (`plotting.py:387`, `:439`). Artifacts that fail `validate_plotspec` raise `ValidationError` before renderer invocation. 3 dedicated boundary tests in `test_plot_plugin_validation_boundary.py` pass. |
| ADR-037 | Closed (2026-06-13) | §4 extension metadata implemented in Task 15: `validate_plugin_meta` (via `plugins/base.py:381-403`) validates `plot_kinds` against allowed values and `plot_modes` against allowed values; both default when absent. 11 tests in `test_plot_extension_metadata.py` pass. |
| STD-001 | Completed | All v0.11.3 bridges closed (Task 1, 2026-06-02); 0 expired remove_by_v0.11.3 records; checker passes; internal bridge dunders renamed; APPROVED_COMPATIBILITY_BRIDGES = {} |
| STD-002 | Completed | WrapCalibratedExplainer numpydoc gap closed in v0.11.3 Task 2; coverage 96.73%, zero pydocstyle violations (2026-06-02) |
| STD-003 | Completed | Monitor for regressions |
| STD-004 | Completed | Monitor for regressions |
| STD-005 | Closed (2026-06-13) | Shares ADR-028 closure: 0 unclassified `warnings.warn` sites (verified 2026-06-13). |

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

_Last gap analysis: 2026-06-11_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Implicit `auto` strategy enables environment-dependent backend selection contrary to ADR decision | 3 | 3 | 9 | `ParallelConfig(strategy="auto")` is the default; when `enabled=True` the 5-step `auto_strategy()` heuristic (`parallel/parallel.py:517-589`) selects `sequential` / `threads` / `processes` / `joblib` based on CPU count, CI detection, task size, workload count, joblib availability, and OS. A caller who sets `enabled=True` without an explicit strategy gets a silently environment-dependent backend — non-deterministic across machines and in violation of ADR-004 §Decision "no automatic strategy selection." Fix: deprecate `strategy="auto"` with an ADR-011 `DeprecationWarning` when `enabled=True AND strategy resolves to "auto"`; require explicit strategy for v1.0.0. Target milestone: v1.0.0-rc. |

### ADR-005 - Explanation Payload Schema

**Compliance verification (2026-06-11):** Reviewed code and RTD - no ADR-005 gaps remain. The former gap 1 (provenance propagation in legacy adapters) is closed on code evidence: `legacy_to_domain` reads and preserves `provenance` (`explanations/adapters.py:36-41`) and `domain_to_legacy` emits it (`explanations/adapters.py:68`). The canonical validator survives the Task-1 alias removal (`schema/validation.py:42` `validate_payload`; the removed symbol was only the deprecated `serialization.validate_payload` alias), `explanation_schema_v1.json` is present, and interval invariants are enforced at serialization (`serialization.py:83` `_validate_invariants`). ADR-005 is fully compliant. No further action required. (Deeper domain-model authority work remains tracked under ADR-008.)

### ADR-006 - Plugin Trust Model

**Compliance verification (2026-06-11):** Reviewed code and RTD - no open ADR-006 gaps. The four `_TRUSTED_*` module-level sets in `plugins/registry.py` are the sole authoritative trust-tracking infrastructure (the v0.11.1 dual-state issue was eliminated when Task 5 Group K removed the list-path API); `check_trust_mutation_primitive.py` passes (2026-06-11). Accepted-registration governance events and list-path removal closed 2026-06-02. Monitor for regressions; no further action required.

### ADR-007 - PlotSpec Abstraction (superseded; see ADR-036/ADR-037)

**Superseded routing note (2026-03-20):** ADR-007 is superseded by ADR-036 and ADR-037. Route canonical PlotSpec contract and validation questions to ADR-036, and visualization extension/rendering governance questions to ADR-037.

### ADR-008 - Explanation Domain Model

_Last gap analysis: 2026-06-11_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Domain model not authoritative source | 5 | 4 | 20 | Re-verified 2026-06-11: `legacy_to_domain`/`domain_to_legacy` are referenced only inside the `explanations` package (adapters + serialization boundary); core explain/predict workflows still operate on legacy dicts. Target milestone: v1.0.0-rc (major refactor; architecture-heavy). |
| 2 | Legacy->domain round-trip fails for conjunctive rules | 4 | 3 | 12 | Re-verified 2026-06-11: `domain_to_legacy` accumulates `rule_features: List[int]` and appends scalar `fr.feature` (`explanations/adapters.py:74`, `:82`), so multi-feature conjunctions cannot round-trip. Target milestone: v1.0.0-rc (follows domain-model authority work). |
| 3 | Structured model/calibration metadata missing | 4 | 3 | 12 | Explanation dataclass lacks dedicated calibration/model descriptor fields (generic `provenance`/`metadata` mappings only). Target milestone: v1.0.0-rc. |

### ADR-009 - Input Preprocessing and Mapping Policy

_Last gap analysis: 2026-06-11_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Mapping export helpers placed on wrapper not explainer | 2 | 3 | 6 | Re-verified 2026-06-11: `export_preprocessor_mapping`/`import_preprocessor_mapping` are defined only on `core/wrap_explainer.py`; no thin `CalibratedExplainer` adapters exist. Target milestone: v1.0.0-rc (public API change). |

### ADR-010 - Optional Dependency Split

**Compliance verification (2026-06-11):** Reviewed code and RTD - no ADR-010 gaps found; `scripts/quality/check_core_extras_parity.py` passes (re-run 2026-06-11). ADR-010 is fully compliant. No further action required.

### ADR-011 - Deprecation and Migration Policy

_Last gap analysis: 2026-06-11_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Task-13 guarded-wrapper deprecations scheduled to survive into v1.0.0, conflicting with ADR-011 §2 binding rule | 3 | 2 | 6 | `explain_guarded_factual` / `explore_guarded_alternatives` (`CalibratedExplainer` + `WrapCalibratedExplainer`, 4 surfaces) were deprecated in v0.11.3 (Task 13, 2026-06-09) with removal targeted "v1.0.0" (ADR-032 decision 1; `docs/migration/deprecations.md:247-250`). ADR-011 §2: "v1.0.0 must ship with zero surviving deprecations" and "No deprecation-removal work may be deferred to v1.0.0". Resolution options: schedule removal inside v0.11.x, or apply the Group L deprecation-reset precedent. Target: v0.11.3 plan Task 15. |
| 2 | Active-deprecations ledger empty / rows mis-filed while active deprecations exist in code | 3 | 2 | 6 | `docs/migration/deprecations.md` "Active deprecations" table has zero rows, yet code actively emits deprecations for: guarded wrappers ×4 (mis-filed in "Removed deprecations (history)" although still present and functional), `ce_agent_utils.narrative_format`, `normalize=True/False` (`calibration/normalization_strategy.py:87-100`), `calibrated_explanations.core.reject` module path (`core/reject.py:32-38`), `core.explain.explain` (`core/explain/__init__.py:25-34`), and entry-point `data_modalities` (`plugins/registry.py:1830-1845`, ADR-033-governed). The history-table header claims listed symbols "have been deleted", which is false for the guarded wrappers. Target: v0.11.3 plan Task 15. |
| 3 | Raw `warnings.warn(DeprecationWarning)` sites bypass the central `deprecate()` helper | 3 | 2 | 6 | ADR-011 §1 mandates emission "via the central `deprecate(msg, *, once_key)` helper". Non-compliant sites: `calibration/normalization_strategy.py:87-100` (no dedup, vague "future release" timeline), `core/reject.py:32-38`, `core/explain/__init__.py:25-34`. (`plugins/registry.py:1835` has manual dedup and an explicit deadline; lowest priority.) Also: two near-duplicate helper modules `utils/deprecations.py` and `utils/deprecation.py` invite future drift. Target: v0.11.3 plan Task 15. |

### ADR-012 - Documentation & Gallery Build Policy

_Last gap analysis: 2026-06-11_

Evidence update (2026-06-11): the former gap framing ("docs build disables notebook execution") is outdated. Notebook execution now exists in two forms: the nightly advisory driver (`ci-nightly.yml` `notebook-exec-report` job runs `scripts/docs/run_notebooks.py --mode advisory --cell-timeout 30 --notebook-timeout 300`) and `nbsphinx_execute = "always"` for non-RTD, non-linkcheck Sphinx builds (`docs/conf.py:218`; RTD renders without execution). The advisory posture on this dev fork is recorded in `ci-nightly.yml:82-84` as deliberate, with blocking-mode enforcement assigned to the upstream Moffran repo at release time — consistent with ADR-012's advisory-mainline/blocking-release split. The rewritten gaps below reflect what actually remains.

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Docs HTML/linkcheck CI job is unwired on PR/main | 3 | 2 | 6 | ADR-012 requires `sphinx-build -W` and linkcheck to run in CI (advisory on mainline). `reusable-build-docs.yml` exists but **no workflow calls it** — the only Sphinx build in CI is the manual-dispatch `maintenance.yml` regen-docs task. Doc rot is currently undetected on PR/main. Wire the reusable builder into `ci-pr.yml` or `ci-nightly.yml` (advisory). Target milestone: v1.0.0-rc. |
| 2 | Per-example runtime ceiling (<30s) not enforced | 2 | 2 | 4 | The nightly driver enforces cell (30s) and notebook (300s) timeouts in advisory mode only; the ADR's <30s-per-example contribution rule has no blocking gate anywhere. Acceptable for the dev fork per the recorded advisory posture; blocking enforcement is a release-branch obligation. Target milestone: v1.0.0-rc. |

### ADR-013 - Interval Calibrator Plugin Strategy

**Compliance verification (2026-06-11):** Reviewed code and RTD - no ADR-013 gaps found; protocol conformance, FAST chain separation, third-party structural conformance, and doc-path corrections all closed 2026-06-02 and guarded by `tests/unit/plugins/test_adr013_interval_gaps.py`. ADR-013 is fully compliant. No further action required.

### ADR-014 - Visualization Plugin Architecture (superseded; see ADR-037)

**Superseded routing note (2026-03-20):** ADR-014 is superseded by ADR-037. Route builder/renderer governance and extension metadata requirements to ADR-037; route canonical PlotSpec semantics to ADR-036.

### ADR-015 - Explanation Plugin Integration

_Last gap analysis: 2026-06-11_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Explainer handle exposes direct `learner` (bypass bridge) | 4 | 2 | 8 | Re-verified 2026-06-11: `ExplainerHandle.learner` property returns the raw underlying learner (`plugins/explanations.py:124-126`), letting plugins bypass the `PredictBridge` invariant enforcement. Restrict direct learner access or document as escape hatch with warnings. Target milestone: v1.0.0-rc (public API semantic). |

### ADR-016 - PlotSpec Separation and Schema (superseded; see ADR-036/ADR-037)

**Superseded routing note (2026-03-20):** ADR-016 is superseded by ADR-036 and ADR-037. Route new work to ADR-036 (canonical PlotSpec contract) and ADR-037 (visualization extension governance).

### ADR-020 - Legacy User API Stability

**Compliance verification (2026-06-11):** Reviewed code and RTD - no ADR-020 gaps found; `tests/unit/api/test_legacy_user_api_contract.py` passes 11/11 after the Task-13 guarded-parameter additions (re-run 2026-06-11). Release-checklist gate, parity tests, and CONTRIBUTING guidance closed 2026-06-02. ADR-020 is fully compliant. No further action required.

### ADR-021 - Calibrated Interval Semantics

**Compliance verification (2026-06-11):** Reviewed code and RTD - no ADR-021 gaps found. Code evidence: the inclusive `low <= predict <= high` invariant is enforced with epsilon tolerance in the predict monitor (`plugins/predict_monitor.py:131-182`, epsilon `1e-9`, hard failure on stark breaches — matching the ADR's soft-coercion non-guarantee), at serialization (`serialization.py:83`), and consistently across validators/bridges/task types (`tests/unit/plugins/test_invariant_consistency.py`). ADR-021 is fully compliant. No further action required.

### ADR-022 - Documentation Information Architecture

*Superseded by Standard-004; see Standard-004 for status.*

### ADR-023 - Matplotlib Coverage Exemption

_Last gap analysis: 2026-06-11_

The exemption itself is correctly in place: `src/calibrated_explanations/viz/matplotlib_adapter.py` is omitted from coverage (`pyproject.toml:152`).

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Stale coverage omit entries reference deleted files | 1 | 1 | 1 | `pyproject.toml:153-156` still omits `_interval_regressor.py`, `_plots.py`, `_plots_legacy.py`, `_venn_abers.py` — none of these files exist anymore. No coverage effect today, but the dead entries misdocument the exemption surface and would silently mask any future file recreated under those names. Delete the four entries. Doc/config-only; suitable for v0.11.3 plan Task 15. |

### ADR-024 - Legacy Plot Input Contracts

*Retired / maintenance reference: `docs/maintenance/legacy-plotting-reference.md`.*

### ADR-025 - Legacy Plot Rendering Semantics

*Retired / maintenance reference: `docs/maintenance/legacy-plotting-reference.md`.*

### ADR-026 - Explanation Plugin Semantics

_Last gap analysis: 2026-06-11_

Evidence update (2026-06-11): context immutability has substantially landed since this gap was scored 12. `ExplanationContext` is a frozen dataclass whose `__post_init__` deep-freezes `categorical_labels`, `interval_settings`, `plot_settings`, and `plugin_config` (`plugins/explanations.py:35-69`); `IntervalCalibratorContext` is frozen with proxied `metadata`/`plugin_config` and a deliberately mutable `plugin_state` scratch area (`plugins/intervals.py:12-43`). The residual gap below is what remains.

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Specific nested context fields remain unfrozen | 3 | 2 | 6 | `ExplanationContext.__post_init__` does not freeze `helper_handles`, `feature_names`, or `categorical_features`; `IntervalCalibratorContext.__post_init__` does not freeze `calibration_splits`, `bins`, `residuals`, `difficulty`, or `fast_flags` — plain dicts/lists passed for these fields stay plugin-mutable. (Array payloads may be deliberate; mappings like `helper_handles`/`bins`/`fast_flags` are not.) Target milestone: v1.0.0-rc. |

### ADR-027 - FAST-Based Feature Filtering

**Compliance verification (2026-06-11):** Reviewed code and RTD - no ADR-027 gaps found; observability-policy and telemetry documentation closed 2026-06-02 in `docs/practitioner/performance-tuning.md`. ADR-027 is fully compliant. No further action required.

### ADR-028 - Logging and Governance Observability

_Last gap analysis: 2026-06-11_

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Warning-policy inventory regressed: 1 unclassified `warnings.warn` site | 2 | 1 | 2 | `python scripts/quality/check_warning_policy.py` (run 2026-06-11) reports 112 call sites with 1 UNCLASSIFIED: `api/config.py:265` — removed-field `UserWarning` added by Task 12 (`ExplainerConfig.task`/`parallel_workers` removal notice). Classify in the allowlist (or migrate to `deprecate()` if treated as a deprecation surface) to restore the zero-unclassified closure state. Target: v0.11.3 plan Task 15. |

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

_Last gap analysis: 2026-06-11_

Re-verified 2026-06-11 against code: parameterized guarded API with deprecated wrappers delegating via `deprecate()`, `GuardedFactualExplanation`/`GuardedAlternativeExplanation` subclasses, single `supports_guarded` plugin metadata field with resolver hard-fail (`plugins/registry.py:1156-1199`), fast-explainer prohibition in `_require_guarded_calibration_alignment`, conjunction joint-probe guarding, and `AlternativeExplanations.filter_by_target_confidence` (Addendum, 2026-06-10) all implemented.

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | `get_guarded_audit` error message recommends the deprecated wrapper methods | 2 | 1 | 2 | `explanations/explanations.py:236-238` raises `ValidationError` advising "Use explain_guarded_factual(...) or explore_guarded_alternatives(...)" — both deprecated by ADR-032 decision 1, which names `explain_factual(..., guarded=True)` / `explore_alternatives(..., guarded=True)` as the canonical API. Update the message text. Target: v0.11.3 plan Task 15. |

### ADR-033 - Modality Extension Plugin Contract and Packaging Strategy

_Last gap analysis: 2026-06-11_

Re-verified 2026-06-11: CLI `--modality` filtering, `vision.py`/`audio.py` shims with `MissingExtensionError`, packaging smoke test, contributor docs, and the v0.11.1 entry-point `DeprecationWarning` for missing `data_modalities` (`plugins/registry.py:1830-1845`) are all delivered. Per ADR-033 §6.2 the shim modules are permanent integration paths (the earlier appendix phrase "shim removal" was incorrect).

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | `data_modalities` default-fallback removal scheduled for v0.12.0/v1.0.0-rc not yet executed | 2 | 2 | 4 | ADR-033 §6.2: at v0.12.0/v1.0.0-rc, plugins without an explicit `data_modalities` declaration must fail `validate_plugin_meta` (`plugins/base.py:352` currently applies the `("tabular",)` default). On schedule — not yet due. Target milestone: v1.0.0-rc. |

### ADR-034 - Centralized Configuration Management

_Last gap analysis: 2026-06-11_

Runtime conformance re-verified 2026-06-11: `check_config_manager_usage.py` reports zero violations (`reports/config_manager_usage_report.json`, generated 2026-06-11); v0.11.3 root-namespace exports (`ExplainerBuilder`/`ExplainerConfig`) and process lifecycle API present.

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Sensitive-value redaction for governance logs/exports (Open Item 1) | 3 | 2 | 6 | Interim posture remains documented as non-redacted (config diagnostics redact secret-like keys per §8; wider governance logs do not). Target milestone: v1.0.0-rc. |
| 2 | `export_effective()` payload schema not versioned for external consumers (Open Item 2) | 3 | 2 | 6 | Target milestone: v1.0.0-rc; must be versioned before external tooling can rely on export. |
| 3 | Status-source conflict between v0.11.3 plan and ADR-034 Open Items | 1 | 1 | 1 | The v0.11.3 plan RC-scope note (2026-05-28) states "sensitive-value redaction declared out of scope; schema versioning already complete", but ADR-034 §Open Items (last edited 2026-06-04, post-dating the note) still lists both as open. Reconcile: either close the ADR Open Items with the declared decisions or correct the plan note. Target: v0.11.3 plan Task 15 (doc-only). |

### ADR-035 - CI Workflow Governance

_Last gap analysis: 2026-06-11_

Evidence (2026-06-11): `python scripts/quality/validate_ci_policy.py --base-sha HEAD~1 --head-sha HEAD --advisory` passes (includes full-SHA action-pin enforcement and reusable-workflow checks; wired as `make check-ci-policy`).

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Advisory-to-required branch-protection flip is not enforceable from repository code alone | 2 | 2 | 4 | **v0.11.3 re-evaluation complete (2026-06-02).** Promotion of `ci-policy/validate-workflows` to a required branch-protection check is platform-governed (requires GitHub administrator action). Validator logic and CODEOWNERS coverage are complete. Recorded as an accepted operational constraint in ADR-035 §v0.11.3 Re-evaluation Record. Pending administrator action; no in-repo work remains. Retarget: monitor / platform action by administrators. |
| 2 | Two governance caveats remain structurally non-automatable (template auto-application and two-maintainer per-path enforcement) | 1 | 2 | 2 | These are documented governance caveats rather than code defects. Track as accepted operational constraints unless platform capabilities change. |

### ADR-036 - PlotSpec Canonical Contract and Validation Boundary

_Last gap analysis: 2026-06-11_

Re-verified 2026-06-11: v0.11.3 default-path promotion implemented (`plotting.py:264-299` is PlotSpec-first with explicit `legacy` opt-out); built-in builders return canonical dataclasses (`viz/builders.py` `build_*_spec -> PlotSpec`/`TriangularPlotSpec`/`GlobalPlotSpec`) and self-validate via `validate_plotspec` at builder exit (`viz/builders.py:376/648/845/1037`).

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | No pipeline-enforced canonical validation between builder output and renderer invocation for third-party plot plugins (§5) | 3 | 2 | 6 | `_render_instance_plot_plugin` / `_render_collection_plot_plugin` pass `plugin.build(context)` output directly to `plugin.render(...)` with no validation step (`plotting.py:360-361`, `411-412`). Base classes only "encourage" validation and do best-effort checks when subclasses call `super()` (`viz/plugins.py:28-35`, `51-65`), and §5 forbids making the renderer responsible for establishing canonical validity. Insert a validation boundary in the orchestration path. Target: v0.11.3 plan Task 15. |

### ADR-037 - Visualization Extension and Rendering Governance

_Last gap analysis: 2026-06-11_

Re-verified 2026-06-11: v0.11.3 default-path promotion and the runtime plot-kind extension prohibition hold (no runtime kind registry exists; plot kinds are core-defined in `viz` builders only).

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Builder/renderer extension metadata lacks mandated supported-plot-kinds/modes declarations (§4) | 3 | 2 | 6 | §4 requires extension metadata to declare supported semantic plot kinds and supported modes; built-in descriptors carry only `capabilities: ["plot:builder"]` + `style` (`plugins/builtins.py:1131-1143`, `1216-1228`) and `validate_plugin_meta` (`plugins/base.py:306-371`) enforces no plot-kind/mode fields, so resolver outcomes depend on implicit `build()`-time intent-type branching — exactly what §4 prohibits. Define and validate the metadata fields, then declare them on built-ins. Target: v0.11.3 plan Task 15. |

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

_Last gap analysis: 2026-06-11_

`check_logging_domains.py` passes (re-run 2026-06-11); domain-logger enforcement and observability examples remain closed.

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Warning-policy inventory regressed: 1 unclassified `warnings.warn` site | 2 | 1 | 2 | Same finding as ADR-028 gap 1: `api/config.py:265` (Task 12 removed-field `UserWarning`) is unclassified in `check_warning_policy.py` (run 2026-06-11: 112 sites, 1 unclassified). Target: v0.11.3 plan Task 15. |
