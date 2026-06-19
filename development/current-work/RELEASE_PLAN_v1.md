> **Active scope:** Master release control surface for v0.11.x → v1.0.0; tracks milestone sequencing, release-blocking conditions, and ADR/Standards governance posture. Moves to `finished-work` when v1.0.0 GA ships and the release series closes.

> **Status note (2025-12-02):** Last edited 2025-12-02 · Archive after v1.0.0 GA · Implementation window: v0.9.0–v1.0.0 ·

# Release Plan to v1.0.0

## Current released version: v0.11.4

> Status: v0.11.4 shipped on 2026-06-19.


Maintainers: Core team
Scope: Concrete steps from v0.6.0 to a stable v1.0.0 with plugin-first execution.

## Repository authority

- `kristinebergs/calibrated_explanations` is an active development mirror.
- CI in the development mirror is responsible for validation and artifact build verification only.
- `Moffran/calibrated_explanations` is authoritative for versions, tags, GitHub releases, PyPI publication, changelog, security advisories, and documentation.

### CI operating scope (mirror repository)

- Required PR path: lint, mypy, core tests, private-member scan, anti-pattern audit, and touched-path governance schema checks.
- Heavy checks (perf, notebook execution, full viz-focused checks, over-testing density, and other heavy/manual/scheduled jobs) stay off the critical PR path unless explicitly promoted by milestone policy.
- Packaging in the development mirror is verification-only: build wheel/sdist, install from built artifacts, and inspect artifact contents.
- Notebook execution is blocking on release branches and advisory/non-blocking outside release-boundary contexts unless explicitly promoted.
- Local reproduction model: `make local-checks-pr` for routine work, `make local-checks` before milestone closure or branch-gate changes.

## Lightweight release control (master)

This document is the release control surface for milestone-boundary planning decisions.
Detailed ADR/Standard status tables, gap inventories, and historical compliance notes are maintained in `development/current-work/RELEASE_PLAN_status_appendix.md`.

### Control snapshot

- **Current released version:** v0.11.4
- **Active detailed milestone:** v1.0.0-rc
- **Next milestone:** v1.0.0-rc
- **Status appendix:** `development/current-work/RELEASE_PLAN_status_appendix.md`

### Release-blocking conditions

A milestone cannot close while any of the following remains open:

- Broken public API contract.
- Failing CI gates.
- Stale or broken docs/examples.
- Unresolved ADR or standards contradictions.
- Any active deprecation scheduled to survive into v1.0.0.
- Open high-severity runtime bug.
- Incomplete milestone scope.

### Packaging metadata maturity control

- The `Development Status` classifier in `pyproject.toml` is release-governed metadata.
- The historical Alpha classifier is stale metadata.
- The classifier must be corrected to `Development Status :: 4 - Beta` no later than v0.11.2.
- This correction does not declare v1.0 stability.
- v1.0.0 release candidates must retain `Development Status :: 4 - Beta`.
- RC release notes must state that the public API is frozen except for release-blocking defects.
- v1.0.0 GA must update the classifier to `Development Status :: 5 - Production/Stable`.
- GA promotion is blocked unless all v1.0.0 release gates are closed:
  - public API contract frozen;
  - CI gates green;
  - docs/examples current and passing;
  - zero active deprecations scheduled to survive into v1.0.0;
  - no unresolved ADR or standards contradiction;
  - no open high-severity runtime bug;
  - release artifacts built and installation-smoke-tested.
- `Development Status :: 6 - Mature` is explicitly out of scope for v1.0.0 and may only be considered after multiple stable post-v1 releases with demonstrated low API churn.


### Active control items

- **Milestone execution control**
  - **Status:** Active milestone is v0.11.4; `development/current-work/v0.11.4_plan.md` is the maintained detailed control surface.
  - **Next action:** Finish v0.11.4 closure validation, then promote v1.0.0-rc at the next milestone boundary.
- **Future milestone discipline**
  - **Status:** v1.0.0-rc planning remains bounded to validation/freeze work; v0.11.4 carries the remaining pre-RC implementation closures.
  - **Next action:** Re-baseline RC status only after v0.11.4 implementation closure is verified.
- **Boundary-update policy**
  - **Status:** Governance/planning docs are updated at milestone boundaries, not on every PR.
  - **Next action:** Apply batched plan-grooming updates at milestone close/open checkpoints to keep process overhead low.
  - **Scope clarification:** This boundary policy applies to milestone plan grooming only; it does not override required same-PR updates to `CONTRIBUTOR_INSTRUCTIONS.md` or other mandated control/checklist files when code/API/governance changes require them.
- **Traceability of completed work**
  - **Status:** Completed items remain in plan/checklist artifacts and are marked complete rather than removed.
  - **Next action:** Continue preserving completed entries for release audit traceability.

## Terminology for improvement plans

These definitions apply across the improvement documents so schedule and gate language stays consistent.

- **Release milestone:** A versioned delivery gate (for example, v0.8.0, v0.9.1, v0.10.0) where checks must be green before shipping or cutting the branch.
- **Plan phase:** A numbered segment inside an uplift plan that groups related work and gates. Phases can span multiple iterations but map to specific release milestones when called out.
- **Stage:** Reserved for ADR-001 stage artifacts and legacy references. Outside ADR-001 material, prefer “phase” for plan sequencing and “milestone” for releases.
- **Gate:** A required check (test run, doc build, waiver review, etc.) that must pass within a phase before the corresponding release milestone can close.

Whenever a document references phases, iterations, or milestones, it uses the definitions above; any “stage” mentions outside ADR-001 should be treated as historical references rather than new schedule constructs.

## ADR gap closure roadmap

The ADR gap analysis enumerates open issues across the architecture. The breakdown below assigns every recorded gap to a remediation strategy and target release before v1.0.0. Severity values cite the unified scoring captured in `development/current-work/RELEASE_PLAN_status_appendix.md`.

## ADR and Standards roadmap summary (gap details in appendix)

Gap-by-gap severity tables now live only in `development/current-work/RELEASE_PLAN_status_appendix.md` to avoid duplicate coverage. This section tracks the top-line status or release alignment for each active ADR. Superseded ADRs are listed only as pointers.

**ADR-001 - Package and Boundary Layout:** Completed; no open appendix gaps.

**ADR-002 - Exception Taxonomy and Validation Contract:** Completed; no open appendix gaps.

**ADR-003 - Caching Strategy:** Completed; no open appendix gaps.

**ADR-004 - Parallel Execution Framework:** Completed (2026-06-17); v0.11.4 deprecated `strategy="auto"` when `enabled=True`, added focused warning tests, and filed the v1.0.0 removal row. No open appendix gaps.

**ADR-005 - Explanation Payload Schema:** Completed (2026-06-11); provenance propagates through both legacy adapters and the canonical `schema.validate_payload` plus serialization invariants are verified in code. No open appendix gaps. (Domain-model authority work continues under ADR-008.)

**ADR-006 - Plugin Trust Model:** Completed (2026-06-17); v0.11.4 closed the checksum trust-elevation bypass and retained keyed trust controls. No open appendix gaps.

**ADR-007 - PlotSpec Abstraction:** Superseded by ADR-036/ADR-037; use ADR-036 for PlotSpec canonical contract and ADR-037 for visualization extension governance.

**ADR-008 - Explanation Domain Model:** Completed (v0.11.4); domain-model authority at the serialization boundary, typed calibration/model descriptors, and multiclass `class_index` preservation are closed. No open appendix gaps.

**ADR-009 - Input Preprocessing and Mapping Policy:** Completed (2026-06-15); JSON-safe mapping export and helper-placement decisions are closed or explicitly documented. No open appendix gaps.

**ADR-010 - Optional Dependency Split:** Completed; core-only vs extras parity checks are in place. No open appendix gaps.

**ADR-011 - Deprecation and Migration Policy:** Completed (2026-06-15). The two-minor default remains normal policy; all active deprecations are filed with v1.0.0 removal ETAs under the binding finalization exception. All three 2026-06-11 reopened gaps closed: (1) guarded wrappers removed in v0.11.3 via finalization exception; (2) active-deprecations ledger rebuilt with 9 correctly filed rows, `make deprecation-closure` passes (0 blocking); (3) raw `DeprecationWarning` sites in `normalization_strategy.py`, `core/reject.py`, `core/explain/__init__.py`, and `core/calibrated_explainer.py` all use `deprecate()` helper. No open appendix gaps.

**ADR-012 - Documentation & Gallery Build Policy:** Accepted; gallery-tooling decision closed (nbconvert, 2026-06-02). Re-evidenced 2026-06-11: notebook execution exists (nightly advisory driver with timeouts; `nbsphinx_execute="always"` on non-RTD builds). Gap 1 closed (v0.11.4): docs HTML/linkcheck CI job wired via `docs-build` job in `ci-nightly.yml` calling `reusable-build-docs.yml`. Gap 2 (per-example runtime ceiling enforcement) remains advisory-only; blocking enforcement is a release-branch obligation. Target: v1.0.0-rc.

**ADR-013 - Interval Calibrator Plugin Strategy:** Completed (2026-06-18); v0.11.4 closed runtime output validation, documented the frozen-context replacement for the legacy adaptor name, and added pre-plugin migration guidance. No open appendix gaps.

**ADR-014 - Visualization Plugin Architecture:** Superseded by ADR-037; use ADR-037 for builder/renderer governance and runtime kind-extension policy.

**ADR-015 - Explanation Plugin Integration:** Completed (2026-06-18); v0.11.4 closed monitor hard-failure, classification bounds enforcement, `ExplainerHandle.learner` deprecation, and broad delegation documentation. No open appendix gaps.

**ADR-016 - PlotSpec Separation and Schema:** Superseded by ADR-036/ADR-037; use ADR-036 for semantic contract and ADR-037 for rendering governance.

**ADR-020 - Legacy User API Stability:** Completed; legacy public API contract tests and contributor workflow guidance are closed. No open appendix gaps.

**ADR-021 - Calibrated Interval Semantics:** Completed; no open appendix gaps.

**ADR-022 - Documentation Information Architecture:** Superseded by Standard-004; see Standard-004 for active status.

**ADR-023 - Matplotlib Coverage Exemption:** Completed; no open appendix gaps.

**ADR-024 - Legacy Plot Input Contracts:** Superseded/retired; maintained in `docs/maintenance/legacy-plotting-reference.md`.

**ADR-025 - Legacy Plot Rendering Semantics:** Superseded/retired; maintained in `docs/maintenance/legacy-plotting-reference.md`.

**ADR-026 - Explanation Plugin Semantics:** Completed (2026-06-18); v0.11.4 closed rule-level batch validation, monitor hard-failure, and documented the trusted `core.*` monitor exemption. No open appendix gaps.

**ADR-027 - FAST-Based Feature Filtering:** Completed (2026-06-18); v0.11.4 reclassified non-strict feature-filter governance events to `DEBUG`. No open appendix gaps.

**ADR-028 - Logging and Governance Observability:** Completed (2026-06-18); warning-policy closure holds, operational loggers are in accepted domains, and `configure_logging()` is implemented. No open appendix gaps.

**ADR-029 - Reject Integration Strategy:** Accepted (2026-01-06); policy enum, strategy registry, and reject envelope direction documented in ADR-029. `RejectResult` → `RejectResultV2` public-API migration: an active `deprecate()` call is present in `explanations/reject.py`; under ADR-011 finalization exception all active deprecations must be closed in v0.11.x. Migration or deprecation reset (removing the active warning and deferring to post-v1.0) must be resolved in v0.11.3 Task 5 (Group L). RC does not implement this; RC only verifies the deprecation ledger is empty.

**ADR-030 - Test Quality Priorities and Enforcement:** Accepted; v0.11.0 delivered full detector extension and CI check-mode enforcement (assertion + determinism checks). Zero-tolerance ratification (marker hygiene, mutation testing policy) targets v0.11.3.

**ADR-031 - Calibrator Serialization & State Persistence:** Completed (2026-06-18); v0.11.4 migrated calibrator primitives to JSON-safe schema v2 with v1 migration warnings and direct round-trip tests. No open appendix gaps.

**ADR-032 - Guarded Explanation Semantics:** Accepted (scoped); schema-compatible representative-point guarded semantics and guarded auditability are authoritative for v0.11.x. Semantic identity, plugin-path identity, and whole-interval certification are explicitly out of scope. **2026-06-11 sweep:** decisions verified in code; one minor open gap (`get_guarded_audit` error message recommends the deprecated wrappers) plus a pending decision on the guarded-wrapper removal schedule (ADR-011 conflict). Closure: v0.11.3 plan Task 15.

**ADR-033 - Modality Extension Plugin Contract and Packaging Strategy:** Completed (2026-06-18); `data_modalities` enforcement is closed and the skipped warning-phase deviation is documented. No open appendix gaps.

**ADR-034 - Centralized Configuration Management:** Accepted (2026-04-07); v0.11.2 runtime conformance closure is complete (Phase B migration + release-plan synchronization). v0.11.3 Task 10 closes remaining gaps (§7 scope boundary addendum, `CE_DEBUG_TRUST_INVARIANTS` governance, perturbation.py lifecycle fix, zombie `config.ini` deletion, `ExplainerConfig.task`/`parallel_workers` removal, root namespace exports). Remaining open items resolved: (a) sensitive-value redaction — declared out of scope for v1.0.0; CE_ env vars are behavioral flags, not secrets; documented in ADR-034 §7; (b) export payload schema versioning — `ResolvedConfigSnapshot` already carries a `schema_version` field; schema versioning is complete; ADR-034 §7 documents the version contract. No deferred v1.0 implementation items remain.

**ADR-036 - PlotSpec Canonical Contract and Validation Boundary:** Completed (2026-06-15); canonical dataclass IR, builder output contract, validation boundary, and forbidden backend-leakage rules established. v0.11.3 Task 6 promoted PlotSpec to the default user-facing plotting path; Task 15 closed the final gap: `validate_plot_artifact()` inserted at both build/render boundaries (`plotting.py:387`, `:439`). No open appendix gaps.

**ADR-037 - Visualization Extension and Rendering Governance:** Completed (2026-06-18); v0.11.4 migrated plugin metadata to six semantic plot kinds, deprecated category vocabulary, and documented `triangular` as internal routing. No open appendix gaps.

**ADR-038 - Call-time Configuration Taxonomy and Naming Conventions:** Accepted with RC graduation item only (2026-06-18); v0.11.4 closed plugin taxonomy policy drift, warning allowlist drift, and unknown wrapper-kwarg visibility. Remaining item: `**kwargs` graduation gate for v1.0.0-rc.

**Standard-001 - Nomenclature Standardization:** Completed; nomenclature guardrails and transitional shim removals are closed. No open appendix gaps.

**Standard-002 - Code Documentation Standardisation:** Completed; docstring coverage and wrapper/public numpydoc closure are complete. No open appendix gaps.

**Standard-003 - Test Coverage Standard:** Completed; no open appendix gaps.

**Standard-004 - Documentation Standard (Audience Hubs):** Completed; no open appendix gaps.

**Standard-005 - Logging and Observability Standard:** Completed (2026-06-18); shares ADR-028 closure with `configure_logging()`, logger-domain compliance, and 0 unclassified warning sites. No open appendix gaps.

## Release milestones


### Uplift status by milestone

| Release | Documentation overhaul (ADR-012/027) | Code docs (Standard-002) | Coverage uplift (Standard-003) | Naming (Standard-001) | Notes |
| --- | --- | --- | --- | --- | --- |
| v0.8.0 | IA restructure landed; Sphinx/linkcheck gates required for merges. | Phase 0 primer circulated; baseline inventory started. | `.coveragerc` drafted; `fail_under=80` staged. | Lint guards enabled; shim inventory captured. | Rollback: revert to pre-IA toctree if Sphinx fails; waivers expire after one iteration. |
| v0.9.1 | Audience hubs sustained; quickstart smoke tests block release. | Phase 1 batches A–C targeted for ≥90%; docs examples aligned with new IA. | XML+Codecov upload required; gating rising toward 90%. | Phase 1 cleanup in progress with measurement checkpoints. | Rollback: pin docs to last green build; coverage waivers require dated follow-up issues. |
| v0.10.0 | Doc gate holds prior bar; no new IA work planned. | Phase 2 blocking check for touched files; waivers time-bounded. | `fail_under=90` enforced; plugin/plotting module thresholds scheduled. | Release mapping added; refactors aligned with legacy API tests. | Risk: refactors (explain plugins, boundary split) may churn coverage; branch cut requires rerunning gates. |
| v0.10.1 | Doc hubs refreshed with telemetry/performance opt-in notes. | Package-wide ≥90% expected; notebook/example lint extended. | Module thresholds hardened; waiver expiry versions mandatory. | Phase metrics reviewed; % renamed modules tracked. | Rollback: if module gates fail, defer release or lower threshold with explicit expiry in checklist. |
| v0.10.2 | No changes planned. | No changes planned. | No changes planned. | No changes planned. | Test quality remediation: fix private-member violations in tests per ADR-030. |
| v0.10.3 | Domain model (ADR-008), Schema (ADR-005), Defaults, Plugin docs (Standard-004), Legacy stability (ADR-020). | No changes planned. | No changes planned. | No changes planned. | ADR gap closure part 1: ADR-005/008/010/020 + Standard-004. |
| v0.11.0 | Modality extension breaking contract/resolver changes (ADR-033); ADR-027 observability policy/examples updates and ADR-028 docs alignment to Standard-005. | Wrapper/public numpydoc closure target (Standard-002). | ADR-030 detector + CI enforcement and ADR-010 core-only vs extras parity checks. | Naming guardrail automation where feasible (Standard-001). | ADR gap-closure maximization milestone: close ADR-004/005/006(partial)/009/011/014/015/020/026/030/031/033; keep only architecture-heavy migrations deferred. |
| v0.11.1 | Notebook execution + runtime ceilings (ADR-012); remaining ADR-027/028 enforcement hardening; ADR-033 additive modality rollout follow-through. | No major new code-doc initiative planned beyond maintenance. | No major new coverage initiative planned beyond maintenance. | Double-underscore mutation cleanup completion tasks (Standard-001). | Registry hardening deferred from v0.11.0: full PluginManager resolution migration, trust-state atomicity unification, governance audit completion, and legacy list deprecation. CI upgrade: decommission legacy workflows. |
| v0.11.2 | Gap audit quick-win docs updates only; no doc-build changes. | Minor maintenance only. | No new coverage work planned. | No new naming work; enforcement maintained. | ConfigManager completion (ADR-034 Phase B), ADR governance sweep, governance dashboard artifact, LIME/SHAP v0.11.2 removal phase (Task 21 execution), deep memory audit (retention/leak fixes), PlotSpec default-promotion follow-up decision (ADR-036/ADR-037), ADR-035 conformance gap remediation, and packaging metadata maturity correction. |
| v0.11.3 | Minimal docs-build changes; Standard-002 numpydoc gap closure. | Close WrapCalibratedExplainer numpydoc blocks (Standard-002). | No new coverage work planned. | Final transitional shim removal (Standard-001). | RC readiness: Standard-001 shim closure, Standard-002 gap, ADR-030 zero-tolerance ratification, OSS perf harness (stretch), RejectResult→V2 migration (Group L, ADR-011 finalization), configuration management contract closure (Task 10), RC upgrade checklist + safe-defaults guide (Task 11). All implementation work that was previously in v1.0.0-rc is now in this milestone. |
| v0.11.4 | ADR-012 release-branch docs hardening plus plugin-contract migration notes. | No broad code-doc initiative; targeted ADR/STD documentation closure only. | No broad coverage initiative; targeted tests added for ADR-031, ADR-038, plugins, logging, CI, and persistence. | No broad naming initiative. | Pre-RC ADR gap closure: Tasks 1-19 closed or explicitly deferred. Major closures include ADR-004, ADR-008, ADR-012, ADR-013, ADR-015, ADR-021, ADR-026, ADR-027, ADR-028/STD-005, ADR-031, ADR-033, ADR-037, ADR-038 hardening, ADR-030/005/006 fixes, documentation migration, capability scaffold, and nightly parity-reference determinism. |
| v1.0.0 | Docs maintenance review; parity checks remain blocking. | Continuous improvement cadence; badge and quarterly reviews. | Waiver backlog should be zero; mutation/fuzzing exploration optional. | Final shim removals verified post-tag; legacy API guard tests green. | Stability declaration: RC contract freeze confirmed, production staging signed off, post-release maintenance cadences scheduled, and packaging classifier promoted to `Development Status :: 5 - Production/Stable` at GA cutover. |

### v0.6.x (stabilisation patches)

- Hardening: add regression tests for plugin parity, schema validation, and
  WrapCalibratedExplainer keyword defaults.
- Documentation polish: refresh plugin guide with registry/CLI examples and note
  compatibility guardrails.
- No behavioural changes beyond docs/tests.
- Coverage readiness: ratify Standard-003, publish `.coveragerc` draft with
  provisional exemptions, and record baseline metrics to size the remediation
  backlog.【F:development/standards/STD-003-test-coverage-standard.md†L1-L74】【F:docs/improvement/archived/test_coverage_assessment.md†L1-L23】

### v0.7.0 (interval & configuration integration)

1. Implement interval plugin resolution and fast-mode reuse per
   `PLUGIN_GAP_CLOSURE_PLAN` step 1, ensuring calibrators resolve via registry and
   trusted fallbacks.【F:docs/improvement/PLUGIN_GAP_CLOSURE_PLAN.md†L24-L43】
2. Surface interval/plot configuration knobs (keywords, env vars, pyproject) and
   propagate telemetry metadata for `interval_source`/`proba_source`.【F:docs/improvement/PLUGIN_GAP_CLOSURE_PLAN.md†L45-L61】
3. Wire CLI console entry point and smoke tests; document usage in README and
   contributing guides.【F:docs/improvement/PLUGIN_GAP_CLOSURE_PLAN.md†L63-L70】
4. Update ADR-013/ADR-015 statuses to Accepted with implementation notes.
5. Ratify Standard-001/Standard-002, publish contributor style excerpts, and land initial
   lint/tooling guardrails for naming and docstring coverage per preparatory
   phase plans.【F:development/finished-work/Standard-001_nomenclature_remediation.md†L20-L28】【F:development/finished-work/code_documentation_uplift.md†L10-L28】
   - 2025-10-07 – Updated test helpers (`tests/conftest.py`, `tests/unit/core/test_calibrated_explainer_interval_plugins.py`) to comply with Ruff naming guardrails, keeping Standard-001 lint checks green.
   - 2025-10-07 – Harmonised `core.validation` docstring spacing with numpy-style guardrails to satisfy Standard-002 pydocstyle checks.
6. Implement Standard-003 phase 1 changes: ship shared `.coveragerc`, enable
   `--cov-fail-under=80` in CI, and document waiver workflow in contributor
   templates.【F:development/standards/STD-003-test-coverage-standard.md†L34-L74】【F:docs/improvement/archived/coverage_uplift_plan.md†L9-L33】

Release gate: parity tests green for factual/alternative/fast, interval override
coverage exercised, CLI packaging verified, and nomenclature/doc lint warnings
live in CI with coverage thresholds enforcing ≥90% package-level coverage.

### v0.8.0 (plot routing, telemetry, and doc IA rollout)

1. Adopt Standard-004 (superseding ADR-022) by restructuring the documentation toctree into the audience-based information architecture (Getting Started, Practitioner, Researcher, Contributor hubs) and shipping the new telemetry concept page plus quickstart refactor per the information architecture plan.【F:development/standards/STD-004-documentation-audience-standard.md†L1-L53】
   - Land the docs sitemap rewrite with a crosswalk checklist (legacy page -> new section) and block merge on green sphinx-build -W, linkcheck, and nav tests to prevent broken routes.
   - Refactor quickstart content into runnable classification and regression guides, wire them into docs smoke tests, and add troubleshooting callouts for supported environments.
   - Publish the telemetry concept page with instrumentation examples, expand the plugin registry and CLI walkthroughs, and sync configuration references (pyproject, env vars, CLI flags) with the new navigation.
   - Record section ownership in docs/OWNERS.md (Overview/Get Started - release manager; How-to/Concepts - runtime tech lead; Extending/Governance - contributor experience lead) and update the pre-release doc checklist so every minor release verifies Standard-004 guardrails.
   - Ship a first-class "Interpret Calibrated Explanations" guide in the practitioner track that walks through reading factual and alternative rule tables, calibrated intervals, and telemetry fields, and cross-link it from README quick-start, release notes, and the upgrade checklist so users immediately grasp why the method matters.
2. Promote PlotSpec builders to default for at least factual/alternative plots
   while keeping legacy style available as fallback.【F:src/calibrated_explanations/core/calibrated_explainer.py†L680-L720】【F:src/calibrated_explanations/viz/builders.py†L150-L208】
3. Ensure explain* APIs emit CE-formatted intervals for both percentile and
   thresholded regression requests. When the mode is regression and
   `threshold` is provided, calibrate the percentile representing
   \(\Pr(y \leq \text{threshold})\) via Venn-Abers and expose the resulting
   probability interval alongside the CE-formatted interval metadata. Extend
   tests covering dict payloads, telemetry fields, and thresholded regression
   fixtures so callers see the calibrated probability interval reflected in the
   API response.【F:src/calibrated_explanations/core/calibrated_explainer.py†L760-L820】
4. Document telemetry schema (interval_source/proba_source/plot_source) for
   downstream integrations and provide examples in docs/plugins.md.
5. Review preprocessing persistence contract (ADR-009) to confirm saved
   preprocessor metadata matches expectations.【F:development/adrs/ADR-009-input-preprocessing-and-mapping-policy.md†L1-L80】
6. Execute Standard-001 Phase 2 renames with legacy shims isolated under a
   `legacy/` namespace and update imports/tests/docs to the canonical module
   names.【F:development/finished-work/Standard-001_nomenclature_remediation.md†L30-L33】
7. Complete Standard-002 baseline remediation by finishing pydocstyle batches C (`explanations/`, `perf/`) and D (`plugins/`), adding module summaries and
   upgrading priority package docstrings to numpydoc format with progress
   tracking.【F:development/finished-work/code_documentation_uplift.md†L17-L92】【F:development/standards/STD-002-code-documentation-standard.md†L17-L62】
8. Extend Standard-003 enforcement to critical-path modules (≥95% coverage) and
   enable Codecov patch gating at ≥85% for PRs touching runtime/calibration
   logic, enable
   `--cov-fail-under=85` in CI.【F:development/standards/STD-003-test-coverage-standard.md†L34-L74】【F:docs/improvement/archived/coverage_uplift_plan.md†L24-L33】
9. **Completed 2025-01-14:** Adopted ADR-023 to exempt `src/calibrated_explanations/viz/matplotlib_adapter.py` from coverage due to matplotlib 3.8.4 lazy loading conflicts with pytest-cov instrumentation. All 639 tests now pass with coverage enabled. Package-wide coverage maintained at 85%+.【F:development/adrs/ADR-023-matplotlib-coverage-exemption.md†L1-L100】

Release gate: PlotSpec default route parity, telemetry docs/tests in place,
documentation architecture and ownership shipped, nomenclature renames shipped
with shims, docstring coverage dashboard shows baseline met, Standard-003
critical-path thresholds pass consistently, and full test suite stability
achieved via ADR-023 exemption.

### v0.9.0 (documentation realignment & targeted runtime polish)

1. **Reintroduce calibrated-explanations-first messaging across entry points.** Update README quickstart, Overview, and practitioner quickstarts so telemetry/PlotSpec steps are collapsed into clearly labelled "Optional extras" callouts. Place probabilistic regression next to classification in every onboarding path and link to interpretation guides and citing.md.
2. **Ship audience-specific landing pages.** Implement practitioner, researcher, and contributor hubs per the information architecture update: add probabilistic regression quickstart + concept guide, interpretation guides mirroring notebooks (factual and alternatives with triangular plots), and a researcher "theory & literature" page with published papers and benchmark references.【F:docs/improvement/documentation_information_architecture.md†L5-L118】
3. **Clarify plugin extensibility narrative.** Revise docs/plugins.md to open with a "hello, calibrated plugin" example that demonstrates preserving calibration semantics, move telemetry/CLI details into optional appendices, and document guardrails tying plugins back to calibrated explanations. Include a prominent pointer to the new `external_plugins/` folder and aggregated installation extras for community plugins.【F:docs/improvement/documentation_review.md†L9-L49】

   - 2025-11-06 – Consolidated the plugin story into a Plugins hub (`docs/plugins.md`), added a practitioner-focused "Use external plugins" guide (`docs/practitioner/advanced/use_plugins.md`), and surfaced the curated `external-plugins` extra in installation docs. Cross-linked the appendix index and ensured practitioner/contributor flows are consistent with Standard-004/ADR-006/ADR-037/ADR-026.
4. **Label telemetry and performance scaffolding as optional tooling.** Move telemetry schema/how-to material into contributor governance sections, ensure practitioner guides mention telemetry only for compliance scenarios, and audit navigation labels to avoid implying these extras are mandatory.【F:docs/improvement/documentation_information_architecture.md†L70-L113】
5. **Highlight research pedigree throughout.** Keep the existing research hub mentions in the Overview, practitioner quickstarts, and probabilistic regression concept pages; ensure they cross-link citing.md and key publications in the relevant sections without introducing new banner UI.【F:docs/improvement/documentation_review.md†L15-L34】
6. **Triangular alternatives plots everywhere alternatives appear.** Update explanation guides, PlotSpec docs, and runtime examples so `explore_alternatives` also introduces the triangular plot and its interpretation.
7. **Complete ADR-012 doc workflow enforcement.** Keep Sphinx `-W`, gallery build, and linkcheck mandatory; extend CI smoke tests to run the refreshed quickstarts and fail if optional extras are presented without labels.【F:development/adrs/ADR-012-documentation-and-gallery-build-policy.md†L1-L80】
8. **Turn Standard-002 tooling fully blocking.** Finish pydocstyle batches E (`viz/`, `viz/plots.py`, `legacy/plotting.py`) and F (`serialization.py`, `core.py`), capture and commit the baseline failure report before flipping enforcement, add the documentation coverage badge, and extend linting to notebooks/examples so the Phase 3 automation backlog is complete.【F:development/finished-work/code_documentation_uplift.md†L24-L92】
   - 2025-10-25 – Added nbqa-powered notebook linting and a 94% docstring
     coverage threshold to the lint workflow, making Standard-002's tooling fully
     blocking for documentation CI.
9. **Advance Standard-001 naming cleanup.** Prune deprecated shims scheduled for removal and ensure naming lint rules stay green on the release branch.【F:development/finished-work/Standard-001_nomenclature_remediation.md†L40-L44】【F:development/standards/STD-001-nomenclature-standardization.md†L28-L37】
10. **Sustain Standard-003 coverage uplift.** Audit waiver inventory, retire expired exemptions, raise non-critical modules toward the 90% floor, enable `--cov-fail-under=88` in CI, and execute the module-level remediation efforts for interval regressors, registry/CLI, plotting, and explanation caching per the dedicated gap plan.【F:docs/improvement/archived/coverage_uplift_plan.md†L34-L111】
11. **Scoped runtime polish for explain performance.** Deliver the opt-in calibrator cache, multiprocessing toggle, and vectorised perturbation handling per ADR-003/ADR-004 analysis so calibrated explanations stay responsive without compromising accuracy. Capture improvements and guidance for plugin authors.【F:development/adrs/ADR-003-caching-key-and-eviction.md†L1-L64】【F:development/adrs/ADR-004-parallel-backend-abstraction.md†L1-L64】【F:src/calibrated_explanations/core/calibrated_explainer.py†L1750-L2150】 See the ADR-004 phase table above for task breakdown and ownership.

      - 2025-11-04 – Implemented opt-in calibrator cache with LRU eviction, multiprocessing toggle via `ParallelExecutor`, and vectorized perturbation handling. Added performance guidance for plugin authors in docs/contributor/plugin-contract.md. Cache and parallel primitives integrated into explain pipeline without altering calibration semantics.
12. **Plugin CLI, discovery, and denylist parity (optional extras).** Extend trust toggles and entry-point discovery to interval/plot plugins, add the `CE_DENY_PLUGIN` registry control highlighted in the OSS scope review, and ship the whole surface as opt-in so calibrated explanations remain usable without telemetry/CLI adoption.
13. **External plugin distribution path.** Document and test an aggregated installation extra (e.g., `pip install calibrated-explanations[external-plugins]`) that installs all supported external plugins, outline curation criteria, and add placeholders in docs and README for community plugin listings.

      - 2025-10-25 – Added a packaging regression test that inspects the
         `external-plugins` extra metadata to guarantee the curated bundle stays
         opt-in with the expected dependency pins.
14. **Explanation export convenience.** Provide `to_json()`/`from_json()` helpers on explanation collections that wrap schema v1 utilities and document them as optional aids for integration teams.
15. **Scope streaming-friendly explanation delivery.** Prototype generator or chunked export paths (or record a formal deferral) so memory-sensitive users know how large batches will be handled, capturing the outcome directly in the OSS scope inventory.【F:docs/improvement/OSS_CE_scope_and_gaps.md†L86-L118】

Release gate: Audience landing pages published with calibrated explanations/probabilistic regression foregrounded, research callouts present on all entry points, telemetry/performance extras labelled optional, docs CI (including quickstart smoke tests, notebook lint, and doc coverage badge) green, Standard-001/018/019 gates enforced, runtime performance enhancements landed without altering calibration outputs, plugin denylist control shipped, streaming plan recorded, and optional plugin extras (CLI/discovery/export) documented as add-ons.

### v0.9.1 (governance & observability hardening)

1. Implement ADR-011 policy mechanics—add the central deprecation helper, author the long-promised migration guide, and publish the structured status table with CI enforcement of the two-release window (see `development/current-work/RELEASE_PLAN_status_appendix.md`).
2. Bring docs CI into compliance with ADR-012 by executing notebooks during builds, installing official extras, timing tutorials, and documenting the chosen gallery tooling so drift is detected early (see `development/current-work/RELEASE_PLAN_status_appendix.md`).
3. Finish Standard-002 obligations by documenting wrapper APIs, interval calibrator signatures, and guard helpers to the mandated numpydoc standard (see `development/current-work/RELEASE_PLAN_status_appendix.md`).
4. Elevate coverage governance to the Standard-003 bar—raise thresholds to ≥90%, add per-module gates for prediction/serialization/registry paths, make the Codecov patch gate blocking, and track expiry metadata for waivers (see `development/current-work/RELEASE_PLAN_status_appendix.md`).
5. Reinforce ADR-020 legacy-API commitments with release checklist gates, regression tests for `explain_factual`/`explore_alternatives`, CONTRIBUTING guidance, and a scripted notebook audit workflow (see `development/current-work/RELEASE_PLAN_status_appendix.md`).
6. Restore visualization safety valves per ADR-023 by running the viz suite in CI, removing ignores, and aligning coverage messaging with the final thresholds (see `development/current-work/RELEASE_PLAN_status_appendix.md`).
7. Update governance collateral and hubs to satisfy Standard-004—embed the parity-review checklist in PR templates, reinstate the task API comparison, and publish the researcher future-work ledger (see `development/current-work/RELEASE_PLAN_status_appendix.md`).
8. Implement ADR-004 v0.9.1 scoped deliverable — ParallelExecutor: create a conservative execution layer that centralizes executor selection heuristics, exposes a minimal config surface (min_instances_for_parallel, min_features_for_parallel, task_size_hint_bytes), honors `CE_PARALLEL` overrides, emits compact decision telemetry (decision, reason, n_instances, n_features, bytes_hint, platform, executor_type), and includes unit tests plus a micro-benchmark harness. This is intentionally small and designed to collect field evidence before any full `ParallelExecutor` rollout in v0.10. 【F:development/adrs/ADR-004-parallel-backend-abstraction.md†L1-L40】

Release gate: Deprecation dashboard live, docs CI runs with notebook execution, coverage/waiver gating enforced at ≥90%, legacy API and parity checklists signed, and visualization tests passing on the release branch (see `development/current-work/RELEASE_PLAN_status_appendix.md`).

### v0.10.0 (runtime boundary realignment)

1. Restructure packages to honour ADR-001—split calibration into its own package, eliminate cross-sibling imports, and formalise sanctioned namespaces with ADR addenda where necessary (see `development/current-work/RELEASE_PLAN_status_appendix.md`).
2. Deliver ADR-002 validation parity by replacing legacy exceptions with taxonomy classes, implementing shared validators, parameter guards, and consistent fit-state handling (see `development/current-work/RELEASE_PLAN_status_appendix.md`).
3. Complete ADR-003 caching deliverables: add invalidation/flush hooks, cache the mandated artefacts, emit telemetry, and align the backend with the cachetools+pympler stack or update the ADR rationale (see `development/current-work/RELEASE_PLAN_status_appendix.md`).
4. Implement ADR-004’s parallel execution backlog—auto strategy heuristics, telemetry with timings/utilisation, context management and cancellation, configuration surfaces, resource guardrails, fallback warnings, and automated benchmarking (see `development/current-work/RELEASE_PLAN_status_appendix.md`). Progress is tracked in the ADR-004 phase table above.
5. Enforce interval safety across bridges and exports to resolve ADR-021 and the ADR-015 predict-bridge gap, ensuring invariants, probability cubes, and serialization policies are honoured (see `development/current-work/RELEASE_PLAN_status_appendix.md`).
6. Align runtime plugin semantics with ADR-026 by adding invariant checks, hardening contexts, and extending telemetry payloads. Also internalise `CalibratedExplainer.explain` to reinforce the facade pattern and prevent public access (see `development/current-work/RELEASE_PLAN_status_appendix.md`).
7. Remove deprecated backward-compatibility alias `_is_thresholded()` from `CalibratedExplanations` class (superseded by `_is_probabilistic_regression()` in v0.9.0). Update any remaining external code or documentation that may reference the old method name. This completes the terminology standardization cycle from ADR-021.【F:development/adrs/ADR-021-calibrated-interval-semantics.md†L119-L159】【F:docs/foundations/concepts/terminology_thresholded_vs_probabilistic_regression.md†L1-L24】
8. Condition source and discretizer branching: introduce `condition_source` configuration and thread it through `CalibratedExplainer`, `CalibratedExplanations`, orchestrators, and explanation instances so condition labels can be derived from either observed labels or calibrated predictions. Update discretizer construction to branch between observed-label and prediction-based label building and propagate the choice into `instantiate_discretizer` with validated defaults. Extend runtime helper tests to exercise both observed- and prediction-based condition sources and update discretizer interface stubs accordingly. Plan the user-visible default change (`condition_source="prediction"`) to land in v0.11.0 (or at latest in `v1.0.0-rc`) with an explicit upgrade note and migration guidance for any callers that relied on the historical observed-label behaviour.
9. Update the Docs with a comprehensive API reference for the public API of `CalibratedExplainer`, `WrapCalibratedExplainer`, `CalibratedExplanations`, `CalibratedExplanation`, `FactualExplanation`, and `AlternativeExplanation` including detailed descriptions of methods, parameters, return types, and usage examples. This will help users understand how to effectively utilize the library's capabilities.【F:docs/api_reference/calibrated_explainer.md†L1-L150】
10. **Anti-Pattern Remediation Phase 1:** Triage and categorize private member usage in tests. Rename and move test utilities (Category B) to public helpers to decouple tests from implementation details. See `docs/improvement/archived/ANTI_PATTERN_REMEDIATION_PLAN.md`.
11. **Close Standard-003 Phase 2 gates.** Execute the coverage uplift roadmap for this milestone: (a) complete the waiver audit with expiry metadata and refresh `.coveragerc`/`[tool.coverage.paths]` so Windows/WSL reports collapse to a single source of truth, (b) raise local + CI invocations (pytest + `make test-cov`) to `--cov-fail-under=90` while enabling the Codecov ≥88 % patch gate, and (c) deliver Iteration 3 remediation from the uplift plan—drive deterministic tests for `plugins/registry.py`, `plugins/builtins.py`, `plugins/cli.py`, and legacy plotting save-routing so trust toggles, CLI error paths, and renderer parity are all exercised before we cut the v0.10.0 branch.【F:docs/improvement/archived/coverage_uplift_plan.md†L24-L119】

Release gate: Package boundaries, validation/caching/parallel tests, interval invariants, terminology cleanup, and updated ADR status notes all green with telemetry dashboards verifying the new signals (see `development/current-work/RELEASE_PLAN_status_appendix.md`).

### v0.10.1 (schema & visualization contracts)

1. Confirm the v1 payload schema as the canonical contract — validate existing `explanation_schema_v1.json`, align validation helpers to payload semantics, and refresh fixtures/docs to reflect payload-first guidance (see `development/current-work/RELEASE_PLAN_status_appendix.md`, ADR-005 section).
2. Finish ADR-036 PlotSpec canonical-contract work: enhance `PlotSpec` dataclasses, validation coverage, and JSON boundary round-trips while preserving canonical dataclass authority (see `development/current-work/RELEASE_PLAN_status_appendix.md`).
3. Finish ADR-037 visualization governance work: harden builder/renderer contracts, metadata/default renderer governance, override handling, validation, CLI utilities, and documentation (see `development/current-work/RELEASE_PLAN_status_appendix.md`).
4. Maintain legacy plotting in the maintenance reference — ensure `docs/maintenance/legacy-plotting-reference.md` is authoritative for legacy behavior; avoid treating ADR-024/ADR-025 as active design gates (see `development/current-work/RELEASE_PLAN_status_appendix.md`).
5. Document dynamically generated visualization classes to close the remaining Standard-002 docstring gap tied to plugin guides (see `development/current-work/RELEASE_PLAN_status_appendix.md`).
6. Prototype streaming-friendly explanation delivery (opt-in) — implement an opt-in, non-breaking generator API for large exports (e.g., `CalibratedExplanations.to_json_stream(chunk_size=256)` or `to_json(stream=True)`) that yields JSON Lines or safe chunked JSON pieces. Collect minimal export telemetry (`export_rows`, `chunk_size`, `mode` (`batch`|`stream`), `peak_memory_mb`, `elapsed_seconds`, `schema_version`, `feature_branch`) and validate the memory profile (reference target: 10k rows < 200 MB at `chunk_size=256`). Mark streaming as experimental until prototype validation completes and record follow-up actions in the release notes.
7. **Anti-Pattern Remediation Phase 2:** Refactor core internal tests (Category A) to use public APIs and remove dead code. This reduces brittleness and improves maintainability. See `docs/improvement/archived/ANTI_PATTERN_REMEDIATION_PLAN.md`.
8. **Open-source readiness plan (v0.10.1 → v1.0.0-rc):** add the following workstream tasks here and track them through the remaining milestones so everything lands before the v1.0.0-rc freeze.
   - **Repository structure & metadata:** add top-level community health files (`CODE_OF_CONDUCT.md`, `SECURITY.md`, and `GOVERNANCE.md`/`MAINTAINERS.md`), link them from the README, and document the maintainer/decision-making model in a lightweight, discoverable format.
   - **Documentation:** expand the API reference coverage beyond `CalibratedExplainer` to include CLI entry points, plugin registry contracts, serialization schema, and visualization APIs; add a README "documentation map" that links to API, architecture, contributor, and changelog pages.
   - **Quality & maintainability:** introduce a dependency vulnerability scan in CI (e.g., `pip-audit` or CodeQL), and add a reproducible dependency constraints/lockfile workflow for dev/CI to reduce drift. Constraints are used ONLY when absolutely necessary to avoid incompatibilities; otherwise, softer ranges from `requirements.txt` are preferred.
   - **Community & contribution:** create a `ROADMAP.md` that summarizes the release plan in contributor-facing language and link it from README/CONTRIBUTING; ensure issue/PR templates reference the new governance and security guidance.
   - **Licensing & governance:** add a contribution licensing statement (DCO or inbound=outbound clause) to CONTRIBUTING and clarify how contributions are licensed under BSD-3-Clause.

Release gate: Payload round-trips verified, PlotSpec/visualization plugin registries fully validated, legacy helpers behaving per ADR maintenance reference, and docs updated with new schema references (see `development/current-work/RELEASE_PLAN_status_appendix.md`)

### v0.10.2 (plugin trust & packaging compliance)

1. Enforce ADR-006 trust controls—manual approval for third-party trust flags, deny-list enforcement, diagnostics for skipped plugins, and documented sandbox warnings (see `development/current-work/RELEASE_PLAN_status_appendix.md`).
2. Close ADR-013 protocol gaps by validating calibrators, returning protocol-compliant FAST outputs, freezing contexts, providing CLI diagnostics, and returning frozen defaults (see `development/current-work/RELEASE_PLAN_status_appendix.md`).
3. Finish ADR-015 integration work: ship an in-tree FAST plugin, rebuild explanation collections with canonical metadata, tighten trust enforcement, align environment variables, and provide immutable plugin handles (see `development/current-work/RELEASE_PLAN_status_appendix.md`).
4. Deliver ADR-010 optional-dependency splits by trimming core dependencies, completing extras/lockfiles, auto-skipping viz tests without extras, updating docs, and extending contributor guidance (see `development/current-work/RELEASE_PLAN_status_appendix.md`).
5. Extend ADR-021/ADR-026 telemetry by surfacing FAST probability cubes, interval dependency hints, and frozen bin metadata in runtime payloads (see `development/current-work/RELEASE_PLAN_status_appendix.md`).
6. **Anti-Pattern Remediation Phase 3:** Enforce zero private member usage in tests via CI/Linting to prevent regression. See `docs/improvement/archived/ANTI_PATTERN_REMEDIATION_PLAN.md`.
7. Finalize ADR-027 implementation: align runtime logging with the observability policy (debug by default; warnings only in strict mode), document metadata exposure for per-instance ignore masks, and provide examples in performance tuning documentation (see `development/current-work/RELEASE_PLAN_status_appendix.md`, ADR-027 section).
8. Adopt ADR-028 logging and governance observability architecture and enforce Standard-005 logging and observability rules across core, plugins, and governance surfaces for v0.10.2-touched code paths, including domain-based logger usage, context propagation, and data minimisation (see `development/current-work/RELEASE_PLAN_status_appendix.md`, ADR-028 and Standard-005 sections).
9. Publish {doc}`adrs/ADR-029-reject-integration-strategy` (Reject Integration Strategy) decisions and open questions, and record the deferred strategy/visualization decisions with follow-up tasks in the v0.10.2 plan.
10. Add an interval summary selection enum to choose between regularized mean (default), mean, lower bound, or upper bound for probabilistic predictions and explanations, and document the task in the v0.10.2 plan.
11. Enforce Step 1 of ADR-030 test quality remediation: fix the 7 identified private-member usage violations in `tests/unit/calibration/test_summaries.py` and `tests/unit/core/test_config_helpers.py` to achieve zero violations per `scripts/detect_test_anti_patterns.py`.
12. Add `PluginDiscoveryReport` diagnostics and expose skipped/untrusted/denied entries via `ce.plugins report` or `ce.plugins list --include-skipped`, plus interval validation CLI commands (gate: CLI + unit tests).
13. Implement pyproject trust allowlist support (`[tool.calibrated_explanations.plugins].trusted`) with packaging guidance and tests for opt-in trust resolution (gate: docs + unit tests).
14. Add parity reference harness fixtures and `parity_compare` helper with a small CI job to validate canonical outputs (gate: parity fixtures and CI run green).
15. Require canonical in-tree FAST explanation plugin registration and enforce trusted-only resolution defaults, logging any explicit override for untrusted plugins (gate: plugin registry tests and legacy parity).
16. Ship plugin diagnostics and packaging documentation updates that reflect ADR-006/ADR-013/ADR-015 enforcement and warn explicitly about in-process, non-sandboxed execution (gate: doc updates in plugins guide)

Release gate: Plugin registries enforce trust and protocol policies, extras install cleanly with documentation parity, runtime telemetry captures interval metadata, FAST/CLI flows succeed end-to-end, and logging/governance observability align with ADR-028 and Standard-005 for all v0.10.2 changes (see `development/current-work/RELEASE_PLAN_status_appendix.md`).


### v0.10.3 (ADR gap closure part 1)

  1. Close ADR-008 domain model authority: run runtime flows on domain objects, fix legacy round-trips, add calibration/model metadata, publish golden fixtures, and harden _safe_pick.
  2. Address ADR-005 semantic payload validation gaps with a strict validator (schema + invariants) and fixture coverage.
  3. Resolve ADR-010 core-only dependency clarity (matplotlib import-time requirement).
  4. Track Standard-004 follow-through for plot plugin authoring/override guidance.
  5. Change default behavior for condition_source to "prediction" in CalibratedExplainer and related components.
  6. Reinforce ADR-020 legacy-API commitments with release checklist gates, regression tests, and a scripted notebook audit workflow.
  7. Add Agent facing helper functions for easier integration with LLMs and AI agents.
  8. Add a reject hardening task to strictly verify and document each reject policy, including ablations evaluation, ensuring solid testing and verification.

  Release gate: ADR-005/008/010/020 gaps closed, condition_source default updated, and plugin authoring docs shipped.

### v0.10.4 (bug fix and CI improvement)

  1. Debug conjunctive explanations. Issue identified that they did not materialize.
  2. Initialize ADR-030 test quality tooling upgrades.
  3. Plotspec hardening.
  4. Narrative hardening. Make compatible with conjunctive explanations.
  5. Upgrade CI with better structure and easier oversight. Run old CI in parallel until v1.0.0-rc.
     - Introduce modular CI with reusable workflows at the top level of `.github/workflows/` (GitHub Actions `workflow_call` requirement) and top-level orchestration under `.github/workflows/`.
       - Validation window: keep old and new CI running in parallel for at least 2 full development cycles (recommended 2 weeks) before flipping branch-protection.
       - Workflows to decommission prior to `v1.0.0-rc`: `test.yml` (compat wrapper), `coverage.yml`, `examples.yml`, and any legacy wrappers that duplicate new reusables. See `development/finished-work/CI-upgrade.md` for the full migration and removal plan.
       - Branch-protection flip: add new checks, verify two consecutive green runs, then remove old checks and monitor for 48 hours (see `development/finished-work/CI-upgrade.md`).

### v0.11.0 (domain model & preprocessing finalisation)

  1. Complete ADR-009 preprocessing automation with auto_encode='auto', unseen-category enforcement, preprocessor mapping export/import helpers (`export_preprocessor_mapping()` / `import_preprocessor_mapping()`), and aligned telemetry.
  2. Deliver ADR-030 test quality tooling upgrades (assertion + determinism checks) and wire them into CI.
  3. Add ADR-031 calibrator persistence: versioned to_primitive/from_primitive contracts plus Explainer.save_state()/load_state().
  4. Harden ADR-026 plugin semantics with strict invariant enforcement, immutable contexts, and telemetry completeness.
  5. Add a conformal guard for guarded (conformal) explanations that extends calibrated-explanations by combining the existing discretizers with a conformalized-data-synthesizer (Meister & Nguyen) to guard explanation rule-conditions from unrealistic perturbations.
  6. Address issue #104, by adding real multiclass support.
  7. Implement ADR-033 contract hardening in core: enforce `plugin_api_version` semver parsing with major-hard/minor-soft compatibility checks, enforce `data_modalities` canonical taxonomy + alias normalization + `x-*` namespace, and ship modality-aware resolver tie-break rules (`priority` then explicit ambiguity failure).
  8. Publish ADR-033 migration notes for v0.11.0 API-breaking behavior (metadata contract enforcement and resolver ambiguity handling), with explicit upgrade guidance.
  9. Registry improvements (ADR-006): introduce the `PluginManager` class as the single source of truth for plugin resolution; define the `PluginTrustPolicy` protocol for operator-replaceable trust decisions.
  10. Clean up `registry.py` public surface: remove over-scoped plot trust management API (`mark_plot_builder_trusted`, `mark_plot_builder_untrusted`, `mark_plot_renderer_trusted`, `mark_plot_renderer_untrusted`, `find_plot_plugin_trusted`, `find_plot_renderer_trusted`) from `__all__` per ADR-037's extension-governance mandate; remove ~30 test-helper wrappers from `__all__`; fix `find_plot_renderer_trusted` return-type inconsistency and dead `include_untrusted` parameters on `list_plot_descriptors` / `_list_descriptors`.
  11. Pattern 1 hardening: empty the private member allow-list (.github/private_member_allowlist.json) as part of the final remediation, and remove production test-helper wrapper exports/re-exports (starting with `plugins/registry.py` and `plugins/__init__.py`) so tests cannot bypass public contracts via runtime scaffolding. CI must block this class of regression via `scripts/quality/check_no_test_helper_exports.py`.
  12. Perform a final ADR, standards, and improvement docs gap closure sweep and update any remaining gaps, ensuring the release appendix reflects reality.
  13. Extend alternative Pareto filtering with a selectable non-dominance cost dimension via `pareto_cost` (default `uncertainty_width`, optional `rule_size`).

  Release gate: ADR-009/026/030/031/033 gaps are closed or explicitly deferred, ADR-033 breaking contract/resolver gates are green (parser/taxonomy/ambiguity behavior), core-only install expectations are verified ahead of v1.0.0-rc, `PluginManager` shell and `PluginTrustPolicy` protocol landed, over-scoped plot trust API and test helpers removed from `__all__`, registry API defects fixed, and private-member allowlist is empty.

### v0.11.1 (hardening)

  1. Migrate plugin resolution logic from `CalibratedExplainer._invoke_explanation_plugin` and related methods into `PluginManager`, making it the authoritative resolver per ADR-006.
  2. Eliminate dual trust state: unify descriptor `trusted` flag and `_TRUSTED_*` identifier sets so all trust mutations are atomic; eliminate divergence risk between `find_*_trusted()` and `list_*_descriptors(trusted_only=True)`.
  3. Complete ADR-028 governance audit coverage: add structured audit log events for every accepted plugin registration (currently only deny/skip paths emit governance logs).
  4. Relocate test helper implementations from `registry.py` to `tests/support/registry_helpers.py` or a dedicated `plugins/_testing.py` internal; eliminate the anti-pattern of exposing test scaffolding through the production module.
  5. Close the legacy `_REGISTRY`/`_TRUSTED` list-based path in the v0.11.x line: keep warnings in v0.11.1, complete migration by v0.11.2, and remove residual list-path compatibility code in v0.11.3. No list-path deprecations may survive into v1.0.0.
  6. Reinforce ADR-012 notebook/gallery execution by documenting the tooling choice and enforcing execution/time ceilings in docs CI.
  7. Close ADR-027/ADR-028 observability enforcement by adding logging standards examples and lint/tests.
  8. Finish Standard-001 nomenclature clean-up by eliminating double-underscore mutations, splitting utilities, and confining transitional shims to legacy/.
  9. Extend governance dashboards to surface lint status alongside preprocessing/domain-model telemetry. → relocated to v0.11.2.
  10. Decommission workflows: `test.yml` (compat wrapper), `coverage.yml`, `examples.yml`, and any legacy wrappers that duplicate new reusables. See `development/current-work/CI-upgrade.md` for the full migration and removal plan. → completed in v0.11.1.
  11. Ship ADR-033 additive UX/migration gates: CLI `--modality`, `vision`/`audio` shims that raise `MissingExtensionError` (`CE base + ImportError`), and a hard closure timeline executed in v0.11.x (`v0.11.1` warning, `v0.11.2` enforcement-ready validation, `v0.11.3` removal of warning fallback paths).
  12. Publish ADR-033 follow-through docs: contributor plugin contract updates, practitioner usage notes, and migration guidance for modality plugins.
  13. Add one ADR-033 packaging smoke test validating extension install + entry-point discovery/import behavior.
  14. Update the Reject Framework within Calibrated Explanations to include other forms of rejection beyond just binary conformal-based rejectors, such as uncertainty-based rejectors that leverage the uncertainty estimates from calibrated explanations to make informed decisions about when to abstain from making a prediction. Document the new rejector types and provide examples of how to implement and use them effectively in practice.
  15. Introduce a centralized package configuration layer via `ConfigManager` as the single runtime source of configuration truth. `ConfigManager` must resolve values from call-site overrides, environment variables, `pyproject.toml`, and hard defaults using one documented precedence contract; expose typed accessors for plugin resolution, telemetry, cache/parallel, and reject configuration; and replace ad-hoc config reads across core/plugins/CLI so behavior is deterministic, testable, and auditable.
   16. Promote ADR-020 from Draft → Accepted: update contract document for v0.11.0 removals, confirm release checklist ADR-020 gate active.
      - 2026-03-03 – ADR-020 promoted to Accepted; `legacy_user_api_contract.md` updated with v0.11.0 removal table (`explain_counterfactual`, `get_explanation`, deprecated aliases).
   17. Promote ADR-028 from Draft → Accepted before enforcement script (Task 7) merges, so task 7 enforces an authoritative policy record.
      - 2026-03-03 – ADR-028 promoted to Accepted.
   18. Update `development/current-work/legacy_user_api_contract.md` for v0.11.0 removals and update status note date.
      - 2026-03-03 – Removed-in-v0.11.0 section added; `.get_explanation(i)` note corrected; status note updated.
   19. Add Standard-005 to ADR/Standards roadmap summary table in `RELEASE_PLAN_v1.md` and confirm v0.11.1 enforcement gate.
      - 2026-03-03 – Standard-005 row added to roadmap summary and Standards appendix; observability gaps assigned to v0.11.1 Task 7.
  20. Add a separate `governance_config_event_schema_v1.json` for `calibrated_explanations.governance.config` lifecycle events (`config.resolve`, `config.export`, `config.validation_failure`) — do NOT modify `governance_event_schema_v1.json` (its `event_name: const` and `decision: enum` are plugin-specific); wire emission into `ConfigManager` at snapshot lifecycle boundaries only; add structured log-capture tests and a CI schema gate against the new config-event schema.
  21. API-bloat removal program (ADR-011 + ADR-037 + ADR-020) — mandatory v0.11.x closure: in v0.11.1, inventory and deprecate every core LIME/SHAP entry point with migration mapping and CI warning coverage; in v0.11.2, remove core exports/wrapper hooks and move runtime usage to plugin-only adapters; in v0.11.3, delete residual compatibility stubs and finalize docs/tests so v1.0.0 carries zero LIME/SHAP deprecations.
  22. PlotSpec hardening + ADR revisioning (ADR-036/ADR-037): harden PlotSpec as a canonical semantic IR by enforcing dataclass-only canonical in-memory representation, strengthening validator boundaries, unifying builder outputs to canonical PlotSpec, splitting rendering/normalization/export/test instrumentation responsibilities, and isolating compatibility handling to explicit serializer/translator boundaries. In the same task, publish and adopt ADR-036 + ADR-037 as the authoritative source of truth and supersede ADR-007/ADR-014/ADR-016 as historical records. Keep legacy plotting as the default public `.plot()` path in v0.11.1 and keep runtime plot-kind extension out of scope for this release.

   Release gate: `PluginManager` owns all plugin resolution; trust state is atomic across descriptor and set; governance audit events cover both accepted and rejected registrations (including `governance.config` events from ConfigManager); test-helper bodies no longer live in the production module; list-path and core LIME/SHAP deprecation inventories are complete with explicit v0.11.2/v0.11.3 removal ownership; CI enforces `CE_DEPRECATIONS=error` for all deprecated paths still active in v0.11.1; ADR-012/027/028/001/033 additive rollout gates (CI/docs/shims/packaging smoke test) are green; ADR-020 and ADR-028 promoted to Accepted; `ConfigManager` is the authoritative configuration entry point with precedence and migration tests green; core package import/public API no longer hard-depends on LIME/SHAP adapters (plugin-only) after v0.11.2; PlotSpec canonical-contract hardening is implemented with boundary-only compatibility translation; ADR-036/ADR-037 are authoritative and ADR-007/014/016 are superseded; legacy plotting remains default in v0.11.1; runtime plot-kind extension remains disabled; and no v0.11.1 deprecation task is allowed to defer closure beyond v0.11.3.

### v0.11.2 (config hardening and ADR governance sweep)

  1. Migrate `cache/cache.py`, `parallel/parallel.py`,
     `core/explain/_feature_filter.py`, `core/prediction/orchestrator.py` to
     `ConfigManager`. Done when all four files are removed from the CI allowlist
     (ADR-034 Phase B).
  2. Finalize ADR-034 (pulled forward from v1.0.0-rc; no RC dependency): confirm
     CI allowlist reaches zero or documents any remaining sanctioned boundaries;
     synchronize ADR-034 implementation status and summary text with the now-accepted ADR.
     Depends on task 1 completing green.
  3. ADR and standards governance sweep (pulled forward from v1.0.0-rc; rolling audit;
     no RC dependency): audit all open gaps in the RELEASE_PLAN_v1.md appendix, confirm
     every gap is assigned to a milestone or marked superseded, refresh gap-analysis
     dates older than 60 days, and close any quick-win gaps that require no feature
     work (e.g., remaining ADR-026 observability gaps).
  4. Governance dashboard extension (relocated from v0.11.1 Task 9; depends on v0.11.1 Tasks 3, 7, and 20 completing): surface lint status and preprocessing/domain telemetry in a machine-readable governance status artifact. This artifact is a derived CI/reporting surface and must not replace the runtime governance event contracts defined for plugin/config lifecycle events. Fits v0.11.2 governance sweep theme; ADR-028 CLEAR.
  5A. LIME/SHAP v0.11.2 removal follow-through (execution of v0.11.1 Task 21 removal phase):
      delete `CalibratedExplainer.{preload,explain,is_enabled}_lime/shap` and
      `WrapCalibratedExplainer.explain_lime/shap`; move deprecation ledger rows from Active
      to Removed history; verify plugin-only replacement paths remain functional.
      Depends on task 3 completing (governance inventory must be current before ledger moves).
  6. Deep memory audit and retention hardening: bounded `reset()`/`close()` lifecycle
     semantics for `latest_explanation`, SHAP/LIME helper caches, and the plugin explanation
     instance cache (max 16 LRU entries); RSS stabilization gate (≤10% delta, ≤64 MB absolute
     growth over 200 iterations). Depends on task 5A (post-removal surface needed for
     accurate memory baseline).
  7. PlotSpec default-promotion follow-up (new in v0.11.2): record the binding v0.11.2
     decision to keep legacy as the default path, document the exact per-function deferral
     outcome for all in-scope plots, and carry the promotion question forward to v0.11.3.
     This follow-up is the explicit decision point governed by ADR-036 and ADR-037.
  8. ADR-035 conformance gap remediation (surfaced by Task 4 red-team):
     GAP 1 — insert §2 rollout status note resolving the advisory/blocking MUST contradiction;
     GAP 2 — add `scripts/local_checks.py` rule to `.github/CODEOWNERS`;
     GAP 4 — add inline-workflow allow-list to `validate_ci_policy.py` covering the five
     pre-reusable workflows (`ci-main.yml`, `ci-nightly.yml`, `deprecation-check.yml`,
     `maintenance.yml`, `update_baseline.yml`), each with a dated `review-by: v0.11.3` rationale.
     Depends on task 4 (Task 4 revealed GAP 2 and its Makefile addition triggered GAP 4 analysis).
  9. PlotSpec semantic/visual mending while non-default (new in v0.11.2): improve PlotSpec
      rendered plots on explicit opt-in paths so they preserve the same explanatory meaning
      as legacy plots without changing user default behavior. This task creates the evidence
      base required for any later default-promotion decision.
 10. Packaging metadata maturity correction: update `pyproject.toml` classifier from
      `Development Status :: 3 - Alpha` to `Development Status :: 4 - Beta`, document
      RC/GA policy controls, and verify package metadata generation reflects Beta
      exactly once with Alpha absent.

  Release gate: All four allowlisted modules migrated and removed from CI allowlist; ADR-034 implementation-status text synchronized with the accepted ADR; governance sweep complete with all appendix gaps assigned or superseded; governance status artifact schema documented and CI-generated as a derived reporting surface; Task 21 v0.11.2 removal phase complete (eight core/wrapper LIME/SHAP symbols deleted, fail-closed tests green, deprecation ledger rows moved); deep memory retention bounded with RSS stabilization proof; PlotSpec default-promotion follow-up completed with an explicit v0.11.2 deferral decision recorded; PlotSpec semantic/visual mending evidence exists for the opt-in path; ADR-035 GAPs 1/2/4 closed with tests; packaging metadata maturity correction completed (Beta in source and generated metadata, Alpha absent); `make local-checks-pr` passes.

### v0.11.3 (RC readiness: Standard closure, ADR-030 ratification, docs gap)

  1. Close Standard-001 (pulled forward from v1.0.0-rc; depends on v0.11.1 Standard-001
     cleanup): remove remaining transitional shims from `legacy/`, confirm naming/tooling
     enforcement green on main.【F:development/finished-work/Standard-001_nomenclature_remediation.md†L40-L44】
  2. Close Standard-002 known gap (implementation pulled forward from v1.0.0-rc; RC only
     verifies): add full numpydoc blocks to `WrapCalibratedExplainer` and remaining
     stable public surfaces identified in the appendix.【F:development/finished-work/code_documentation_uplift.md†L24-L92】【F:development/standards/STD-002-code-documentation-standard.md†L43-L62】
  3. Ratify ADR-030 zero-tolerance enforcement (pulled forward from v1.0.0; ratification
     should inform RC readiness): confirm extended anti-pattern scans are CI-blocking,
     document marker hygiene rules in the ADR, and declare mutation testing optional for
     core modules.【F:development/adrs/ADR-030-test-quality-priorities-and-enforcement.md†L1-L50】
  4. (Stretch) OSS performance harness template (pulled forward from v1.0.0-rc;
     self-contained): provide a reusable harness for semi/fully-online latency
     measurements with README guidance and a sample run (gate: sample run documented
     and reproducible).
  5. Full deprecation ledger closure (LIME/SHAP + RejectResult): (a) delete any
     residual v0.11.3-owned Task-21 LIME/SHAP compatibility adapters, move all
     remaining Task-21 rows from Active to Removed history, and verify zero active
     Task-21 deprecations survive into v1.0.0; (b) resolve Group L — the active
     `deprecate()` call in `reject_result_v2_to_legacy()` targeting "v1.0.0-rc"
     violates ADR-011's finalization exception and must be removed here (either
     complete the `RejectResult`→`RejectResultV2` migration, or remove the active
     warning and re-target migration to v1.1+).
  6. PlotSpec default-promotion re-evaluation after v0.11.2 mending: revisit the
      default-path question only after reviewing the v0.11.2 deferral note and plot
      mending evidence. Proposed candidate outcome: broad promotion (1C-family) if
      semantic/visual parity evidence is sufficient; otherwise record another explicit
      deferral rather than promoting by momentum.
  7. Optional `uv` workflow support: adopt `uv` narrowly as an optional contributor
     setup and CI install-acceleration path while keeping `pip install
     calibrated-explanations` as the canonical user install. Add a pinned
     `uv-install-smoke` validation lane, record pip-vs-uv install timing evidence,
     refresh or remove/document `uv.lock`, and verify ADR-010/012/028/030/033 plus
     Standard-001/003/004/005 compliance before closure.
  8. Reject hardening for v0.11.3: difficulty-aware reject evaluation,
     experimental strategy containment, paper-aligned singleton reporting, and
     validity metadata. Full scope, subtasks, and red-team findings tracked in
     Task 8 of `development/finished-work/v0.11.3_plan.md`.
  9. Close v0.11.3 ADR contract gaps not otherwise owned: close the remaining
     v0.11.3-targeted ADR/standard appendix gaps not already covered by Tasks
     1-7, including ADR-006 accepted-registration audit events and legacy
     registry-list closure; ADR-009 JSON-safe mapping export and helper-placement
     documentation; ADR-010 core-only vs extras parity automation; ADR-011
     serializer/compatibility shim deprecation closure; ADR-012 gallery tooling
     decision documentation; ADR-013 interval protocol/fallback strictness and
     third-party harness coverage; ADR-015 invariant-enforcement consistency;
     ADR-020 legacy API release-checklist, parity, and contributor-workflow
     follow-through; ADR-027 FAST observability docs/examples; ADR-029 reject
     lifecycle/config surface decision or explicit deferral; ADR-028/STD-005
     warning-to-logging correction for fallback/degraded-state visibility; and
     ADR-035 branch-protection re-evaluation. Milestone closure is blocked until every
     status-appendix row targeted to v0.11.3 is either implemented, explicitly
     superseded/deferred with rationale, or owned by a concrete task in this
     milestone.
  10. Configuration management contract closure: remove `task` and
      `parallel_workers` fields from `ExplainerConfig` (only `perf_parallel_workers`
      is wired; the removed fields have no applicable targets); promote
      `ExplainerBuilder` and `ExplainerConfig` to the root `calibrated_explanations`
      namespace; add `CE_DEBUG_TRUST_INVARIANTS` to `_KNOWN_ENV_KEYS`; delete the
      zombie `utils/configurations/config.ini`; document the `ExplainerBuilder`/
      env-var precedence rule in `perf_cache()`/`perf_parallel()` docstrings and
      ADR-034 §7. No RC deferrals permitted. Full scope in Task 10 of
      `development/finished-work/v0.11.3_plan.md`.
  11. Reject hardening — docstrings, kwarg documentation, and alternatives integration
      test: expand both reject collection class docstrings from single-line stubs to
      full numpydoc blocks; document `reject_policy` in the `explore_alternatives` and
      `explain_reject` calling-method docstrings; add an integration-level test through
      `explore_alternatives` with a reject policy. `RejectAlternativeExplanations`
      exported from the root namespace. Full scope in Task 11 of
      `development/finished-work/v0.11.3_plan.md`.
  12. RC readiness documentation: produce `docs/upgrade/v1.0.0-upgrade-checklist.md`
      (covering all API changes from Groups A–L, env vars, pyproject settings,
      `ExplainerBuilder` wiring, caching controls, parallel controls, plugin testing,
      and reject framework) and `docs/guides/safe-defaults.md` (safe-by-default
      settings for RC pilot testers). These artifacts must exist before RC pilot
      testing begins; RC only verifies they are accurate. Full scope in Task 12 of
      `development/finished-work/v0.11.3_plan.md`.
  13. Normalize guarded explanations as a parameterized explanation policy: replace the
      parallel `explain_guarded_factual` / `explore_guarded_alternatives` public methods
      with `explain_factual(..., guarded=True)` and `explore_alternatives(..., guarded=True)`
      as the canonical entry points; deprecate the old method names as compatibility
      wrappers (removal target v1.0.0); add `guarded: bool` to `ExplanationRequest` and
      `supports_guarded: bool` to plugin metadata; update ADR-032 and
      `CONTRIBUTOR_INSTRUCTIONS.md`. Full scope in Task 13 of
      `development/finished-work/v0.11.3_plan.md`.
  14. Parameter naming consistency hardening and CI drift protection: deliver a
      CI-blocking script (`scripts/quality/check_parameter_naming.py`) that enforces
      a parameter naming policy across public API signatures (banning removed aliases
      `alpha`/`alphas` and names with no governed definition; blocking internal-only
      names `y_threshold`/`sigma`/`interval_width` from appearing in public
      signatures); document the `threshold` → `y_threshold` internal alias at the
      `IntervalRegressor` call site; add consistent numpydoc `Parameters` entries for
      `threshold`, `confidence`, and `significance`; produce a canonical parameter
      reference page (`docs/foundations/concepts/parameter-reference.md`) with a
      disambiguation table for the three confusable floats-in-(0,1). Full scope in
      Task 14 of `development/finished-work/v0.11.3_plan.md`.
  Release gate: Standard-001 naming lint green with all transitional shims removed; Standard-002 WrapCalibratedExplainer numpydoc gap closed and docstring coverage ≥90%; ADR-030 zero-tolerance enforcement CI-blocking with ratification note in ADR; PlotSpec default promotion is re-evaluated against the v0.11.2 mending evidence and either promoted with synchronized docs/tests or explicitly deferred again; ADR-028/STD-005 fallback visibility is log-first with any remaining `UserWarning` paths justified; all remaining deprecations from v0.10.x/v0.11.x are removed and migration docs moved to Removed history (including Task 5 Group L — no `deprecate()` call may target v1.0.0-rc or later); all v0.11.3-targeted status-appendix gaps are either closed, superseded, or explicitly deferred with rationale; no status-appendix row still says `Target milestone: v0.11.3` unless it corresponds to an incomplete v0.11.3 task that blocks milestone closure; Task 10 config management contract closure complete (`task` and `parallel_workers` removed from `ExplainerConfig`, root namespace exports present, `CE_DEBUG_TRUST_INVARIANTS` in `_KNOWN_ENV_KEYS`, zombie `config.ini` deleted, ADR-034 §7 written); Task 11 reject collection docstrings complete, `reject_policy` documented in calling-method docstrings, `RejectAlternativeExplanations` in root namespace, integration test through `explore_alternatives` green; Task 12 RC readiness documentation present and content-complete (`docs/upgrade/v1.0.0-upgrade-checklist.md` and `docs/guides/safe-defaults.md`); Task 13 `explain_factual(guarded=True)` / `explore_alternatives(guarded=True)` canonical, old methods deprecated, ADR-032 updated; Task 14 parameter naming CI script green and wired into `local-checks-pr`, parameter reference doc present; `make local-checks-pr` passes.
  Packaging workflow gate: optional `uv` support is documented and validated with
  lockfile handling, timing evidence, and CI follow-up completed before v0.11.3 closes.
  Status 2026-05-12: completed with optional constraint-based `uv pip` support,
  stale `uv.lock` removal, a pinned `uv-install-smoke` PR lane, and local timing
  evidence (`pip_install_seconds=209`, `uv_install_seconds=53`).

  **Milestone closure — 2026-06-13:** v0.11.3 is complete. All 18 tasks implemented and verified. Key closures: Standard-001 shim removal (Task 1), Standard-002 numpydoc gap (Task 2), ADR-030 zero-tolerance ratification (Task 3), full deprecation ledger closure including `data_modalities` fail-closed enforcement (Tasks 5, 15, 17, 18), PlotSpec default promotion (Task 6), `uv` workflow support (Task 7), reject hardening (Tasks 8, 11), config management contract (Task 10), RC upgrade checklist and safe-defaults guide (Task 12), guarded explanation parameterization (Task 13), parameter naming CI guard (Task 14), ADR gap closure sweep (Task 15), constraint minimization (Task 16), call-time configuration taxonomy `GuardedOptions`/`reject_confidence` (Task 17), release preparation (Task 18). Gates: `make local-checks-pr` ✅, `make deprecation-closure` ✅ (9 v1.0.0 permitted, 0 blocking), Sphinx strict build ✅, warning policy ✅ (0 unclassified), `uv-install-smoke` ✅. Version tagged: `0.11.3`. CHANGELOG converted. ADR/appendix rows closed. Next: v1.0.0-rc.

**Known pre-RC implementation gaps — v0.11.4 closure status.** These were the pre-RC patch items that motivated v0.11.4. Their detailed closure evidence is tracked in `development/current-work/v0.11.4_plan.md` and `development/current-work/RELEASE_PLAN_status_appendix.md`; only the explicitly deferred RC/post-v1 items remain open.

| Gap | ADR | Severity | Decision required |
|-----|-----|----------|-------------------|
| `strategy="auto"` silently selects backend when `enabled=True` | ADR-004 gap 1 | 9 | **Closed v0.11.4:** deprecation warning plus v1.0.0 removal ledger row |
| Domain model authority (3 sub-gaps) | ADR-008 gaps 1/2/3 | 20/14/12 | **Closed v0.11.4:** domain-authoritative serialization, typed descriptors, and multiclass metadata preservation |
| `ExplainerHandle.learner` direct bypass | ADR-015 gap 2 | 8 | **Closed v0.11.4:** deprecation warning retained and broad delegation documented as accepted risk |
| Unfrozen nested context fields / rule-level semantics | ADR-026 gap 1/3 | 6/9 | **Closed v0.11.4:** nested freezing verified; rule-level validation and trusted built-in exemption documented |
| Docs HTML/linkcheck CI job wired (nightly advisory) | ADR-012 Gap 1 | - | **Closed v0.11.4:** `docs-build` job added to `ci-nightly.yml`; release-branch strict docs workflow added in Task 16 |


### v1.0.0-rc (release candidate readiness)

<!-- Removed items (evidence in parentheses):
  - #6 Institutionalise Standard-003 → STALE: appendix confirms "STD-003 fully compliant (2026-02-27): no further action required"; CI enforcement already in place
  - #7 Promote ADR-026 / deprecate ADR-024/025 → STALE: ADR-026 already Accepted (Status: 2026-01-12); ADR-024/025 already Retired with superseded-prefix files
  - #10 ADR gap closure audit → moved to v0.11.2
  - #11 condition_source default change → DONE: delivered in v0.10.3 CHANGELOG
  - #12 ADR-030 tooling extension → DONE: delivered in v0.11.0 CHANGELOG (assertion + determinism checks; script reorganized to scripts/quality/)
  - #13 ADR-031 calibrator persistence → DONE: delivered in v0.11.0 CHANGELOG (to_primitive/from_primitive + WrapCalibratedExplainer save/load)
  - #15 ADR/standards docs gap closure → moved to v0.11.2
  - #16 ADR-034 post-acceptance conformance closure → moved to v0.11.2
  Items #3 (Standard-001 shim removal) and #4 (Standard-002 gap closure) moved to v0.11.3.
  Item #14 (OSS performance harness) moved to v0.11.3.
  Item #5 (versioned documentation preview and public doc-quality dashboards) → declared explicitly out of scope for v1.0.0. Versioned documentation hosting has been deferred since v0.8.0. A static current-docs approach is sufficient for v1.0.0-rc and GA; versioned hosting and automated dashboard infrastructure is a v1.1+ concern. Removed from RC release gate.
  Item #6 (RC upgrade checklist) → moved to v0.11.3 Task 11. Upgrade checklist must exist before RC testing starts; creating it during RC defeats its purpose. RC only verifies the checklist is present and accurate.
  "document safe defaults for RC adopters" (sub-item of old item #3) → moved to v0.11.3 Task 11. Documentation must exist before RC so pilot testers know what to validate.
  ADR-029 RejectResult→V2 migration → moved to v0.11.3 Task 5 (Group L). An active deprecate() call is present; ADR-011 finalization exception requires all active deprecations to close in v0.11.x.
  ADR-034 deferred items → resolved without RC work: sensitive-value redaction declared out of scope for v1.0.0; export schema versioning already implemented (ResolvedConfigSnapshot.schema_version).
-->

> **RC posture:** v1.0.0-rc is a validation and freeze milestone only. No implementation work is permitted here. If a release-blocking defect requires a code fix, the fix is considered an emergency patch, not a planned RC task.

1. Confirm Explanation Schema v1 is content-complete and frozen (any schema gaps
   must be resolved in v0.11.3 before RC); publish the compatibility statement
   communicating that only patch updates will follow.【F:docs/schema_v1.md†L1-L120】
2. Reconfirm wrap interfaces and exception taxonomy against v0.6.x contracts;
   update README and CHANGELOG with a release-candidate compatibility note.【F:src/calibrated_explanations/core/wrap_explainer.py†L260-L471】【F:src/calibrated_explanations/core/exceptions.py†L1-L63】
3. Validate caching/parallel toggles in staging; verify (do not document) that
   telemetry captures cache hits/misses and worker utilisation metrics as
   documented in v0.11.3 Task 11 safe-defaults guide.【F:development/adrs/ADR-003-caching-key-and-eviction.md†L28-L64】【F:development/adrs/ADR-004-parallel-backend-abstraction.md†L25-L64】
4. Verify Standard-002 compliance at ≥90% docstring coverage holds for the RC
   branch (the WrapCalibratedExplainer gap was closed in v0.11.3; this is a
   verification gate, not an implementation task).
5. Verify the upgrade checklist produced in v0.11.3 Task 11 is accurate for the
   RC build; make minor corrections only if a release-blocking defect changed a
   documented behavior. Confirm documentation is current and navigable. Backport
   documentation to downstream extension repositories so they are compatible with
   the v1.0.0-rc API before GA tag.
6. Confirm release-candidate packaging metadata remains
   `Development Status :: 4 - Beta` and publish RC release notes stating the
   public API is frozen except for release-blocking defects.

Release gate: Explanation Schema v1 frozen and compatibility statement published; wrap interface and exception taxonomy compatibility confirmed against v0.6.x; caching/parallel staging validation signed off and telemetry verified against v0.11.3 documentation; Standard-002 ≥90% verified; upgrade checklist present, accurate, and reviewed; deprecation ledger is empty (zero active deprecations; verified as closed by v0.11.3); RC package metadata is `Development Status :: 4 - Beta`; RC release notes state the public API freeze posture.
### v1.0.0 (stability declaration)

<!-- Removed item #6 (Ratify ADR-030): moved to v0.11.3 so ratification informs RC readiness. -->

1. Announce the stable plugin/telemetry contracts and publish the final
   compatibility statement across README, CHANGELOG, and docs hub.
2. Promote packaging metadata from `Development Status :: 4 - Beta` to `Development Status :: 5 - Production/Stable`, only after all v1.0.0 release gates are closed and release-blocking defects are zero.
<!-- Removed "backport documentation to downstream extension repositories" from item 3: this is implementation work and must complete before GA. Moved to v1.0.0-rc item 5. -->
<!-- Removed item 4 (staging validation): this is v1.0.0-rc work, not GA work. RC release gate already requires "caching/parallel staging validation signed off". At GA, staging validation is a confirmed pre-condition, not a new activity. -->
3. Tag the v1.0.0 release and circulate the upgrade checklist to partners with
   caching and parallelisation guidance.
4. Confirm that staging validation signed off at v1.0.0-rc remains valid — no new
   staging runs are performed at GA unless a release-blocking defect was patched
   after RC cut. Confirm zero active deprecations (verified by v0.11.3 Task 5
   Group L resolution and the empty deprecation ledger gate at RC).
5. Confirm Standard-001/Standard-002 guardrails remain enforced post-tag, monitor the
   caching/parallel telemetry dashboards, and schedule maintenance cadences
   (coverage/docstring audits, performance regression sweeps) for the first
   patch release.
6. Verify documentation hubs are current, navigable, and reflect v1.0.0 API state. Versioned documentation hosting and automated dashboard infrastructure are explicitly targeted at v1.1+ and are not a GA gate — a static docs approach is used for v1.0.0.

Release gate: Tagged release artifacts available; documentation hubs current and navigable (including downstream extension repositories confirmed at RC); caching/parallel toggles operating within documented guardrails; staging validation from v1.0.0-rc confirmed valid (not re-run at GA unless a post-RC release-blocking defect was patched); zero active deprecations confirmed; post-release maintenance cadences scheduled; packaging classifier promoted to `Development Status :: 5 - Production/Stable` at GA cutover. (Versioned documentation hosting is NOT a v1.0.0 gate — see v1.1 target note above.)

## Standard-003 integration analysis

- **Scope alignment:** The release milestones already emphasise testing and
  documentation maturity; Standard-003 adds explicit quantitative coverage gates that
   complement Standard-001/Standard-002 quality goals without altering plugin-focused
   scope.【F:development/standards/STD-003-test-coverage-standard.md†L34-L74】
- **Milestone sequencing:** Early v0.6.x tasks capture baseline metrics and
   prepare `.coveragerc`, v0.7.0 introduces CI thresholds, v0.8.0 widens
   enforcement to critical paths and patch checks, and v0.9.0 retires waivers
   ahead of the release candidate. This staging keeps debt burn-down parallel to
   existing plugin/doc improvements.【F:docs/improvement/archived/coverage_uplift_plan.md†L11-L48】
- **Release readiness:** By v1.0.0, coverage gating is embedded in branch
   policies and telemetry/documentation communications, ensuring Standard-003 remains
   sustainable beyond the initial rollout.【F:development/standards/STD-003-test-coverage-standard.md†L34-L74】【F:docs/improvement/archived/coverage_uplift_plan.md†L11-L48】

## Post-1.0 considerations

- Continue monitoring caching and parallel execution telemetry to determine
  whether the opt-in defaults can graduate to on-by-default in v1.1, updating
  ADR-003/ADR-004 rollout notes as needed.【F:development/adrs/ADR-003-caching-key-and-eviction.md†L28-L64】【F:development/adrs/ADR-004-parallel-backend-abstraction.md†L25-L64】
- Evaluate additional renderer plugins (plotly) after verifying PlotSpec default
  adoption.
- **`transform_to_numeric` root-namespace deprecation (v1.1+ ADR-011 cycle):**
  `transform_to_numeric` is currently in `__all__` as a public API symbol.
  Its core purpose (categorical-to-numeric encoding) is now largely covered by
  `WrapCalibratedExplainer`'s `auto_encode` / `preprocessor` config and
  `export_preprocessor_mapping()` / `import_preprocessor_mapping()`. Once the
  `auto_encode='auto'` mapping-persistence path (ADR-009 pending item) is complete,
  `transform_to_numeric` should be deprecated from the root namespace and moved to
  `calibrated_explanations.utils` only, then removed in a v1.2+ cycle.
  See ADR-009 §Implementation status for the full rationale and migration path.
- **ADR-009 pending items — `auto_encode='auto'` path and unseen-category policy (v1.1+):**
  The wrapper preprocessing surface decision (wrapper-only, core stays numeric) is implemented
  and stable for v1.0.0. Two ADR-009 pending items remain open: the `auto_encode='auto'`
  automatic encoding mode with deterministic mapping storage, and the unseen-category policy
  behavior (`'error'` default / `'ignore'` opt-in) with documentation. These are the
  prerequisite work before `transform_to_numeric` can be deprecated (see bullet above).
  Target: v1.1+.
- **ADR-029 — Reject strategy lifecycle hooks and full config surface (v1.1+):**
  ADR-029 documents three lifecycle hooks (`pre_apply_hook`, `pre_emit_hook`, `post_emit_hook`)
  and a full strategy configuration surface beyond the policy enum. These were deferred from
  Task 8 (reject hardening) as research-dependent. The `RejectResult`→`RejectResultV2` public
  API migration closed in v0.11.3 (Task 5 Group L); the hook surface and extended config are
  post-v1.0 scope. A separate v1.1+ ADR or ADR-029 amendment is needed before implementation.
- **ADR-029 / Task 8 — C3 reject scenario (research-dependent, v1.1+):**
  The "C3" confidence-region reject scenario (non-singleton reject regions; probabilistic
  reject boundaries) was documented in Task 8 planning but deferred: it requires upstream
  research results before a stable API surface can be committed. Not a v1.0.0 gate.
  Track under the ADR-029 lifecycle-hooks work above.
- **ADR-033 — Timeseries modality (separate ADR needed, v1.1+):**
  ADR-033 establishes the entry-point plugin modality contract (`data_modalities` key,
  `('tabular',)` as the v1.0.0 baseline). Timeseries support requires a dedicated new ADR
  covering data layout (rolling windows, indexing semantics), calibration applicability, and
  plugin registration conventions. Explicitly out of scope for v1.0.0.
- **ADR-034 — Governance-log sensitive-value redaction (v1.1+):**
  ADR-034 item "sensitive-value redaction" was declared out of scope for v1.0.0 (comment in
  RELEASE_PLAN_v1.md RC removed items). The current governance log emits raw config values;
  redaction of secrets or PII from governance events needs a v1.1+ implementation pass with a
  configurable redaction policy.
- **ADR-034 — `export_effective()` full schema stability contract (v1.1+):**
  `ResolvedConfigSnapshot.schema_version` is implemented, but the full guarantee that
  `export_effective()` output is a stability-versioned, consumer-safe schema contract (not just
  a diagnostic dump) has not been formalized. Requires a schema stability statement and a
  breaking-change policy for `export_effective()` output format. Target: v1.1+.
- **`multi_labels_enabled` / `interval_summary` — graduation from `**kwargs` to explicit typed surface (v1.1+):**
  Both parameters are currently consumed from `**kwargs` in `explain_factual` and
  `explore_alternatives` (documented as `[EXPERIMENTAL]` per ADR-038 §3). Before the
  multi-label surface can leave experimental status, `multi_labels_enabled` and
  `interval_summary` must be promoted to explicit keyword-only arguments and, if 3+
  multi-label tuning parameters are bundled together, wrapped in a `MultiLabelOptions`
  dataclass per ADR-038 §2c. The `**kwargs` forwarding path must be removed at that
  point per ADR-038 §3 graduation gate.
- **ADR-012 — sphinx-gallery adoption for executable example docs (v1.1+):**
  Current docs pipeline uses `nbconvert` for notebook execution (nightly-advisory with
  timeouts). ADR-012 identified `sphinx-gallery` as the preferred tool for executable,
  gallery-style documentation examples. Adoption is deferred until after the docs HTML/linkcheck
  CI job is wired (RC pre-condition above) and the v1.0.0 doc state is stable.

## Detailed status material relocation

Detailed ADR/Standards severity tables, gap inventories, and compliance-history notes were moved to `development/current-work/RELEASE_PLAN_status_appendix.md`.
Use that appendix as the single detailed status source; keep this master document focused on release control and milestone execution framing.
