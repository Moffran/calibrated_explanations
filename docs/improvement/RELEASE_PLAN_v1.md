> **Status note (2025-12-02):** Last edited 2025-12-02 · Archive after v1.0.0 GA · Implementation window: v0.9.0–v1.0.0 ·

# Release Plan to v1.0.0

## Current released version: v0.11.0

> Status: v0.11.0 shipped on 2026-03-02.


Maintainers: Core team
Scope: Concrete steps from v0.6.0 to a stable v1.0.0 with plugin-first execution.

## Terminology for improvement plans

These definitions apply across the improvement documents so schedule and gate language stays consistent.

- **Release milestone:** A versioned delivery gate (for example, v0.8.0, v0.9.1, v0.10.0) where checks must be green before shipping or cutting the branch.
- **Plan phase:** A numbered segment inside an uplift plan that groups related work and gates. Phases can span multiple iterations but map to specific release milestones when called out.
- **Stage:** Reserved for ADR-001 stage artifacts and legacy references. Outside ADR-001 material, prefer “phase” for plan sequencing and “milestone” for releases.
- **Gate:** A required check (test run, doc build, waiver review, etc.) that must pass within a phase before the corresponding release milestone can close.

Whenever a document references phases, iterations, or milestones, it uses the definitions above; any “stage” mentions outside ADR-001 should be treated as historical references rather than new schedule constructs.

## ADR gap closure roadmap

The ADR gap analysis enumerates open issues across the architecture. The breakdown below assigns every recorded gap to a remediation strategy and target release before v1.0.0. Severity values cite the unified scoring captured in the ADR status appendix of this document.

## ADR and Standards roadmap summary (gap details in appendix)

Gap-by-gap severity tables now live only in the ADR status appendix to avoid duplicate coverage. This section tracks the top-line status or release alignment for each active ADR. Superseded ADRs are listed only as pointers.

**ADR-001 - Package and Boundary Layout:** Completed; no open appendix gaps.

**ADR-002 - Exception Taxonomy and Validation Contract:** Completed; no open appendix gaps.

**ADR-003 - Caching Strategy:** Completed; no open appendix gaps.

**ADR-004 - Parallel Execution Framework:** Partially complete; v0.11.0 target is to close naming alignment and finalize `strategy="auto"` policy handling. Any remaining behavioral redesign defers to v0.11.1+.

**ADR-005 - Explanation Payload Schema:** Partially complete; v0.11.0 target is to close legacy adapter provenance propagation.

**ADR-006 - Plugin Trust Model:** Partially complete; v0.11.0 includes `PluginManager` shell, `PluginTrustPolicy`, public-surface cleanup, and accepted-registration audit events. Trust-state atomicity unification and legacy-list deprecation remain v0.11.1.

**ADR-007 - PlotSpec Abstraction:** Superseded by ADR-036/ADR-037; use ADR-036 for PlotSpec canonical contract and ADR-037 for visualization extension governance.

**ADR-008 - Explanation Domain Model:** Partially complete; major domain-authoritative migration and full round-trip/golden parity remain v0.11.1+ due cross-cutting scope.

**ADR-009 - Input Preprocessing and Mapping Policy:** Partially complete; v0.11.0 target is to close JSON-safe mapping export hardening and document helper placement decisions.

**ADR-010 - Optional Dependency Split:** Partially complete; v0.11.0 target is to add automated core-only vs extras parity checks.

**ADR-011 - Deprecation and Migration Policy:** Partially complete; v0.11.0 target is deprecation warning coverage for active compatibility shims. Legacy registry-list deprecation path remains v0.11.1.

**ADR-012 - Documentation & Gallery Build Policy:** Accepted; notebook execution/runtime ceilings and gallery-tooling decision documentation remain v0.11.1 hardening work.

**ADR-013 - Interval Calibrator Plugin Strategy:** Partially complete; protocol/signature alignment and fallback-chain strictness remain v0.11.1+.

**ADR-014 - Visualization Plugin Architecture:** Superseded by ADR-037; use ADR-037 for builder/renderer governance and runtime kind-extension policy.

**ADR-015 - Explanation Plugin Integration:** Partially complete; v0.11.0 target is invariant-enforcement consistency and bridge-surface hardening in line with ADR-026 tasking.

**ADR-016 - PlotSpec Separation and Schema:** Superseded by ADR-036/ADR-037; use ADR-036 for semantic contract and ADR-037 for rendering governance.

**ADR-020 - Legacy User API Stability:** Accepted (2026-03-03); `legacy_user_api_contract.md` updated for v0.11.0 removals. Remaining open appendix gaps (parity assertions, contributor workflow guidance) targeted v0.11.1.

**ADR-021 - Calibrated Interval Semantics:** Completed; no open appendix gaps.

**ADR-022 - Documentation Information Architecture:** Superseded by Standard-004; see Standard-004 for active status.

**ADR-023 - Matplotlib Coverage Exemption:** Completed; no open appendix gaps.

**ADR-024 - Legacy Plot Input Contracts:** Superseded/retired; maintained in `docs/maintenance/legacy-plotting-reference.md`.

**ADR-025 - Legacy Plot Rendering Semantics:** Superseded/retired; maintained in `docs/maintenance/legacy-plotting-reference.md`.

**ADR-026 - Explanation Plugin Semantics:** Partially complete; v0.11.0 target is to close context immutability and telemetry dependency metadata gaps.

**ADR-027 - FAST-Based Feature Filtering:** Partially complete; v0.11.0 target is observability policy/docs/examples closure.

**ADR-028 - Logging and Governance Observability:** Accepted (2026-03-03); enforcement tooling for domain-logger naming and Standard-005 observability example alignment remain v0.11.1 open gaps.

**ADR-029 - Reject Integration Strategy:** Accepted (2026-01-06); policy enum, strategy registry, and reject envelope direction documented in ADR-029. `RejectResult` → `RejectResultV2` public-API migration targeted for v1.0.0-rc; deprecation warning active from v0.11.x.

**ADR-030 - Test Quality Priorities and Enforcement:** Accepted; v0.11.0 delivered full detector extension and CI check-mode enforcement (assertion + determinism checks). Zero-tolerance ratification (marker hygiene, mutation testing policy) targets v0.11.3.

**ADR-031 - Calibrator Serialization & State Persistence:** Completed in v0.11.0; versioned `to_primitive`/`from_primitive` contracts plus `WrapCalibratedExplainer` save/load delivered. No open appendix gaps.

**ADR-032 - Guarded Explanation Semantics:** Accepted (scoped); schema-compatible representative-point guarded semantics and guarded auditability are authoritative for v0.11.x. Semantic identity, plugin-path identity, and whole-interval certification are explicitly out of scope.

**ADR-033 - Modality Extension Plugin Contract and Packaging Strategy:** Accepted; split across v0.11.0 (breaking metadata/resolver semantics) and v0.11.1 (CLI/shims/docs/packaging hardening).

**ADR-034 - Centralized Configuration Management:** Proposed (2026-03-03); Phase A complete in v0.11.1 (ConfigManager established, plugin/registry/logging paths migrated). Phase B in v0.11.2: migrate `cache`, `parallel`, `_feature_filter`, `orchestrator` and close allowlist. Promoted to Accepted after Phase B verification.

**ADR-036 - PlotSpec Canonical Contract and Validation Boundary:** Accepted (2026-03-20); canonical dataclass IR, builder output contract, validation boundary, and forbidden backend-leakage rules established. v0.11.1 delivers contract foundation. v0.11.2 follow-up: PlotSpec default-promotion readiness-gate definition and policy decision.

**ADR-037 - Visualization Extension and Rendering Governance:** Accepted (2026-03-20); builder/renderer contracts, deterministic extension metadata requirements, and default-path posture established. Legacy plotting remains default and PlotSpec opt-in in v0.11.1. Runtime plot-kind extension explicitly deferred. v0.11.2 follow-up: revisit default-path promotion and tighten readiness gate; runtime kind-extension policy decision.

**Standard-001 - Nomenclature Standardization:** Partially complete; v0.11.0 delivered naming guardrail automation and private-member allowlist emptied. Double-underscore mutation cleanup targets v0.11.1; final transitional shim removal targets v0.11.3.

**Standard-002 - Code Documentation Standardisation:** Partially complete; v0.11.0 target is wrapper/public numpydoc closure. Known gap (WrapCalibratedExplainer numpydoc blocks) targets v0.11.3.

**Standard-003 - Test Coverage Standard:** Completed; no open appendix gaps.

**Standard-004 - Documentation Standard (Audience Hubs):** Completed; no open appendix gaps.

**Standard-005 - Logging and Observability Standard:** Accepted (2026-01-15); enforcement tooling for domain-logger naming and observability example alignment with Standard-005 rules targeted v0.11.1 (ADR-028 open gaps 1–2).

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
| v0.11.2 | Gap audit quick-win docs updates only; no doc-build changes. | Minor maintenance only. | No new coverage work planned. | No new naming work; enforcement maintained. | ConfigManager completion (ADR-034 Phase B), ADR governance sweep, deep memory audit (retention/leak fixes), and PlotSpec default-promotion follow-up decision (ADR-036/ADR-037). |
| v0.11.3 | Minimal docs-build changes; Standard-002 numpydoc gap closure. | Close WrapCalibratedExplainer numpydoc blocks (Standard-002). | No new coverage work planned. | Final transitional shim removal (Standard-001). | RC readiness: Standard-001 shim closure, Standard-002 gap, ADR-030 zero-tolerance ratification, OSS perf harness (stretch). |
| v1.0.0 | Docs maintenance review; parity checks remain blocking. | Continuous improvement cadence; badge and quarterly reviews. | Waiver backlog should be zero; mutation/fuzzing exploration optional. | Final shim removals verified post-tag; legacy API guard tests green. | Stability declaration: RC contract freeze confirmed, production staging signed off, post-release maintenance cadences scheduled. |

### v0.6.x (stabilisation patches)

- Hardening: add regression tests for plugin parity, schema validation, and
  WrapCalibratedExplainer keyword defaults.
- Documentation polish: refresh plugin guide with registry/CLI examples and note
  compatibility guardrails.
- No behavioural changes beyond docs/tests.
- Coverage readiness: ratify Standard-003, publish `.coveragerc` draft with
  provisional exemptions, and record baseline metrics to size the remediation
  backlog.【F:docs/standards/STD-003-test-coverage-standard.md†L1-L74】【F:docs/improvement/archived/test_coverage_assessment.md†L1-L23】

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
   phase plans.【F:docs/improvement/Standard-001_nomenclature_remediation.md†L20-L28】【F:docs/improvement/code_documentation_uplift.md†L10-L28】
   - 2025-10-07 – Updated test helpers (`tests/conftest.py`, `tests/unit/core/test_calibrated_explainer_interval_plugins.py`) to comply with Ruff naming guardrails, keeping Standard-001 lint checks green.
   - 2025-10-07 – Harmonised `core.validation` docstring spacing with numpy-style guardrails to satisfy Standard-002 pydocstyle checks.
6. Implement Standard-003 phase 1 changes: ship shared `.coveragerc`, enable
   `--cov-fail-under=80` in CI, and document waiver workflow in contributor
   templates.【F:docs/standards/STD-003-test-coverage-standard.md†L34-L74】【F:docs/improvement/archived/coverage_uplift_plan.md†L9-L33】

Release gate: parity tests green for factual/alternative/fast, interval override
coverage exercised, CLI packaging verified, and nomenclature/doc lint warnings
live in CI with coverage thresholds enforcing ≥90% package-level coverage.

### v0.8.0 (plot routing, telemetry, and doc IA rollout)

1. Adopt Standard-004 (superseding ADR-022) by restructuring the documentation toctree into the audience-based information architecture (Getting Started, Practitioner, Researcher, Contributor hubs) and shipping the new telemetry concept page plus quickstart refactor per the information architecture plan.【F:docs/standards/STD-004-documentation-audience-standard.md†L1-L53】
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
   preprocessor metadata matches expectations.【F:docs/improvement/adrs/ADR-009-input-preprocessing-and-mapping-policy.md†L1-L80】
6. Execute Standard-001 Phase 2 renames with legacy shims isolated under a
   `legacy/` namespace and update imports/tests/docs to the canonical module
   names.【F:docs/improvement/Standard-001_nomenclature_remediation.md†L30-L33】
7. Complete Standard-002 baseline remediation by finishing pydocstyle batches C (`explanations/`, `perf/`) and D (`plugins/`), adding module summaries and
   upgrading priority package docstrings to numpydoc format with progress
   tracking.【F:docs/improvement/code_documentation_uplift.md†L17-L92】【F:docs/standards/STD-002-code-documentation-standard.md†L17-L62】
8. Extend Standard-003 enforcement to critical-path modules (≥95% coverage) and
   enable Codecov patch gating at ≥85% for PRs touching runtime/calibration
   logic, enable
   `--cov-fail-under=85` in CI.【F:docs/standards/STD-003-test-coverage-standard.md†L34-L74】【F:docs/improvement/archived/coverage_uplift_plan.md†L24-L33】
9. **Completed 2025-01-14:** Adopted ADR-023 to exempt `src/calibrated_explanations/viz/matplotlib_adapter.py` from coverage due to matplotlib 3.8.4 lazy loading conflicts with pytest-cov instrumentation. All 639 tests now pass with coverage enabled. Package-wide coverage maintained at 85%+.【F:docs/improvement/adrs/ADR-023-matplotlib-coverage-exemption.md†L1-L100】

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
7. **Complete ADR-012 doc workflow enforcement.** Keep Sphinx `-W`, gallery build, and linkcheck mandatory; extend CI smoke tests to run the refreshed quickstarts and fail if optional extras are presented without labels.【F:docs/improvement/adrs/ADR-012-documentation-and-gallery-build-policy.md†L1-L80】
8. **Turn Standard-002 tooling fully blocking.** Finish pydocstyle batches E (`viz/`, `viz/plots.py`, `legacy/plotting.py`) and F (`serialization.py`, `core.py`), capture and commit the baseline failure report before flipping enforcement, add the documentation coverage badge, and extend linting to notebooks/examples so the Phase 3 automation backlog is complete.【F:docs/improvement/code_documentation_uplift.md†L24-L92】
   - 2025-10-25 – Added nbqa-powered notebook linting and a 94% docstring
     coverage threshold to the lint workflow, making Standard-002's tooling fully
     blocking for documentation CI.
9. **Advance Standard-001 naming cleanup.** Prune deprecated shims scheduled for removal and ensure naming lint rules stay green on the release branch.【F:docs/improvement/Standard-001_nomenclature_remediation.md†L40-L44】【F:docs/standards/STD-001-nomenclature-standardization.md†L28-L37】
10. **Sustain Standard-003 coverage uplift.** Audit waiver inventory, retire expired exemptions, raise non-critical modules toward the 90% floor, enable `--cov-fail-under=88` in CI, and execute the module-level remediation efforts for interval regressors, registry/CLI, plotting, and explanation caching per the dedicated gap plan.【F:docs/improvement/archived/coverage_uplift_plan.md†L34-L111】
11. **Scoped runtime polish for explain performance.** Deliver the opt-in calibrator cache, multiprocessing toggle, and vectorised perturbation handling per ADR-003/ADR-004 analysis so calibrated explanations stay responsive without compromising accuracy. Capture improvements and guidance for plugin authors.【F:docs/improvement/adrs/ADR-003-caching-key-and-eviction.md†L1-L64】【F:docs/improvement/adrs/ADR-004-parallel-backend-abstraction.md†L1-L64】【F:src/calibrated_explanations/core/calibrated_explainer.py†L1750-L2150】 See the ADR-004 phase table above for task breakdown and ownership.

      - 2025-11-04 – Implemented opt-in calibrator cache with LRU eviction, multiprocessing toggle via ParallelExecutor facade, and vectorized perturbation handling. Added performance guidance for plugin authors in docs/contributor/plugin-contract.md. Cache and parallel primitives integrated into explain pipeline without altering calibration semantics.
12. **Plugin CLI, discovery, and denylist parity (optional extras).** Extend trust toggles and entry-point discovery to interval/plot plugins, add the `CE_DENY_PLUGIN` registry control highlighted in the OSS scope review, and ship the whole surface as opt-in so calibrated explanations remain usable without telemetry/CLI adoption.
13. **External plugin distribution path.** Document and test an aggregated installation extra (e.g., `pip install calibrated-explanations[external-plugins]`) that installs all supported external plugins, outline curation criteria, and add placeholders in docs and README for community plugin listings.

      - 2025-10-25 – Added a packaging regression test that inspects the
         `external-plugins` extra metadata to guarantee the curated bundle stays
         opt-in with the expected dependency pins.
14. **Explanation export convenience.** Provide `to_json()`/`from_json()` helpers on explanation collections that wrap schema v1 utilities and document them as optional aids for integration teams.
15. **Scope streaming-friendly explanation delivery.** Prototype generator or chunked export paths (or record a formal deferral) so memory-sensitive users know how large batches will be handled, capturing the outcome directly in the OSS scope inventory.【F:docs/improvement/OSS_CE_scope_and_gaps.md†L86-L118】

Release gate: Audience landing pages published with calibrated explanations/probabilistic regression foregrounded, research callouts present on all entry points, telemetry/performance extras labelled optional, docs CI (including quickstart smoke tests, notebook lint, and doc coverage badge) green, Standard-001/018/019 gates enforced, runtime performance enhancements landed without altering calibration outputs, plugin denylist control shipped, streaming plan recorded, and optional plugin extras (CLI/discovery/export) documented as add-ons.

### v0.9.1 (governance & observability hardening)

1. Implement ADR-011 policy mechanics—add the central deprecation helper, author the long-promised migration guide, and publish the structured status table with CI enforcement of the two-release window (see ADR status appendix in this document).
2. Bring docs CI into compliance with ADR-012 by executing notebooks during builds, installing official extras, timing tutorials, and documenting the chosen gallery tooling so drift is detected early (see ADR status appendix in this document).
3. Finish Standard-002 obligations by documenting wrapper APIs, interval calibrator signatures, and guard helpers to the mandated numpydoc standard (see ADR status appendix in this document).
4. Elevate coverage governance to the Standard-003 bar—raise thresholds to ≥90%, add per-module gates for prediction/serialization/registry paths, make the Codecov patch gate blocking, and track expiry metadata for waivers (see ADR status appendix in this document).
5. Reinforce ADR-020 legacy-API commitments with release checklist gates, regression tests for `explain_factual`/`explore_alternatives`, CONTRIBUTING guidance, and a scripted notebook audit workflow (see ADR status appendix in this document).
6. Restore visualization safety valves per ADR-023 by running the viz suite in CI, removing ignores, and aligning coverage messaging with the final thresholds (see ADR status appendix in this document).
7. Update governance collateral and hubs to satisfy Standard-004—embed the parity-review checklist in PR templates, reinstate the task API comparison, and publish the researcher future-work ledger (see ADR status appendix in this document).
8. Implement ADR-004 v0.9.1 scoped deliverable — ParallelExecutor: create a conservative execution layer that centralizes executor selection heuristics, exposes a minimal config surface (min_instances_for_parallel, min_features_for_parallel, task_size_hint_bytes), honors `CE_PARALLEL` overrides, emits compact decision telemetry (decision, reason, n_instances, n_features, bytes_hint, platform, executor_type), and includes unit tests plus a micro-benchmark harness. This is intentionally small and designed to collect field evidence before any full `ParallelExecutor` rollout in v0.10. 【F:docs/improvement/adrs/ADR-004-parallel-backend-abstraction.md†L1-L40】

Release gate: Deprecation dashboard live, docs CI runs with notebook execution, coverage/waiver gating enforced at ≥90%, legacy API and parity checklists signed, and visualization tests passing on the release branch (see ADR status appendix in this document).

### v0.10.0 (runtime boundary realignment)

1. Restructure packages to honour ADR-001—split calibration into its own package, eliminate cross-sibling imports, and formalise sanctioned namespaces with ADR addenda where necessary (see ADR status appendix in this document).
2. Deliver ADR-002 validation parity by replacing legacy exceptions with taxonomy classes, implementing shared validators, parameter guards, and consistent fit-state handling (see ADR status appendix in this document).
3. Complete ADR-003 caching deliverables: add invalidation/flush hooks, cache the mandated artefacts, emit telemetry, and align the backend with the cachetools+pympler stack or update the ADR rationale (see ADR status appendix in this document).
4. Implement ADR-004’s parallel execution backlog—auto strategy heuristics, telemetry with timings/utilisation, context management and cancellation, configuration surfaces, resource guardrails, fallback warnings, and automated benchmarking (see ADR status appendix in this document). Progress is tracked in the ADR-004 phase table above.
5. Enforce interval safety across bridges and exports to resolve ADR-021 and the ADR-015 predict-bridge gap, ensuring invariants, probability cubes, and serialization policies are honoured (see ADR status appendix in this document).
6. Align runtime plugin semantics with ADR-026 by adding invariant checks, hardening contexts, and extending telemetry payloads. Also internalise `CalibratedExplainer.explain` to reinforce the facade pattern and prevent public access (see ADR status appendix in this document).
7. Remove deprecated backward-compatibility alias `_is_thresholded()` from `CalibratedExplanations` class (superseded by `_is_probabilistic_regression()` in v0.9.0). Update any remaining external code or documentation that may reference the old method name. This completes the terminology standardization cycle from ADR-021.【F:docs/improvement/adrs/ADR-021-calibrated-interval-semantics.md†L119-L159】【F:docs/foundations/concepts/terminology_thresholded_vs_probabilistic_regression.md†L1-L24】
8. Condition source and discretizer branching: introduce `condition_source` configuration and thread it through `CalibratedExplainer`, `CalibratedExplanations`, orchestrators, and explanation instances so condition labels can be derived from either observed labels or calibrated predictions. Update discretizer construction to branch between observed-label and prediction-based label building and propagate the choice into `instantiate_discretizer` with validated defaults. Extend runtime helper tests to exercise both observed- and prediction-based condition sources and update discretizer interface stubs accordingly. Plan the user-visible default change (`condition_source="prediction"`) to land in v0.11.0 (or at latest in `v1.0.0-rc`) with an explicit upgrade note and migration guidance for any callers that relied on the historical observed-label behaviour.
9. Update the Docs with a comprehensive API reference for the public API of `CalibratedExplainer`, `WrapCalibratedExplainer`, `CalibratedExplanations`, `CalibratedExplanation`, `FactualExplanation`, and `AlternativeExplanation` including detailed descriptions of methods, parameters, return types, and usage examples. This will help users understand how to effectively utilize the library's capabilities.【F:docs/api_reference/calibrated_explainer.md†L1-L150】
10. **Anti-Pattern Remediation Phase 1:** Triage and categorize private member usage in tests. Rename and move test utilities (Category B) to public helpers to decouple tests from implementation details. See `docs/improvement/archived/ANTI_PATTERN_REMEDIATION_PLAN.md`.
11. **Close Standard-003 Phase 2 gates.** Execute the coverage uplift roadmap for this milestone: (a) complete the waiver audit with expiry metadata and refresh `.coveragerc`/`[tool.coverage.paths]` so Windows/WSL reports collapse to a single source of truth, (b) raise local + CI invocations (pytest + `make test-cov`) to `--cov-fail-under=90` while enabling the Codecov ≥88 % patch gate, and (c) deliver Iteration 3 remediation from the uplift plan—drive deterministic tests for `plugins/registry.py`, `plugins/builtins.py`, `plugins/cli.py`, and legacy plotting save-routing so trust toggles, CLI error paths, and renderer parity are all exercised before we cut the v0.10.0 branch.【F:docs/improvement/archived/coverage_uplift_plan.md†L24-L119】

Release gate: Package boundaries, validation/caching/parallel tests, interval invariants, terminology cleanup, and updated ADR status notes all green with telemetry dashboards verifying the new signals (see ADR status appendix in this document).

### v0.10.1 (schema & visualization contracts)

1. Confirm the v1 payload schema as the canonical contract — validate existing `explanation_schema_v1.json`, align validation helpers to payload semantics, and refresh fixtures/docs to reflect payload-first guidance (see ADR-005 status appendix).
2. Finish ADR-036 PlotSpec canonical-contract work: enhance `PlotSpec` dataclasses, validation coverage, and JSON boundary round-trips while preserving canonical dataclass authority (see ADR status appendix in this document).
3. Finish ADR-037 visualization governance work: harden builder/renderer contracts, metadata/default renderer governance, override handling, validation, CLI utilities, and documentation (see ADR status appendix in this document).
4. Maintain legacy plotting in the maintenance reference — ensure `docs/maintenance/legacy-plotting-reference.md` is authoritative for legacy behavior; avoid treating ADR-024/ADR-025 as active design gates (see ADR status appendix).
5. Document dynamically generated visualization classes to close the remaining Standard-002 docstring gap tied to plugin guides (see ADR status appendix in this document).
6. Prototype streaming-friendly explanation delivery (opt-in) — implement an opt-in, non-breaking generator API for large exports (e.g., `CalibratedExplanations.to_json_stream(chunk_size=256)` or `to_json(stream=True)`) that yields JSON Lines or safe chunked JSON pieces. Collect minimal export telemetry (`export_rows`, `chunk_size`, `mode` (`batch`|`stream`), `peak_memory_mb`, `elapsed_seconds`, `schema_version`, `feature_branch`) and validate the memory profile (reference target: 10k rows < 200 MB at `chunk_size=256`). Mark streaming as experimental until prototype validation completes and record follow-up actions in the release notes.
7. **Anti-Pattern Remediation Phase 2:** Refactor core internal tests (Category A) to use public APIs and remove dead code. This reduces brittleness and improves maintainability. See `docs/improvement/archived/ANTI_PATTERN_REMEDIATION_PLAN.md`.
8. **Open-source readiness plan (v0.10.1 → v1.0.0-rc):** add the following workstream tasks here and track them through the remaining milestones so everything lands before the v1.0.0-rc freeze.
   - **Repository structure & metadata:** add top-level community health files (`CODE_OF_CONDUCT.md`, `SECURITY.md`, and `GOVERNANCE.md`/`MAINTAINERS.md`), link them from the README, and document the maintainer/decision-making model in a lightweight, discoverable format.
   - **Documentation:** expand the API reference coverage beyond `CalibratedExplainer` to include CLI entry points, plugin registry contracts, serialization schema, and visualization APIs; add a README "documentation map" that links to API, architecture, contributor, and changelog pages.
   - **Quality & maintainability:** introduce a dependency vulnerability scan in CI (e.g., `pip-audit` or CodeQL), and add a reproducible dependency constraints/lockfile workflow for dev/CI to reduce drift. Constraints are used ONLY when absolutely necessary to avoid incompatibilities; otherwise, softer ranges from `requirements.txt` are preferred.
   - **Community & contribution:** create a `ROADMAP.md` that summarizes the release plan in contributor-facing language and link it from README/CONTRIBUTING; ensure issue/PR templates reference the new governance and security guidance.
   - **Licensing & governance:** add a contribution licensing statement (DCO or inbound=outbound clause) to CONTRIBUTING and clarify how contributions are licensed under BSD-3-Clause.

Release gate: Payload round-trips verified, PlotSpec/visualization plugin registries fully validated, legacy helpers behaving per ADR maintenance reference, and docs updated with new schema references (see ADR status appendix in this document)

### v0.10.2 (plugin trust & packaging compliance)

1. Enforce ADR-006 trust controls—manual approval for third-party trust flags, deny-list enforcement, diagnostics for skipped plugins, and documented sandbox warnings (see ADR status appendix in this document).
2. Close ADR-013 protocol gaps by validating calibrators, returning protocol-compliant FAST outputs, freezing contexts, providing CLI diagnostics, and returning frozen defaults (see ADR status appendix in this document).
3. Finish ADR-015 integration work: ship an in-tree FAST plugin, rebuild explanation collections with canonical metadata, tighten trust enforcement, align environment variables, and provide immutable plugin handles (see ADR status appendix in this document).
4. Deliver ADR-010 optional-dependency splits by trimming core dependencies, completing extras/lockfiles, auto-skipping viz tests without extras, updating docs, and extending contributor guidance (see ADR status appendix in this document).
5. Extend ADR-021/ADR-026 telemetry by surfacing FAST probability cubes, interval dependency hints, and frozen bin metadata in runtime payloads (see ADR status appendix in this document).
6. **Anti-Pattern Remediation Phase 3:** Enforce zero private member usage in tests via CI/Linting to prevent regression. See `docs/improvement/archived/ANTI_PATTERN_REMEDIATION_PLAN.md`.
7. Finalize ADR-027 implementation: align runtime logging with the observability policy (debug by default; warnings only in strict mode), document metadata exposure for per-instance ignore masks, and provide examples in performance tuning documentation (see ADR-027 status appendix).
8. Adopt ADR-028 logging and governance observability architecture and enforce Standard-005 logging and observability rules across core, plugins, and governance surfaces for v0.10.2-touched code paths, including domain-based logger usage, context propagation, and data minimisation (see ADR-028 status appendix and Standard-005).
9. Publish {doc}`adrs/ADR-029-reject-integration-strategy` (Reject Integration Strategy) decisions and open questions, and record the deferred strategy/visualization decisions with follow-up tasks in the v0.10.2 plan.
10. Add an interval summary selection enum to choose between regularized mean (default), mean, lower bound, or upper bound for probabilistic predictions and explanations, and document the task in the v0.10.2 plan.
11. Enforce Step 1 of ADR-030 test quality remediation: fix the 7 identified private-member usage violations in `tests/unit/calibration/test_summaries.py` and `tests/unit/core/test_config_helpers.py` to achieve zero violations per `scripts/detect_test_anti_patterns.py`.
12. Add `PluginDiscoveryReport` diagnostics and expose skipped/untrusted/denied entries via `ce.plugins report` or `ce.plugins list --include-skipped`, plus interval validation CLI commands (gate: CLI + unit tests).
13. Implement pyproject trust allowlist support (`[tool.calibrated_explanations.plugins].trusted`) with packaging guidance and tests for opt-in trust resolution (gate: docs + unit tests).
14. Add parity reference harness fixtures and `parity_compare` helper with a small CI job to validate canonical outputs (gate: parity fixtures and CI run green).
15. Require canonical in-tree FAST explanation plugin registration and enforce trusted-only resolution defaults, logging any explicit override for untrusted plugins (gate: plugin registry tests and legacy parity).
16. Ship plugin diagnostics and packaging documentation updates that reflect ADR-006/ADR-013/ADR-015 enforcement and warn explicitly about in-process, non-sandboxed execution (gate: doc updates in plugins guide)

Release gate: Plugin registries enforce trust and protocol policies, extras install cleanly with documentation parity, runtime telemetry captures interval metadata, FAST/CLI flows succeed end-to-end, and logging/governance observability align with ADR-028 and Standard-005 for all v0.10.2 changes (see ADR status appendix in this document).


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
       - Workflows to decommission prior to `v1.0.0-rc`: `test.yml` (compat wrapper), `coverage.yml`, `examples.yml`, and any legacy wrappers that duplicate new reusables. See `docs/improvement/CI-upgrade.md` for the full migration and removal plan.
       - Branch-protection flip: add new checks, verify two consecutive green runs, then remove old checks and monitor for 48 hours (see `docs/improvement/CI-upgrade.md`).

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
  5. Introduce `deprecate()` calls with a v0.13.0/v1.0.0 sunset date for the legacy `_REGISTRY`/`_TRUSTED` list-based path per ADR-011's two-minor-release requirement; document the migration path to the identifier-keyed descriptor dicts.
  6. Reinforce ADR-012 notebook/gallery execution by documenting the tooling choice and enforcing execution/time ceilings in docs CI.
  7. Close ADR-027/ADR-028 observability enforcement by adding logging standards examples and lint/tests.
  8. Finish Standard-001 nomenclature clean-up by eliminating double-underscore mutations, splitting utilities, and confining transitional shims to legacy/.
  9. Extend governance dashboards to surface lint status alongside preprocessing/domain-model telemetry. → relocated to v0.11.2.
  10. Decommission workflows: `test.yml` (compat wrapper), `coverage.yml`, `examples.yml`, and any legacy wrappers that duplicate new reusables. See `docs/improvement/CI-upgrade.md` for the full migration and removal plan. → relocated to v0.11.2.
  11. Ship ADR-033 additive UX/migration gates: CLI `--modality`, `vision`/`audio` shims that raise `MissingExtensionError` (`CE base + ImportError`), and version-pinned shim timeline (`v0.11.1` warn, `v1.0.0-rc` remove).
  12. Publish ADR-033 follow-through docs: contributor plugin contract updates, practitioner usage notes, and migration guidance for modality plugins.
  13. Add one ADR-033 packaging smoke test validating extension install + entry-point discovery/import behavior.
  14. Update the Reject Framework within Calibrated Explanations to include other forms of rejection beyond just binary conformal-based rejectors, such as uncertainty-based rejectors that leverage the uncertainty estimates from calibrated explanations to make informed decisions about when to abstain from making a prediction. Document the new rejector types and provide examples of how to implement and use them effectively in practice.
  15. Introduce a centralized package configuration layer via `ConfigManager` as the single runtime source of configuration truth. `ConfigManager` must resolve values from call-site overrides, environment variables, `pyproject.toml`, and hard defaults using one documented precedence contract; expose typed accessors for plugin resolution, telemetry, cache/parallel, and reject configuration; and replace ad-hoc config reads across core/plugins/CLI so behavior is deterministic, testable, and auditable.
   16. Promote ADR-020 from Draft → Accepted: update contract document for v0.11.0 removals, confirm release checklist ADR-020 gate active.
      - 2026-03-03 – ADR-020 promoted to Accepted; `legacy_user_api_contract.md` updated with v0.11.0 removal table (`explain_counterfactual`, `get_explanation`, deprecated aliases).
   17. Promote ADR-028 from Draft → Accepted before enforcement script (Task 7) merges, so task 7 enforces an authoritative policy record.
      - 2026-03-03 – ADR-028 promoted to Accepted.
   18. Update `docs/improvement/legacy_user_api_contract.md` for v0.11.0 removals and update status note date.
      - 2026-03-03 – Removed-in-v0.11.0 section added; `.get_explanation(i)` note corrected; status note updated.
   19. Add Standard-005 to ADR/Standards roadmap summary table in `RELEASE_PLAN_v1.md` and confirm v0.11.1 enforcement gate.
      - 2026-03-03 – Standard-005 row added to roadmap summary and Standards appendix; observability gaps assigned to v0.11.1 Task 7.
  20. Extend `GovernanceEvent` schema (introduced in Task 3 for plugin events) to cover `calibrated_explanations.governance.config` lifecycle events (resolve, export, validation-failure) per ADR-034 §4 + ADR-028; add caplog tests and CI governance-event schema gate.
  21. API-bloat removal program for v1.0.0 (ADR-011 + ADR-037 + ADR-020): keep LIME/SHAP as plugin-only surfaces by removing `explain_lime`/`explain_shap` and related wrapper/export hooks from core imports/public API, move adapters to plugin modules with lazy imports, and split heavy dependencies into optional extras. Execute under the deprecation protocol: add explicit deprecation warnings + migration docs in v0.11.1, enforce warning coverage and `CE_DEPRECATIONS=error` on deprecated-core paths, then remove the deprecated core entry points before cutting v1.0.0.
  22. PlotSpec hardening + ADR revisioning (ADR-036/ADR-037): harden PlotSpec as a canonical semantic IR by enforcing dataclass-only canonical in-memory representation, strengthening validator boundaries, unifying builder outputs to canonical PlotSpec, splitting rendering/normalization/export/test instrumentation responsibilities, and isolating compatibility handling to explicit serializer/translator boundaries. In the same task, publish and adopt ADR-036 + ADR-037 as the authoritative source of truth and supersede ADR-007/ADR-014/ADR-016 as historical records. Keep legacy plotting as the default public `.plot()` path in v0.11.1 and keep runtime plot-kind extension out of scope for this release.

   Release gate: `PluginManager` owns all plugin resolution; trust state is atomic across descriptor and set; governance audit events cover both accepted and rejected registrations (including `governance.config` events from ConfigManager); test-helper bodies no longer live in the production module; legacy list deprecation warnings are active and CI enforces `CE_DEPRECATIONS=error` for tests that exercise the legacy path; ADR-012/027/028/001/033 additive rollout gates (CI/docs/shims/packaging smoke test) are green; ADR-020 and ADR-028 promoted to Accepted; `ConfigManager` is the authoritative configuration entry point with precedence and migration tests green; core package import/public API no longer hard-depends on LIME/SHAP adapters (plugin-only); PlotSpec canonical-contract hardening is implemented with boundary-only compatibility translation; ADR-036/ADR-037 are authoritative and ADR-007/014/016 are superseded; legacy plotting remains default in v0.11.1; and runtime plot-kind extension remains disabled.

### v0.11.2 (config hardening and ADR governance sweep)

  1. Migrate `cache/cache.py`, `parallel/parallel.py`,
     `core/explain/_feature_filter.py`, `core/prediction/orchestrator.py` to
     `ConfigManager`. Done when all four files are removed from the CI allowlist
     (ADR-034 Phase B).
  2. Finalize ADR-034 (pulled forward from v1.0.0-rc; no RC dependency): confirm
     CI allowlist reaches zero or documents any remaining sanctioned boundaries;
     promote ADR-034 status from Proposed to Accepted with an implementation summary.
     Depends on task 1 completing green.
  3. ADR and standards governance sweep (pulled forward from v1.0.0-rc; rolling audit;
     no RC dependency): audit all open gaps in the RELEASE_PLAN_v1.md appendix, confirm
     every gap is assigned to a milestone or marked superseded, refresh gap-analysis
     dates older than 60 days, and close any quick-win gaps that require no feature
     work (e.g., remaining ADR-026 observability gaps).
  4. Governance dashboard extension (relocated from v0.11.1 Task 9; depends on v0.11.1 Tasks 3 and 7 completing): surface lint status and preprocessing/domain telemetry in a machine-readable governance status artifact. Fits v0.11.2 governance sweep theme; ADR-028 CLEAR.
  5. CI workflow decommissioning (relocated from v0.11.1 Task 10; depends on v0.11.1 Task 6 replacement workflows stabilizing): remove legacy wrapper workflows (`test.yml`, `coverage.yml`, `examples.yml`) and update branch protection references. ADR-011 does not apply (CI wrappers are not public API).
  6. Deep memory audit and retention hardening.
  7. PlotSpec default-promotion follow-up (new in v0.11.2): revisit whether PlotSpec should move from opt-in to default path; define and enforce a stricter readiness gate; update PlotSpec plots and flip defaults only if that gate is satisfied. This follow-up is the explicit decision point governed by ADR-036 and ADR-037.

  Release gate: All four allowlisted modules migrated and removed from CI allowlist; ADR-034 status promoted to Accepted; governance sweep complete with all appendix gaps assigned or superseded; governance status artifact schema documented and CI-generated; legacy CI wrapper workflows removed with replacement workflows confirmed green; PlotSpec default-promotion follow-up completed with explicit readiness-gate decision recorded; `make local-checks-pr` passes.

### v0.11.3 (RC readiness: Standard closure, ADR-030 ratification, docs gap)

  1. Close Standard-001 (pulled forward from v1.0.0-rc; depends on v0.11.1 Standard-001
     cleanup): remove remaining transitional shims from `legacy/`, confirm naming/tooling
     enforcement green on main.【F:docs/improvement/Standard-001_nomenclature_remediation.md†L40-L44】
  2. Close Standard-002 known gap (implementation pulled forward from v1.0.0-rc; RC only
     verifies): add full numpydoc blocks to `WrapCalibratedExplainer` and remaining
     stable public surfaces identified in the appendix.【F:docs/improvement/code_documentation_uplift.md†L24-L92】【F:docs/standards/STD-002-code-documentation-standard.md†L43-L62】
  3. Ratify ADR-030 zero-tolerance enforcement (pulled forward from v1.0.0; ratification
     should inform RC readiness): confirm extended anti-pattern scans are CI-blocking,
     document marker hygiene rules in the ADR, and declare mutation testing optional for
     core modules.【F:docs/improvement/adrs/ADR-030-test-quality-priorities-and-enforcement.md†L1-L50】
  4. (Stretch) OSS performance harness template (pulled forward from v1.0.0-rc;
     self-contained): provide a reusable harness for semi/fully-online latency
     measurements with README guidance and a sample run (gate: sample run documented
     and reproducible).

  Release gate: Standard-001 naming lint green with all transitional shims removed; Standard-002 WrapCalibratedExplainer numpydoc gap closed and docstring coverage ≥90%; ADR-030 zero-tolerance enforcement CI-blocking with ratification note in ADR; `make local-checks-pr` passes.

### v1.0.0-rc (release candidate readiness)

<!-- Removed items (evidence in parentheses):
  - #6 Institutionalise Standard-003 → STALE: appendix confirms "STD-003 fully compliant (2026-02-27): no further action required"; CI enforcement already in place
  - #7 Promote ADR-026 / deprecate ADR-024/025 → STALE: ADR-026 already Accepted (Status: 2026-01-12); ADR-024/025 already Retired with superseded-prefix files
  - #10 ADR gap closure audit → moved to v0.11.2
  - #11 condition_source default change → DONE: delivered in v0.10.3 CHANGELOG
  - #12 ADR-030 tooling extension → DONE: delivered in v0.11.0 CHANGELOG (assertion + determinism checks; script reorganized to scripts/quality/)
  - #13 ADR-031 calibrator persistence → DONE: delivered in v0.11.0 CHANGELOG (to_primitive/from_primitive + WrapCalibratedExplainer save/load)
  - #15 ADR/standards docs gap closure → moved to v0.11.2
  - #16 ADR-034 finalization → moved to v0.11.2
  Items #3 (Standard-001 shim removal) and #4 (Standard-002 gap closure) moved to v0.11.3.
  Item #14 (OSS performance harness) moved to v0.11.3.
-->

1. Freeze Explanation Schema v1, publish draft compatibility statement, and
   communicate that only patch updates will follow for the schema.【F:docs/schema_v1.md†L1-L120】
2. Reconfirm wrap interfaces and exception taxonomy against v0.6.x contracts,
   updating README & CHANGELOG with a release-candidate compatibility note.【F:src/calibrated_explanations/core/wrap_explainer.py†L260-L471】【F:src/calibrated_explanations/core/exceptions.py†L1-L63】
3. Validate the new caching/parallel toggles in staging, document safe defaults
   for RC adopters, and ensure telemetry captures cache hits/misses and worker
   utilisation metrics for release sign-off.【F:docs/improvement/adrs/ADR-003-caching-key-and-eviction.md†L28-L64】【F:docs/improvement/adrs/ADR-004-parallel-backend-abstraction.md†L25-L64】
4. Verify Standard-002 compliance at ≥90% docstring coverage holds for the RC branch
   (the WrapCalibratedExplainer gap was closed in v0.11.3; this is a verification gate,
   not an implementation task).
5. Launch the versioned documentation preview and public doc-quality dashboards
   (coverage badge, doc lint, notebook lint) described in the information
   architecture plan so stakeholders can validate the structure ahead of GA.【F:docs/improvement/documentation_information_architecture.md†L108-L118】
6. Provide an RC upgrade checklist covering environment variables, pyproject
   settings, CLI usage, caching controls, and plugin integration testing
   expectations.

Release gate: Schema v1 freeze documented; wrap interface and exception taxonomy compatibility confirmed; caching/parallel staging validation signed off; Standard-002 ≥90% verified; versioned docs preview and doc-quality dashboards live; upgrade checklist ready for pilot customers.

### v1.0.0 (stability declaration)

<!-- Removed item #6 (Ratify ADR-030): moved to v0.11.3 so ratification informs RC readiness. -->

1. Announce the stable plugin/telemetry contracts and publish the final
   compatibility statement across README, CHANGELOG, and docs hub.
2. Tag the v1.0.0 release, backport documentation to downstream extension
   repositories, and circulate the upgrade checklist to partners with caching
   and parallelisation guidance.
3. Validate telemetry, plugin registries, cache behaviour, and worker scaling in
   production-like staging, signing off with no pending high-priority bugs.
4. Confirm Standard-001/Standard-002 guardrails remain enforced post-tag, monitor the
   caching/parallel telemetry dashboards, and schedule maintenance cadences
   (coverage/docstring audits, performance regression sweeps) for the first
   patch release.
5. Finalise versioned documentation hosting and publish long-term dashboard
   links (coverage, doc lint, notebooks) so the IA plan’s success metrics are met
   when GA lands.【F:docs/improvement/documentation_information_architecture.md†L108-L118】

Release gate: Tagged release artifacts available, documentation hubs updated with
versioned hosting and public dashboards, caching/parallel toggles operating
within documented guardrails, staging validation signed off, and post-release
maintenance cadences scheduled.

## Standard-003 integration analysis

- **Scope alignment:** The release milestones already emphasise testing and
  documentation maturity; Standard-003 adds explicit quantitative coverage gates that
   complement Standard-001/Standard-002 quality goals without altering plugin-focused
   scope.【F:docs/standards/STD-003-test-coverage-standard.md†L34-L74】
- **Milestone sequencing:** Early v0.6.x tasks capture baseline metrics and
   prepare `.coveragerc`, v0.7.0 introduces CI thresholds, v0.8.0 widens
   enforcement to critical paths and patch checks, and v0.9.0 retires waivers
   ahead of the release candidate. This staging keeps debt burn-down parallel to
   existing plugin/doc improvements.【F:docs/improvement/archived/coverage_uplift_plan.md†L11-L48】
- **Release readiness:** By v1.0.0, coverage gating is embedded in branch
   policies and telemetry/documentation communications, ensuring Standard-003 remains
   sustainable beyond the initial rollout.【F:docs/standards/STD-003-test-coverage-standard.md†L34-L74】【F:docs/improvement/archived/coverage_uplift_plan.md†L11-L48】

## Post-1.0 considerations

- Continue monitoring caching and parallel execution telemetry to determine
  whether the opt-in defaults can graduate to on-by-default in v1.1, updating
  ADR-003/ADR-004 rollout notes as needed.【F:docs/improvement/adrs/ADR-003-caching-key-and-eviction.md†L28-L64】【F:docs/improvement/adrs/ADR-004-parallel-backend-abstraction.md†L25-L64】
- Evaluate additional renderer plugins (plotly) after verifying PlotSpec default
  adoption.
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
