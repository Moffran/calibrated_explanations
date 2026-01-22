> **Status note (2025-12-02):** Last edited 2025-12-02 · Archive after v1.0.0 GA · Implementation window: v0.9.0–v1.0.0 ·

# Release Plan to v1.0.0

## Current released version: v0.10.2

> Status: v0.10.2 shipped on 2026-01-22.


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

## ADR roadmap summary (gap details in appendix)

Gap-by-gap severity tables now live only in the ADR status appendix to avoid duplicate coverage. This section tracks the top-line status or release alignment for each active ADR. Superseded ADRs are listed only as pointers.

**ADR-001 – Package and Boundary Layout:** Completed; boundary realignment delivered per ADR-001 decisions. See appendix for retired gap table.
**ADR-002 – Exception Taxonomy and Validation Contract:** Completed with taxonomy adoption and validator parity; appendix holds the consolidated gap status.
**ADR-003 – Caching Strategy:** Completed in v0.10.0. Opt-in in-process LRU cache implemented with deterministic keys, eviction controls, and safe fallbacks; docs and tests updated.
**ADR-004 - Parallel Execution Framework:** Completed with `ParallelExecutor` and instance-parallel execution; remaining work is documentation/name alignment (`ParallelFacade` legacy references) targeted for v0.11.0.
**ADR-005 - Explanation Payload Schema:** Clarified to favour a payload-first v1 contract for v0.10.1; envelope and richer registry deferred. Remaining work (strict validator + fixtures) targeted for v0.11.0.
**ADR-006 – Plugin Trust Model:** Completed. Trust/deny controls, diagnostics, and governance logging are in place; appendix updated to reflect closure.
**ADR-007 – PlotSpec Abstraction:** Completed. `PlotSpec` IR, schema/versioning, validation, and headless export support implemented; registry extensions are optional.
**ADR-008 – Explanation Domain Model:** Domain-model hardening targeted for v0.11.0; ADR clarifies domain/legacy round-trips and remaining structured metadata gaps.
**ADR-009 – Input Preprocessing and Mapping Policy:** Automation and preprocessing persistence align to v0.11.0; appendix lists outstanding enforcement gaps.
**ADR-010 - Core vs Evaluation Split:** Completed for extras; remaining action is to verify core-only installs do not require matplotlib at import time and align CI accordingly (target v0.11.0).
**ADR-011 – Deprecation and Migration Policy:** Completed: central `deprecate()` helper and migration guidance implemented; CI gates added for deprecation enforcement.
**ADR-012 - Documentation & Gallery Build Policy:** Accepted. Notebooks/gallery rendering clarified as advisory on mainline and blocking on release branches; remaining work (executed notebooks + runtime ceilings) targeted for v0.11.0.
**ADR-013 – Interval Calibrator Plugin Strategy:** Completed. Protocol validation and CLI diagnostics implemented; remaining notes are documentation-only.
**ADR-014 – Visualization Plugin Architecture:** Completed. Plot plugin registries, validation hooks, and CLI helpers implemented; docs alignment remains.
**ADR-015 – Explanation Plugin Integration:** Completed for in-tree FAST plugin and trust enforcement; remaining hardening tracked under ADR-026.
**ADR-016 – PlotSpec Separation and Schema:** Completed with schema metadata and validation hooks in builders/serializers.
**ADR-030 – Test Quality Priorities and Enforcement:** Accepted. Baseline enforcement in CI; extended tooling (determinism/assertion checks) planned for v0.11.0.
**ADR-020 - Legacy User API Stability:** Legacy contract enforcement and parity tests tracked across v0.9.1-v0.10.x release gates; remaining release checklist + audit workflow targeted for v0.11.0.
**ADR-021 – Calibrated Interval Semantics:** Completed in v0.10.0; invariant enforcement and JSON-safe serialization implemented.
**ADR-022 – Documentation Information Architecture:** Superseded by Standard-004; see Standard-004 for IA expectations.
**ADR-023 – Matplotlib Coverage Exemption:** Completed with coverage exemptions and a CI viz-only job in place.
**ADR-024 – Legacy Plot Input Contracts:** Deprecated; legacy plotting maintained in `docs/maintenance/legacy-plotting-reference.md` and gaps retired.
**ADR-025 – Legacy Plot Rendering Semantics:** Deprecated; pixel-level rendering semantics moved to maintenance reference and gaps retired.
**ADR-026 - Explanation Plugin Semantics:** Partially complete; strict invariant enforcement and immutability remain open and targeted for v0.11.0 (see appendix).
**ADR-027 - FAST-Based Feature Filtering:** Partially complete; remaining observability policy and examples targeted for v0.11.0 (see appendix).
**ADR-028 - Logging and Governance Observability:** Partially complete; remaining enforcement tooling and examples targeted for v0.11.0 (see appendix).
**ADR-031 – Calibrator Serialization & State Persistence:** Drafted for v0.11.x; versioned calibrator primitives and explainer save/load contracts to be implemented after mapping persistence work.
**Standard-001 – Nomenclature Standardization:** Remediation plan in `Standard-001_nomenclature_remediation.md`; phased renames and shims tracked across releases.
**Standard-002 – Code Documentation Standardisation:** Accepted and rolled out in batches; numpydoc/pydocstyle enforcement staged with release branches treated as blocking once baselines met.
**Standard-003 – Test Coverage Standard:** Accepted. Package floor and critical-path thresholds defined (90% package, 95% critical paths); CI/gates staged and waiver governance enforced per appendix.
**Standard-004 - Documentation Standard (Audience Hubs):** Completed (IA + guardrails). Plot plugin authoring/override guidance follow-through targeted for v0.11.0.

## Release milestones

### Uplift status by milestone

| Release | Documentation overhaul (ADR-012/027) | Code docs (Standard-002) | Coverage uplift (Standard-003) | Naming (Standard-001) | Notes |
| --- | --- | --- | --- | --- | --- |
| v0.8.0 | IA restructure landed; Sphinx/linkcheck gates required for merges. | Phase 0 primer circulated; baseline inventory started. | `.coveragerc` drafted; `fail_under=80` staged. | Lint guards enabled; shim inventory captured. | Rollback: revert to pre-IA toctree if Sphinx fails; waivers expire after one iteration. |
| v0.9.1 | Audience hubs sustained; quickstart smoke tests block release. | Phase 1 batches A–C targeted for ≥90%; docs examples aligned with new IA. | XML+Codecov upload required; gating rising toward 90%. | Phase 1 cleanup in progress with measurement checkpoints. | Rollback: pin docs to last green build; coverage waivers require dated follow-up issues. |
| v0.10.0 | Doc gate holds prior bar; no new IA work planned. | Phase 2 blocking check for touched files; waivers time-bounded. | `fail_under=90` enforced; plugin/plotting module thresholds scheduled. | Release mapping added; refactors aligned with legacy API tests. | Risk: refactors (explain plugins, boundary split) may churn coverage; branch cut requires rerunning gates. |
| v0.10.1 | Doc hubs refreshed with telemetry/performance opt-in notes. | Package-wide ≥90% expected; notebook/example lint extended. | Module thresholds hardened; waiver expiry versions mandatory. | Phase metrics reviewed; % renamed modules tracked. | Rollback: if module gates fail, defer release or lower threshold with explicit expiry in checklist. |
| v0.10.2 | No changes planned. | No changes planned. | No changes planned. | No changes planned. | Test quality remediation: fix private-member violations in tests per ADR-030. |
| v0.11.0 | Notebook execution + runtime ceilings (ADR-012); observability enforcement docs (ADR-027/028). | No changes planned. | No changes planned. | No changes planned. | ADR gap closure milestone: ADR-004/005/008/009/010/012/020/026/027/028/030/031 + Standard-004 follow-through. |
| v1.0.0 | Docs maintenance review; parity checks remain blocking. | Continuous improvement cadence; badge and quarterly reviews. | Waiver backlog should be zero; mutation/fuzzing exploration optional. | Final shim removals completed; legacy API guard tests green. | Test quality ratification: zero-tolerance enforcement for new quality rules per ADR-030; risks surface inline in milestone gates; rollback paths documented per gate. |

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
   templates.【F:docs/standards/STD-003-test-coverage-standard.md†L34-L74】【F:docs/improvement/coverage_uplift_plan.md†L9-L33】

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
   `--cov-fail-under=85` in CI.【F:docs/standards/STD-003-test-coverage-standard.md†L34-L74】【F:docs/improvement/coverage_uplift_plan.md†L24-L33】
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

   - 2025-11-06 – Consolidated the plugin story into a Plugins hub (`docs/plugins.md`), added a practitioner-focused "Use external plugins" guide (`docs/practitioner/advanced/use_plugins.md`), and surfaced the curated `external-plugins` extra in installation docs. Cross-linked the appendix index and ensured practitioner/contributor flows are consistent with Standard-004/ADR-006/ADR-014/ADR-026.
4. **Label telemetry and performance scaffolding as optional tooling.** Move telemetry schema/how-to material into contributor governance sections, ensure practitioner guides mention telemetry only for compliance scenarios, and audit navigation labels to avoid implying these extras are mandatory.【F:docs/improvement/documentation_information_architecture.md†L70-L113】
5. **Highlight research pedigree throughout.** Keep the existing research hub mentions in the Overview, practitioner quickstarts, and probabilistic regression concept pages; ensure they cross-link citing.md and key publications in the relevant sections without introducing new banner UI.【F:docs/improvement/documentation_review.md†L15-L34】
6. **Triangular alternatives plots everywhere alternatives appear.** Update explanation guides, PlotSpec docs, and runtime examples so `explore_alternatives` also introduces the triangular plot and its interpretation.
7. **Complete ADR-012 doc workflow enforcement.** Keep Sphinx `-W`, gallery build, and linkcheck mandatory; extend CI smoke tests to run the refreshed quickstarts and fail if optional extras are presented without labels.【F:docs/improvement/adrs/ADR-012-documentation-and-gallery-build-policy.md†L1-L80】
8. **Turn Standard-002 tooling fully blocking.** Finish pydocstyle batches E (`viz/`, `viz/plots.py`, `legacy/plotting.py`) and F (`serialization.py`, `core.py`), capture and commit the baseline failure report before flipping enforcement, add the documentation coverage badge, and extend linting to notebooks/examples so the Phase 3 automation backlog is complete.【F:docs/improvement/code_documentation_uplift.md†L24-L92】
   - 2025-10-25 – Added nbqa-powered notebook linting and a 94% docstring
     coverage threshold to the lint workflow, making Standard-002's tooling fully
     blocking for documentation CI.
9. **Advance Standard-001 naming cleanup.** Prune deprecated shims scheduled for removal and ensure naming lint rules stay green on the release branch.【F:docs/improvement/Standard-001_nomenclature_remediation.md†L40-L44】【F:docs/standards/STD-001-nomenclature-standardization.md†L28-L37】
10. **Sustain Standard-003 coverage uplift.** Audit waiver inventory, retire expired exemptions, raise non-critical modules toward the 90% floor, enable `--cov-fail-under=88` in CI, and execute the module-level remediation efforts for interval regressors, registry/CLI, plotting, and explanation caching per the dedicated gap plan.【F:docs/improvement/coverage_uplift_plan.md†L34-L111】
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
8. Implement ADR-004 v0.9.1 scoped deliverable — ParallelFacade: create a conservative facade that centralizes executor selection heuristics, exposes a minimal config surface (min_instances_for_parallel, min_features_for_parallel, task_size_hint_bytes), honors `CE_PARALLEL` overrides, emits compact decision telemetry (decision, reason, n_instances, n_features, bytes_hint, platform, executor_type), and includes unit tests plus a micro-benchmark harness. This is intentionally small and designed to collect field evidence before any full `ParallelExecutor` rollout in v0.10. 【F:docs/improvement/adrs/ADR-004-parallel-backend-abstraction.md†L1-L40】

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
10. **Anti-Pattern Remediation Phase 1:** Triage and categorize private member usage in tests. Rename and move test utilities (Category B) to public helpers to decouple tests from implementation details. See `docs/improvement/ANTI_PATTERN_REMEDIATION_PLAN.md`.
11. **Close Standard-003 Phase 2 gates.** Execute the coverage uplift roadmap for this milestone: (a) complete the waiver audit with expiry metadata and refresh `.coveragerc`/`[tool.coverage.paths]` so Windows/WSL reports collapse to a single source of truth, (b) raise local + CI invocations (pytest + `make test-cov`) to `--cov-fail-under=90` while enabling the Codecov ≥88 % patch gate, and (c) deliver Iteration 3 remediation from the uplift plan—drive deterministic tests for `plugins/registry.py`, `plugins/builtins.py`, `plugins/cli.py`, and legacy plotting save-routing so trust toggles, CLI error paths, and renderer parity are all exercised before we cut the v0.10.0 branch.【F:docs/improvement/coverage_uplift_plan.md†L24-L119】

Release gate: Package boundaries, validation/caching/parallel tests, interval invariants, terminology cleanup, and updated ADR status notes all green with telemetry dashboards verifying the new signals (see ADR status appendix in this document).

### v0.10.1 (schema & visualization contracts)

1. Confirm the v1 payload schema as the canonical contract — validate existing `explanation_schema_v1.json`, align validation helpers to payload semantics, and refresh fixtures/docs to reflect payload-first guidance (see ADR-005 status appendix).
2. Finish ADR-007 and ADR-016 schema work: enhance `PlotSpec` dataclasses, registries, validation coverage, JSON round-trips, and headless export paths (see ADR status appendix in this document).
3. Restore ADR-014 visualization plugin architecture with working fallback builders, helper base classes, metadata/default renderers, override handling, validation, CLI utilities, and documentation (see ADR status appendix in this document).
4. Maintain legacy plotting in the maintenance reference — ensure `docs/maintenance/legacy-plotting-reference.md` is authoritative for legacy behavior; avoid treating ADR-024/ADR-025 as active design gates (see ADR status appendix).
5. Document dynamically generated visualization classes to close the remaining Standard-002 docstring gap tied to plugin guides (see ADR status appendix in this document).
6. Prototype streaming-friendly explanation delivery (opt-in) — implement an opt-in, non-breaking generator API for large exports (e.g., `CalibratedExplanations.to_json_stream(chunk_size=256)` or `to_json(stream=True)`) that yields JSON Lines or safe chunked JSON pieces. Collect minimal export telemetry (`export_rows`, `chunk_size`, `mode` (`batch`|`stream`), `peak_memory_mb`, `elapsed_seconds`, `schema_version`, `feature_branch`) and validate the memory profile (reference target: 10k rows < 200 MB at `chunk_size=256`). Mark streaming as experimental until prototype validation completes and record follow-up actions in the release notes.
7. **Anti-Pattern Remediation Phase 2:** Refactor core internal tests (Category A) to use public APIs and remove dead code. This reduces brittleness and improves maintainability. See `docs/improvement/ANTI_PATTERN_REMEDIATION_PLAN.md`.
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
6. **Anti-Pattern Remediation Phase 3:** Enforce zero private member usage in tests via CI/Linting to prevent regression. See `docs/improvement/ANTI_PATTERN_REMEDIATION_PLAN.md`.
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

### v0.11.0 (domain model & preprocessing finalisation)

1. Close ADR-008 domain model authority—run runtime flows on domain objects, fix legacy round-trips, add calibration/model metadata, publish golden fixtures, and harden `_safe_pick` (see ADR status appendix in this document).
2. Complete ADR-009 preprocessing automation with `auto_encode='auto'`, unseen-category enforcement, mapping export/import helpers, dtype diagnostics, and aligned telemetry/docs (see ADR status appendix in this document).
3. Deliver ADR-030 test quality tooling upgrades (assertion + determinism checks) and wire them into CI for v1.0.0 readiness.
4. Add ADR-031 calibrator persistence: versioned `to_primitive`/`from_primitive` contracts plus `Explainer.save_state()`/`load_state()` (gate: round-trip tests and schema version policy).
5. Address ADR-005 semantic payload validation gaps with a strict validator (schema + invariants) and fixture coverage.
6. Reinforce ADR-012 notebook/gallery execution by documenting the tooling choice and enforcing execution/time ceilings in docs CI.
7. Add ADR-020 legacy API stability gates (release checklist + notebook/API audit workflow).
8. Harden ADR-026 plugin semantics with strict invariant enforcement, immutable contexts, and telemetry completeness.
9. Resolve ADR-004 naming drift and ADR-010 core-only dependency clarity (matplotlib import-time requirement).
10. Close ADR-027/ADR-028 observability enforcement by adding logging standards examples and lint/tests.
11. Track Standard-004 follow-through for plot plugin authoring/override guidance.
12. Finish Standard-001 nomenclature clean-up by eliminating double-underscore mutations, splitting utilities, reporting lint telemetry, and confining transitional shims to `legacy/` (see ADR status appendix in this document).
13. Extend governance dashboards to surface lint status alongside preprocessing/domain-model telemetry, ensuring ongoing monitoring after v1.0.0 (see ADR status appendix in this document).
14. Change default behavior for `condition_source` to `"prediction"` in `CalibratedExplainer` and related components. Update all relevant documentation, including the upgrade checklist, to inform users of this change and provide guidance on how to adjust their implementations if they previously relied on the default `"observed"` setting. This change aims to enhance the consistency of calibrated explanations by basing condition labels on model predictions rather than observed labels. Plan for this change to be communicated clearly in the v1.0.0 release notes and upgrade guides (see ADR status appendix in this document).
15. Empty the private member allow-list (`.github/private_member_allowlist.json`) as part of the final Pattern 1 remediation hardening. All remaining private member usages in tests must be refactored to public APIs or justified as permanent exceptions in the remediation plan.
16. Perform a final ADR, standards, and improvement docs gap closure sweep and update the appendix for any remaining gaps.

Release gate: ADR-005/008/009/012/020/026/030/031 gaps are closed or explicitly deferred, observability enforcement is in place (ADR-027/028), and core-only install expectations are verified ahead of v1.0.0-rc (see ADR status appendix in this document)


### v1.0.0-rc (release candidate readiness)

1. Freeze Explanation Schema v1, publish draft compatibility statement, and
   communicate that only patch updates will follow for the schema.【F:docs/schema_v1.md†L1-L120】
2. Reconfirm wrap interfaces and exception taxonomy against v0.6.x contracts,
   updating README & CHANGELOG with a release-candidate compatibility note.【F:src/calibrated_explanations/core/wrap_explainer.py†L260-L471】【F:src/calibrated_explanations/core/exceptions.py†L1-L63】
3. Close Standard-001 by removing remaining transitional shims and ensure naming/tooling
   enforcement is green on the release branch.【F:docs/improvement/Standard-001_nomenclature_remediation.md†L40-L44】
4. Maintain Standard-002 compliance at ≥90% docstring coverage and outline the
   ongoing maintenance workflow in the RC changelog section.【F:docs/improvement/code_documentation_uplift.md†L24-L92】【F:docs/standards/STD-002-code-documentation-standard.md†L43-L62】
5. Validate the new caching/parallel toggles in staging, document safe defaults
   for RC adopters, and ensure telemetry captures cache hits/misses and worker
   utilisation metrics for release sign-off.【F:docs/improvement/adrs/ADR-003-caching-key-and-eviction.md†L28-L64】【F:docs/improvement/adrs/ADR-004-parallel-backend-abstraction.md†L25-L64】
6. Institutionalise Standard-003 by baking coverage checks into release branch
   policies, publishing a health dashboard (Codecov badge + waiver log), and
   enforcing `--cov-fail-under=90` in CI.【F:docs/standards/STD-003-test-coverage-standard.md†L34-L74】【F:docs/improvement/coverage_uplift_plan.md†L24-L33】
7. Promote ADR-026 from Draft to Accepted with implementation summaries so plugin semantics remain authoritative before the freeze; ADR-024 and ADR-025 are deprecated and their legacy parity guidance moved to the maintenance reference (see ADR status appendix).
8. Launch the versioned documentation preview and public doc-quality dashboards
   (coverage badge, doc lint, notebook lint) described in the information
   architecture plan so stakeholders can validate the structure ahead of GA.【F:docs/improvement/documentation_information_architecture.md†L108-L118】
9. Provide an RC upgrade checklist covering environment variables, pyproject
   settings, CLI usage, caching controls, and plugin integration testing
   expectations.
10. Audit the ADR gap closure roadmap to confirm every gap is either implemented or superseded with an updated ADR decision before promoting the RC branch, recording outcomes in the status log (see ADR status appendix in this document).
11. Finish Standard-001 nomenclature clean-up by eliminating double-underscore mutations, splitting utilities, reporting lint telemetry, and confining transitional shims to `legacy/` (see ADR status appendix in this document).
12. Extend governance dashboards to surface lint status alongside preprocessing/domain-model telemetry, ensuring ongoing monitoring after v1.0.0 (see ADR status appendix in this document).
13. Change default behavior for `condition_source` to `"prediction"` in `CalibratedExplainer` and related components. Update all relevant documentation, including the upgrade checklist, to inform users of this change and provide guidance on how to adjust their implementations if they previously relied on the default `"observed"` setting. This change aims to enhance the consistency of calibrated explanations by basing condition labels on model predictions rather than observed labels. Plan for this change to be communicated clearly in the v1.0.0 release notes and upgrade guides (see ADR status appendix in this document).
14. Empty the private member allow-list (`.github/private_member_allowlist.json`) as part of the final Pattern 1 remediation hardening. All remaining private member usages in tests must be refactored to public APIs or justified as permanent exceptions in the remediation plan.
15. Extend test quality tooling per ADR-030: update `scripts/anti-pattern-analysis/detect_test_anti_patterns.py` to flag tests without assertions and unseeded random usage, preparing for zero-tolerance enforcement in v1.0.0.
16. Add ADR-031 calibrator persistence: versioned `to_primitive`/`from_primitive` contracts plus `Explainer.save_state()`/`load_state()` (gate: round-trip tests and schema version policy).
17. Provide an OSS performance harness template for semi/fully-online latency measurements with README guidance (gate: sample run and documented workflow).
18. Perform an ADR, standards, and improvement docs gap closure analysis and implementation for any remaining gaps.

Release gate: All schema/contract freezes documented, nomenclature and docstring
lint suites blocking green, PlotSpec/plugin ADRs promoted, versioned docs preview
and doc-quality dashboards live, caching/parallel telemetry dashboards reviewed,
coverage dashboards live, ADR gap closure log signed off, and upgrade checklist
ready for pilot customers.

### v1.0.0 (stability declaration)

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
6. Ratify ADR-030 test quality enforcement: ensure extended anti-pattern scans are in CI, marker hygiene is enforced, and mutation testing is optional for core modules.

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
  existing plugin/doc improvements.【F:docs/improvement/coverage_uplift_plan.md†L11-L48】
- **Release readiness:** By v1.0.0, coverage gating is embedded in branch
  policies and telemetry/documentation communications, ensuring Standard-003 remains
  sustainable beyond the initial rollout.【F:docs/standards/STD-003-test-coverage-standard.md†L34-L74】【F:docs/improvement/coverage_uplift_plan.md†L11-L48】

## Post-1.0 considerations

- Continue monitoring caching and parallel execution telemetry to determine
  whether the opt-in defaults can graduate to on-by-default in v1.1, updating
  ADR-003/ADR-004 rollout notes as needed.【F:docs/improvement/adrs/ADR-003-caching-key-and-eviction.md†L28-L64】【F:docs/improvement/adrs/ADR-004-parallel-backend-abstraction.md†L25-L64】
- Evaluate additional renderer plugins (plotly) after verifying PlotSpec default
  adoption.
- Plan schema v2 requirements with downstream consumers before making breaking
  changes.

# Appendix ADR Gap Status

## ADR status appendix (unified severity tables)

The unified severity scales and per-ADR tables below replace the standalone `ADR-gap-analysis.md` to keep a single source of truth inside the release plan.

This document consolidates the individual ADR findings into a single reference that
uses a unified two-axis severity scale. Every gap is ranked within its ADR so the
highest-impact remediation items appear first.

## Unified Severity Scales

* **Violation impact (1–5)** – How seriously the observed gap violates the ADR.
  * 5 – Directly contradicts the ADR and blocks its intended outcome.
  * 4 – Major erosion of the ADR goal; functionality works but is fragile or
    inconsistent with the decision.
  * 3 – Noticeable drift that weakens guarantees or increases medium-term risk.
  * 2 – Minor divergence, largely cosmetic or limited in blast radius.
  * 1 – Informational observation; no required action.
* **Code scope (1–5)** – Breadth of code affected by the gap.
  * 5 – Cross-cutting impact spanning multiple top-level packages or the
    majority of runtime paths.
  * 4 – Multiple modules or a critical hub within a top-level package.
  * 3 – Confined to a single package or subsystem.
  * 2 – A couple of modules or helper utilities.
  * 1 – Single file or narrowly scoped helper.

**Unified severity** is the product of the two scores. Entries are sorted in
descending order of this product within each ADR.

---

## ADR-001 – Package and Boundary Layout

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Calibration layer remains embedded in `core` | 0 | 0 | 0 | **COMPLETED.** Calibration extracted to `calibration` package with compatibility shim in `core`. |
| 2 | Core imports downstream siblings directly | 0 | 0 | 0 | **COMPLETED.** Core imports are clean; lazy imports used where necessary. |
| 3 | Cache and parallel boundaries not split | 0 | 0 | 0 | **COMPLETED.** `cache` and `parallel` packages created. |
| 4 | Schema validation package missing | 0 | 0 | 0 | **COMPLETED.** `schema` package created. |
| 5 | Public API surface overly broad | 0 | 0 | 0 | **COMPLETED.** `__init__` exports cleaned up (verified in code). |
| 6 | Extra top-level namespaces lack ADR coverage | 0 | 0 | 0 | **COMPLETED.** Namespaces documented and rationalized. |

## ADR-002 – Exception Taxonomy and Validation Contract

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Legacy `ValueError`/`RuntimeError` usage in core and plugins | 0 | 0 | 0 | **COMPLETED.** `CalibratedError` hierarchy adopted. |
| 2 | Validation API contract not implemented | 0 | 0 | 0 | **COMPLETED.** `validate_inputs` implemented per ADR. |
| 3 | Structured error payload helpers absent | 0 | 0 | 0 | **COMPLETED.** `explain_exception` and details dicts implemented. |
| 4 | `validate_param_combination` is a no-op | 0 | 0 | 0 | **COMPLETED.** Implemented in `api/params.py` (verified via `wrap_explainer` import). |
| 5 | Fit-state and alias handling inconsistent | 0 | 0 | 0 | **COMPLETED.** `check_is_fitted` and `NotFittedError` used consistently. |

## ADR-003 – Caching Strategy

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Automatic invalidation & flush hooks missing | 0 | 0 | 0 | **COMPLETED.** `flush()` and `reset_version()` implemented. |
| 2 | Required artefacts not cached | 0 | 0 | 0 | **COMPLETED.** `CalibratorCache` handles artefacts. |
| 3 | Governance & documentation (STRATEGY_REV) absent | 0 | 0 | 0 | **COMPLETED.** Governance artefacts documented. |
| 4 | Telemetry integration incomplete | 0 | 0 | 0 | **COMPLETED.** Telemetry hooks implemented in `cache.py`. |
| 5 | Backend diverges from cachetools + pympler stack | 0 | 0 | 0 | **COMPLETED.** `cachetools` backend adopted. |

## ADR-004 – Parallel Execution Framework

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | ParallelFacade (conservative chooser) missing | 0 | 0 | 0 | **COMPLETED.** `ParallelExecutor` facade implemented with basic `_auto_strategy`. |
| 2 | Workload-aware auto strategy absent | 0 | 0 | 0 | **COMPLETED.** Auto strategy now consumes workload hints (`work_items`, `task_size_hint_bytes`, granularity) and defaults to serial for tiny batches. |
| 3 | Telemetry lacks timings and utilisation metrics | 0 | 0 | 0 | **COMPLETED.** `ParallelMetrics` tracks durations, worker counts, and utilisation; telemetry emitted via `_emit`. |
| 4 | Context management & cancellation missing | 4 | 4 | 16 | **COMPLETED.** `__enter__`/`__exit__` and cancellation support implemented. |
| 5 | Configuration surface incomplete | 4 | 3 | 12 | **COMPLETED.** `ParallelConfig` adds `task_size_hint_bytes`, `force_serial_on_failure`, `instance_chunk_size`, `feature_chunk_size`. |
| 6 | Resource guardrails ignore cgroup/CI limits | 0 | 0 | 0 | **COMPLETED.** Guardrails enforce `max_workers` bounds and auto-strategy fallbacks under constrained environments (cgroup/CI). |
| 7 | Fallback warnings not emitted | 4 | 2 | 8 | **COMPLETED.** Telemetry and `force_serial_on_failure` emit fallback visibility for users. |
| 8 | Testing and benchmarking coverage limited | 0 | 0 | 0 | **COMPLETED.** Workload-hint resolution covered by unit tests; benchmark harness implemented in `evaluation/parallel_ablation.py`. |
| 9 | Documentation for strategies & troubleshooting lacking | 0 | 0 | 0 | **COMPLETED.** Practitioner playbook updated with ADR-004-complete guardrails and workload-driven chooser guidance. |
| 10 | Documentation naming drift (`ParallelFacade` vs `ParallelExecutor`) | 2 | 2 | 4 | Legacy docs still mention `ParallelFacade`; standardize references on `ParallelExecutor` (or add an alias) and update documentation accordingly. |

### ADR-004 phase tracking (release alignment)

| Phase | Release target | Alignment summary | Status |
| --- | --- | --- | --- |
| Phase 0 – Foundations | v0.9.0 runtime polish | Documentation ownership and telemetry schema groundwork in place ahead of toggle rollout. | **COMPLETED.** Owner assignments completed; telemetry schema design finished. |
| Phase 1 – Configuration Surface | v0.9.0 runtime polish | Defines chunk-size and configuration knobs promised as opt-in runtime controls. | **COMPLETED.** `ParallelConfig` extended with chunk/size hints and failure toggles. |
| Phase 2 – Executor & Plugin Refactor | v0.10.0 runtime realignment | Payload sharing, batching hooks, and lifecycle management for ADR-004. | **COMPLETED.** Context manager and pooling lifecycle complete; payload sharing hooks merged. |
| Phase 3 – Workload-aware Strategy | v0.10.0 runtime realignment | Workload estimator and adaptive gating. | **COMPLETED.** Auto strategy consumes work-item hints and size estimates, defaulting to serial for small batches. |
| Phase 4 – Testing & Benchmarking | v0.10.0 runtime realignment | Spawn lifecycle coverage and automated benchmark reporting. | **COMPLETED.** Auto-strategy heuristics covered by unit tests; perf harness implemented and baseline results generated. |
| Phase 5 – Rollout & Documentation | v0.10.0 release prep / v1.0.0-RC readiness | User guidance, changelog, and telemetry artefacts for release checklists. | **COMPLETED.** Practitioner playbook and release status updated for ADR-004 completion. <br> ⚠️ **Update:** Granularity and FeatureParallel removal added to v1.0.0-rc scope. |

Alignment note: Parallel is treated as a shared service with domain-specific runtime wrappers (e.g., explain) expected to wrap
the shared `ParallelExecutor` so heuristics and chunking remain co-located with domain executors while respecting ADR-001
boundaries. ADR-004 now documents this expectation.

## ADR-003 – Caching Strategy

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Cache not implemented | 5 | 4 | 20 | **COMPLETED.** Opt-in in-process LRU cache implemented with deterministic keys and configurable eviction. |
| 2 | Key semantics not stable | 4 | 3 | 12 | **COMPLETED.** Tuple-based keys with namespace, version_tag, and payload hash ensure reproducibility. |
| 3 | Eviction not configurable | 3 | 3 | 9 | **COMPLETED.** LRU with max_items; TTL optional; backend uses cachetools. |
| 4 | Failure modes not handled | 3 | 2 | 6 | **COMPLETED.** Cache misses fall back to recomputation with warnings; no crashes. |
| 5 | Telemetry missing | 2 | 2 | 4 | **COMPLETED.** Optional debug logging; no mandatory telemetry per ADR. |

## ADR-005 – Explanation Payload Schema

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Documentation & fixtures out of date | 3 | 2 | 6 | Docs/tests still reference legacy shapes instead of the v1 payload contract. |
| 2 | Validation helper limited to JSON Schema | 3 | 3 | 9 | `validate_payload` does not enforce semantic invariants beyond schema shape; interval checks live in serializers. |
| 3 | Schema version guidance inconsistent | 2 | 2 | 4 | `schema_version` is recommended but not consistently documented across docs/fixtures. |
| 4 | Provenance/metadata extension guidance missing | 2 | 2 | 4 | Payload allows `provenance`/`metadata`, but usage guidance is sparse and inconsistent. |

## ADR-006 – Plugin Trust Model

*Gaps removed from appendix after completion; see ADR-006 for implementation details.*

## ADR-007 – PlotSpec Abstraction

*Gaps removed from appendix after completion; see ADR-007 for implementation details.*

## ADR-008 – Explanation Domain Model

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Domain model not authoritative source | 5 | 4 | 20 | Core workflows still operate on legacy dicts, with domain objects produced only at serialization boundaries. |
| 2 | Legacy-to-domain round-trip fails for conjunctive rules | 4 | 3 | 12 | `domain_to_legacy` casts features to scalars, breaking conjunction support. |
| 3 | Structured model/calibration metadata absent | 4 | 3 | 12 | Explanation dataclass lacks dedicated fields for calibration parameters and model descriptors. **STATUS 2025-11-04: Factual/alternative payload structures clarified with formal definitions in ADR-008; implementation work remains.** |
| 4 | Golden fixture parity tests missing | 3 | 2 | 6 | Absence of byte-level fixtures weakens regression detection for adapters. |
| 5 | `_safe_pick` silently duplicates data | 3 | 2 | 6 | Interval helper duplicates endpoints rather than flagging inconsistencies, risking misreported uncertainty. |

## ADR-009 – Preprocessing Pipeline

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Automatic encoding pathway unimplemented | 5 | 4 | 20 | `auto_encode='auto'` has no effect; built-in encoder never runs. |
| 2 | Unseen-category policy ignored | 4 | 3 | 12 | Configuration captures the flag but no enforcement occurs during inference. |
| 3 | DataFrame/dtype validation incomplete | 3 | 3 | 9 | Validators coerce to NumPy without inspecting categorical columns, missing ADR-required diagnostics. |
| 4 | Telemetry docs mismatch emitted fields | 2 | 2 | 4 | Documentation references `identifier` while runtime payload exposes `transformer_id`. |

## ADR-010 – Optional Dependency Split

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Core-only install still requires matplotlib in CI | 3 | 2 | 6 | Core dependencies are lean and extras are defined, but CI’s “core-only” job still installs matplotlib due to import-time requirements; clarify whether matplotlib is truly required at import time. |
| 2 | Contributor guidance on extras outdated | 2 | 2 | 4 | Documentation should confirm the current extras split and the minimal-core install path. |

## ADR-011 – Deprecation Policy

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Central `deprecate()` helper missing | 0 | 0 | 0 | **COMPLETED.** Helper implemented in `utils/deprecations.py`. |
| 2 | Migration guide absent | 0 | 0 | 0 | **COMPLETED.** Guide published at `docs/migration/deprecations.md`. |
| 3 | Release plan lacks status table | 0 | 0 | 0 | **COMPLETED.** Status table added to `RELEASE_PLAN_v1.md`. |
| 4 | CI gates for deprecation policy missing | 0 | 0 | 0 | **COMPLETED.** `deprecation-check.yml` enforces `CE_DEPRECATIONS=error`. |

## ADR-012 – Documentation & Gallery Build Policy

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Notebooks never rendered in docs CI | 5 | 4 | 20 | Docs workflow skips sphinx-gallery/nbconvert execution, so notebook breakage ships undetected. |
| 2 | Docs build ignores `[viz]`/`[notebooks]` extras | 0 | 0 | 0 | **COMPLETED.** Docs CI installs `[notebooks,viz,eval]` extras alongside doc requirements. |
| 3 | Example runtime ceiling unenforced | 3 | 3 | 9 | No automation times notebooks/examples, so the <30 s headless contract can regress silently. |
| 4 | Gallery tooling decision undocumented | 2 | 2 | 4 | ADR-required choice between sphinx-gallery/nbconvert is not recorded, leaving contributors without guidance. |

## ADR-013 – Interval Calibrator Plugin Strategy

*Gaps removed from appendix after completion; see ADR-013 for implementation details.*

## ADR-014 – Visualization Plugin Architecture

*Gaps removed from appendix after completion; remaining documentation updates tracked under Standard-004 follow-through.*

## ADR-015 – Explanation Plugin Integration

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | In-tree FAST plugin missing | 0 | 0 | 0 | **COMPLETED.** Built-in FAST explanation plugin is registered and trusted by default. |
| 2 | Collection reconstruction bypassed | 0 | 0 | 0 | **COMPLETED.** Batch reconstruction rebuilds canonical explanation collections with metadata. |
| 3 | Trust enforcement during resolution lax | 0 | 0 | 0 | **COMPLETED.** Registry enforces trust/deny rules with explicit override warnings. |
| 4 | Predict bridge omits interval invariants | 3 | 2 | 6 | Invariants are monitored with warnings; consider strict-mode enforcement in ADR-026 follow-ups. |
| 5 | Environment variable names diverge | 0 | 0 | 0 | **COMPLETED.** Resolver honors `CE_EXPLANATION_PLUGIN[_FAST]` naming. |
| 6 | Helper handles expose mutable explainer | 3 | 2 | 6 | Immutable plugin handles remain a hardening task shared with ADR-026. |

## ADR-016 – PlotSpec Separation and Schema

*Gaps removed from appendix after completion; see ADR-016 for implementation details.*

## ADR-020 – Legacy User API Stability

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Release checklist omits legacy API gate | 4 | 3 | 12 | No checkbox ensures contract/tests stay in sync before shipping. |
| 2 | Wrapper regression tests miss parity on key methods | 4 | 3 | 12 | `explain_factual`/`explore_alternatives` signatures and normalisation aren’t asserted, risking drift. |
| 3 | Contributor workflow ignores contract document | 3 | 3 | 9 | CONTRIBUTING never directs authors to update the canonical contract alongside code changes. |
| 4 | Notebook audit process undefined | 3 | 2 | 6 | ADR’s periodic notebook review step has no script or checklist entry. |

## ADR-021 – Calibrated Interval Semantics

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Interval invariants never enforced | 0 | 0 | 0 | **COMPLETED.** Invariant enforcement implemented in `PredictBridgeMonitor`, `PredictionOrchestrator`, and `CalibratedExplanation` (with warnings for violations). |
| 2 | FAST explanations drop probability cubes | 0 | 0 | 0 | **COMPLETED.** `FastExplanationPipeline` updated to include `__full_probabilities__` in prediction output. |
| 3 | JSON export stores live callables | 0 | 0 | 0 | **COMPLETED.** `_jsonify` helper updated to safely handle callables during serialization. |

## ADR-022 – Documentation Information Architecture (Superseded)

*Superseded by Standard-004. See Standard-004 section for active gaps.*

## ADR-023 – Matplotlib Coverage Exemption

*Gaps removed from appendix after completion; see ADR-023 for implementation details.*

## ADR-024 – Legacy Plot Input Contracts

*Superseded by `docs/maintenance/legacy-plotting-reference.md`. Gaps retired as of 2025-10-18.*

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | `_plot_global` ignores `show=False` | 5 | 3 | 15 | Helper always calls `plt.show()`, violating the headless contract and breaking CI/headless runs. |
| 2 | `_plot_global` lacks save parameters | 4 | 3 | 12 | Helper cannot honour ADR-shared save semantics, leaving plots unsaveable through the documented interface. |
| 3 | Save-path concatenation drift undocumented | 2 | 2 | 4 | Helper now normalises directories, diverging from ADR guidance without updated docs. |

## ADR-025 – Legacy Plot Rendering Semantics

*Superseded by `docs/maintenance/legacy-plotting-reference.md`. Gaps retired as of 2025-10-18.*

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Matplotlib guard allows silent skips | 4 | 3 | 12 | Helpers return early without requiring Matplotlib even when file output is requested, hiding failures. |
| 2 | Regression axis not forced symmetric | 4 | 3 | 12 | `_plot_regression` sets raw min/max limits instead of the symmetric range promised by the ADR. |
| 3 | Interval backdrop disabled | 3 | 3 | 9 | Commented-out `fill_betweenx` leaves regression interval visuals inconsistent with documented design. |
| 4 | One-sided interval warning untested | 3 | 2 | 6 | Guard exists but lacks coverage, so regressions could ship unnoticed. |

## ADR-026 – Explanation Plugin Semantics

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | `explain` method remains public | 0 | 0 | 0 | **COMPLETED.** Legacy `explain` is a deprecated shim; the main explainer does not expose a public `explain` method. |
| 2 | Predict bridge skips interval invariant checks | 0 | 0 | 0 | **COMPLETED.** Invariant enforcement implemented in `PredictBridgeMonitor` (with warnings for violations). |
| 3 | Explanation context exposes mutable dicts | 4 | 3 | 12 | Context builder embeds plain dicts despite the frozen contract, enabling plugin-side mutation. **STATUS 2025-11-04: Frozen context requirement clarified in ADR-026 subsection 1; enforcement remains.** |
| 4 | Telemetry omits interval dependency hints | 3 | 2 | 6 | Batch telemetry drops `interval_dependencies`, reducing observability. |
| 5 | Mondrian bins left mutable in requests | 2 | 2 | 4 | `ExplanationRequest` stores caller-supplied bins verbatim, violating the immutability promise. |

## ADR-027 – FAST-Based Feature Filtering

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Observability policy alignment undocumented | 2 | 2 | 4 | Document governance logging expectations for feature filtering and confirm examples in the observability standard. |
| 2 | Feature-filter telemetry examples sparse | 2 | 2 | 4 | Add or link examples in practitioner/contributor guides showing emitted metadata. |

## ADR-028 – Logging and Governance Observability

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Enforcement tooling for logger domains missing | 2 | 2 | 4 | Add lint/tests to confirm domain logger usage per Standard-005 guidance. |
| 2 | Observability examples need alignment | 2 | 2 | 4 | Ensure docs/examples match Standard-005 expectations for structured logging. |

## Standard-001 – Nomenclature Standardization

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Double-underscore fields still mutated outside legacy | 5 | 4 | 20 | Core helpers touch `__` attributes directly, contravening the ADR’s hard ban. |
| 2 | Naming guardrails lack automated enforcement | 4 | 4 | 16 | Ruff/pre-commit configuration does not fail on new snake-case or `__` violations, leaving the policy unenforced. |
| 3 | Kitchen-sink `utils/helper.py` persists | 3 | 3 | 9 | Monolithic helper resists the topic-focused split mandated for Phase 1. |
| 4 | Telemetry for lint drift missing | 3 | 3 | 9 | No runtime metrics capture naming debt, undermining the ADR’s governance plan. |
| 5 | Transitional shims remain first-class | 3 | 2 | 6 | `_legacy_explain` and similar helpers live alongside active modules instead of isolated legacy shims. |

## Standard-002 – Documentation Standardisation

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Wrapper public APIs lack full numpydoc blocks | 4 | 3 | 12 | `WrapCalibratedExplainer` methods omit Parameters/Returns/Raises despite being the stable user surface. |
| 2 | `IntervalRegressor.__init__` docstring outdated | 4 | 2 | 8 | Documented parameters no longer exist, misleading users of the calibrator. |
| 3 | `IntervalRegressor.bins` setter undocumented | 3 | 2 | 6 | Public mutator ships without a summary or type guidance. |
| 4 | Guard helpers missing summaries | 2 | 2 | 4 | `_assert_fitted`/`_assert_calibrated` lack the mandated one-line docstrings. |
| 5 | Nested combined-plot plugin classes undocumented | 2 | 2 | 4 | Dynamically returned classes expose blank docstrings, hurting plugin discoverability. |

## Standard-003 – Test Coverage Standard

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Coverage floor still enforced at 88 % | 0 | 0 | 0 | **COMPLETED.** Coverage gates now enforce ≥90% package-level coverage. |
| 2 | Critical modules below 95 % without gates | 0 | 0 | 0 | **COMPLETED.** Per-module 95% gates enforced via `scripts/check_coverage_gates.py`. |
| 3 | Codecov patch gate optional | 0 | 0 | 0 | **COMPLETED.** Patch gating is enforced in CI. |
| 4 | Public API packages under-tested | 2 | 2 | 4 | Continue monitoring public API shims to maintain ≥90% coverage. |
| 5 | Exemptions lack expiry metadata | 2 | 2 | 4 | Ensure waiver/omit entries include expiry metadata where applicable. |

## Standard-004 – Documentation Standard (Audience Hubs)

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Audience-based navigation structure not implemented | 0 | 0 | 0 | **COMPLETED.** Hubs (`practitioner`, `researcher`, `contributor`) implemented in `docs/index.md`. |
| 2 | PR template lacks parity review gate | 0 | 0 | 0 | **COMPLETED.** Checklist item added to PR template. |
| 3 | "Task API comparison" reference missing | 0 | 0 | 0 | **COMPLETED.** Task API comparison is linked from practitioner guidance. |
| 4 | Telemetry concept page lacks substance | 0 | 0 | 0 | **COMPLETED.** Telemetry concept documentation exists in foundations. |
| 5 | Researcher future-work ledger absent | 0 | 0 | 0 | **COMPLETED.** Researcher future-work ledger published. |
| 6 | Plot plugin authoring/override guidance incomplete | 2 | 2 | 4 | Ensure plot plugin authoring docs cover renderer overrides and validation hooks (Standard-004 follow-through). |
