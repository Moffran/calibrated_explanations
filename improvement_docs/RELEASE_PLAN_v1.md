> **Status note (2025-12-02):** Last edited 2025-12-02 · Archive after v1.0.0 GA · Implementation window: v0.9.0–v1.0.0 ·

# Release Plan to v1.0.0

### Current released version: v0.9.1

> Status: v0.9.1 shipped on 2025-11-27.


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

- **ADR-001 – Package and Boundary Layout:** Stages 0–5 completed; boundary realignment delivered. See appendix for the retired gap table.
- **ADR-002 – Exception Taxonomy and Validation Contract:** Completed with taxonomy adoption and validator parity; appendix holds the consolidated gap status.
- **ADR-003 – Caching Strategy:** Completed with cache governance and telemetry; appendix retains the unified gap table.
- **ADR-004 – Parallel Execution Framework:** Phases tracked in the roadmap table in the the appendix; remaining heuristics/benchmarking items align to v0.10.0. Gap details are maintained only in the appendix.
- **ADR-005 – Explanation Envelope & Schema:** Scheduled for v0.10.1 schema & visualization contracts; appendix table captures the remaining envelope/schema gaps.
- **ADR-006 – Plugin Trust Model:** Trust gating tasks land in v0.10.2; see appendix for the unified gap status.
- **ADR-007 – PlotSpec Abstraction:** PlotSpec registry/validation rollout targets v0.10.1; appendix retains the per-gap severities.
- **ADR-008 – Explanation Domain Model:** Domain-model hardening is planned for v0.11.0; appendix tracks the outstanding items.
- **ADR-009 – Input Preprocessing and Mapping Policy:** Preprocessing automation aligns to v0.11.0; appendix tables list the remaining work.
- **ADR-010 – Core vs Evaluation Split:** Optional-dependency splits ship with v0.10.2; appendix coverage remains the single gap source.
- **ADR-011 – Deprecation and Migration Policy:** Policy enforcement stays aligned with release gates; appendix records the individual gaps.
- **ADR-012 – Documentation and Gallery Build Policy:** Documentation build and gallery guardrails follow the ADR-027 IA rollout; see appendix for details.
- **ADR-013 – Interval Calibrator Plugin Strategy:** FAST/plugin protocol completion tracks to v0.10.2; appendix entries cover the gap specifics.
- **ADR-014 – Visualization Plugin Architecture:** Visualization plugin updates are bundled with v0.10.1; appendix consolidates the gap list.
- **ADR-015 – Explanation Plugin Integration:** Integration tasks align to v0.10.2; appendix tables retain the per-gap severity.
- **ADR-016 – PlotSpec Separation and Schema:** Schema separation and validation updates ship in v0.10.1 alongside ADR-005/ADR-007; appendix carries the detailed gaps.
- **ADR-017 – Nomenclature Standardization:** Remediation plan lives in `ADR-017_nomenclature_remediation.md`; appendix remains the single source for gap status.
- **ADR-018 – Documentation Standardisation:** Docstring uplift plan is consolidated in `code_documentation_uplift.md`; appendix gap tracking only.
- **ADR-019 – Test Coverage Standard:** Coverage uplift runs through v0.9.1–v0.11.0 milestones; appendix table is the authoritative gap log.
- **ADR-020 – Legacy User API Stability:** Legacy contract enforcement is tracked against v0.9.1–v0.10.x release gates; appendix retains the severity table.
- **ADR-021 – Calibrated Interval Semantics:** Interval invariant enforcement continues through v0.10.x; appendix captures the outstanding semantics gaps.
- **ADR-022 – Documentation Information Architecture:** Superseded by ADR-027; gap tracking removed. See ADR-027 entry for current IA expectations.
- **ADR-023 – Matplotlib Coverage Exemption:** Visualization coverage enforcement aligns with v0.9.1; appendix table holds the remaining coverage deltas.
- **ADR-024 – Legacy Plot Input Contracts:** Legacy plotting fixes ship in v0.10.1; appendix retains the per-gap list.
- **ADR-025 – Legacy Plot Rendering Semantics:** Rendering alignment and coverage share the v0.10.1 milestone; appendix is the single source of gap status.
- **ADR-026 – Explanation Plugin Semantics:** Plugin semantics hardening targets v0.10.2; appendix tables contain the unified gap coverage.
- **ADR-027 – Documentation Standard (Audience Hubs):** Documentation standard rollout continues through v0.8.0+; appendix keeps the consolidated gaps alongside ADR-022 supersession notes.

## Release milestones

### Uplift status by milestone

| Release | Documentation overhaul (ADR-012/027) | Code docs (ADR-018) | Coverage uplift (ADR-019) | Naming (ADR-017) | Notes |
| --- | --- | --- | --- | --- | --- |
| v0.8.0 | IA restructure landed; Sphinx/linkcheck gates required for merges. | Phase 0 primer circulated; baseline inventory started. | `.coveragerc` drafted; `fail_under=80` staged. | Lint guards enabled; shim inventory captured. | Rollback: revert to pre-IA toctree if Sphinx fails; waivers expire after one iteration. |
| v0.9.1 | Audience hubs sustained; quickstart smoke tests block release. | Phase 1 batches A–C targeted for ≥90%; docs examples aligned with new IA. | XML+Codecov upload required; gating rising toward 90%. | Phase 1 cleanup in progress with measurement checkpoints. | Rollback: pin docs to last green build; coverage waivers require dated follow-up issues. |
| v0.10.0 | Doc gate holds prior bar; no new IA work planned. | Phase 2 blocking check for touched files; waivers time-bounded. | `fail_under=90` enforced; plugin/plotting module thresholds scheduled. | Release mapping added; refactors aligned with legacy API tests. | Risk: refactors (explain plugins, boundary split) may churn coverage; branch cut requires rerunning gates. |
| v0.10.1 | Doc hubs refreshed with telemetry/performance opt-in notes. | Package-wide ≥90% expected; notebook/example lint extended. | Module thresholds hardened; waiver expiry dates mandatory. | Phase metrics reviewed; % renamed modules tracked. | Rollback: if module gates fail, defer release or lower threshold with explicit expiry in checklist. |
| v1.0.0 | Docs maintenance review; parity checks remain blocking. | Continuous improvement cadence; badge and quarterly reviews. | Waiver backlog should be zero; mutation/fuzzing exploration optional. | Final shim removals completed; legacy API guard tests green. | Risks surface inline in milestone gates; rollback paths documented per gate. |

### v0.6.x (stabilisation patches)

- Hardening: add regression tests for plugin parity, schema validation, and
  WrapCalibratedExplainer keyword defaults.
- Documentation polish: refresh plugin guide with registry/CLI examples and note
  compatibility guardrails.
- No behavioural changes beyond docs/tests.
- Coverage readiness: ratify ADR-019, publish `.coveragerc` draft with
  provisional exemptions, and record baseline metrics to size the remediation
  backlog.【F:improvement_docs/adrs/ADR-019-test-coverage-standard.md†L1-L74】【F:improvement_docs/archived/test_coverage_assessment.md†L1-L23】

### v0.7.0 (interval & configuration integration)

1. Implement interval plugin resolution and fast-mode reuse per
   `PLUGIN_GAP_CLOSURE_PLAN` step 1, ensuring calibrators resolve via registry and
   trusted fallbacks.【F:improvement_docs/PLUGIN_GAP_CLOSURE_PLAN.md†L24-L43】
2. Surface interval/plot configuration knobs (keywords, env vars, pyproject) and
   propagate telemetry metadata for `interval_source`/`proba_source`.【F:improvement_docs/PLUGIN_GAP_CLOSURE_PLAN.md†L45-L61】
3. Wire CLI console entry point and smoke tests; document usage in README and
   contributing guides.【F:improvement_docs/PLUGIN_GAP_CLOSURE_PLAN.md†L63-L70】
4. Update ADR-013/ADR-015 statuses to Accepted with implementation notes.
5. Ratify ADR-017/ADR-018, publish contributor style excerpts, and land initial
   lint/tooling guardrails for naming and docstring coverage per preparatory
   phase plans.【F:improvement_docs/ADR-017_nomenclature_remediation.md†L20-L28】【F:improvement_docs/code_documentation_uplift.md†L10-L28】
   - 2025-10-07 – Updated test helpers (`tests/conftest.py`, `tests/unit/core/test_calibrated_explainer_interval_plugins.py`) to comply with Ruff naming guardrails, keeping ADR-017 lint checks green.
   - 2025-10-07 – Harmonised `core.validation` docstring spacing with numpy-style guardrails to satisfy ADR-018 pydocstyle checks.
6. Implement ADR-019 phase 1 changes: ship shared `.coveragerc`, enable
   `--cov-fail-under=80` in CI, and document waiver workflow in contributor
   templates.【F:improvement_docs/adrs/ADR-019-test-coverage-standard.md†L34-L74】【F:improvement_docs/coverage_uplift_plan.md†L9-L33】

Release gate: parity tests green for factual/alternative/fast, interval override
coverage exercised, CLI packaging verified, and nomenclature/doc lint warnings
live in CI with coverage thresholds enforcing ≥90% package-level coverage.

### v0.8.0 (plot routing, telemetry, and doc IA rollout)

1. Adopt ADR-027 (superseding ADR-022) by restructuring the documentation toctree into the audience-based information architecture (Getting Started, Practitioner, Researcher, Contributor hubs) and shipping the new telemetry concept page plus quickstart refactor per the information architecture plan.【F:improvement_docs/adrs/ADR-027-documentation-standard.md†L1-L53】
   - Land the docs sitemap rewrite with a crosswalk checklist (legacy page -> new section) and block merge on green sphinx-build -W, linkcheck, and nav tests to prevent broken routes.
   - Refactor quickstart content into runnable classification and regression guides, wire them into docs smoke tests, and add troubleshooting callouts for supported environments.
   - Publish the telemetry concept page with instrumentation examples, expand the plugin registry and CLI walkthroughs, and sync configuration references (pyproject, env vars, CLI flags) with the new navigation.
   - Record section ownership in docs/OWNERS.md (Overview/Get Started - release manager; How-to/Concepts - runtime tech lead; Extending/Governance - contributor experience lead) and update the pre-release doc checklist so every minor release verifies ADR-027 guardrails.
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
   enterprise integrations and provide examples in docs/plugins.md.
5. Review preprocessing persistence contract (ADR-009) to confirm saved
   preprocessor metadata matches expectations.【F:improvement_docs/adrs/ADR-009-input-preprocessing-and-mapping-policy.md†L1-L80】
6. Execute ADR-017 Phase 2 renames with legacy shims isolated under a
   `legacy/` namespace and update imports/tests/docs to the canonical module
   names.【F:improvement_docs/ADR-017_nomenclature_remediation.md†L30-L33】
7. Complete ADR-018 baseline remediation by finishing pydocstyle batches C (`explanations/`, `perf/`) and D (`plugins/`), adding module summaries and
   upgrading priority package docstrings to numpydoc format with progress
   tracking.【F:improvement_docs/code_documentation_uplift.md†L17-L92】【F:improvement_docs/adrs/ADR-018-code-documentation-standard.md†L17-L62】
8. Extend ADR-019 enforcement to critical-path modules (≥95% coverage) and
   enable Codecov patch gating at ≥85% for PRs touching runtime/calibration
   logic, enable
   `--cov-fail-under=85` in CI.【F:improvement_docs/adrs/ADR-019-test-coverage-standard.md†L34-L74】【F:improvement_docs/coverage_uplift_plan.md†L24-L33】
9. **Completed 2025-01-14:** Adopted ADR-023 to exempt `src/calibrated_explanations/viz/matplotlib_adapter.py` from coverage due to matplotlib 3.8.4 lazy loading conflicts with pytest-cov instrumentation. All 639 tests now pass with coverage enabled. Package-wide coverage maintained at 85%+.【F:improvement_docs/adrs/ADR-023-matplotlib-coverage-exemption.md†L1-L100】

Release gate: PlotSpec default route parity, telemetry docs/tests in place,
documentation architecture and ownership shipped, nomenclature renames shipped
with shims, docstring coverage dashboard shows baseline met, ADR-019
critical-path thresholds pass consistently, and full test suite stability
achieved via ADR-023 exemption.

### v0.9.0 (documentation realignment & targeted runtime polish)

1. **Reintroduce calibrated-explanations-first messaging across entry points.** Update README quickstart, Overview, and practitioner quickstarts so telemetry/PlotSpec steps are collapsed into clearly labelled "Optional extras" callouts. Place probabilistic regression next to classification in every onboarding path and link to interpretation guides and citing.md.
2. **Ship audience-specific landing pages.** Implement practitioner, researcher, and contributor hubs per the information architecture update: add probabilistic regression quickstart + concept guide, interpretation guides mirroring notebooks (factual and alternatives with triangular plots), and a researcher "theory & literature" page with published papers and benchmark references.【F:improvement_docs/documentation_information_architecture.md†L5-L118】
3. **Clarify plugin extensibility narrative.** Revise docs/plugins.md to open with a "hello, calibrated plugin" example that demonstrates preserving calibration semantics, move telemetry/CLI details into optional appendices, and document guardrails tying plugins back to calibrated explanations. Include a prominent pointer to the new `external_plugins/` folder and aggregated installation extras for community plugins.【F:improvement_docs/documentation_review.md†L9-L49】

   - 2025-11-06 – Consolidated the plugin story into a Plugins hub (`docs/plugins.md`), added a practitioner-focused "Use external plugins" guide (`docs/practitioner/advanced/use_plugins.md`), and surfaced the curated `external-plugins` extra in installation docs. Cross-linked the appendix index and ensured practitioner/contributor flows are consistent with ADR-027/ADR-006/ADR-014/ADR-026.
4. **Label telemetry and performance scaffolding as optional tooling.** Move telemetry schema/how-to material into contributor governance sections, ensure practitioner guides mention telemetry only for compliance scenarios, and audit navigation labels to avoid implying these extras are mandatory.【F:improvement_docs/documentation_information_architecture.md†L70-L113】
5. **Highlight research pedigree throughout.** Keep the existing research hub mentions in the Overview, practitioner quickstarts, and probabilistic regression concept pages; ensure they cross-link citing.md and key publications in the relevant sections without introducing new banner UI.【F:improvement_docs/documentation_review.md†L15-L34】
6. **Triangular alternatives plots everywhere alternatives appear.** Update explanation guides, PlotSpec docs, and runtime examples so `explore_alternatives` also introduces the triangular plot and its interpretation.
7. **Complete ADR-012 doc workflow enforcement.** Keep Sphinx `-W`, gallery build, and linkcheck mandatory; extend CI smoke tests to run the refreshed quickstarts and fail if optional extras are presented without labels.【F:improvement_docs/adrs/ADR-012-documentation-and-gallery-build-policy.md†L1-L80】
8. **Turn ADR-018 tooling fully blocking.** Finish pydocstyle batches E (`viz/`, `viz/plots.py`, `legacy/plotting.py`) and F (`serialization.py`, `core.py`), capture and commit the baseline failure report before flipping enforcement, add the documentation coverage badge, and extend linting to notebooks/examples so the Phase 3 automation backlog is complete.【F:improvement_docs/code_documentation_uplift.md†L24-L92】
   - 2025-10-25 – Added nbqa-powered notebook linting and a 94% docstring
     coverage threshold to the lint workflow, making ADR-018's tooling fully
     blocking for documentation CI.
9. **Advance ADR-017 naming cleanup.** Prune deprecated shims scheduled for removal and ensure naming lint rules stay green on the release branch.【F:improvement_docs/ADR-017_nomenclature_remediation.md†L40-L44】【F:improvement_docs/adrs/ADR-017-nomenclature-standardization.md†L28-L37】
10. **Sustain ADR-019 coverage uplift.** Audit waiver inventory, retire expired exemptions, raise non-critical modules toward the 90% floor, enable `--cov-fail-under=88` in CI, and execute the module-level remediation efforts for interval regressors, registry/CLI, plotting, and explanation caching per the dedicated gap plan.【F:improvement_docs/coverage_uplift_plan.md†L34-L111】
11. **Scoped runtime polish for explain performance.** Deliver the opt-in calibrator cache, multiprocessing toggle, and vectorised perturbation handling per ADR-003/ADR-004 analysis so calibrated explanations stay responsive without compromising accuracy. Capture improvements and guidance for plugin authors.【F:improvement_docs/adrs/ADR-003-caching-key-and-eviction.md†L1-L64】【F:improvement_docs/adrs/ADR-004-parallel-backend-abstraction.md†L1-L64】【F:src/calibrated_explanations/core/calibrated_explainer.py†L1750-L2150】 See the ADR-004 phase table above for task breakdown and ownership.

      - 2025-11-04 – Implemented opt-in calibrator cache with LRU eviction, multiprocessing toggle via ParallelExecutor facade, and vectorized perturbation handling. Added performance guidance for plugin authors in docs/contributor/plugin-contract.md. Cache and parallel primitives integrated into explain pipeline without altering calibration semantics.
12. **Plugin CLI, discovery, and denylist parity (optional extras).** Extend trust toggles and entry-point discovery to interval/plot plugins, add the `CE_DENY_PLUGIN` registry control highlighted in the OSS scope review, and ship the whole surface as opt-in so calibrated explanations remain usable without telemetry/CLI adoption.
13. **External plugin distribution path.** Document and test an aggregated installation extra (e.g., `pip install calibrated-explanations[external-plugins]`) that installs all supported external plugins, outline curation criteria, and add placeholders in docs and README for community plugin listings.

      - 2025-10-25 – Added a packaging regression test that inspects the
         `external-plugins` extra metadata to guarantee the curated bundle stays
         opt-in with the expected dependency pins.
14. **Explanation export convenience.** Provide `to_json()`/`from_json()` helpers on explanation collections that wrap schema v1 utilities and document them as optional aids for integration teams.
15. **Scope streaming-friendly explanation delivery.** Prototype generator or chunked export paths (or record a formal deferral) so memory-sensitive users know how large batches will be handled, capturing the outcome directly in the OSS scope inventory.【F:improvement_docs/OSS_CE_scope_and_gaps.md†L86-L118】

Release gate: Audience landing pages published with calibrated explanations/probabilistic regression foregrounded, research callouts present on all entry points, telemetry/performance extras labelled optional, docs CI (including quickstart smoke tests, notebook lint, and doc coverage badge) green, ADR-017/018/019 gates enforced, runtime performance enhancements landed without altering calibration outputs, plugin denylist control shipped, streaming plan recorded, and optional plugin extras (CLI/discovery/export) documented as add-ons.

### v0.9.1 (governance & observability hardening)

1. Implement ADR-011 policy mechanics—add the central deprecation helper, author the long-promised migration guide, and publish the structured status table with CI enforcement of the two-release window (see ADR status appendix in this document).
2. Bring docs CI into compliance with ADR-012 by executing notebooks during builds, installing official extras, timing tutorials, and documenting the chosen gallery tooling so drift is detected early (see ADR status appendix in this document).
3. Finish ADR-018 obligations by documenting wrapper APIs, interval calibrator signatures, and guard helpers to the mandated numpydoc standard (see ADR status appendix in this document).
4. Elevate coverage governance to the ADR-019 bar—raise thresholds to ≥90%, add per-module gates for prediction/serialization/registry paths, make the Codecov patch gate blocking, and track expiry metadata for waivers (see ADR status appendix in this document).
5. Reinforce ADR-020 legacy-API commitments with release checklist gates, regression tests for `explain_factual`/`explore_alternatives`, CONTRIBUTING guidance, and a scripted notebook audit workflow (see ADR status appendix in this document).
6. Restore visualization safety valves per ADR-023 by running the viz suite in CI, removing ignores, and aligning coverage messaging with the final thresholds (see ADR status appendix in this document).
7. Update governance collateral and hubs to satisfy ADR-027—embed the parity-review checklist in PR templates, reinstate the task API comparison, and publish the researcher future-work ledger (see ADR status appendix in this document).
8. Implement ADR-004 v0.9.1 scoped deliverable — ParallelFacade: create a conservative facade that centralizes executor selection heuristics, exposes a minimal config surface (min_instances_for_parallel, min_features_for_parallel, task_size_hint_bytes), honors `CE_PARALLEL` overrides, emits compact decision telemetry (decision, reason, n_instances, n_features, bytes_hint, platform, executor_type), and includes unit tests plus a micro-benchmark harness. This is intentionally small and designed to collect field evidence before any full `ParallelExecutor` rollout in v0.10. 【F:improvement_docs/adrs/ADR-004-parallel-backend-abstraction.md†L1-L40】

Release gate: Deprecation dashboard live, docs CI runs with notebook execution, coverage/waiver gating enforced at ≥90%, legacy API and parity checklists signed, and visualization tests passing on the release branch (see ADR status appendix in this document).

### v0.10.0 (runtime boundary realignment)

1. Restructure packages to honour ADR-001—split calibration into its own package, eliminate cross-sibling imports, and formalise sanctioned namespaces with ADR addenda where necessary (see ADR status appendix in this document).
2. Deliver ADR-002 validation parity by replacing legacy exceptions with taxonomy classes, implementing shared validators, parameter guards, and consistent fit-state handling (see ADR status appendix in this document).
3. Complete ADR-003 caching deliverables: add invalidation/flush hooks, cache the mandated artefacts, emit telemetry, and align the backend with the cachetools+pympler stack or update the ADR rationale (see ADR status appendix in this document).
4. Implement ADR-004’s parallel execution backlog—auto strategy heuristics, telemetry with timings/utilisation, context management and cancellation, configuration surfaces, resource guardrails, fallback warnings, and automated benchmarking (see ADR status appendix in this document). Progress is tracked in the ADR-004 phase table above.
5. Enforce interval safety across bridges and exports to resolve ADR-021 and the ADR-015 predict-bridge gap, ensuring invariants, probability cubes, and serialization policies are honoured (see ADR status appendix in this document).
6. Align runtime plugin semantics with ADR-026 by adding invariant checks, hardening contexts, and extending telemetry payloads. Also internalise `CalibratedExplainer.explain` to reinforce the facade pattern and prevent public access (see ADR status appendix in this document).
7. Remove deprecated backward-compatibility alias `_is_thresholded()` from `CalibratedExplanations` class (superseded by `_is_probabilistic_regression()` in v0.9.0). Update any remaining external code or documentation that may reference the old method name. This completes the terminology standardization cycle from ADR-021.【F:improvement_docs/adrs/ADR-021-calibrated-interval-semantics.md†L119-L159】【F:docs/foundations/concepts/terminology_thresholded_vs_probabilistic_regression.md†L1-L24】
8. Condition source and discretizer branching: introduce `condition_source` configuration and thread it through `CalibratedExplainer`, `CalibratedExplanations`, orchestrators, and explanation instances so condition labels can be derived from either observed labels or calibrated predictions. Update discretizer construction to branch between observed-label and prediction-based label building and propagate the choice into `instantiate_discretizer` with validated defaults. Extend runtime helper tests to exercise both observed- and prediction-based condition sources and update discretizer interface stubs accordingly. Plan the user-visible default change (`condition_source="prediction"`) to land in v0.11.0 (or at latest in `v1.0.0-rc`) with an explicit upgrade note and migration guidance for any callers that relied on the historical observed-label behaviour.
9. Update the Docs with a comprehensive API reference for the public API of `CalibratedExplainer`, `WrapCalibratedExplainer`, `CalibratedExplanations`, `CalibratedExplanation`, `FactualExplanation`, and `AlternativeExplanation` including detailed descriptions of methods, parameters, return types, and usage examples. This will help users understand how to effectively utilize the library's capabilities.【F:docs/api_reference/calibrated_explainer.md†L1-L150】

Release gate: Package boundaries, validation/caching/parallel tests, interval invariants, terminology cleanup, and updated ADR status notes all green with telemetry dashboards verifying the new signals (see ADR status appendix in this document).

### v0.10.1 (schema & visualization contracts)

1. Implement the ADR-005 envelope—introduce the structured payload, discriminant registry, provenance metadata, mandatory schema versioning, and refreshed fixtures/docs (see ADR status appendix in this document).
2. Finish ADR-007 and ADR-016 schema work: enhance PlotSpec dataclasses, registries, validation coverage, JSON round-trips, and headless export paths (see ADR status appendix in this document).
3. Restore ADR-014 visualization plugin architecture with working fallback builders, helper base classes, metadata/default renderers, override handling, validation, CLI utilities, and documentation (see ADR status appendix in this document).
4. Realign legacy plotting helpers with ADR-024/ADR-025 by honouring `show=False`, implementing save parameters, reinstating symmetric axes and interval backdrops, enforcing Matplotlib guards, and adding missing coverage (see ADR status appendix in this document).
5. Document dynamically generated visualization classes to close the remaining ADR-018 docstring gap tied to plugin guides (see ADR status appendix in this document).
6. Prototype streaming-friendly explanation delivery (opt-in) — implement an opt-in, non-breaking generator API for large exports (e.g., `CalibratedExplanations.to_json_stream(chunk_size=256)` or `to_json(stream=True)`) that yields JSON Lines or safe chunked JSON pieces. Collect minimal export telemetry (`export_rows`, `chunk_size`, `mode` (`batch`|`stream`), `peak_memory_mb`, `elapsed_seconds`, `schema_version`, `feature_branch`) and validate the memory profile (reference target: 10k rows < 200 MB at `chunk_size=256`). Mark streaming as experimental until prototype validation completes and record follow-up actions in the release notes.

Release gate: Envelope round-trips verified, PlotSpec/visualization plugin registries fully validated, legacy helpers behaving per ADR contracts, and docs updated with new schema references (see ADR status appendix in this document)

### v0.10.2 (plugin trust & packaging compliance)

1. Enforce ADR-006 trust controls—manual approval for third-party trust flags, deny-list enforcement, diagnostics for skipped plugins, and documented sandbox warnings (see ADR status appendix in this document).
2. Close ADR-013 protocol gaps by validating calibrators, returning protocol-compliant FAST outputs, freezing contexts, providing CLI diagnostics, and returning frozen defaults (see ADR status appendix in this document).
3. Finish ADR-015 integration work: ship an in-tree FAST plugin, rebuild explanation collections with canonical metadata, tighten trust enforcement, align environment variables, and provide immutable plugin handles (see ADR status appendix in this document).
4. Deliver ADR-010 optional-dependency splits by trimming core dependencies, completing extras/lockfiles, auto-skipping viz tests without extras, updating docs, and extending contributor guidance (see ADR status appendix in this document).
5. Extend ADR-021/ADR-026 telemetry by surfacing FAST probability cubes, interval dependency hints, and frozen bin metadata in runtime payloads (see ADR status appendix in this document).

Release gate: Plugin registries enforce trust and protocol policies, extras install cleanly with documentation parity, runtime telemetry captures interval metadata, and FAST/CLI flows succeed end-to-end (see ADR status appendix in this document).

### v0.11.0 (domain model & preprocessing finalisation)

1. Make the ADR-008 domain model authoritative—run runtime flows on domain objects, fix legacy round-trips, add calibration/model metadata, publish golden fixtures, and harden `_safe_pick` (see ADR status appendix in this document).
2. Complete ADR-009 preprocessing automation with built-in encoding, unseen-category enforcement, dtype diagnostics, and aligned telemetry/docs (see ADR status appendix in this document).
3. Finish ADR-017 nomenclature clean-up by eliminating double-underscore mutations, splitting utilities, reporting lint telemetry, and confining transitional shims to `legacy/` (see ADR status appendix in this document).
4. Extend governance dashboards to surface lint status alongside preprocessing/domain-model telemetry, ensuring ongoing monitoring after v1.0.0 (see ADR status appendix in this document).
5. Change default behavior for `condition_source` to `"prediction"` in `CalibratedExplainer` and related components. Update all relevant documentation, including the upgrade checklist, to inform users of this change and provide guidance on how to adjust their implementations if they previously relied on the default `"observed"` setting. This change aims to enhance the consistency of calibrated explanations by basing condition labels on model predictions rather than observed labels. Plan for this change to be communicated clearly in the v1.0.0 release notes and upgrade guides (see ADR status appendix in this document).

Release gate: Domain/preprocessing pipelines operate on ADR-compliant models with telemetry coverage, naming lint metrics published, and no outstanding ADR exceptions ahead of v1.0.0-rc (see ADR status appendix in this document)


### v1.0.0-rc (release candidate readiness)

1. Freeze Explanation Schema v1, publish draft compatibility statement, and
   communicate that only patch updates will follow for the schema.【F:docs/schema_v1.md†L1-L120】
2. Reconfirm wrap interfaces and exception taxonomy against v0.6.x contracts,
   updating README & CHANGELOG with a release-candidate compatibility note.【F:src/calibrated_explanations/core/wrap_explainer.py†L260-L471】【F:src/calibrated_explanations/core/exceptions.py†L1-L63】
3. Close ADR-017 by removing remaining transitional shims and ensure naming/tooling
   enforcement is green on the release branch.【F:improvement_docs/ADR-017_nomenclature_remediation.md†L40-L44】
4. Maintain ADR-018 compliance at ≥90% docstring coverage and outline the
   ongoing maintenance workflow in the RC changelog section.【F:improvement_docs/code_documentation_uplift.md†L24-L92】【F:improvement_docs/adrs/ADR-018-code-documentation-standard.md†L43-L62】
5. Validate the new caching/parallel toggles in staging, document safe defaults
   for RC adopters, and ensure telemetry captures cache hits/misses and worker
   utilisation metrics for release sign-off.【F:improvement_docs/adrs/ADR-003-caching-key-and-eviction.md†L28-L64】【F:improvement_docs/adrs/ADR-004-parallel-backend-abstraction.md†L25-L64】 See [Parallel Execution Improvement Plan – Phase 5](parallel_execution_improvement_plan.md#phase-5--rollout--documentation-week-15-16) for rollout and documentation activities.
6. Institutionalise ADR-019 by baking coverage checks into release branch
   policies, publishing a health dashboard (Codecov badge + waiver log), and
   enforcing `--cov-fail-under=90` in CI.【F:improvement_docs/adrs/ADR-019-test-coverage-standard.md†L34-L74】【F:improvement_docs/coverage_uplift_plan.md†L24-L33】
7. Promote ADR-024/ADR-025/ADR-026 from Draft to Accepted with implementation
   summaries so PlotSpec and plugin semantics remain authoritative before the
   freeze.【F:improvement_docs/adrs/ADR-024-plotspec-inputs.md†L1-L80】【F:improvement_docs/adrs/ADR-025-plotspec-rendering.md†L1-L90】【F:improvement_docs/adrs/ADR-026-explanation-plugins.md†L1-L86】
8. Launch the versioned documentation preview and public doc-quality dashboards
   (coverage badge, doc lint, notebook lint) described in the information
   architecture plan so stakeholders can validate the structure ahead of GA.【F:improvement_docs/documentation_information_architecture.md†L108-L118】
9. Provide an RC upgrade checklist covering environment variables, pyproject
   settings, CLI usage, caching controls, and plugin integration testing
   expectations.
10. Audit the ADR gap closure roadmap to confirm every gap is either implemented or superseded with an updated ADR decision before promoting the RC branch, recording outcomes in the status log (see ADR status appendix in this document).

Release gate: All schema/contract freezes documented, nomenclature and docstring
lint suites blocking green, PlotSpec/plugin ADRs promoted, versioned docs preview
and doc-quality dashboards live, caching/parallel telemetry dashboards reviewed,
coverage dashboards live, ADR gap closure log signed off, and upgrade checklist
ready for pilot customers.

### v1.0.0 (stability declaration)

1. Announce the stable plugin/telemetry contracts and publish the final
   compatibility statement across README, CHANGELOG, and docs hub.
2. Tag the v1.0.0 release, backport documentation to enterprise extension
   repositories, and circulate the upgrade checklist to partners with caching
   and parallelisation guidance.
3. Validate telemetry, plugin registries, cache behaviour, and worker scaling in
   production-like staging, signing off with no pending high-priority bugs.
4. Confirm ADR-017/ADR-018 guardrails remain enforced post-tag, monitor the
   caching/parallel telemetry dashboards, and schedule maintenance cadences
   (coverage/docstring audits, performance regression sweeps) for the first
   patch release.
5. Finalise versioned documentation hosting and publish long-term dashboard
   links (coverage, doc lint, notebooks) so the IA plan’s success metrics are met
   when GA lands.【F:improvement_docs/documentation_information_architecture.md†L108-L118】

Release gate: Tagged release artifacts available, documentation hubs updated with
versioned hosting and public dashboards, caching/parallel toggles operating
within documented guardrails, staging validation signed off, and post-release
maintenance cadences scheduled.

## ADR-019 integration analysis

- **Scope alignment:** The release milestones already emphasise testing and
  documentation maturity; ADR-019 adds explicit quantitative coverage gates that
  complement ADR-017/ADR-018 quality goals without altering plugin-focused
  scope.【F:improvement_docs/adrs/ADR-019-test-coverage-standard.md†L34-L74】
- **Milestone sequencing:** Early v0.6.x tasks capture baseline metrics and
  prepare `.coveragerc`, v0.7.0 introduces CI thresholds, v0.8.0 widens
  enforcement to critical paths and patch checks, and v0.9.0 retires waivers
  ahead of the release candidate. This staging keeps debt burn-down parallel to
  existing plugin/doc improvements.【F:improvement_docs/coverage_uplift_plan.md†L11-L48】
- **Release readiness:** By v1.0.0, coverage gating is embedded in branch
  policies and telemetry/documentation communications, ensuring ADR-019 remains
  sustainable beyond the initial rollout.【F:improvement_docs/adrs/ADR-019-test-coverage-standard.md†L34-L74】【F:improvement_docs/coverage_uplift_plan.md†L11-L48】

## Post-1.0 considerations

- Continue monitoring caching and parallel execution telemetry to determine
  whether the opt-in defaults can graduate to on-by-default in v1.1, updating
  ADR-003/ADR-004 rollout notes as needed.【F:improvement_docs/adrs/ADR-003-caching-key-and-eviction.md†L28-L64】【F:improvement_docs/adrs/ADR-004-parallel-backend-abstraction.md†L25-L64】
- Evaluate additional renderer plugins (plotly) after verifying PlotSpec default
  adoption.
- Plan schema v2 requirements with enterprise consumers before making breaking
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
| 3 | Telemetry lacks timings and utilisation metrics | 5 | 4 | 20 | **COMPLETED.** `ParallelMetrics` tracks durations and worker counts; telemetry emitted via `_emit`. |
| 4 | Context management & cancellation missing | 4 | 4 | 16 | **COMPLETED.** `__enter__`/`__exit__` and cancellation support implemented. |
| 5 | Configuration surface incomplete | 4 | 3 | 12 | **COMPLETED.** `ParallelConfig` adds `task_size_hint_bytes`, `force_serial_on_failure`, `instance_chunk_size`, `feature_chunk_size`. |
| 6 | Resource guardrails ignore cgroup/CI limits | 4 | 3 | 12 | **COMPLETED.** Guardrails enforce `max_workers` bounds and auto-strategy fallbacks under constrained environments. |
| 7 | Fallback warnings not emitted | 4 | 2 | 8 | **COMPLETED.** Telemetry and `force_serial_on_failure` emit fallback visibility for users. |
| 8 | Testing and benchmarking coverage limited | 0 | 0 | 0 | **COMPLETED.** Workload-hint resolution and auto-strategy heuristics now covered by targeted unit tests. |
| 9 | Documentation for strategies & troubleshooting lacking | 0 | 0 | 0 | **COMPLETED.** Practitioner playbook updated with ADR-004-complete guardrails and workload-driven chooser guidance. |

### ADR-004 phase tracking (release alignment)

| Phase | Release target | Alignment summary | Status |
| --- | --- | --- | --- |
| Phase 0 – Foundations | v0.9.0 runtime polish | Documentation ownership and telemetry schema groundwork in place ahead of toggle rollout. | ✅ Owner assignments completed; telemetry schema design finished. |
| Phase 1 – Configuration Surface | v0.9.0 runtime polish | Defines chunk-size and configuration knobs promised as opt-in runtime controls. | ✅ `ParallelConfig` extended with chunk/size hints and failure toggles. |
| Phase 2 – Executor & Plugin Refactor | v0.10.0 runtime realignment | Payload sharing, batching hooks, and lifecycle management for ADR-004. | ✅ Context manager and pooling lifecycle complete; payload sharing hooks merged. |
| Phase 3 – Workload-aware Strategy | v0.10.0 runtime realignment | Workload estimator and adaptive gating. | ✅ Auto strategy consumes work-item hints and size estimates, defaulting to serial for small batches. |
| Phase 4 – Testing & Benchmarking | v0.10.0 runtime realignment | Spawn lifecycle coverage and automated benchmark reporting. | ✅ Auto-strategy heuristics covered by unit tests; perf harness tracked via monitoring backlog. |
| Phase 5 – Rollout & Documentation | v0.10.0 release prep / v1.0.0-RC readiness | User guidance, changelog, and telemetry artefacts for release checklists. | ✅ Practitioner playbook and release status updated for ADR-004 completion. |

Alignment note: Parallel is treated as a shared service with domain-specific runtime wrappers (e.g., explain) expected to wrap
the shared `ParallelExecutor` so heuristics and chunking remain co-located with domain executors while respecting ADR-001
boundaries. ADR-004 now documents this expectation.

## ADR-005 – Explanation Envelope & Schema

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | ADR-compliant envelope absent | 5 | 4 | 20 | Serializer emits flat payloads without `type`, `generator`, `meta`, or nested `payload` blocks. |
| 2 | Enumerated type registry missing | 5 | 3 | 15 | No discriminant `type` field or per-type schema files exist. |
| 3 | Generator provenance (`parameters_hash`) missing | 4 | 3 | 12 | Exports lack provenance metadata required for reproducibility. |
| 4 | Validation helper misaligned | 4 | 3 | 12 | `validate_payload` skips semantic checks and does not enforce the envelope contract. |
| 5 | Schema version optional | 3 | 3 | 9 | Callers can omit `schema_version`, weakening compatibility guarantees. |
| 6 | Documentation & fixtures out of date | 3 | 2 | 6 | Docs and tests still describe the legacy flat schema, impeding migration. |

## ADR-006 – Plugin Trust Model

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Trust flag from third-party metadata auto-enables plugins | 4 | 4 | 16 | Registry honours `trusted=True` from packages without operator approval, bypassing opt-in controls. |
| 2 | Deny list not enforced during discovery | 3 | 3 | 9 | `CE_DENY_PLUGIN` is ignored when loading entry points, allowing denied plugins to register. |
| 3 | Untrusted entry-point metadata unavailable for diagnostics | 3 | 2 | 6 | Skipped plugins leave no audit trail, limiting operator visibility. |
| 4 | “No sandbox” warning undocumented | 2 | 2 | 4 | User-facing docs omit the ADR-required reminder that plugins execute without isolation. |

## ADR-007 – PlotSpec Abstraction

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | PlotSpec schema lacks kind/encoding/version fields | 5 | 3 | 15 | Dataclass cannot represent ADR-required structure, blocking new plot families. |
| 2 | Backend dispatcher & registry missing | 4 | 3 | 12 | Rendering hard-codes the matplotlib adapter with no extensible registry. |
| 3 | Plugin extensibility hooks absent | 4 | 3 | 12 | Plugins cannot register kinds/default renderers as ADR envisioned. |
| 4 | Kind-aware validation incomplete | 3 | 3 | 9 | `validate_plotspec` only understands bar bodies, leaving other kinds unchecked. |
| 5 | JSON round-trip inconsistent for non-bar plots | 3 | 2 | 6 | Triangular/global builders emit dicts outside the versioned serializer contract. |
| 6 | Headless export support missing | 2 | 2 | 4 | Adapter lacks byte-returning interfaces for remote rendering use cases. |

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
| 1 | Core dependency list still heavy | 5 | 4 | 20 | Mandatory dependencies include `ipython`, `lime`, and `matplotlib`, contradicting the lean-core mandate. |
| 2 | Evaluation extra incomplete | 4 | 3 | 12 | `[eval]` extra omits packages (`xgboost`, `lime`) required by evaluation scripts and docs. |
| 3 | Visualization tests not auto-skipped without extras | 4 | 3 | 12 | `pytest.mark.viz` cases run regardless of matplotlib availability, causing failures on core installs. |
| 4 | Evaluation environment lockfile missing | 3 | 2 | 6 | No `environment.yml`/requirements file accompanies the evaluation README. |
| 5 | Extras documentation inaccurate | 3 | 2 | 6 | README and researcher docs promise packages not actually bundled in extras. |
| 6 | Contributor guidance on extras absent | 2 | 2 | 4 | CONTRIBUTING lacks instructions for working with lean core vs. extras. |

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
| 2 | Docs build ignores `[viz]`/`[notebooks]` extras | 4 | 3 | 12 | CI installs bespoke requirements instead of project extras, letting dependency drift go unnoticed. |
| 3 | Example runtime ceiling unenforced | 3 | 3 | 9 | No automation times notebooks/examples, so the <30 s headless contract can regress silently. |
| 4 | Gallery tooling decision undocumented | 2 | 2 | 4 | ADR-required choice between sphinx-gallery/nbconvert is not recorded, leaving contributors without guidance. |

## ADR-013 – Interval Calibrator Plugin Strategy

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Runtime skips protocol validation for calibrators | 5 | 4 | 20 | Resolved plugins are not checked against classification/regression protocols or interval invariants before use. |
| 2 | FAST plugin returns non-protocol collections | 5 | 3 | 15 | `FastIntervalCalibratorPlugin.create` yields lists instead of protocol objects, breaking ADR guarantees. |
| 3 | Interval context remains mutable | 4 | 3 | 12 | Context builders hand plugins plain dicts, allowing mutation despite the read-only requirement. **STATUS 2025-11-04: Read-only contract documented in ADR-013 interval propagation section; enforcement remains.** |
| 4 | Legacy default plugin rebuilds calibrators | 3 | 3 | 9 | Default plugin reinstantiates calibrators rather than returning frozen instances as mandated. **STATUS 2025-11-04: Frozen instance requirement clarified in ADR-013; cache/registry integration remains.** |
| 5 | CLI interval validation commands missing | 2 | 2 | 4 | `ce.plugins explain-interval` and related commands are absent, reducing observability of runtime validation. |

## ADR-014 – Visualization Plugin Architecture

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Legacy fallback builder/renderer inert | 5 | 3 | 15 | Legacy style fallback returns empty results, breaking guaranteed `.plot()` behaviour. |
| 2 | Helper base classes (`viz/plugins.py`) missing | 4 | 3 | 12 | Third-party authors lack shared lifecycle/validation helpers. |
| 3 | Metadata lacks `default_renderer` | 4 | 3 | 12 | Registry cannot chain builders to preferred renderers. |
| 4 | Renderer override resolution incomplete | 4 | 3 | 12 | Environment variables and explicit overrides are ignored when selecting renderers. |
| 5 | Dedicated `PlotPluginError` absent | 3 | 2 | 6 | Failures bubble up as generic configuration errors instead of plot-specific diagnostics. |
| 6 | Default renderer skips `validate_plotspec` | 3 | 2 | 6 | Invalid specs can pass through unchecked before rendering. |
| 7 | CLI helpers not implemented | 3 | 2 | 6 | Required commands (`ce.plugins list --plots`, `validate-plot`, `set-default`) are missing. |
| 8 | Documentation for plot plugins lacking | 2 | 2 | 4 | No authoring guide or migration notes accompany the new plugin system. |

## ADR-015 – Explanation Plugin Integration

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | In-tree FAST plugin missing | 5 | 3 | 15 | Default `explain_fast` path fails without external plugins, contradicting ADR defaults. |
| 2 | Collection reconstruction bypassed | 4 | 3 | 12 | `CalibratedExplanations.from_batch` returns plugin-provided containers instead of rebuilding collections with metadata. |
| 3 | Trust enforcement during resolution lax | 4 | 3 | 12 | Resolver activates untrusted plugins unless the operator intervenes, bypassing ADR safeguards. |
| 4 | Predict bridge omits interval invariants | 4 | 3 | 12 | `LegacyPredictBridge` does not enforce `low ≤ predict ≤ high`, risking incorrect outputs. **STATUS 2025-11-04: Bridge contract and invariant enforcement requirements documented in ADR-026 subsections 2a/2b; implementation remains.** |
| 5 | Environment variable names diverge | 3 | 2 | 6 | Resolver expects mode-specific keys instead of `CE_EXPLANATION_PLUGIN[_FAST]` documented in the ADR. |
| 6 | Helper handles expose mutable explainer | 3 | 2 | 6 | Plugins receive direct access to the explainer instance, undermining the intended immutable context. **STATUS 2025-11-04: Immutability requirement clarified in ADR-026; enforcement remains.** |

## ADR-016 – PlotSpec Separation and Schema

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | PlotSpec dataclass lacks `kind`/`mode`/`feature_order` | 5 | 3 | 15 | Core schema from ADR-016 never landed, so adapters must guess fundamental metadata. |
| 2 | Feature indices discarded during dict conversion | 4 | 3 | 12 | Builders renumber `feature_order`, breaking parity with caller-supplied indices. |
| 3 | Validator still enforces legacy envelope | 4 | 3 | 12 | `viz.serializers.validate_plotspec` ignores ADR-016 invariants and is never invoked by builders. |
| 4 | Builders skip validation hooks | 3 | 3 | 9 | PlotSpec payloads return without structural checks, letting malformed specs leak to adapters. |
| 5 | `save_behavior` metadata unimplemented | 3 | 2 | 6 | Save hints stay in ad hoc dict manipulation instead of declared dataclass fields. |

## ADR-017 – Nomenclature Standardization

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Double-underscore fields still mutated outside legacy | 5 | 4 | 20 | Core helpers touch `__` attributes directly, contravening the ADR’s hard ban. |
| 2 | Naming guardrails lack automated enforcement | 4 | 4 | 16 | Ruff/pre-commit configuration does not fail on new snake-case or `__` violations, leaving the policy unenforced. |
| 3 | Kitchen-sink `utils/helper.py` persists | 3 | 3 | 9 | Monolithic helper resists the topic-focused split mandated for Phase 1. |
| 4 | Telemetry for lint drift missing | 3 | 3 | 9 | No runtime metrics capture naming debt, undermining the ADR’s governance plan. |
| 5 | Transitional shims remain first-class | 3 | 2 | 6 | `_legacy_explain` and similar helpers live alongside active modules instead of isolated legacy shims. |

## ADR-018 – Documentation Standardisation

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Wrapper public APIs lack full numpydoc blocks | 4 | 3 | 12 | `WrapCalibratedExplainer` methods omit Parameters/Returns/Raises despite being the stable user surface. |
| 2 | `IntervalRegressor.__init__` docstring outdated | 4 | 2 | 8 | Documented parameters no longer exist, misleading users of the calibrator. |
| 3 | `IntervalRegressor.bins` setter undocumented | 3 | 2 | 6 | Public mutator ships without a summary or type guidance. |
| 4 | Guard helpers missing summaries | 2 | 2 | 4 | `_assert_fitted`/`_assert_calibrated` lack the mandated one-line docstrings. |
| 5 | Nested combined-plot plugin classes undocumented | 2 | 2 | 4 | Dynamically returned classes expose blank docstrings, hurting plugin discoverability. |

## ADR-019 – Test Coverage Standard

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Coverage floor still enforced at 88 % | 5 | 4 | 20 | CI, local tooling, and the release checklist gate at 88 %, missing the ADR-mandated 90 % threshold. |
| 2 | Critical modules below 95 % without gates | 5 | 3 | 15 | Prediction helpers, serialization, and registry lack per-path enforcement and sit under target coverage. |
| 3 | Codecov patch gate optional | 4 | 4 | 16 | Patch status stays informational, so sub-88 % diffs can merge contrary to ADR policy. |
| 4 | Public API packages under-tested | 4 | 3 | 12 | `__init__` shims and gateway modules remain far from the guardrail, risking unnoticed regressions. |
| 5 | Exemptions lack expiry metadata | 3 | 2 | 6 | `.coveragerc` omissions omit review dates, weakening waiver governance. |

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
| 1 | Interval invariants never enforced | 5 | 4 | 20 | Prediction bridges return payloads without checking `low ≤ predict ≤ high`, undermining safety guarantees. **STATUS 2025-11-04 (CRITICAL): Invariant contract clarified uniformly across three levels (prediction, feature-weight, scenario) in ADR-021 subsections 4a-4b; enforcement remains for v0.10.0.** |
| 2 | FAST explanations drop probability cubes | 4 | 3 | 12 | `explain_fast` omits `__full_probabilities__`, so ADR-promised metadata is missing for FAST runs. |
| 3 | JSON export stores live callables | 3 | 2 | 6 | `_collection_metadata` serializes `assign_threshold` functions, breaking downstream tooling expectations. |

## ADR-022 – Documentation Information Architecture (Superseded)

*Superseded by ADR-027. See ADR-027 section for active gaps.*

## ADR-023 – Matplotlib Coverage Exemption

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Visualization tests never run in CI | 5 | 4 | 20 | Workflows skip the promised `pytest --no-cov -m viz` job, so regressions go undetected. |
| 2 | Pytest ignores block viz suite entirely | 5 | 3 | 15 | `pytest.ini` `--ignore` entries prevent even manual runs, nullifying the safety valve. |
| 3 | Coverage threshold messaging inconsistent | 3 | 4 | 12 | ADR cites an 85 % floor but tooling enforces 88 %, confusing contributors and waiver reviews. |

## ADR-024 – Legacy Plot Input Contracts

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | `_plot_global` ignores `show=False` | 5 | 3 | 15 | Helper always calls `plt.show()`, violating the headless contract and breaking CI/headless runs. |
| 2 | `_plot_global` lacks save parameters | 4 | 3 | 12 | Helper cannot honour ADR-shared save semantics, leaving plots unsaveable through the documented interface. |
| 3 | Save-path concatenation drift undocumented | 2 | 2 | 4 | Helper now normalises directories, diverging from ADR guidance without updated docs. |

## ADR-025 – Legacy Plot Rendering Semantics

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Matplotlib guard allows silent skips | 4 | 3 | 12 | Helpers return early without requiring Matplotlib even when file output is requested, hiding failures. |
| 2 | Regression axis not forced symmetric | 4 | 3 | 12 | `_plot_regression` sets raw min/max limits instead of the symmetric range promised by the ADR. |
| 3 | Interval backdrop disabled | 3 | 3 | 9 | Commented-out `fill_betweenx` leaves regression interval visuals inconsistent with documented design. |
| 4 | One-sided interval warning untested | 3 | 2 | 6 | Guard exists but lacks coverage, so regressions could ship unnoticed. |

## ADR-026 – Explanation Plugin Semantics

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | `explain` method remains public | 5 | 4 | 20 | `CalibratedExplainer.explain` is exposed as a public method, but ADR-026 defines it as an internal orchestration primitive that must not be invoked directly. |
| 2 | Predict bridge skips interval invariant checks | 5 | 3 | 15 | `_PredictBridgeMonitor` never enforces `low ≤ predict ≤ high`, letting malformed intervals through. **STATUS 2025-11-04 (CRITICAL): Calibration contract and validation requirements clarified in ADR-026 subsections 2a/2b/3a/3b; enforcement remains for v0.10.0.** |
| 3 | Explanation context exposes mutable dicts | 4 | 3 | 12 | Context builder embeds plain dicts despite the frozen contract, enabling plugin-side mutation. **STATUS 2025-11-04: Frozen context requirement clarified in ADR-026 subsection 1; enforcement remains.** |
| 4 | Telemetry omits interval dependency hints | 3 | 2 | 6 | Batch telemetry drops `interval_dependencies`, reducing observability. |
| 5 | Mondrian bins left mutable in requests | 2 | 2 | 4 | `ExplanationRequest` stores caller-supplied bins verbatim, violating the immutability promise. |

## ADR-027 – Documentation Standard (Audience Hubs)

| Rank | Gap | Violation | Scope | Unified severity | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | Audience-based navigation structure not implemented | 0 | 0 | 0 | **COMPLETED.** Hubs (`practitioner`, `researcher`, `contributor`) implemented in `docs/index.md`. |
| 2 | PR template lacks parity review gate | 0 | 0 | 0 | **COMPLETED.** Checklist item added to PR template. |
| 3 | “Task API comparison” reference missing | 3 | 3 | 9 | Get Started hub omits the mandated comparison link, weakening practitioner onboarding. |
| 4 | Telemetry concept page lacks substance | 4 | 2 | 8 | Flesh out telemetry concept content (required by ADR-027 advanced tracks). |
| 5 | Researcher future-work ledger absent | 3 | 2 | 6 | Researcher advanced hub lacks the promised roadmap tied to literature references. |
