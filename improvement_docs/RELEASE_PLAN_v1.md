> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after v1.0.0 GA · Implementation window: v0.9.0–v1.0.0.

# Release Plan to v1.0.0

Maintainers: Core team
Scope: Concrete steps from v0.6.0 to a stable v1.0.0 with plugin-first execution.

## Current baseline (v0.6.0)

- Plugin orchestration is live for factual/alternative/fast modes with trusted
  in-tree adapters and runtime validation while legacy flows remain available via
  `_use_plugin=False`.【F:src/calibrated_explanations/core/calibrated_explainer.py†L388-L420】【F:src/calibrated_explanations/core/calibrated_explainer.py†L520-L606】
- WrapCalibratedExplainer preserves the public batch-first predict/predict_proba
  and explain helpers with keyword-compatible behaviour, ensuring enterprise
  wrappers keep working.【F:src/calibrated_explanations/core/wrap_explainer.py†L260-L471】
- Exception taxonomy and schema commitments are stable (`core.exceptions`,
  schema v1 docs/tests).【F:src/calibrated_explanations/core/exceptions.py†L1-L63】【F:docs/schema_v1.md†L1-L120】
- Interval plugins are defined but not yet resolved through the runtime; plot
  routing still defaults to the legacy adapter path.【F:src/calibrated_explanations/plugins/intervals.py†L1-L80】【F:src/calibrated_explanations/core/calibration_helpers.py†L1-L78】【F:src/calibrated_explanations/core/calibrated_explainer.py†L680-L720】

## Guiding principles

1. **Keep calibrated explanations front and centre.** Every milestone must begin
   with factual explanations (`explain_factual`) and alternative exploration
   (`explore_alternatives`), ensuring users can obtain calibrated answers before
   encountering any extras.
2. **Highlight probabilistic regression alongside classification.** The plan
   must showcase probabilistic and interval regression next to binary and
   multiclass classification so our differentiator is unmistakable, using the
   existing notebooks and quickstarts as canonical references.【F:notebooks/demo_probabilistic_regression.ipynb†L1-L20】
3. **Introduce alternatives with their triangular plots.** Whenever alternatives
   appear in docs or runtime work, pair them with the triangular alternative plot
   narrative and clarify that fast explanations live in an external plugin lane,
   not in the core onboarding.
4. **Use simple, reproducible examples first.** Tutorials should mirror the
   README and notebooks and layer in complexity only when discussing plugins or
   extras.【F:README.md†L1-L140】
5. **Deliver a clear, audience-led structure.** Follow the practitioner,
   researcher, and contributor journeys from the information architecture so
   navigation reinforces the calibrated-explanations-first story.【F:improvement_docs/documentation_information_architecture.md†L40-L118】
6. **Plan for external plugins.** Provide placeholders in documentation for
   listing community plugins, maintain an `external_plugins/` folder, and spell
   out how aggregated installation paths (e.g., extras) keep them discoverable.
7. **Ground messaging in published research.** Promote the research pedigree of
   calibrated explanations and probabilistic regression on every major landing
   page, citing the papers catalogued in `citing.md`.【F:docs/citing.md†L1-L140】
8. **Champion the plugin contract.** Keep the plugin system visible as the means
   to extend the framework—emphasising calibration guardrails and the ease of
   building aligned extensions.【F:improvement_docs/PLUGIN_GAP_CLOSURE_PLAN.md†L24-L70】
9. **Label extras as optional.** Telemetry, performance tooling, dashboards, and
   other peripherals remain opt-in so they never dilute the calibrated
   explanation narrative.

## Gap analysis against improvement plans

- **Triangular alternative plots missing from the release milestones.** The
  documentation review and information architecture plans expect every
  alternatives narrative to pair with the triangular plot guidance, but the
  release plan previously omitted the work; v0.9.0 now carries an explicit task
  to wire the storytelling and PlotSpec updates together.【F:improvement_docs/documentation_information_architecture.md†L58-L87】【F:improvement_docs/documentation_review.md†L1-L34】
- **External plugin ecosystem lacking structure.** The plugin gap plan and doc
  review call for discoverability improvements, yet the release plan had no
  tasks for placeholders or aggregated installation; v0.9.0 now introduces the
  `external_plugins/` folder, documentation hooks, and an installation extra to
  keep community plugins aligned.【F:improvement_docs/PLUGIN_GAP_CLOSURE_PLAN.md†L60-L94】【F:improvement_docs/documentation_review.md†L24-L49】
- **Research-forward messaging not tied to milestones.** Supporting documents
  highlight the need for research callouts and citation pathways, so v0.9.0 now
  commits to reinforcing "Backed by research" content on all landing pages.
- **Fast explanations treated as core in some guides.** Plans now ensure fast
  explanation guidance is reframed as an external plugin topic during v0.9.0 so
  core docs stay focused on calibrated factual and alternative flows.【F:improvement_docs/documentation_information_architecture.md†L44-L76】

## Release milestones

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
   phase plans.【F:improvement_docs/nomenclature_standardization_plan.md†L5-L13】【F:improvement_docs/documentation_standardization_plan.md†L7-L22】
   - 2025-10-07 – Updated test helpers (`tests/conftest.py`, `tests/unit/core/test_calibrated_explainer_interval_plugins.py`) to comply with Ruff naming guardrails, keeping ADR-017 lint checks green.
   - 2025-10-07 – Harmonised `core.validation` docstring spacing with numpy-style guardrails to satisfy ADR-018 pydocstyle checks.
6. Implement ADR-019 phase 1 changes: ship shared `.coveragerc`, enable
   `--cov-fail-under=80` in CI, and document waiver workflow in contributor
   templates.【F:improvement_docs/adrs/ADR-019-test-coverage-standard.md†L34-L74】【F:improvement_docs/test_coverage_standardization_plan.md†L9-L27】

Release gate: parity tests green for factual/alternative/fast, interval override
coverage exercised, CLI packaging verified, and nomenclature/doc lint warnings
live in CI with coverage thresholds enforcing ≥90% package-level coverage.

### v0.8.0 (plot routing, telemetry, and doc IA rollout)

1. Adopt ADR-022 by restructuring the documentation toctree into the role-based information architecture, assigning section owners, and shipping the new telemetry concept page plus quickstart refactor per the information architecture plan.【F:improvement_docs/adrs/ADR-022-documentation-information-architecture.md†L1-L73】【F:improvement_docs/documentation_information_architecture.md†L1-L129】
   - Land the docs sitemap rewrite with a crosswalk checklist (legacy page -> new section) and block merge on green sphinx-build -W, linkcheck, and nav tests to prevent broken routes.
   - Refactor quickstart content into runnable classification and regression guides, wire them into docs smoke tests, and add troubleshooting callouts for supported environments.
   - Publish the telemetry concept page with instrumentation examples, expand the plugin registry and CLI walkthroughs, and sync configuration references (pyproject, env vars, CLI flags) with the new navigation.
   - Record section ownership in docs/OWNERS.md (Overview/Get Started - release manager; How-to/Concepts - runtime tech lead; Extending/Governance - contributor experience lead) and update the pre-release doc checklist so every minor release verifies ADR-022 guardrails.
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
   names.【F:improvement_docs/nomenclature_standardization_plan.md†L15-L24】
7. Complete ADR-018 baseline remediation by finishing pydocstyle batches C (`explanations/`, `perf/`) and D (`plugins/`), adding module summaries and
   upgrading priority package docstrings to numpydoc format with progress
   tracking.【F:improvement_docs/documentation_standardization_plan.md†L16-L22】【F:improvement_docs/adrs/ADR-018-code-documentation-standard.md†L17-L62】【F:improvement_docs/pydocstyle_breakdown.md†L26-L27】
8. Extend ADR-019 enforcement to critical-path modules (≥95% coverage) and
   enable Codecov patch gating at ≥85% for PRs touching runtime/calibration
   logic, enable
   `--cov-fail-under=85` in CI.【F:improvement_docs/adrs/ADR-019-test-coverage-standard.md†L34-L74】【F:improvement_docs/test_coverage_standardization_plan.md†L15-L27】
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
4. **Label telemetry and performance scaffolding as optional tooling.** Move telemetry schema/how-to material into contributor governance sections with "Optional" badges, ensure practitioner guides mention telemetry only for compliance scenarios, and audit navigation labels to avoid implying these extras are mandatory.【F:improvement_docs/documentation_information_architecture.md†L70-L113】
5. **Highlight research pedigree throughout.** Add "Backed by research" callouts to Overview, practitioner quickstarts, and probabilistic regression concept pages; cross-link citing.md and key publications in relevant sections.【F:improvement_docs/documentation_review.md†L15-L34】
6. **Triangular alternatives plots everywhere alternatives appear.** Update explanation guides, PlotSpec docs, and runtime examples so any mention of `explore_alternatives` introduces the triangular plot and its interpretation, reinforcing calibrated decision boundaries.
7. **Complete ADR-012 doc workflow enforcement.** Keep Sphinx `-W`, gallery build, and linkcheck mandatory; extend CI smoke tests to run the refreshed quickstarts and fail if optional extras are presented without labels.【F:improvement_docs/adrs/ADR-012-documentation-and-gallery-build-policy.md†L1-L80】
8. **Turn ADR-018 tooling fully blocking.** Finish pydocstyle batches E (`viz/`, `viz/plots.py`, `legacy/_plots_legacy.py`) and F (`serialization.py`, `core.py`), capture and commit the baseline failure report before flipping enforcement, add the documentation coverage badge, and extend linting to notebooks/examples so the Phase 3 automation backlog is complete.【F:improvement_docs/documentation_standardization_plan.md†L29-L41】【F:improvement_docs/pydocstyle_breakdown.md†L28-L33】
9. **Advance ADR-017 naming cleanup.** Prune deprecated shims scheduled for removal and ensure naming lint rules stay green on the release branch.【F:improvement_docs/nomenclature_standardization_plan.md†L25-L33】【F:improvement_docs/adrs/ADR-017-nomenclature-standardization.md†L28-L37】
10. **Sustain ADR-019 coverage uplift.** Audit waiver inventory, retire expired exemptions, raise non-critical modules toward the 90% floor, enable `--cov-fail-under=88` in CI, and execute the module-level remediation sprints for interval regressors, registry/CLI, plotting, and explanation caching per the dedicated gap plan.【F:improvement_docs/test_coverage_gap_plan.md†L5-L120】
11. **Scoped runtime polish for explain performance.** Deliver the opt-in calibrator cache, multiprocessing toggle, and vectorised perturbation handling per ADR-003/ADR-004 analysis so calibrated explanations stay responsive without compromising accuracy. Capture improvements and guidance for plugin authors.【F:improvement_docs/adrs/ADR-003-caching-key-and-eviction.md†L1-L64】【F:improvement_docs/adrs/ADR-004-parallel-backend-abstraction.md†L1-L64】【F:src/calibrated_explanations/core/calibrated_explainer.py†L1750-L2150】
12. **Plugin CLI, discovery, and denylist parity (optional extras).** Extend trust toggles and entry-point discovery to interval/plot plugins, add the `CE_DENY_PLUGIN` registry control highlighted in the OSS scope review, and ship the whole surface as opt-in so calibrated explanations remain usable without telemetry/CLI adoption.【F:improvement_docs/OSS_CE_scope_and_gaps.md†L68-L110】
13. **External plugin distribution path.** Document and test an aggregated installation extra (e.g., `pip install calibrated-explanations[external-plugins]`) that installs all supported external plugins, outline curation criteria, and add placeholders in docs and README for community plugin listings.
14. **Explanation export convenience.** Provide `to_json()`/`from_json()` helpers on explanation collections that wrap schema v1 utilities and document them as optional aids for integration teams.
15. **Scope streaming-friendly explanation delivery.** Prototype generator or chunked export paths (or record a formal deferral) so memory-sensitive users know how large batches will be handled, capturing the outcome directly in the OSS scope inventory.【F:improvement_docs/OSS_CE_scope_and_gaps.md†L86-L118】

Release gate: Audience landing pages published with calibrated explanations/probabilistic regression foregrounded, research callouts present on all entry points, telemetry/performance extras labelled optional, docs CI (including quickstart smoke tests, notebook lint, and doc coverage badge) green, ADR-017/018/019 gates enforced, runtime performance enhancements landed without altering calibration outputs, plugin denylist control shipped, streaming plan recorded, and optional plugin extras (CLI/discovery/export) documented as add-ons.

### v1.0.0-rc (release candidate readiness)

1. Freeze Explanation Schema v1, publish draft compatibility statement, and
   communicate that only patch updates will follow for the schema.【F:docs/schema_v1.md†L1-L120】
2. Reconfirm wrap interfaces and exception taxonomy against v0.6.x contracts,
   updating README & CHANGELOG with a release-candidate compatibility note.【F:src/calibrated_explanations/core/wrap_explainer.py†L260-L471】【F:src/calibrated_explanations/core/exceptions.py†L1-L63】
3. Close ADR-017 by removing remaining transitional shims and ensure naming/tooling
   enforcement is green on the release branch.【F:improvement_docs/nomenclature_standardization_plan.md†L25-L33】
4. Maintain ADR-018 compliance at ≥90% docstring coverage and outline the
   ongoing maintenance workflow in the RC changelog section.【F:improvement_docs/documentation_standardization_plan.md†L29-L34】【F:improvement_docs/adrs/ADR-018-code-documentation-standard.md†L43-L62】
5. Validate the new caching/parallel toggles in staging, document safe defaults
   for RC adopters, and ensure telemetry captures cache hits/misses and worker
   utilisation metrics for release sign-off.【F:improvement_docs/adrs/ADR-003-caching-key-and-eviction.md†L28-L64】【F:improvement_docs/adrs/ADR-004-parallel-backend-abstraction.md†L25-L64】
6. Institutionalise ADR-019 by baking coverage checks into release branch
   policies, publishing a health dashboard (Codecov badge + waiver log), and
   enforcing `--cov-fail-under=90` in CI.【F:improvement_docs/adrs/ADR-019-test-coverage-standard.md†L34-L74】【F:improvement_docs/test_coverage_standardization_plan.md†L21-L27】
7. Promote ADR-024/ADR-025/ADR-026 from Draft to Accepted with implementation
   summaries so PlotSpec and plugin semantics remain authoritative before the
   freeze.【F:improvement_docs/adrs/ADR-024-plotspec-inputs.md†L1-L80】【F:improvement_docs/adrs/ADR-025-plotspec-rendering.md†L1-L90】【F:improvement_docs/adrs/ADR-026-explanation-plugins.md†L1-L86】
8. Launch the versioned documentation preview and public doc-quality dashboards
   (coverage badge, doc lint, notebook lint) described in the information
   architecture plan so stakeholders can validate the structure ahead of GA.【F:improvement_docs/documentation_information_architecture.md†L108-L118】
9. Provide an RC upgrade checklist covering environment variables, pyproject
   settings, CLI usage, caching controls, and plugin integration testing
   expectations.

Release gate: All schema/contract freezes documented, nomenclature and docstring
lint suites blocking green, PlotSpec/plugin ADRs promoted, versioned docs preview
and doc-quality dashboards live, caching/parallel telemetry dashboards reviewed,
coverage dashboards live, and upgrade checklist ready for pilot customers.

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
  existing plugin/doc improvements.【F:improvement_docs/test_coverage_standardization_plan.md†L9-L27】
- **Release readiness:** By v1.0.0, coverage gating is embedded in branch
  policies and telemetry/documentation communications, ensuring ADR-019 remains
  sustainable beyond the initial rollout.【F:improvement_docs/adrs/ADR-019-test-coverage-standard.md†L34-L74】

## Post-1.0 considerations

- Continue monitoring caching and parallel execution telemetry to determine
  whether the opt-in defaults can graduate to on-by-default in v1.1, updating
  ADR-003/ADR-004 rollout notes as needed.【F:improvement_docs/adrs/ADR-003-caching-key-and-eviction.md†L28-L64】【F:improvement_docs/adrs/ADR-004-parallel-backend-abstraction.md†L25-L64】
- Evaluate additional renderer plugins (plotly) after verifying PlotSpec default
  adoption.
- Plan schema v2 requirements with enterprise consumers before making breaking
  changes.
