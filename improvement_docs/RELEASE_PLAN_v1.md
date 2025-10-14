# Release Plan to v1.0.0

Last updated: 2025-10-05
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

- Maintain compatibility for the v0.6.x OSS series (no breaking contract changes
  before v0.7). Honour ADR-005 schema, the WrapCalibratedExplainer public
  surface (`fit`, `calibrate`, `explain_factual`, `explore_alternatives`,
  `predict`, `predict_proba`, plotting helpers, and uncertainty/threshold
  options), and the core exceptions.
- Keep plugin trust and telemetry hooks intact: mode/task metadata, interval and
  plot hints, `PredictBridge` monitoring.
- Avoid scope creep (no new ML strategies) so we can reach v1.0.0 with a polished
  plugin stack, documentation, and support tooling.
- Uphold ADR-017/ADR-018 naming and documentation conventions so contributor
  workflows, linting, and prose stay aligned with the evolving plugin-first
  architecture.【F:improvement_docs/adrs/ADR-017-nomenclature-standardization.md†L1-L37】【F:improvement_docs/adrs/ADR-018-code-documentation-standard.md†L1-L62】
- Adopt ADR-019 coverage guardrails as part of CI quality gates, keeping
  remediation milestones in sync with the coverage standardization plan.【F:improvement_docs/adrs/ADR-019-test-coverage-standard.md†L1-L74】【F:improvement_docs/test_coverage_standardization_plan.md†L1-L27】

## Release milestones

### v0.6.x (stabilisation patches)

- Hardening: add regression tests for plugin parity, schema validation, and
  WrapCalibratedExplainer keyword defaults.
- Documentation polish: refresh plugin guide with registry/CLI examples and note
  compatibility guardrails.
- No behavioural changes beyond docs/tests.
- Coverage readiness: ratify ADR-019, publish `.coveragerc` draft with
  provisional exemptions, and record baseline metrics to size the remediation
  backlog.【F:improvement_docs/adrs/ADR-019-test-coverage-standard.md†L1-L74】【F:improvement_docs/test_coverage_assessment.md†L1-L23】

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

### v0.9.0 (docs, packaging, performance polish)

1. Finalise documentation workflow per ADR-012 (CI build, gallery/linkcheck) and
   ensure plugin/telemetry pages are cross-linked.【F:improvement_docs/adrs/ADR-012-documentation-and-gallery-build-policy.md†L1-L80】
2. Publish plugin authoring guide + cookiecutter or scaffolding tasks (stretch
   from plugin gap plan).【F:improvement_docs/PLUGIN_GAP_CLOSURE_PLAN.md†L72-L78】
3. Ship scoped performance enhancements aligned with ADR-003/ADR-004: deliver an
   opt-in calibrator cache with eviction policy documentation, wire the
   multiprocessing backend toggle for `CalibratedExplainer.explain`, and update
   both ADRs to Accepted with implementation notes and rollout guidance.【F:improvement_docs/adrs/ADR-003-caching-key-and-eviction.md†L1-L64】【F:improvement_docs/adrs/ADR-004-parallel-backend-abstraction.md†L1-L64】【F:src/calibrated_explanations/core/calibrated_explainer.py†L1750-L2150】
   - explain repeatedly scans the entire perturbed_feature array and recomputes counts in pure Python loops. For every feature, instance, and candidate value, the code rebuilds boolean masks by iterating across the full perturbed_feature array and recalculates calibration counts from scratch, both for categorical and numeric features. This creates roughly quadratic complexity in the number of perturbations and explains the observed slowness; vectorizing these lookups or pre-grouping the perturbations would reduce the runtime drastically without leaving Python.
   - _explain_predict_step builds perturbation arrays with repeated np.concatenate, causing repeated full copies. The helper invoked by explain appends each new perturbation by concatenating NumPy arrays inside nested loops for every feature and candidate value. Because np.concatenate copies its inputs, this turns perturbation generation into an O(n²) process. Accumulating perturbations in Python lists and concatenating once (or preallocating) would greatly reduce overhead.
4. Address `CalibratedExplainer.explain` fallback performance by batching perturbation
   generation, vectorising aggregation loops, and expanding regression tests so the
   Python implementation meets release SLAs without a C/Cython port. Capture the outcome
   in release notes and guidance for plugin authors.【F:src/calibrated_explanations/core/calibrated_explainer.py†L1750-L2150】
5. Publish migration notes summarising plugin configuration defaults and
   remaining legacy escapes.
6. Turn ADR-018 tooling on by finishing pydocstyle batches E (`viz/`, `viz/plots.py`, `legacy/_plots_legacy.py`) and F (`serialization.py`, `core.py`), then making docstring linting blocking in CI, adding
   coverage gates for touched modules, and wiring badges/reporting into the docs
   workflow.【F:improvement_docs/documentation_standardization_plan.md†L24-L34】【F:improvement_docs/adrs/ADR-018-code-documentation-standard.md†L17-L62】【F:improvement_docs/pydocstyle_breakdown.md†L28-L29】
7. Advance ADR-017 enforcement by pruning deprecated shims scheduled for removal
   and locking naming lint rules in the release branch.【F:improvement_docs/nomenclature_standardization_plan.md†L25-L33】【F:improvement_docs/adrs/ADR-017-nomenclature-standardization.md†L28-L37】
8. Audit ADR-019 waiver inventory, trim expired exemptions, and raise
   non-critical modules toward the 90% floor to reduce debt before the v1 RC, and enable
   `--cov-fail-under=88` in CI.

Release gate: Docs CI green, packaging metadata includes CLI, caching/parallel
controls implemented with ADR updates merged, migration guide available,
docstring lint gates passing, ADR-019 waivers documented, and no outstanding
deprecated naming shims slated for removal.

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
7. Provide an RC upgrade checklist covering environment variables, pyproject
   settings, CLI usage, caching controls, and plugin integration testing
   expectations.

Release gate: All schema/contract freezes documented, nomenclature and docstring
lint suites blocking green, caching/parallel telemetry dashboards reviewed,
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

Release gate: Tagged release artifacts available, documentation hubs updated,
caching/parallel toggles operating within documented guardrails, staging
validation signed off, and post-release maintenance cadences scheduled.

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
