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
  before v0.7). Honour ADR-005 schema, WrapCalibratedExplainer surface, and core
  exceptions.
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

### v0.8.0 (plot routing & telemetry completeness)

1. Promote PlotSpec builders to default for at least factual/alternative plots
   while keeping legacy style available as fallback.【F:src/calibrated_explanations/core/calibrated_explainer.py†L680-L720】【F:src/calibrated_explanations/viz/builders.py†L150-L208】
2. Ensure explain* APIs emit CE-formatted intervals for both percentile and
   thresholded regression requests. When the mode is regression and
   `threshold` is provided, calibrate the percentile representing
   \(\Pr(y \leq \text{threshold})\) via Venn-Abers and expose the resulting
   probability interval alongside the CE-formatted interval metadata. Extend
   tests covering dict payloads, telemetry fields, and thresholded regression
   fixtures so callers see the calibrated probability interval reflected in the
   API response.【F:src/calibrated_explanations/core/calibrated_explainer.py†L760-L820】
3. Document telemetry schema (interval_source/proba_source/plot_source) for
   enterprise integrations and provide examples in docs/plugins.md.
4. Review preprocessing persistence contract (ADR-009) to confirm saved
   preprocessor metadata matches expectations.【F:improvement_docs/adrs/ADR-009-input-preprocessing-and-mapping-policy.md†L1-L80】
5. Execute ADR-017 Phase 2 renames with legacy shims isolated under a
   `legacy/` namespace and update imports/tests/docs to the canonical module
   names.【F:improvement_docs/nomenclature_standardization_plan.md†L15-L24】
6. Complete ADR-018 baseline remediation by finishing pydocstyle batches C (`explanations/`, `perf/`) and D (`plugins/`), adding module summaries and
   upgrading priority package docstrings to numpydoc format with progress
   tracking.【F:improvement_docs/documentation_standardization_plan.md†L16-L22】【F:improvement_docs/adrs/ADR-018-code-documentation-standard.md†L17-L62】【F:improvement_docs/pydocstyle_breakdown.md†L26-L27】
7. Extend ADR-019 enforcement to critical-path modules (≥95% coverage) and
   enable Codecov patch gating at ≥85% for PRs touching runtime/calibration
   logic, enable
   `--cov-fail-under=85` in CI.【F:improvement_docs/adrs/ADR-019-test-coverage-standard.md†L34-L74】【F:improvement_docs/test_coverage_standardization_plan.md†L15-L27】

Release gate: PlotSpec default route parity, telemetry docs/tests in place,
nomenclature renames shipped with shims, docstring coverage dashboard shows
baseline met, and ADR-019 critical-path thresholds pass consistently.

### v0.9.0 (docs, packaging, performance polish)

1. Finalise documentation workflow per ADR-012 (CI build, gallery/linkcheck) and
   ensure plugin/telemetry pages are cross-linked.【F:improvement_docs/adrs/ADR-012-documentation-and-gallery-build-policy.md†L1-L80】
2. Publish plugin authoring guide + cookiecutter or scaffolding tasks (stretch
   from plugin gap plan).【F:improvement_docs/PLUGIN_GAP_CLOSURE_PLAN.md†L72-L78】
3. Reassess optional perf features (caching/parallel) and either mark ADR-003/004
   as deferred beyond v1 or land minimal opt-in implementations guarded by docs.【F:improvement_docs/adrs/ADR-003-caching-key-and-eviction.md†L1-L64】【F:improvement_docs/adrs/ADR-004-parallel-backend-abstraction.md†L1-L64】
4. Publish migration notes summarising plugin configuration defaults and
   remaining legacy escapes.
5. Turn ADR-018 tooling on by finishing pydocstyle batches E (`viz/`, `_plots.py`, `_plots_legacy.py`) and F (`serialization.py`, `core.py`), then making docstring linting blocking in CI, adding
   coverage gates for touched modules, and wiring badges/reporting into the docs
   workflow.【F:improvement_docs/documentation_standardization_plan.md†L24-L34】【F:improvement_docs/adrs/ADR-018-code-documentation-standard.md†L17-L62】【F:improvement_docs/pydocstyle_breakdown.md†L28-L29】
6. Advance ADR-017 enforcement by pruning deprecated shims scheduled for removal
   and locking naming lint rules in the release branch.【F:improvement_docs/nomenclature_standardization_plan.md†L25-L33】【F:improvement_docs/adrs/ADR-017-nomenclature-standardization.md†L28-L37】
7. Audit ADR-019 waiver inventory, trim expired exemptions, and raise
   non-critical modules toward the 90% floor to reduce debt before the v1 RC, and enable
   `--cov-fail-under=88` in CI.

Release gate: Docs CI green, packaging metadata includes CLI, migration guide
available, docstring lint gates passing, ADR-019 waivers documented, and no
outstanding deprecated naming shims slated for removal.

### v1.0.0 (stability declaration)

1. Announce stable plugin/telemetry contracts and freeze Explanation Schema v1
   (patch-only updates afterwards).【F:docs/schema_v1.md†L1-L120】
2. Verify wrap interfaces and exception taxonomy unchanged, update README &
   CHANGELOG with compatibility statement.【F:src/calibrated_explanations/core/wrap_explainer.py†L260-L471】【F:src/calibrated_explanations/core/exceptions.py†L1-L63】
3. Provide upgrade checklist covering environment variables, pyproject settings,
   and CLI usage.
4. Tag release and backport documentation to enterprise extension repositories.
5. Close ADR-017 by removing remaining transitional shims and confirm
   nomenclature/tooling enforcement is stable post-release.【F:improvement_docs/nomenclature_standardization_plan.md†L25-L33】
6. Keep ADR-018 compliance at ≥90% docstring coverage and document the ongoing
   maintenance process in changelog and docs.【F:improvement_docs/documentation_standardization_plan.md†L29-L34】【F:improvement_docs/adrs/ADR-018-code-documentation-standard.md†L43-L62】
7. Institutionalise ADR-019 by baking coverage checks into release branch
   policies and publishing a health dashboard (Codecov badge + waiver log) in
   the docs. Enforce
   `--cov-fail-under=90` in CI.【F:improvement_docs/adrs/ADR-019-test-coverage-standard.md†L34-L74】【F:improvement_docs/test_coverage_standardization_plan.md†L21-L27】

Release gate: No pending high-priority bugs, docs/tests/telemetry stable, plugin
registry feature-complete for explanation/interval/plot categories, naming and
docstring standards locked, and ADR-019 guardrails part of release branch policy.

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

- Consider enabling caching/parallel features by default once performance data
  supports it (may require updated ADRs).【F:improvement_docs/adrs/ADR-003-caching-key-and-eviction.md†L1-L64】【F:improvement_docs/adrs/ADR-004-parallel-backend-abstraction.md†L1-L64】
- Evaluate additional renderer plugins (plotly) after verifying PlotSpec default
  adoption.
- Plan schema v2 requirements with enterprise consumers before making breaking
  changes.
