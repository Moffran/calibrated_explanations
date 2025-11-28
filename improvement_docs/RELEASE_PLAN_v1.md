> **Status note (2025-11-28):** Last edited 2025-11-28 · Archive after v1.0.0 GA · Implementation window: v0.9.0–v1.0.0 · **UPDATE: ADR-001 Stages 0–1c completed 2025-11-28** (see `improvement_docs/adrs/ADR-001-STAGE-1-COMPLETION-REPORT.md`).

# Release Plan to v1.0.0

### Current released version: v0.9.0

> Status: v0.9.0 shipped on 2025-11-07.


Maintainers: Core team
Scope: Concrete steps from v0.6.0 to a stable v1.0.0 with plugin-first execution.

## ADR gap closure roadmap

The ADR gap analysis enumerates open issues across the architecture. The breakdown below assigns every recorded gap to a remediation strategy and target release before v1.0.0. Severity values cite the unified scoring in `ADR-gap-analysis.md`.【F:improvement_docs/ADR-gap-analysis.md†L1-L291】

### ADR-001 – Package and Boundary Layout

**Implementation Status**: ✅ Stages 0–1c **COMPLETED** (2025-11-28). See `ADR-001-STAGE-1-COMPLETION-REPORT.md` for details.

- ✅ **Calibration layer remains embedded in `core`** (severity 20, critical) → `v0.10.0 runtime boundary realignment`. COMPLETED: Calibration extracted to top-level package with compatibility shim.
- ⏳ **Core imports downstream siblings directly** (severity 20, critical) → `v0.10.0 runtime boundary realignment`. DEFERRED to Stage 2 (CalibratedExplainer refactor). 
- ✅ **Cache and parallel boundaries not split** (severity 12, high) → `v0.10.0 runtime boundary realignment`. COMPLETED: Split into separate packages with perf shim.
- ✅ **Schema validation package missing** (severity 6, medium) → `v0.10.1 schema & visualization contracts`. COMPLETED: Schema validation package created.
- ⏳ **Public API surface overly broad** (severity 6, medium) → `v0.10.0 runtime boundary realignment`. DEFERRED to Stage 3.
- ⏳ **Extra top-level namespaces lack ADR coverage** (severity 6, medium) → `v0.10.0 runtime boundary realignment`. DEFERRED to Stage 4.

### ADR-002 – Exception Taxonomy and Validation Contract
- **Legacy `ValueError`/`RuntimeError` usage in core and plugins** (severity 20, critical) → `v0.10.0 runtime boundary realignment`. Replace direct raises with taxonomy classes and add regression tests for calibration, plugin, and prediction flows.
- **Validation API contract not implemented** (severity 16, critical) → `v0.10.0 runtime boundary realignment`. Implement shared validation entry points that wrappers and plugins can reuse, aligned with ADR signatures.
- **Structured error payload helpers absent** (severity 12, high) → `v0.10.0 runtime boundary realignment`. Add helpers for diagnostics payloads and wire them through explain/export surfaces.
- **`validate_param_combination` is a no-op** (severity 9, high) → `v0.10.0 runtime boundary realignment`. Implement parameter guardrails and document enforcement in migration notes.
- **Fit-state and alias handling inconsistent** (severity 6, medium) → `v0.10.0 runtime boundary realignment`. Harmonise wrappers with `canonicalize_kwargs` and extend contract tests.

### ADR-003 – Caching Strategy
- **Automatic invalidation & flush hooks missing** (severity 20, critical) → `v0.10.0 runtime boundary realignment`. Track cache versions, expose manual flush APIs, and update docs per ADR-003.
- **Required artefacts not cached** (severity 16, critical) → `v0.10.0 runtime boundary realignment`. Extend caching to calibration summaries and explanation tensors with eviction rules.
- **Governance & documentation (STRATEGY_REV) absent** (severity 12, high) → `v0.10.0 runtime boundary realignment`. Add governance artefacts (strategy revision log, release checklist hooks) and, if scope diverges, update ADR-003 rationale.
- **Telemetry integration incomplete** (severity 9, high) → `v0.10.0 runtime boundary realignment`. Emit hit/miss counters via telemetry and ensure docs reflect the signals.
- **Backend diverges from cachetools + pympler stack** (severity 9, high) → `v0.10.0 runtime boundary realignment`. Standardise on the mandated libraries or amend ADR-003 with rationale before RC.

### ADR-004 – Parallel Execution Framework

> Tracking: [Parallel Execution Improvement Plan](parallel_execution_improvement_plan.md#release-plan-alignment) (Phases 0–5)

- **Workload-aware auto strategy absent** (severity 20, critical) → deferred to `v0.10.0 runtime boundary realignment`. The v0.9.1 plan introduces a conservative `ParallelFacade` to centralize selection heuristics and collect decision telemetry; the full `_auto_strategy` implementation will be revisited for v0.10 after field evidence is gathered.
- **Telemetry lacks timings and utilisation metrics** (severity 20, critical) → phased: v0.9.1 will require compact decision telemetry (decision, reason, n_instances, n_features, bytes_hint, platform, executor_type). Collection of per-task timings and worker utilisation is deferred to v0.10.
- **Context management & cancellation missing** (severity 16, critical) → deferred to `v0.10.0 runtime boundary realignment`. v0.9.1 does not add cooperative cancellation but will document expectations.
- **Configuration surface incomplete** (severity 12, high) → phased: add a small conservative config surface in v0.9.1 (min_features_for_parallel, min_instances_for_parallel, task_size_hint_bytes) exposed via the facade; richer flags like `force_serial_on_failure` and advanced injection hooks remain v0.10 scope.
- **Resource guardrails ignore cgroup/CI limits** (severity 12, high) → deferred to `v0.10.0 runtime boundary realignment`. The facade will use conservative defaults to avoid oversubscription and expose overrides for CI/staging in v0.9.1.
- **Fallback warnings not emitted** (severity 8, medium) → v0.9.1: the facade will emit structured decision telemetry and warn when it forces a serial fallback. Full structured warnings/telemetry integration will be extended in v0.10.
- **Testing and benchmarking coverage limited** (severity 9, high) → v0.9.1: add unit tests for facade logic and a micro-benchmark harness (evaluation/parallel_ablation.py) for evidence-driven decisions; exhaustive lifecycle tests (spawn/fork/joblib) remain v0.10 work.
- **Documentation for strategies & troubleshooting lacking** (severity 6, medium) → v0.9.1: ship minimal guidance for the facade (env var matrix, decision heuristics) and record the v0.10 plan in ADR notes.

### ADR-005 – Explanation Envelope & Schema

- **ADR-compliant envelope absent** (severity 20, critical) → `v0.10.1 schema & visualization contracts`. Replace flat payloads with envelope structure and migrate serializers/tests.
- **Enumerated type registry missing** (severity 15, critical) → `v0.10.1 schema & visualization contracts`. Build discriminant registry and versioned schema files.
- **Generator provenance (`parameters_hash`) missing** (severity 12, high) → `v0.10.1 schema & visualization contracts`. Persist provenance metadata and document usage.
- **Validation helper misaligned** (severity 12, high) → `v0.10.1 schema & visualization contracts`. Align helpers with envelope semantics and enforce them in CI.
- **Schema version optional** (severity 9, high) → `v0.10.1 schema & visualization contracts`. Make `schema_version` mandatory with migration warnings.
- **Documentation & fixtures out of date** (severity 6, medium) → `v0.10.1 schema & visualization contracts`. Refresh docs/tests and update ADR-005 with acceptance notes.

### ADR-006 – Plugin Trust Model

- **Trust flag from third-party metadata auto-enables plugins** (severity 16, critical) → `v0.10.2 plugin trust & packaging compliance`. Require explicit operator approval before honouring third-party trust flags.
- **Deny list not enforced during discovery** (severity 9, high) → `v0.10.2 plugin trust & packaging compliance`. Enforce `CE_DENY_PLUGIN` across discovery paths with tests.
- **Untrusted entry-point metadata unavailable for diagnostics** (severity 6, medium) → `v0.10.2 plugin trust & packaging compliance`. Record skipped plugin metadata in telemetry/logging.
- **“No sandbox” warning undocumented** (severity 4, low) → `v0.10.2 plugin trust & packaging compliance`. Document the warning prominently and add governance checklist entries.

### ADR-007 – PlotSpec Abstraction

- **PlotSpec schema lacks kind/encoding/version fields** (severity 15, critical) → `v0.10.1 schema & visualization contracts`. Extend the dataclass so new plot families can express ADR-required structure.【F:improvement_docs/ADR-gap-analysis.md†L97-L102】
- **Backend dispatcher & registry missing** (severity 12, high) → `v0.10.1 schema & visualization contracts`. Introduce an extensible registry instead of hard-coding the matplotlib adapter.【F:improvement_docs/ADR-gap-analysis.md†L97-L102】
- **Plugin extensibility hooks absent** (severity 12, high) → `v0.10.1 schema & visualization contracts`. Provide registration hooks for kinds/default renderers per the ADR.【F:improvement_docs/ADR-gap-analysis.md†L97-L102】
- **Kind-aware validation incomplete** (severity 9, high) → `v0.10.1 schema & visualization contracts`. Extend validators to all plot kinds.【F:improvement_docs/ADR-gap-analysis.md†L100-L101】
- **JSON round-trip inconsistent for non-bar plots** (severity 6, medium) → `v0.10.1 schema & visualization contracts`. Harden serializers/deserializers with fixtures.【F:improvement_docs/ADR-gap-analysis.md†L101-L102】
- **Headless export support missing** (severity 4, low) → `v0.10.1 schema & visualization contracts`. Provide byte-based export path for remote rendering.【F:improvement_docs/ADR-gap-analysis.md†L101-L102】

### ADR-008 – Explanation Domain Model

- **Domain model not authoritative source** (severity 20, critical) → `v0.11.0 domain model & preprocessing finalisation`. Shift runtime flows to use domain objects natively and keep legacy dicts as adapters.
- **Legacy-to-domain round-trip fails for conjunctive rules** (severity 12, high) → `v0.11.0 domain model & preprocessing finalisation`. Fix conversion helpers and add fixtures.
- **Structured model/calibration metadata absent** (severity 12, high) → `v0.11.0 domain model & preprocessing finalisation`. Extend dataclasses with calibration/model metadata per ADR-008. *Status 2025-11-04: Factual/alternative payload structures clarified with formal definitions; implementation work remains.*
- **Golden fixture parity tests missing** (severity 6, medium) → `v0.11.0 domain model & preprocessing finalisation`. Add byte-level fixtures for serialization/regression coverage.
- **`_safe_pick` silently duplicates data** (severity 6, medium) → `v0.11.0 domain model & preprocessing finalisation`. Enforce invariant checks with targeted exceptions.

### ADR-009 – Preprocessing Pipeline

- **Automatic encoding pathway unimplemented** (severity 20, critical) → `v0.11.0 domain model & preprocessing finalisation`. Implement the built-in encoder and integrate with pipeline defaults.
- **Unseen-category policy ignored** (severity 12, high) → `v0.11.0 domain model & preprocessing finalisation`. Enforce unseen-category handling with tests and docs.
- **DataFrame/dtype validation incomplete** (severity 9, high) → `v0.11.0 domain model & preprocessing finalisation`. Extend validators for categorical dtypes and add diagnostics.
- **Telemetry docs mismatch emitted fields** (severity 4, low) → `v0.11.0 domain model & preprocessing finalisation`. Align docs and telemetry payloads.

### ADR-010 – Optional Dependency Split

- **Core dependency list still heavy** (severity 20, critical) → `v0.10.2 plugin trust & packaging compliance`. Trim core dependencies and move extras-only libraries behind extras.
- **Evaluation extra incomplete** (severity 12, high) → `v0.10.2 plugin trust & packaging compliance`. Complete `[eval]` extra with required packages and lockfiles.
- **Visualization tests not auto-skipped without extras** (severity 12, high) → `v0.10.2 plugin trust & packaging compliance`. Mark viz tests with skip conditions tied to extras.
- **Evaluation environment lockfile missing** (severity 6, medium) → `v0.10.2 plugin trust & packaging compliance`. Publish lockfile/requirements for evaluation workflows.
- **Extras documentation inaccurate** (severity 6, medium) → `v0.10.2 plugin trust & packaging compliance`. Synchronise docs with extras definitions.
- **Contributor guidance on extras absent** (severity 4, low) → `v0.10.2 plugin trust & packaging compliance`. Update CONTRIBUTING with lean-core instructions.

### ADR-011 – Deprecation Policy

- **Central `deprecate()` helper missing** (severity 15, critical) → `v0.9.1 governance & observability hardening`. Implement helper and ensure all callsites adopt it.
- **Migration guide absent** (severity 15, critical) → `v0.9.1 governance & observability hardening`. Author the migration guide referenced in CHANGELOG.
- **Release plan lacks status table** (severity 12, high) → `v0.9.1 governance & observability hardening`. Add the structured table (this document update) and keep it current per release.
- **CI gates for deprecation policy missing** (severity 12, high) → `v0.9.1 governance & observability hardening`. Add automation enforcing the two-minor-release window and migration-note checks.

**Status (2025-11-19):**

- **Central `deprecate()` helper implemented:** Done. The helper is available at `src/calibrated_explanations/utils/deprecations.py` and has been adopted in representative callsites across the codebase. Remaining callsites identified as runtime warnings (non-deprecation) were intentionally left unchanged.
- **Migration guide authored:** Done. See `docs/migration/deprecations.md` for migration steps, example edits, and a deprecation status table.
- **Structured status table added to this plan:** Done (this section updated). The status table below summarises progress and next actions.
- **CI gates for deprecation policy:** In progress. Unit tests for the deprecation helper were added (`tests/unit/test_utils_deprecations.py`) and a docs smoke test was added. Full CI automation enforcing the two-minor-release window requires CI workflow updates and will be scheduled as a follow-up task.

**ADR-011 Status Table (deprecation items)**

| Item | Target | Status | Notes |
|---|---:|---|---|
| Central `deprecate()` helper | v0.9.1 | ✅ Done | Implemented at `src/.../utils/deprecations.py`; once-per-key semantics and env override `CE_DEPRECATIONS` supported.
| Migration guide (user-facing) | v0.9.1 | ✅ Done | `docs/migration/deprecations.md` provides migration steps and a matrix of deprecated symbols.
| Repository-wide sweep of callsites | v0.9.1 | ✅ Representative | Representative callsites converted; remaining `warnings.warn` messages are runtime notices not deprecations.
| Unit tests for helper | v0.9.1 | ✅ Done | `tests/unit/test_utils_deprecations.py` validates emission and error-mode behaviour.
| Docs smoke/link check | v0.9.1 | ✅ Added | `tests/doc/test_deprecations_doc_smoke.py` validates the migration doc is present and basic links parse.
| CI gating automation (two-minor-release enforcement) | v0.9.1 | ⏳ In progress | Requires CI workflow change; recommended follow-up PR to add workflow changes and governance hooks.

### ADR-012 – Documentation & Gallery Build Policy

- **Notebooks never rendered in docs CI** (severity 20, critical) → `v0.9.1 governance & observability hardening`. Enable notebook execution in docs CI.
- **Docs build ignores `[viz]`/`[notebooks]` extras** (severity 12, high) → `v0.9.1 governance & observability hardening`. Use project extras in workflows.
- **Example runtime ceiling unenforced** (severity 9, high) → `v0.9.1 governance & observability hardening`. Add timing checks for tutorials/examples.
- **Gallery tooling decision undocumented** (severity 4, low) → `v0.9.1 governance & observability hardening`. Record the chosen tooling in ADR updates.

### ADR-013 – Interval Calibrator Plugin Strategy

- **Runtime skips protocol validation for calibrators** (severity 20, critical) → `v0.10.2 plugin trust & packaging compliance`. Enforce protocol validation before activation.
- **FAST plugin returns non-protocol collections** (severity 15, critical) → `v0.10.2 plugin trust & packaging compliance`. Refactor FAST plugin outputs to protocol objects.
- **Interval context remains mutable** (severity 12, high) → `v0.10.2 plugin trust & packaging compliance`. Supply frozen contexts and document immutability. *Status 2025-11-04: Read-only requirement clarified in ADR-013 interval propagation section.*
- **Legacy default plugin rebuilds calibrators** (severity 9, high) → `v0.10.2 plugin trust & packaging compliance`. Return frozen instances per ADR guidance. *Status 2025-11-04: Frozen instance requirement clarified in ADR-013.*
- **CLI interval validation commands missing** (severity 4, low) → `v0.10.2 plugin trust & packaging compliance`. Add CLI commands for validation and document usage.

### ADR-014 – Visualization Plugin Architecture

- **Legacy fallback builder/renderer inert** (severity 15, critical) → `v0.10.1 schema & visualization contracts`. Restore fallback path to produce guaranteed plots.
- **Helper base classes missing** (severity 12, high) → `v0.10.1 schema & visualization contracts`. Implement shared base classes and lifecycle docs.
- **Metadata lacks `default_renderer`** (severity 12, high) → `v0.10.1 schema & visualization contracts`. Populate metadata and ensure registry respects defaults.
- **Renderer override resolution incomplete** (severity 12, high) → `v0.10.1 schema & visualization contracts`. Honour overrides/env vars when selecting renderers.
- **Dedicated `PlotPluginError` absent** (severity 6, medium) → `v0.10.1 schema & visualization contracts`. Add dedicated exception and use it in plugin errors.
- **Default renderer skips `validate_plotspec`** (severity 6, medium) → `v0.10.1 schema & visualization contracts`. Enforce validation before rendering.
- **CLI helpers not implemented** (severity 6, medium) → `v0.10.1 schema & visualization contracts`. Ship CLI utilities for plot plugin management.
- **Documentation for plot plugins lacking** (severity 4, low) → `v0.10.1 schema & visualization contracts`. Publish authoring guide updates.

### ADR-015 – Explanation Plugin Integration

- **In-tree FAST plugin missing** (severity 15, critical) → `v0.10.2 plugin trust & packaging compliance`. Provide default FAST plugin aligned with ADR.
- **Collection reconstruction bypassed** (severity 12, high) → `v0.10.2 plugin trust & packaging compliance`. Ensure collections rebuild with canonical metadata.
- **Trust enforcement during resolution lax** (severity 12, high) → `v0.10.2 plugin trust & packaging compliance`. Harden resolver trust checks.
- **Predict bridge omits interval invariants** (severity 12, high) → `v0.10.0 runtime boundary realignment`. Enforce interval invariants inside prediction bridge. *Status 2025-11-04: Explicit payload structures and invariant contracts documented in ADR-015 subsections 2a/2b.*
- **Environment variable names diverge** (severity 6, medium) → `v0.10.2 plugin trust & packaging compliance`. Align env var names and document them.
- **Helper handles expose mutable explainer** (severity 6, medium) → `v0.10.2 plugin trust & packaging compliance`. Provide immutable handles to plugins. *Status 2025-11-04: Plugin immutability requirement clarified in ADR-015.*

### ADR-016 – PlotSpec Separation and Schema

- **PlotSpec dataclass lacks required fields** (severity 15, critical) → `v0.10.1 schema & visualization contracts`. Extend dataclass with `kind`, `mode`, and `feature_order`.
- **Feature indices discarded during dict conversion** (severity 12, high) → `v0.10.1 schema & visualization contracts`. Preserve indices in conversion helpers.
- **Validator still enforces legacy envelope** (severity 12, high) → `v0.10.1 schema & visualization contracts`. Update validators to new schema and mandate usage.
- **Builders skip validation hooks** (severity 9, high) → `v0.10.1 schema & visualization contracts`. Require builders to call validation.
- **`save_behavior` metadata unimplemented** (severity 6, medium) → `v0.10.1 schema & visualization contracts`. Implement metadata field and docs.

### ADR-017 – Nomenclature Standardization

- **Double-underscore fields still mutated outside legacy** (severity 20, critical) → `v0.11.0 domain model & preprocessing finalisation`. Purge non-legacy double-underscore access and add lint enforcement.
- **Naming guardrails lack automated enforcement** (severity 16, critical) → `v0.9.1 governance & observability hardening`. Turn Ruff/pre-commit checks into blockers and document process.
- **Kitchen-sink `utils/helper.py` persists** (severity 9, high) → `v0.11.0 domain model & preprocessing finalisation`. Split helpers by topic and deprecate legacy names via ADR-011 gates.
- **Telemetry for lint drift missing** (severity 9, high) → `v0.11.0 domain model & preprocessing finalisation`. Capture lint status metrics in telemetry/governance dashboards.
- **Transitional shims remain first-class** (severity 6, medium) → `v0.11.0 domain model & preprocessing finalisation`. Confine shims to `legacy/` and stage removals per deprecation policy.

### ADR-018 – Documentation Standardisation

- **Wrapper public APIs lack numpydoc blocks** (severity 12, high) → `v0.9.1 governance & observability hardening`. Author full numpydoc content for wrapper APIs.
- **`IntervalRegressor.__init__` docstring outdated** (severity 8, medium) → `v0.9.1 governance & observability hardening`. Update docstring and add regression tests.
- **`IntervalRegressor.bins` setter undocumented** (severity 6, medium) → `v0.9.1 governance & observability hardening`. Document setter semantics.
- **Guard helpers missing summaries** (severity 4, low) → `v0.9.1 governance & observability hardening`. Add one-line summaries per ADR.
- **Nested combined-plot plugin classes undocumented** (severity 4, low) → `v0.10.1 schema & visualization contracts`. Document dynamically generated classes alongside plugin guide.

### ADR-019 – Test Coverage Standard

- **Coverage floor still enforced at 88%** (severity 20, critical) → `v0.9.1 governance & observability hardening`. Raise thresholds to 90% and enforce in CI.
- **Critical modules below 95% without gates** (severity 15, critical) → `v0.9.1 governance & observability hardening`. Add per-module gates for prediction, serialization, and registry modules.
- **Codecov patch gate optional** (severity 16, critical) → `v0.9.1 governance & observability hardening`. Make patch gate blocking for ADR-covered areas.
- **Public API packages under-tested** (severity 12, high) → `v0.9.1 governance & observability hardening`. Expand tests for gateway modules with coverage tracking.
- **Exemptions lack expiry metadata** (severity 6, medium) → `v0.9.1 governance & observability hardening`. Add expiry metadata to `.coveragerc` and release checklist.

### ADR-020 – Legacy User API Stability

- **Release checklist omits legacy API gate** (severity 12, high) → `v0.9.1 governance & observability hardening`. Add checklist gate verifying legacy compatibility.
- **Wrapper regression tests miss parity on key methods** (severity 12, high) → `v0.9.1 governance & observability hardening`. Add regression tests for `explain_factual`/`explore_alternatives`.
- **Contributor workflow ignores contract document** (severity 9, high) → `v0.9.1 governance & observability hardening`. Update CONTRIBUTING to reference the contract.
- **Notebook audit process undefined** (severity 6, medium) → `v0.9.1 governance & observability hardening`. Automate notebook audit scripts and integrate into release checklist.

### ADR-021 – Calibrated Interval Semantics

- **Interval invariants never enforced** (severity 20, critical) → `v0.10.0 runtime boundary realignment`. Enforce invariants in prediction bridges and serializers. *Status 2025-11-04: Invariant contract clarified uniformly across three levels (prediction, feature-weight, scenario) in ADR-021 subsections 4a-4b; enforcement remains for v0.10.0.*
- **FAST explanations drop probability cubes** (severity 12, high) → `v0.10.2 plugin trust & packaging compliance`. Extend FAST exports with probability metadata.
- **JSON export stores live callables** (severity 6, medium) → `v0.10.0 runtime boundary realignment`. Serialize immutable metadata instead of callables.

### ADR-022 – Documentation Information Architecture

- **Seven-section navigation not implemented** (severity 20, critical) → `v0.9.0 documentation realignment`. Complete navigation restructure per IA plan.
- **“Extending the library” lane missing** (severity 12, high) → `v0.9.0 documentation realignment`. Ship contributor/extension lane in navigation.
- **Telemetry concept page lacks substance** (severity 8, medium) → `v0.9.0 documentation realignment`. Flesh out telemetry concept content.

### ADR-023 – Matplotlib Coverage Exemption

- **Visualization tests never run in CI** (severity 20, critical) → `v0.9.1 governance & observability hardening`. Add viz-only CI job.
- **Pytest ignores block viz suite entirely** (severity 15, critical) → `v0.9.1 governance & observability hardening`. Remove ignores and mark tests appropriately.
- **Coverage threshold messaging inconsistent** (severity 12, high) → `v0.9.1 governance & observability hardening`. Align docs/tooling messaging with final threshold.

### ADR-024 – Legacy Plot Input Contracts

- **`_plot_global` ignores `show=False`** (severity 15, critical) → `v0.10.1 schema & visualization contracts`. Honour `show=False` across helpers.
- **`_plot_global` lacks save parameters** (severity 12, high) → `v0.10.1 schema & visualization contracts`. Implement save-path parameters.
- **Save-path concatenation drift undocumented** (severity 4, low) → `v0.10.1 schema & visualization contracts`. Update docs to match behaviour or revert to ADR contract.

### ADR-025 – Legacy Plot Rendering Semantics

- **Matplotlib guard allows silent skips** (severity 12, high) → `v0.10.1 schema & visualization contracts`. Fail loudly or document fallback semantics.
- **Regression axis not forced symmetric** (severity 12, high) → `v0.10.1 schema & visualization contracts`. Reinstate symmetric axis behaviour.
- **Interval backdrop disabled** (severity 9, high) → `v0.10.1 schema & visualization contracts`. Restore backdrop shading with coverage.
- **One-sided interval warning untested** (severity 6, medium) → `v0.10.1 schema & visualization contracts`. Add tests for warning emission.

### ADR-026 – Explanation Plugin Semantics

- **Predict bridge skips interval invariant checks** (severity 15, critical) → `v0.10.0 runtime boundary realignment`. Add invariant checks and tests. *Status 2025-11-04: Calibration contract and validation requirements clarified in ADR-026 subsections 2a/2b/3a/3b.*
- **Explanation context exposes mutable dicts** (severity 12, high) → `v0.10.2 plugin trust & packaging compliance`. Return frozen contexts to plugins. *Status 2025-11-04: Frozen context requirement clarified in ADR-026 subsection 1.*
- **Telemetry omits interval dependency hints** (severity 6, medium) → `v0.10.2 plugin trust & packaging compliance`. Extend telemetry payloads with dependency hints.
- **Mondrian bins left mutable in requests** (severity 4, low) → `v0.10.2 plugin trust & packaging compliance`. Freeze bins within request objects.

### ADR-027 – Documentation Standard (Audience Hubs)

- **PR template lacks parity review gate** (severity 15, critical) → `v0.9.1 governance & observability hardening`. Update template/checklist with parity review.
- **“Task API comparison” reference missing** (severity 9, high) → `v0.9.1 governance & observability hardening`. Restore comparison link in practitioner hub.
- **Researcher future-work ledger absent** (severity 6, medium) → `v0.9.1 governance & observability hardening`. Publish roadmap ledger tied to literature references.


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

   - 2025-11-06 – Consolidated the plugin story into a Plugins hub (`docs/plugins.md`), added a practitioner-focused "Use external plugins" guide (`docs/practitioner/advanced/use_plugins.md`), and surfaced the curated `external-plugins` extra in installation docs. Cross-linked the appendix index and ensured practitioner/contributor flows are consistent with ADR-027/ADR-006/ADR-014/ADR-026.
4. **Label telemetry and performance scaffolding as optional tooling.** Move telemetry schema/how-to material into contributor governance sections, ensure practitioner guides mention telemetry only for compliance scenarios, and audit navigation labels to avoid implying these extras are mandatory.【F:improvement_docs/documentation_information_architecture.md†L70-L113】
5. **Highlight research pedigree throughout.** Keep the existing research hub mentions in the Overview, practitioner quickstarts, and probabilistic regression concept pages; ensure they cross-link citing.md and key publications in the relevant sections without introducing new banner UI.【F:improvement_docs/documentation_review.md†L15-L34】
6. **Triangular alternatives plots everywhere alternatives appear.** Update explanation guides, PlotSpec docs, and runtime examples so `explore_alternatives` also introduces the triangular plot and its interpretation.
7. **Complete ADR-012 doc workflow enforcement.** Keep Sphinx `-W`, gallery build, and linkcheck mandatory; extend CI smoke tests to run the refreshed quickstarts and fail if optional extras are presented without labels.【F:improvement_docs/adrs/ADR-012-documentation-and-gallery-build-policy.md†L1-L80】
8. **Turn ADR-018 tooling fully blocking.** Finish pydocstyle batches E (`viz/`, `viz/plots.py`, `legacy/plotting.py`) and F (`serialization.py`, `core.py`), capture and commit the baseline failure report before flipping enforcement, add the documentation coverage badge, and extend linting to notebooks/examples so the Phase 3 automation backlog is complete.【F:improvement_docs/documentation_standardization_plan.md†L29-L41】【F:improvement_docs/pydocstyle_breakdown.md†L28-L33】
   - 2025-10-25 – Added nbqa-powered notebook linting and a 94% docstring
     coverage threshold to the lint workflow, making ADR-018's tooling fully
     blocking for documentation CI.
9. **Advance ADR-017 naming cleanup.** Prune deprecated shims scheduled for removal and ensure naming lint rules stay green on the release branch.【F:improvement_docs/nomenclature_standardization_plan.md†L25-L33】【F:improvement_docs/adrs/ADR-017-nomenclature-standardization.md†L28-L37】
10. **Sustain ADR-019 coverage uplift.** Audit waiver inventory, retire expired exemptions, raise non-critical modules toward the 90% floor, enable `--cov-fail-under=88` in CI, and execute the module-level remediation sprints for interval regressors, registry/CLI, plotting, and explanation caching per the dedicated gap plan.【F:improvement_docs/test_coverage_gap_plan.md†L5-L120】
11. **Scoped runtime polish for explain performance.** Deliver the opt-in calibrator cache, multiprocessing toggle, and vectorised perturbation handling per ADR-003/ADR-004 analysis so calibrated explanations stay responsive without compromising accuracy. Capture improvements and guidance for plugin authors.【F:improvement_docs/adrs/ADR-003-caching-key-and-eviction.md†L1-L64】【F:improvement_docs/adrs/ADR-004-parallel-backend-abstraction.md†L1-L64】【F:src/calibrated_explanations/core/calibrated_explainer.py†L1750-L2150】 See [Parallel Execution Improvement Plan – Phase 0](parallel_execution_improvement_plan.md#phase-0--foundations-week-01) and [Phase 1](parallel_execution_improvement_plan.md#phase-1--configuration-surface-week-24) for task breakdown and ownership.

      - 2025-11-04 – Implemented opt-in calibrator cache with LRU eviction, multiprocessing toggle via ParallelExecutor facade, and vectorized perturbation handling. Added performance guidance for plugin authors in docs/contributor/plugin-contract.md. Cache and parallel primitives integrated into explain pipeline without altering calibration semantics.
12. **Plugin CLI, discovery, and denylist parity (optional extras).** Extend trust toggles and entry-point discovery to interval/plot plugins, add the `CE_DENY_PLUGIN` registry control highlighted in the OSS scope review, and ship the whole surface as opt-in so calibrated explanations remain usable without telemetry/CLI adoption.【F:improvement_docs/OSS_CE_scope_and_gaps.md†L68-L110】
13. **External plugin distribution path.** Document and test an aggregated installation extra (e.g., `pip install calibrated-explanations[external-plugins]`) that installs all supported external plugins, outline curation criteria, and add placeholders in docs and README for community plugin listings.

      - 2025-10-25 – Added a packaging regression test that inspects the
         `external-plugins` extra metadata to guarantee the curated bundle stays
         opt-in with the expected dependency pins.
14. **Explanation export convenience.** Provide `to_json()`/`from_json()` helpers on explanation collections that wrap schema v1 utilities and document them as optional aids for integration teams.
15. **Scope streaming-friendly explanation delivery.** Prototype generator or chunked export paths (or record a formal deferral) so memory-sensitive users know how large batches will be handled, capturing the outcome directly in the OSS scope inventory.【F:improvement_docs/OSS_CE_scope_and_gaps.md†L86-L118】

Release gate: Audience landing pages published with calibrated explanations/probabilistic regression foregrounded, research callouts present on all entry points, telemetry/performance extras labelled optional, docs CI (including quickstart smoke tests, notebook lint, and doc coverage badge) green, ADR-017/018/019 gates enforced, runtime performance enhancements landed without altering calibration outputs, plugin denylist control shipped, streaming plan recorded, and optional plugin extras (CLI/discovery/export) documented as add-ons.

### v0.9.1 (governance & observability hardening)

1. Implement ADR-011 policy mechanics—add the central deprecation helper, author the long-promised migration guide, and publish the structured status table with CI enforcement of the two-release window.【F:improvement_docs/ADR-gap-analysis.md†L138-L142】
2. Bring docs CI into compliance with ADR-012 by executing notebooks during builds, installing official extras, timing tutorials, and documenting the chosen gallery tooling so drift is detected early.【F:improvement_docs/ADR-gap-analysis.md†L145-L150】
3. Finish ADR-018 obligations by documenting wrapper APIs, interval calibrator signatures, and guard helpers to the mandated numpydoc standard.【F:improvement_docs/ADR-gap-analysis.md†L210-L214】
4. Elevate coverage governance to the ADR-019 bar—raise thresholds to ≥90%, add per-module gates for prediction/serialization/registry paths, make the Codecov patch gate blocking, and track expiry metadata for waivers.【F:improvement_docs/ADR-gap-analysis.md†L220-L224】
5. Reinforce ADR-020 legacy-API commitments with release checklist gates, regression tests for `explain_factual`/`explore_alternatives`, CONTRIBUTING guidance, and a scripted notebook audit workflow.【F:improvement_docs/ADR-gap-analysis.md†L230-L233】
6. Restore visualization safety valves per ADR-023 by running the viz suite in CI, removing ignores, and aligning coverage messaging with the final thresholds.【F:improvement_docs/ADR-gap-analysis.md†L255-L257】
7. Update governance collateral and hubs to satisfy ADR-027—embed the parity-review checklist in PR templates, reinstate the task API comparison, and publish the researcher future-work ledger.【F:improvement_docs/ADR-gap-analysis.md†L289-L291】
8. Implement ADR-004 v0.9.1 scoped deliverable — ParallelFacade: create a conservative facade that centralizes executor selection heuristics, exposes a minimal config surface (min_instances_for_parallel, min_features_for_parallel, task_size_hint_bytes), honors `CE_PARALLEL` overrides, emits compact decision telemetry (decision, reason, n_instances, n_features, bytes_hint, platform, executor_type), and includes unit tests plus a micro-benchmark harness. This is intentionally small and designed to collect field evidence before any full `ParallelExecutor` rollout in v0.10. 【F:improvement_docs/adrs/ADR-004-parallel-backend-abstraction.md†L1-L40】【F:improvement_docs/ADR-gap-analysis.md†L60-L72】

Release gate: Deprecation dashboard live, docs CI runs with notebook execution, coverage/waiver gating enforced at ≥90%, legacy API and parity checklists signed, and visualization tests passing on the release branch.【F:improvement_docs/ADR-gap-analysis.md†L138-L257】【F:improvement_docs/ADR-gap-analysis.md†L289-L291】

### v0.10.0 (runtime boundary realignment)

1. Restructure packages to honour ADR-001—split calibration into its own package, eliminate cross-sibling imports, and formalise sanctioned namespaces with ADR addenda where necessary.【F:improvement_docs/ADR-gap-analysis.md†L33-L38】
2. Deliver ADR-002 validation parity by replacing legacy exceptions with taxonomy classes, implementing shared validators, parameter guards, and consistent fit-state handling.【F:improvement_docs/ADR-gap-analysis.md†L44-L48】
3. Complete ADR-003 caching deliverables: add invalidation/flush hooks, cache the mandated artefacts, emit telemetry, and align the backend with the cachetools+pympler stack or update the ADR rationale.【F:improvement_docs/ADR-gap-analysis.md†L54-L58】
4. Implement ADR-004’s parallel execution backlog—auto strategy heuristics, telemetry with timings/utilisation, context management and cancellation, configuration surfaces, resource guardrails, fallback warnings, and automated benchmarking.【F:improvement_docs/ADR-gap-analysis.md†L64-L71】 Track deliverables in [Parallel Execution Improvement Plan – Phases 2–4](parallel_execution_improvement_plan.md#phase-2--executor--plugin-refactor-week-58).
5. Enforce interval safety across bridges and exports to resolve ADR-021 and the ADR-015 predict-bridge gap, ensuring invariants, probability cubes, and serialization policies are honoured.【F:improvement_docs/ADR-gap-analysis.md†L239-L241】【F:improvement_docs/ADR-gap-analysis.md†L179-L182】
6. Align runtime plugin semantics with ADR-026 by adding invariant checks, hardening contexts, and extending telemetry payloads.【F:improvement_docs/ADR-gap-analysis.md†L280-L282】
7. Remove deprecated backward-compatibility alias `_is_thresholded()` from `CalibratedExplanations` class (superseded by `_is_probabilistic_regression()` in v0.9.0). Update any remaining external code or documentation that may reference the old method name. This completes the terminology standardization cycle from ADR-021.【F:improvement_docs/adrs/ADR-021-calibrated-interval-semantics.md†L119-L159】【F:TERMINOLOGY_ANALYSIS_THRESHOLDED_VS_PROBABILISTIC_REGRESSION.md†L1-L720】

Release gate: Package boundaries, validation/caching/parallel tests, interval invariants, terminology cleanup, and updated ADR status notes all green with telemetry dashboards verifying the new signals.【F:improvement_docs/ADR-gap-analysis.md†L33-L282】

### v0.10.1 (schema & visualization contracts)

1. Implement the ADR-005 envelope—introduce the structured payload, discriminant registry, provenance metadata, mandatory schema versioning, and refreshed fixtures/docs.【F:improvement_docs/ADR-gap-analysis.md†L77-L82】
2. Finish ADR-007 and ADR-016 schema work: enhance PlotSpec dataclasses, registries, validation coverage, JSON round-trips, and headless export paths.【F:improvement_docs/ADR-gap-analysis.md†L97-L102】【F:improvement_docs/ADR-gap-analysis.md†L190-L194】
3. Restore ADR-014 visualization plugin architecture with working fallback builders, helper base classes, metadata/default renderers, override handling, validation, CLI utilities, and documentation.【F:improvement_docs/ADR-gap-analysis.md†L166-L173】
4. Realign legacy plotting helpers with ADR-024/ADR-025 by honouring `show=False`, implementing save parameters, reinstating symmetric axes and interval backdrops, enforcing Matplotlib guards, and adding missing coverage.【F:improvement_docs/ADR-gap-analysis.md†L263-L274】
5. Document dynamically generated visualization classes to close the remaining ADR-018 docstring gap tied to plugin guides.【F:improvement_docs/ADR-gap-analysis.md†L214】

Release gate: Envelope round-trips verified, PlotSpec/visualization plugin registries fully validated, legacy helpers behaving per ADR contracts, and docs updated with new schema references.【F:improvement_docs/ADR-gap-analysis.md†L77-L274】

### v0.10.2 (plugin trust & packaging compliance)

1. Enforce ADR-006 trust controls—manual approval for third-party trust flags, deny-list enforcement, diagnostics for skipped plugins, and documented sandbox warnings.【F:improvement_docs/ADR-gap-analysis.md†L88-L91】
2. Close ADR-013 protocol gaps by validating calibrators, returning protocol-compliant FAST outputs, freezing contexts, providing CLI diagnostics, and returning frozen defaults.【F:improvement_docs/ADR-gap-analysis.md†L156-L160】
3. Finish ADR-015 integration work: ship an in-tree FAST plugin, rebuild explanation collections with canonical metadata, tighten trust enforcement, align environment variables, and provide immutable plugin handles.【F:improvement_docs/ADR-gap-analysis.md†L179-L184】
4. Deliver ADR-010 optional-dependency splits by trimming core dependencies, completing extras/lockfiles, auto-skipping viz tests without extras, updating docs, and extending contributor guidance.【F:improvement_docs/ADR-gap-analysis.md†L127-L132】
5. Extend ADR-021/ADR-026 telemetry by surfacing FAST probability cubes, interval dependency hints, and frozen bin metadata in runtime payloads.【F:improvement_docs/ADR-gap-analysis.md†L240-L241】【F:improvement_docs/ADR-gap-analysis.md†L280-L283】

Release gate: Plugin registries enforce trust and protocol policies, extras install cleanly with documentation parity, runtime telemetry captures interval metadata, and FAST/CLI flows succeed end-to-end.【F:improvement_docs/ADR-gap-analysis.md†L88-L283】

### v0.11.0 (domain model & preprocessing finalisation)

1. Make the ADR-008 domain model authoritative—run runtime flows on domain objects, fix legacy round-trips, add calibration/model metadata, publish golden fixtures, and harden `_safe_pick`.【F:improvement_docs/ADR-gap-analysis.md†L108-L112】
2. Complete ADR-009 preprocessing automation with built-in encoding, unseen-category enforcement, dtype diagnostics, and aligned telemetry/docs.【F:improvement_docs/ADR-gap-analysis.md†L118-L121】
3. Finish ADR-017 nomenclature clean-up by eliminating double-underscore mutations, splitting utilities, reporting lint telemetry, and confining transitional shims to `legacy/`.【F:improvement_docs/ADR-gap-analysis.md†L200-L204】
4. Extend governance dashboards to surface lint status alongside preprocessing/domain-model telemetry, ensuring ongoing monitoring after v1.0.0.【F:improvement_docs/ADR-gap-analysis.md†L203-L204】

Release gate: Domain/preprocessing pipelines operate on ADR-compliant models with telemetry coverage, naming lint metrics published, and no outstanding ADR exceptions ahead of v1.0.0-rc.【F:improvement_docs/ADR-gap-analysis.md†L108-L204】


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
   utilisation metrics for release sign-off.【F:improvement_docs/adrs/ADR-003-caching-key-and-eviction.md†L28-L64】【F:improvement_docs/adrs/ADR-004-parallel-backend-abstraction.md†L25-L64】 See [Parallel Execution Improvement Plan – Phase 5](parallel_execution_improvement_plan.md#phase-5--rollout--documentation-week-15-16) for rollout and documentation activities.
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
10. Audit the ADR gap closure roadmap to confirm every gap is either implemented or superseded with an updated ADR decision before promoting the RC branch, recording outcomes in the status log.【F:improvement_docs/RELEASE_PLAN_v1.md†L74-L274】

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
