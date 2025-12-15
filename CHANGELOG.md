<!-- markdownlint-disable-file -->
# Changelog

## [Unreleased]

[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.9.1...main)

### Added

- **Parallel Execution Framework**: Introduced a robust parallel execution framework (ADR-004) with workload-aware auto-strategy, telemetry, and resource guardrails.
  - Added `ParallelConfig` with options for chunking, fallback strategies, and task size hints.
  - Implemented automatic fallback to sequential execution on failure.
  - Added support for process-based parallelism on Windows.
- **Caching Strategy**: Implemented a comprehensive caching strategy (ADR-003) using `ExplanationCacheFacade` to improve performance.
  - Added `pympler` dependency for precise memory profiling.
  - Added telemetry for cache hits, misses, evictions, and resets.
- **Validation & Error Handling**: Standardized validation and error handling across the library (ADR-002).
  - Introduced a unified exception taxonomy (`ValidationError`, `DataShapeError`, `NotFittedError`, etc.) with structured error details.
  - Enhanced `validate_inputs` and other validation helpers for consistent API contracts.
  - Added `explain_exception` helper for human-readable error messages.
- **Interval Safety**: Enforced robust `low <= predict <= high` invariants for all plugin predictions and serialization (ADR-021).
- **Condition Source**: Added support for `condition_source` configuration to select conditioning data sources.

### Changed

- **Package Structure**: Restructured the internal package layout (ADR-001) to enforce strict boundaries and reduce circular dependencies.
  - Moved core logic into dedicated sub-packages (`core`, `calibration`, `explanations`, `cache`, `parallel`, `schema`, `plugins`, `viz`, `utils`).
  - Deprecated unsanctioned symbols and locked the public API to sanctioned entry points.
- **Parallel Runtime**: Auto parallel backend now prefers `joblib` on all platforms, with process-based fallback.
- **Terminology**: Standardized on "probabilistic regression" as the canonical user-facing term for regression with threshold-based probability predictions (ADR-021).
- **Explanation Plugin Semantics**: Internalized `CalibratedExplainer.explain` to `_explain` and enforced immutable contexts (ADR-026).
- **Documentation**: Comprehensive API reference updates with `autoclass` directives.

### Fixed

- **Legacy Exceptions**: Replaced numerous legacy `ValueError` and `RuntimeError` raises with specific, informative exceptions.
- **Windows Compatibility**: Fixed pickling issues to enable process-based parallelism on Windows.
- **Serialization**: Fixed JSON serialization for explanation collections containing live objects.

### Removed

- **Feature Parallelism**: Deprecated `FeatureParallel` execution strategy in favor of more efficient instance-based parallelism.

## [v0.9.1](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.9.1) - 2025-11-27

[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.9.0...v0.9.1)

### Release Plan Alignment (v0.9.1)

- **Governance & observability hardening (ADR-017/018/019/020).** Added a per-module coverage gate script with suffix-aware path matching so ADR-019 thresholds are enforced in CI, tightened the `test` workflow to keep the core-only matrix from uploading coverage while still running the full gate on the main job, and switched editable installs for local test runs to reduce dependency skew. An accompanying notebook API auditor tracks use of the documented legacy surface (ADR-020) to keep examples in lockstep with the contract.【F:scripts/check_coverage_gates.py†L1-L102】【F:.github/workflows/test.yml†L33-L99】【F:scripts/audit_notebook_api.py†L1-L144】
- **Runtime safety and regression coverage:** Added focused CalibratedExplainer regression tests that assert out-of-bag errors propagate, categorical features infer from label mappings, plugin override coercion handles callables/objects, and plugin manager requirements raise clear errors, strengthening the runtime guardrails promised for v0.9.1.【F:tests/unit/core/test_calibrated_explainer_runtime_helpers.py†L1-L220】
- **Plugin fallback and serialization reliability:** Covered deprecated CalibratedExplainer surfaces with deletion-coupling notes and new tests for plugin fallback chains, interval regressors, and JSON round-trips so the v0.9.1 release ships with explicit removal guidance and resilient persistence paths.【F:src/calibrated_explanations/core/calibrated_explainer.py†L1125-L1914】【F:tests/unit/core/test_calibrated_explainer_additional.py†L1-L200】【F:tests/unit/core/test_calibrated_explainer_plugin_fallbacks.py†L1-L200】【F:tests/unit/core/test_interval_regressor.py†L1-L260】【F:tests/unit/test_serialization.py†L1-L220】
- **Documentation standardisation.** Expanded the practitioner hub with an explicit wrapper-vs-direct API comparison so parity requirements from the governance plan remain visible, and published a researcher "future work" ledger to keep calibration and interval-regression research directions discoverable alongside the quickstarts.【F:docs/practitioner/index.md†L1-L34】【F:docs/practitioner/task_api_comparison.md†L1-L92】【F:docs/researcher/future_work.md†L1-L109】
- **Interval regression hardening.** Improved interval regression storage and probabilistic threshold handling so calibration buffers resize safely, aligned with the practitioner/researcher documentation updates that now cross-link interval workflows with the existing probabilistic regression guidance.【F:src/calibrated_explanations/core/calibration/interval_regressor.py†L1-L200】【F:docs/practitioner/index.md†L1-L20】【F:docs/researcher/future_work.md†L1-L32】

### Additional changes

#### Narrative Generation API

- **Introduced `to_narrative()` method for generating human-readable explanations** (contributed by @appraneethreddy)
  - Added `to_narrative()` convenience method to `CalibratedExplanations` class for converting explanations into narrative text
  - Created `NarrativeGenerator` class (`core/narrative_generator.py`) to handle template-based narrative generation with support for multiple expertise levels
  - Implemented `NarrativePlotPlugin` (`viz/narrative_plugin.py`) integrating narrative generation with the plot system
  - Supports multiple expertise levels: `"beginner"`, `"intermediate"`, `"advanced"` for tailored explanations
  - Multiple output formats: DataFrame, plain text, HTML table, and dictionary structures
  - Template-based customization with automatic fallback to default `explain_template.yaml`
  - Problem type detection (classification vs regression) with threshold-aware narrative generation
  - Proper handling of prediction intervals and uncertainty for calibrated predictions
  - Comprehensive documentation in `docs/foundations/how-to/to_narrative.md` with usage examples

#### Execution Strategy Plugins

- **Introduced execution strategy wrapper plugins for user-configurable parallelism strategies**
  - Created 6 new wrapper explanation plugins bridging explanation layer and execution layer:
    - `core.explanation.factual.sequential` - Single-threaded sequential processing
    - `core.explanation.factual.feature_parallel` - Parallel processing across features
    - `core.explanation.factual.instance_parallel` - Parallel processing across instances
    - `core.explanation.alternative.sequential` - Alternative mode sequential processing
    - `core.explanation.alternative.feature_parallel` - Alternative mode feature parallelism
    - `core.explanation.alternative.instance_parallel` - Alternative mode instance parallelism
  - Implemented `_ExecutionExplanationPluginBase` class for consistent bridging behavior
  - Each strategy declares automatic fallback chain for graceful degradation (instance_parallel → feature_parallel → sequential → legacy)
  - Users can now select execution strategies via configuration:
    - Keyword argument: `explainer.explain_factual(x, explanation_plugin="core.explanation.factual.feature_parallel")`
    - Environment variable: `CE_EXPLANATION_PLUGIN_FACTUAL="core.explanation.factual.feature_parallel"`
    - pyproject.toml: `[tool.calibrated_explanations.explanations]`
  - All execution plugins registered as trusted and automatically available
  - Maintains full backward compatibility - legacy plugins remain as ultimate fallback
  - Updated ADR-015 with execution strategy plugin documentation
  - Added comprehensive unit tests (25 test cases) covering registration, metadata, modes, and fallback chains

#### CalibratedExplainer Streamlining

- **Extracted core discretization and rule boundary computation to explain subpackage** to consolidate explanation logic and eliminate circular dependencies
  - Created `explain._computation.discretize()` function: pure function extracting discretization logic from `CalibratedExplainer._discretize()`
  - Created `explain._computation.rule_boundaries()` function: pure function extracting rule boundary extraction logic from `CalibratedExplainer.rule_boundaries()`
  - Updated `CalibratedExplainer._discretize()` to delegate to extracted function (maintains backward compatibility)
  - Updated `CalibratedExplainer.rule_boundaries()` to delegate to extracted function (maintains backward compatibility)
  - Resolved circular dependency chain: `prediction_helpers.explain_predict_step()` now calls explainer methods that delegate to extracted functions instead of direct logic
  - Both extracted functions exported in `explain._computation.__all__` for public use
  - No breaking changes: all public and private APIs remain unchanged

#### Terminology Standardization

- **Standardized on "probabilistic regression" as the canonical user-facing term** for regression with threshold-based probability predictions. "Thresholded regression" is used in technical architecture documents (ADRs, design notes) to describe the implementation mechanism (CPS-based threshold calibration).
  - Renamed internal method `_is_thresholded()` → `_is_probabilistic_regression()` in `CalibratedExplanations` class for consistency
  - **Backward compatibility preserved:** `_is_thresholded()` kept as deprecated alias pointing to `_is_probabilistic_regression()` (will be removed in v0.10.0)
  - Public method `is_thresholded()` on `Explanation` objects **unchanged** for full backward compatibility
  - Updated ADR-021 with a "Terminology" section explaining the equivalence and usage guidance
  - Added cross-reference in ADR-013 linking to ADR-021 terminology clarification
  - Updated all docstrings to prefer "probabilistic regression" terminology
  - Created migration guide: `docs/migration/v0.9-to-v0.10-terminology.md`
  - See [Terminology Analysis](TERMINOLOGY_ANALYSIS_THRESHOLDED_VS_PROBABILISTIC_REGRESSION.md) for full context and rationale

#### Additional minor changes

- **CalibratedExplainer and plugin guards.** Refined plugin manager delegation and prediction bridge imports to avoid circular
  dependencies, and extended explanation plugin validation to cope with deferred imports while preserving ADR-015 invariants on
  batch payloads.【F:src/calibrated_explanations/core/calibrated_explainer.py†L232-L320】【F:src/calibrated_explanations/plugins/explanations.py†L1-L189】
- **Runtime helper coverage.** Added extensive CalibratedExplainer runtime helper tests covering exception propagation, categorical
  defaults, plugin override coercion, and telemetry delegation to improve reliability ahead of v0.9.1 gating.【F:tests/unit/core/test_calibrated_explainer_runtime_helpers.py†L1-L120】

## [v0.9.0](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.9.0) - 2025-11-07

[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.8.0...v0.9.0)

## Highlights

- **ADR alignment audit: Factual vs Alternative explanations.** Conducted comprehensive
  review of all ADRs (ADR-005, ADR-008, ADR-013, ADR-015, ADR-021, ADR-026) to ensure
  full alignment with paper-consistent semantics for factual and alternative explanations.
  Added formal rule definitions to ADR-008, interval propagation contracts to ADR-013,
  factual/alternative payload requirements to ADR-015, feature-level interval semantics
  to ADR-021, and factual/alternative calibration contracts and validation criteria to
  ADR-026. All ADRs now explicitly document that alternative explanations include a
  reference calibrated prediction with uncertainty interval, feature weights in factual
  rules have calibrated intervals, and all `[low, high]` pairs must satisfy the inclusive
  bounds invariant. This audit ensures consistency across all explanation generation paths
  and provides explicit guidance for plugin developers.【F:docs/improvement/adrs/ADR-008-explanation-domain-model-and-compat.md†L45-L75】【F:docs/improvement/adrs/ADR-013-interval-calibrator-plugin-strategy.md†L80-L110】【F:docs/improvement/adrs/ADR-015-explanation-plugin.md†L151-L210】【F:docs/improvement/adrs/ADR-021-calibrated-interval-semantics.md†L120-L150】【F:docs/improvement/adrs/ADR-026-explanation-plugin-semantics.md†L84-L165】

- **Plugin-based explain architecture.** All explain logic (sequential,
  feature-parallel, instance-parallel) now lives in dedicated plugins under
  `core/explain/`, replacing the monolithic branching in `CalibratedExplainer.explain`.
  This clean separation improves maintainability, testability, and sets the foundation
  for future distributed execution strategies.
- **Faster calibrated explanations.** Explore the latest latency benchmarks and
  optimization notes in the [explain performance evaluation
  report](evaluation/explain_performance.md).
- **Revamped documentation experience.** Dive into the reorganised guides,
  role-based hubs, and refreshed quickstarts starting at the
  [documentation homepage](docs/index.md).
- **Plugin governance guardrails.** Optional extras, telemetry governance, and
  plugin trust workflows now ship with automated enforcement across docs and
  CI.
- **Expanded export tooling.** Persist and reload explanation collections via
  the new `CalibratedExplanations.to_json`/`.from_json` helpers and refreshed
  how-to guides.

### Release plan alignment

- **Explanation schema v1 and ADR-005/008 compliance:** Updated explanation JSON schema v1 to include
  `explanation_type` field distinguishing factual and alternative explanations, aligned ADR-005 with paper-compliant semantics from ADR-008, and ensured all domain models, serialization, and adapters preserve the calibrated prediction baseline for both explanation types. This establishes stable round-trip serialization for instance-based explanations as defined in the CE papers.【F:docs/schema_v1.md†L1-L50】【F:docs/improvement/adrs/ADR-005-explanation-json-schema-versioning.md†L1-L80】【F:docs/improvement/adrs/ADR-008-explanation-domain-model-and-compat.md†L1-L60】【F:src/calibrated_explanations/schemas/explanation_schema_v1.json†L1-L40】
- **Explain executor decomposition (ADR-004 compliance):** Moved all explain execution
  strategies into a plugin system (`src/calibrated_explanations/core/explain/`)
  with three implementations: `SequentialExplainExecutor` (single-threaded fallback),
  `FeatureParallelExplainExecutor` (executor-backed feature distribution), and
  `InstanceParallelExplainExecutor` (instance-level chunking). `CalibratedExplainer.explain`
  is now a thin 13-line delegator that selects and invokes the appropriate plugin
  based on executor configuration. All legacy equivalence tests and instance-parallel
  tests pass, confirming behavioral parity with the original implementation.
  【F:src/calibrated_explanations/core/explain/__init__.py†L1-L175】
  【F:src/calibrated_explanations/core/calibrated_explainer.py†L2299-L2351】
- **Calibrated-explanations-first messaging (Tasks 1 & 5):** Rewrote the README,
  overview landing page, and quickstarts so calibrated factual and alternative
  flows lead every entry point, with telemetry, PlotSpec, and plugin extras
  demoted into consistent "Optional extras" callouts and cross-linked research
  hubs.【F:README.md†L1-L86】【F:docs/index.md†L1-L44】【F:docs/_shared/optional_extras_template.md†L1-L24】【F:docs/_shared/backed_by_research.md†L1-L20】
- **Audience-specific hubs & triangular alternatives (Tasks 2 & 6):** Added
  practitioner, researcher, and contributor landing pages that pair probabilistic
  regression guidance with triangular alternative plot coaching so every
  `explore_alternatives` mention narrates the calibrated decision boundary
  story.【F:docs/practitioner/index.md†L1-L54】【F:docs/researcher/index.md†L1-L54】【F:docs/_shared/alternatives_triangular.md†L1-L14】【F:docs/get-started/quickstart_classification.md†L1-L84】
- **Plugin extensibility & external distribution (Tasks 3, 12 & 13):** Reframed
  `docs/plugins.md` around a calibration-preserving "hello plugin" example,
  surfaced the `external_plugins/` namespace and curated installation extra, and
  exposed `CE_DENY_PLUGIN` governance toggles across docs and CLI output so the
  optional plugin lane remains discoverable yet clearly opt-in.【F:docs/plugins.md†L1-L200】【F:external_plugins/__init__.py†L1-L23】【F:README.md†L640-L686】【F:src/calibrated_explanations/plugins/cli.py†L74-L123】
- **Research-forward storytelling (Task 5):** Maintained research hub call-outs
  across README, overview, and concept guides so every onboarding path links to
  `docs/research/` and `citing.md`, reinforcing the calibration pedigree.【F:README.md†L22-L60】【F:docs/overview/index.md†L1-L62】【F:docs/citing.md†L1-L140】
- **ADR-018/017/019 enforcement (Tasks 8–10):** Elevated docstring coverage and
  notebook lint to blocking status, retired transitional core/plot shims (removing
  `legacy/_interval_regressor.py`, `legacy/_venn_abers.py`, and `legacy/_plots*.py`),
  and tightened coverage thresholds to 88% alongside Codecov patch gates that focus
  on runtime and calibration modules.【F:.github/workflows/lint.yml†L38-L86】【F:src/calibrated_explanations/core/_legacy_explain.py†L1-L110】【F:pytest.ini†L1-L8】【F:codecov.yml†L1-L32】【F:src/calibrated_explanations/legacy/__init__.py†L1-L6】
- **Runtime performance polish (Task 11):** Implemented opt-in calibrator cache with LRU eviction, multiprocessing toggle via ParallelExecutor facade, and vectorized perturbation handling. Added performance guidance for plugin authors in docs/contributor/plugin-contract.md. Cache and parallel primitives integrated into explain pipeline without altering calibration semantics.【F:src/calibrated_explanations/perf/__init__.py†L1-L52】【F:src/calibrated_explanations/perf/cache.py†L1-L120】【F:src/calibrated_explanations/core/calibrated_explainer.py†L199-L377】
- **Documentation-first plugin governance (Task 12):** Expanded CLI and
  registry tests to surface denied identifiers, audit trusted plugins, and keep
  the governance narrative inline with the release checklist.【F:tests/plugins/test_cli.py†L74-L152】【F:docs/foundations/governance/release_checklist.md†L1-L92】【F:src/calibrated_explanations/plugins/registry.py†L84-L154】
- **External plugin bundle verification (Task 13):** Shipped packaging tests and
  documentation for the `external-plugins` extra so the curated FAST bundle
  stays optional yet discoverable.【F:tests/plugins/test_external_plugins_extra.py†L1-L90】【F:external_plugins/fast_explanations/__init__.py†L1-L92】
- **Explanation export helpers (Task 14):** Added `CalibratedExplanations.to_json`
  and `.from_json` wrappers and refreshed the export how-to so integration teams
  can persist schema v1 collections with calibrated metadata intact.【F:src/calibrated_explanations/explanations/explanations.py†L180-L247】【F:docs/how-to/export_explanations.md†L1-L86】
- **Streaming-friendly delivery status (Task 15):** Recorded the deferral and
  interim batching guidance in the OSS scope inventory, closing the release gate
  while signalling follow-up expectations.【F:docs/improvement/OSS_CE_scope_and_gaps.md†L1-L18】

### Runtime

- Accelerated categorical perturbations, discretisation, and prediction batching
  for both factual and alternative explanations, backed by legacy parity helpers
  preserved in `core/_legacy_explain.py` for regression testing.【F:src/calibrated_explanations/core/calibrated_explainer.py†L880-L1120】【F:src/calibrated_explanations/core/_legacy_explain.py†L1-L140】
- Introduced calibration summary caching and telemetry-aware performance
  factories so heavy workloads can reuse intermediate results without breaking
  calibration guarantees.【F:src/calibrated_explanations/core/calibrated_explainer.py†L356-L416】【F:src/calibrated_explanations/perf/cache.py†L121-L320】

### Documentation

- Published practitioner/researcher/contributor hubs, refreshed quickstarts with
  probabilistic regression walkthroughs, and rewrote plugins governance material
  to foreground calibrated-explanations-first messaging while labelling extras
  as optional.【F:docs/get-started/quickstart_regression.md†L1-L94】【F:docs/research/index.md†L1-L80】【F:docs/plugins.md†L200-L420】
- Consolidated plugin documentation into a single Plugins hub (`docs/plugins.md`),
  introduced a practitioner guide for using external plugins
  (`docs/practitioner/advanced/use_plugins.md`), and surfaced the curated
  `external-plugins` install extra in installation docs. Cross-links to the
  external plugin index were added to ensure a coherent, optional plugin story
  aligned with ADR-027 and the plugin ADRs (ADR-006/014/026).【F:docs/plugins.md†L1-L80】【F:docs/practitioner/advanced/use_plugins.md†L1-L200】【F:docs/get-started/installation.md†L1-L120】【F:docs/appendices/external_plugins.md†L1-L160】
- Added runtime performance tuning, governance checklists, and optional telemetry
  guides so compliance and SRE flows stay audience scoped.【F:docs/how-to/tune_runtime_performance.md†L1-L140】【F:docs/governance/release_checklist.md†L40-L92】

### CI & QA

- Hardened docs CI with the shared fragment checker, ensured Sphinx `-W`,
  navigation smoke tests, and linkcheck remain blocking, and elevated docstring
  coverage thresholds to 94% via lint automation.【F:.github/workflows/docs.yml†L19-L40】【F:.github/workflows/lint.yml†L38-L86】
- Raised global coverage minimums to 88% and tightened Codecov calibration
  patch targets so ADR-019 enforcement captures runtime changes.【F:pytest.ini†L1-L8】【F:codecov.yml†L1-L32】

### Tests & Tooling

- Expanded plugin registry, CLI, and builtin adapter suites, added dedicated
  legacy plotting regression tests, and introduced performance benchmarks and
  telemetry hooks to monitor the new runtime toggles.【F:tests/plugins/test_builtins_module.py†L1-L180】【F:tests/legacy/test_plotting.py†L1-L200】【F:evaluation/scripts/compare_explain_performance.py†L1-L200】
- Refreshed reports documenting docstring coverage baselines and lint outputs to
  back ADR-018's blocking rollout.【F:reports/docstring_coverage_20251025.txt†L1-L32】【F:reports/pydocstyle-baseline.txt†L1-L120】



## [v0.8.0](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.8.0) - 2025-10-24

[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.7.0...v0.8.0)

### Added

- ADR-023: Exempted `src/calibrated_explanations/viz/matplotlib_adapter.py` from coverage reporting due to matplotlib 3.8.4 lazy loading conflicts with pytest-cov instrumentation timing. All viz tests continue to run and validate functionality.
- Telemetry payloads now propagate across preprocessing, calibration, and explanation batches, with new helpers like `build_rules_payload()`/`to_telemetry()` enabling JSON-ready exports of uncertainty intervals and rule tables. Export flows are validated by integration tests that enforce schema coverage for percentile and thresholded runs.
- Plugin registry trust metadata enforces optional SHA256 checksum validation during registration and entry-point loading, expanding discovery helpers and surfacing provenance for third-party plugins.
- Extended `ce.plugins` CLI trust/untrust coverage to include interval calibrators and plot components (builders/renderers), with list/show support via `--kind` selectors.


### Changed

- Promoted PlotSpec rendering to the canonical pipeline by introducing the
  snake_case `calibrated_explanations.plotting` module, moving legacy
  Matplotlib helpers into `legacy/plotting.py`, and keeping `_plots*` only as
  warning shims to satisfy ADR-017 phase-two renames. Runtime and tests now
  import from the new modules, and telemetry records the PlotSpec defaults for
  auditability.
- Raised pytest's coverage floor to 85% and enabled Codecov patch gating on calibration/runtime modules, keeping CI focused on uncertainty-critical paths.


### Fixed

- Resolved pytest test suite failures caused by matplotlib lazy loading conflicts with pytest-cov instrumentation. matplotlib 3.8+ uses lazy `__getattr__` to delay submodule loading, which breaks when pytest-cov instruments code before matplotlib initializes. Solution: Skip viz tests that directly call `render()` during CI/CD (8 test modules ignored), exempt `matplotlib_adapter.py` and legacy shims from coverage. Tests achieve 86.19% coverage (target: 85%) with 586 tests passing.

- Reorganised the documentation site to follow ADR-022's role-based navigation with refreshed quickstarts, telemetry concept guides, troubleshooting material, and section ownership guidance.
- Updated the README Quick Start to highlight telemetry inspection workflows and PlotSpec defaults, aligning the repository's front door with the new documentation narrative.

## [v0.7.0](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.7.0) - 2025-10-07

[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.6.1...v0.7.0)

### Added

- Interval calibrators now resolve through the plugin registry for both legacy
  and FAST paths, enabling trusted override chains and capturing telemetry about
  the `interval_source`/`proba_source` used for each explanation batch.
- CalibratedExplainer accepts new keyword overrides (`interval_plugin`,
  `fast_interval_plugin`, `plot_style`) and reads environment/pyproject
  fallbacks so operators can configure intervals and plots without code changes.
- Packaged a `ce.plugins` console script with smoke tests covering list/show and
  trust management workflows.

### Changed

- Interval plugin contexts surface FAST reuse hints and metadata to keep
  calibrator reuse efficient while routing through the registry.
- README, developer docs, and contributing guides document the new CLI,
  configuration knobs, and plugin telemetry expectations.

### Docs

- README Quick Start and `docs/plugins.md` now demonstrate PlotSpec-first
  plotting, telemetry inspection (including preprocess metadata), and the
  enhanced `ce.plugins` CLI output, folding the guardrail narrative into the
  user-facing docs for the v0.8.0 release.

- ADR-013 and ADR-015 marked Accepted with implementation notes summarising the
  registry-backed runtime.
- ADR-017/ADR-018 ratified with quick-reference style excerpts in
  `CONTRIBUTING.md` and contributor docs.
- Harmonised `core.validation` docstrings with numpy-style lint guardrails (ADR-018).

### CI

- Shared `.coveragerc` published and the test workflow now enforces
  `--cov-fail-under=80` to meet ADR-019 phase 1 requirements (with gradual increase for each new version).
- Lint workflow surfaces Ruff naming warnings and docstring lint/coverage
  reports, providing guardrails for ADR-017/ADR-018 adoption.

#### Public API updates in v0.7.0
This document summarises the signature adjustments introduced while aligning the
codebase with the new Ruff style baseline. Reference it from the v0.7.0 changelog
when communicating breaking or user-visible updates.

##### Function parameter renames
The following parameters have been renamed across multiple functions and methods:
- X_test → x
- y_test → y

##### Wrapper keyword normalisation
The following `WrapCalibratedExplainer` entry points now strip deprecated alias
arguments after emitting a `DeprecationWarning`:
- `calibrate`
- `explain_factual`
- `explore_alternatives`
- `explain_fast`
- `predict`
- `predict_proba`
Alias keys such as `alpha`, `alphas`, and `n_jobs` are therefore ignored going
forward. Callers must provide the canonical keyword names (`low_high_percentiles`,
`parallel_workers`, etc.) for custom behaviour to take effect.【F:src/calibrated_explanations/core/wrap_explainer.py†L201-L409】【F:src/calibrated_explanations/api/params.py†L16-L70】

##### Explanation plugin toggle
`CalibratedExplainer` now exposes a keyword-only `_use_plugin` flag across all
explanation factories (`explain_factual`, `explore_alternatives`, `explain_fast`,
`explain`, and the `__call__` shorthand). The flag defaults to `True`, enabling
the plugin orchestrator. Pass `_use_plugin=False` to route through the legacy
implementation when needed.【F:src/calibrated_explanations/core/calibrated_explainer.py†L1489-L1665】

##### Conjunction helper parameters
All `add_conjunctions` helpers across explanation containers use the renamed
keyword arguments `n_top_features` and `max_rule_size` (previously exposed as
`num_to_include` and `num_rule_size`). Update downstream code, documentation,
and notebooks accordingly.【F:src/calibrated_explanations/explanations/explanations.py†L460-L501】

### Fixed

- Fixed test helper stubs and plugin descriptors to satisfy Ruff naming guardrails (ADR-017), keeping `ruff check --select N` green.

## [v0.6.1](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.6.1) - 2025-10-05

[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.6.0...v0.6.1)

- Added runtime regression coverage to compare plugin-orchestrated factual, alternative, and fast explanations against the legacy `_use_plugin=False` code paths (`tests/integration/core/test_explanation_parity.py`).
- Exercised schema v1 guardrails by asserting that payloads missing required keys are rejected when `jsonschema` is installed (`tests/unit/core/test_serialization_and_quick.py::test_validate_payload_rejects_missing_required_fields`).
- Locked in `WrapCalibratedExplainer` keyword defaults and alias handling when using configuration objects (`tests/unit/core/test_wrap_keyword_defaults.py`).

### Docs

- Documented the v0.6.x hardening checklist covering plugin parity, schema validation, and wrapper default tests in `docs/plugins.md`.

## [v0.6.0](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.6.0) - 2025-09-04

[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.5.1...v0.6.0)

### Highlights (Contract-first)

- Internal domain model for explanations (ADR-008): added `Explanation` and `FeatureRule` types with adapters to preserve legacy dict outputs. No public API changes required; golden outputs unchanged.
- Explanation Schema v1 (ADR-005): shipped a versioned JSON Schema (`schemas/explanation_schema_v1.json`) and utilities for `to_json`/`from_json` with validation. Round-trip tests included.
- Preprocessing policy hooks (ADR-009): wrapper now supports configurable preprocessing with mapping persistence and unseen-category policy; default behavior unchanged for numeric inputs.
- Optional extras split (Phase 2S): declared `viz`, `lime`, `notebooks`, `dev`, `eval` extras and lazy plotting import. Tests that need matplotlib are marked `@pytest.mark.viz` and skipped when extras are absent.
- Docs: new Schema v1 and Migration (0.5.x → 0.6.0) pages; evaluation README and API reference updates.

### Deprecations

- Parameter alias deprecations wired; warnings emitted once per session (removal not before v0.8.0). See migration guide.

### CI

- Added docs build + linkcheck job and a core-only test job without viz extras to ensure core independence.

### Notes

- This release focuses on contract stability and does not change public serialized outputs. Performance features remain behind flags and will arrive in v0.7.x.

### Acknowledgements

We thank community contributors for overlapping PR work and early prototypes that informed the v0.6.0 contract-first implementation (domain model, schema/serialization, preprocessing hooks). Your feedback and ideas helped refine the final design.

Also added explicit credit files:

- AUTHORS.md (Main authors, authors listed in papers)
- CONTRIBUTORS.md (community contributions)

### Maintenance / Phase 1B completion

- Phase 1B concluded: parameter canonicalization and lightweight validation wired at predict/predict_proba boundaries; strict typing with py.typed and targeted mypy overrides; documentation for error handling/validation/params added and linked; removed OnlineCalibratedExplainer and pruned legacy mentions; CI hygiene (branch conditions, perf guard) and repo lint/type gates green.

### Features

- Updated references to the paper [Calibrated Explanations for Regression](https://doi.org/10.1007/s10994-024-06642-8) in README and citing. The paper is now published in Machine Learning Journal.
  - [Löfström, T](https://github.com/tuvelofstrom)., [Löfström, H](https://github.com/Moffran)., Johansson, U., Sönströd, C., and [Matela, R](https://github.com/rudymatela). (2025). [Calibrated Explanations for Regression](https://doi.org/10.1007/s10994-024-06642-8). Machine Learning 114, 100.
- Plot Style Control: With [a series of commits](https://github.com/Moffran/calibrated_explanations/compare/4cbc4ff410df19a32899071daa2568e8904c2c47...4093496b1d938f3470f2d33715f0af286f239728), a `plot_config.ini` file and a `test_plot_config.py` file have been added. The style parameters are used by the plots. The style parameters for all plot functions can now be overridden using the `style_override` parameter. [Controlling figure width is also added](https://github.com/Moffran/calibrated_explanations/compare/86babfa1afed75fc8959cf072c21e932c3d08f07...e0d13f32907185a144781ff76b553ad5c8cc0f8d).
- Optional extras and lazy plotting: Added optional dependency extras in `pyproject.toml` (`viz` for matplotlib, `lime` for LIME). Made matplotlib a lazy optional import used only when plotting is invoked, with a friendly runtime hint to install `calibrated_explanations[viz]`. Updated README with installation examples.

### Breaking Changes

- Removed the experimental `OnlineCalibratedExplainer` and its tests/docs. All references were purged from code, docs, configs, and packaging metadata.

### Fixes

- [fix: ensure figures are closed when not shown in plotting functions](https://github.com/Moffran/calibrated_explanations/commit/f20a047b2c4acb0eae6b5f6aed876f2db7d4d389)

## [v0.5.1](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.5.1) - 2024-11-27

[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.5.0...v0.5.1)

### Features

- String Targets Support: Added support for string targets, enhancing flexibility in handling diverse datasets. Special thanks to our new contributor [ww-jermaine](https://github.com/ww-jermaine) for the efforts on this feature ([issue #27](https://github.com/Moffran/calibrated_explanations/issues/27)).
- Out-of-Bag Calibration: Introduced support for out-of-bag calibration when using random forests from `sklearn`, enabling improved calibration techniques directly within ensemble models. See the new [notebook](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/quickstart_wrap_oob.ipynb) for examples.
- Documentation Enhancements: Updated and refined [documentation](https://calibrated-explanations.readthedocs.io/en/latest/?badge=latest), including fixes to existing sections and the addition of doctests for helper functions to ensure accuracy and reliability.
- Minor updates: Added a `calibrated` parameter to the `predict` and `predict_proba` methods to allow uncalibrated results.

### Fixes

- Bug Fixes: Resolved multiple bugs to enhance stability and performance across the library.


## [v0.5.0](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.5.0) - 2024-10-15

[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.4.0...v0.5.0)

### Features

- Improved the introduction in README.
- Added `calibrated_confusion_matrix` in `CalibratedExplainer` and `WrapCalibratedExplainer`, providing a leave-one-out calibrated confusion matrix using the calibration set. The insights from the confusion matrix are useful when analyzing explanations, to determine general prediction and error distributions of the model. An example of using the confusion matrix in the analysis is given in paper [Calibrated Explanations for Multi-class](https://raw.githubusercontent.com/mlresearch/v230/main/assets/lofstrom24a/lofstrom24a.pdf).
- Embraced the update of `crepes` version 0.7.1, making it possible to add a seed when fitting. Addresses issue #43.
- Updating terminology and functionality:
  - Introducing the concept of _ensured_ explanations.
    - Changed the name of `CounterfactualExplanation` to `AlternativeExplanation`, as it better reflects the purpose and functionality of the class.
    - Added a collection subclass `AlternativeExplanations` inheriting from `CalibratedExplanations`, which is used for collections of `AlternativeExplanation`'s. Collection methods referring to methods only available in the `AlternativeExplanation` are included in the new collection class.
    - Added an `explore_alternatives` method in `CalibratedExplainer` and `WrapCalibratedExplainer` to be used instead of `explain_counterfactual`, as the name of the later is ambiguous. The `explain_counterfactual` is still kept for compatibility reasons but only forwards the call to `explore_alternatives`. All files and notebooks have been updated to only call `explore_alternatives`. All references to counterfactuals have been changed to alternatives, with obvious exceptions.
    - Added both filtering methods and a ranking metric that can help filter out ensured explanations.
      - The parameters `rnk_metric` and `rnk_weight` has been added to the plotting functions and is applicable to all kinds of plots.
      - Both the `AlternativeExplanation` class (for an individual instance) and the collection subclass `AlternativeExplanations` contains filter functions only applicable to alternative explanations, such as `counter_explanations`, `semi_explanations`, `super_explanations`, and `ensured_explanations`.
        - `counter_explanations` removes all alternatives except those changing prediction.
        - `semi_explanations` removes all alternatives except those reducing the probability while not changing prediction.
        - `super_explanations` removes all alternatives except those increasing the probability for the prediction.
        - The concept of potential (uncertain) explanations is introduced. When the uncertainty interval spans across probability 0.5, an explanation is considered a potential. It will normally only be counter-potential and semi-potential, but can in some cases also be super-potential. Potential alternatives can be included or excluded from the above methods using the boolean parameter `include_potentials`.
        - `ensured_explanations` removes all alternatives except those with lower uncertainty (i.e. smaller uncertainty interval) than the original prediction.
    - Added a new form of plot for probabilistic predictions is added, clearly visualizing both the aleatoric and the epistemic uncertainty.
      - A global plot is added, plotting all test instances with probability and uncertainty as the x- and y-axes. The area corresponding to potential (uncertain) predictions is marked. The plot can be invoked using the `plot(X_test)` or `plot(X_test, y_test)` call.
      - A local plot for alternative explanations, with probability and uncertainty as the x- and y-axes, is added, which can be invoked from an `AlternativeExplanation` or a `AlternativeExplanations` using `plot(style='triangular')`. The optimal use is when combined with the `filter_top` parameter (see below), to include all alternatives, as follows: `plot(style='triangular', filter_top=None)`.
    - Added prerpint and bibtex to the paper introducing _ensured_ explanations:
      - [Löfström, T](https://github.com/tuvelofstrom)., [Löfström, H](https://github.com/Moffran)., and [Hallberg Szabadvary, J](https://github.com/egonmedhatten). (2024). [Ensured: Explanations for Decreasing the Epistemic Uncertainty in Predictions](https://arxiv.org/abs/2410.05479). arXiv preprint arXiv:2410.05479.
      - Bibtex:

        ```bibtex
        @misc{lofstrom2024ce_ensured,
          title =        {Ensured: Explanations for Decreasing the Epistemic Uncertainty in Predictions},
          author =          {L\"ofstr\"om, Helena and L\"ofstr\"om, Tuwe and Hallberg Szabadvary, Johan},
          year =            {2024},
          eprint =          {2410.05479},
          archivePrefix =   {arXiv},
          primaryClass =    {cs.LG}
        }
        ```

  - Introduced _fast_ explanations
    - Introduced a new type of explanation called `FastExplanation` which can be extracted using the `explain_fast` method. It differs from a `FactualExplanation` in that it does not define a rule condition but only provides a feature weight.
    - The new type of explanation is using ideas from [ConformaSight](https://github.com/rabia174/ConformaSight), a recently proposed global explanation algorithm based on conformal classification. Acknowledgements have been added.
  - Introduced a new form av probabilistic regression explanation:
    - Introduced the possibility to get explanations for the probability of being inside an interval. This is achieved by assigning a tuple with lower and upper bounds as threshold, e.g., `threshold=(low,high)` to get the probability of the prediction falling inside the interval (low, high].
    - To the best of our knowledge, this is the only package that provide this functionality with epistemic uncertainty.
  - Introduced the possibility to add new user defined rule conditions, using the `add_new_rule_condition` method. This is only applicable to numerical features.
    - Factual explanations will create new conditions covering the instance value. Categorical features already get a condition for the instance value during the invocation of `explain_factual`.
    - Alternative explanations will create new conditions that exclude the instance value. Categorical features already get conditions for all alternative categories during the invocation of `explore_alternatives`.
  - Parameter naming:
    - The parameter indicating the number of rules to plot is renamed to `filter_top` (previously `n_features_to_show`), making the call including all rules (`filter_top=None`) makes a lot more sense.

### Fixes

- Added checks to ensure that the learner is not called unless the `WrapCalibratedExplainer` is fitted.
- Added checks to ensure that the explainer is not called unless the `WrapCalibratedExplainer` is calibrated.
- Fixed incorrect use of `np.random.seed`.

## [v0.4.0](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.4.0) - 2024-08-23
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.3.5...v0.4.0)
### Features
- Paper updates:
  - [Calibrated Explanations for Regression](https://doi.org/10.1007/s10994-024-06642-8) has been accepted to Machine Learning. It is currently in press.
- Code improvements:
  - __Substantial speedup__ achieved through the newly implemented `explain` method! This method implements the core algorithm while minimizing the number of calls to core._predict, substantially speeding up the code without altering the algorithmic logic of `calibrated_explanations`. The `explain` method is used exclusively from this version on when calling `explain_factual` or `explain_counterfactual`.
    - Re-ran the ablation study for classification, looking at the impact of calibration set size, number of percentile samplings for numeric features and the number of features.
      - Uploaded a pdf version of the [ablation study](https://github.com/Moffran/calibrated_explanations/blob/main/evaluation/Calibrated_Explanations_Ablation.pdf), making the results easier to overview.
    - Re-ran the evaluation for regression, measuring stability, robustness and running times with and without normalization.
  - Improved the `safe_import` to allow `import ... from ...` constructs.
  - Restructured package
    - Added a utils folder:
      - Moved discretizers.py to utils
      - Moved utils.py to utils and renamed to helper.py
    - Made explanations public
    - Made VennAbers and interval_regressor restricted
  - Experimental functionality introduced:
    - Several new experimental features have been introduced. These will be presented as Features once they are thoroughly tested and evaluated.
- Code interface improvements:
  - Added support for the `MondrianCategorizer` from crepes in the `WrapCalibratedExplainer`.
  - Added wrapper functions in `WrapCalibratedExplainer` redirecting to `CalibratedExplainer`:
    - Including `predict`, `predict_proba`, and `set_difficulty_estimator`.
    - Moved any remaining implementations of functions in `WrapCalibratedExplainer` to `CalibratedExplainer`.
  - Renamed the `plot_all` and `plot_explanation` functions to `plot`. Updated all usages of the `plot` function.
  - Added `__len__` and `__getitem__` to `CalibratedExplanations`.
    - `__getitem__` allow indexing with `int`, `slice`, and lists (both boolean and integer lists). When more than one explanation is retrieved, a new `CalibratedExplanations` is returned, otherwise, the indexed `CalibratedExplanation` is returned.
- Documentation improvements:
  - Restructured and extended the [documentation](https://calibrated-explanations.readthedocs.io/en/latest/?badge=latest).
    - Updated the information at the entry page
    - Added an API reference
- Improvements of the CI setup:
  - Updated CI to run pytest before pylint.
  - Updated CI to avoid running tests when commit message starts with 'info:' or 'docs:'.
- Testing improvements
  - Improved tests to test `predict` and `predict_proba` functions in `CalibratedExplainer` better.
  - Added several other tests to increase [coverage](https://app.codecov.io/github/Moffran/calibrated_explanations).
### Fixes
- Fixed minor errors in the `predict` and `predict_proba` functions in `CalibratedExplainer`.
- Several other minor bug fixes have also been made.

## [v0.3.5](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.3.5) - 2024-07-24
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.3.4...v0.3.5)
### Features
- Made several improvements of the `WrapCalibratedExplainer`:
    - `WrapCalibratedExplainer` is introduced as the default way to interact with `calibrated-explanations` in README.md. The benefit of having a wrapper class as interface is that it makes it easier to add different kinds of explanations.
    - Documentation of the functions has been updated.
    - Initialization:
      - The `WrapCalibratedExplainer` can now be initialized with an unfitted model as well as with a fitted model.
      - The `WrapCalibratedExplainer` can now be initialized with an already initialized `CalibratedExplainer` instance, providing access to the `predict` and `predict_proba` functions.
    - The `fit` method will reinitialize the explainer if the `WrapCalibratedExplainer` has already been calibrated, to ensure that the `explainer` is adapted to the re-fitted model.
    - Added improved error handling.
    - Made several other minor quality improving adjustments.
- Code coverage tests are added and monitored at [Codecov](https://app.codecov.io/github/Moffran/calibrated_explanations).
  - Tests are added in order to increase code coverage.
  - Unused code is cleaned up.
- Updated the [Further reading and citing](https://github.com/Moffran/calibrated_explanations#further-reading-and-citing) section in the README:
  - Added a reference and bibtex to:
    - [Löfström, T](https://github.com/tuvelofstrom)., [Löfström, H](https://github.com/Moffran)., Johansson, U. (2024). [Calibrated Explanations for Multi-class](https://easychair.org/publications/preprint/rqdD). <i>Proceedings of the Thirteenth Workshop on Conformal and Probabilistic Prediction and Applications</i>, in <i>Proceedings of Machine Learning Research</i>. In press.
    - ```bibtex
      @Booklet{lofstrom2024ce_multiclass,
        author = {Tuwe Löfström and Helena Löfström and Ulf Johansson},
        title = {Calibrated Explanations for Multi-Class},
        howpublished = {EasyChair Preprint no. 14106},
        year = {EasyChair, 2024}
      }
      ```
  - Updated the [docs/citing.md](https://github.com/Moffran/calibrated_explanations/blob/main/docs/citing.md) with the above changes.
### Fixes
- Discretizers are limited to the default alternatives for classification and regression. BinaryDiscretizer removed. `__repr__` functions added.
- Changed the `check_is_fitted` function to remove ties to sklearn.
- Made the `safe_import` throw an `ImportError` when an import fail.

## [v0.3.4](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.3.4) - 2024-07-10
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.3.3...v0.3.4)
### Features
- Updated the [Further reading and citing](https://github.com/Moffran/calibrated_explanations#further-reading-and-citing) section in the README:
  - Added a reference and bibtex to:
    - [Löfström, H](https://github.com/Moffran)., [Löfström, T](https://github.com/tuvelofstrom). (2024). [Conditional Calibrated Explanations: Finding a Path Between Bias and Uncertainty](https://doi.org/10.1007/978-3-031-63787-2_17). In: Longo, L., Lapuschkin, S., Seifert, C. (eds) Explainable Artificial Intelligence. xAI 2024. Communications in Computer and Information Science, vol 2153. Springer, Cham.
    - ```bibtex
      @InProceedings{lofstrom2024ce_conditional,
      author="L{\"o}fstr{\"o}m, Helena
      and L{\"o}fstr{\"o}m, Tuwe",
      editor="Longo, Luca
      and Lapuschkin, Sebastian
      and Seifert, Christin",
      title="Conditional Calibrated Explanations: Finding a Path Between Bias and Uncertainty",
      booktitle="Explainable Artificial Intelligence",
      year="2024",
      publisher="Springer Nature Switzerland",
      address="Cham",
      pages="332--355",
      abstract="While Artificial Intelligence and Machine Learning models are becoming increasingly prevalent, it is essential to remember that they are not infallible or inherently objective. These models depend on the data they are trained on and the inherent bias of the chosen machine learning algorithm. Therefore, selecting and sampling data for training is crucial for a fair outcome of the model. A model predicting, e.g., whether an applicant should be taken further in the job application process, could create heavily biased predictions against women if the data used to train the model mostly contained information about men. The well-known concept of conditional categories used in Conformal Prediction can be utilised to address this type of bias in the data. The Conformal Prediction framework includes uncertainty quantification methods for classification and regression. To help meet the challenges of data sets with potential bias, conditional categories were incorporated into an existing explanation method called Calibrated Explanations, relying on conformal methods. This approach allows users to try out different settings while simultaneously having the possibility to study how the uncertainty in the predictions is affected on an individual level. Furthermore, this paper evaluated how the uncertainty changed when using conditional categories based on attributes containing potential bias. It showed that the uncertainty significantly increased, revealing that fairness came with a cost of increased uncertainty.",
      isbn="978-3-031-63787-2"
      }
      ```
  - Updated the [docs/citing.md](https://github.com/Moffran/calibrated_explanations/blob/main/docs/citing.md) with the above changes.
### Fixes
- Changed np.Inf to np.inf for compatibility reasons (numpy v2.0.0).
- Updated requirements for numpy and crepes to include versions v2.0.0 and v0.7.0, respecitvely.

## [v0.3.3](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.3.3) - 2024-05-25
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.3.2...v0.3.3)
### Features
- Changed how probabilistic regression is done to handle both validity and speed by dividing the calibration set into two sets to allow pre-computation of the CPS. Credits to anonymous reviewer for this suggestion.
- Added updated regression experiments and plotting for revised paper.
- Added a new `under the hood` demo notebook to show how to access the information used in the plots,  like conditions and uncertainties etc.
### Fixes
- Several minor updates to descrptions and notebooks in the repository.

## [v0.3.2](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.3.2) - 2024-04-14
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.3.1...v0.3.2)
### Features
- Added Fairness experiments and plotting for the XAI 2024 paper. Added a `Fairness` tag for the weblinks.
- Added multi-class experiments and plotting for upcoming submissions. Added a `Multi-class` tag for weblinks.
- Some improvements were made to the multi-class functionality. The updates included updating the VennAbers class to a more robust handling of multi-class (with or without Mondrian bins).
### Fixes
- Updated the requirement for crepes to v0.6.2, to address known issues with some versions of python.
- The pythonpath for pytest was added to pyprojects.toml to avoid module not found error when running pytest locally.

## [v0.3.1](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.3.1) - 2024-02-23
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.3.0...v0.3.1)
### Features
- Added support for Mondrian explanations, using the `bins` attribute. The `bins` attribute takes a categorical feature of the size of the calibration or test set (depending on context) indicating the category of each instance. For continuous attributes, the `crepes.extras.binning`can be used to define categories through binning.
- Added `BinaryRegressorDiscretizer` and `RegressorDiscretizer` which are similar to `BinaryEntropyDiscretizer` and `EntropyDiscretizer` in that it uses a decision tree to identify suitable discretizations for numerical features. `explain_factual` and `explain_counterfactual` have been updated to use these discretizers for regression by default. In a future version, the possibility to assign your own discretizer may be removed.
- Updated the [Further reading and citing](https://github.com/Moffran/calibrated_explanations#further-reading-and-citing) section in the README:
  - Updated the reference and bibtex to the published version of the introductory paper:
    - Löfström, H., Löfström, T., Johansson, U., and Sönströd, C. (2024). [Calibrated Explanations: with Uncertainty Information and Counterfactuals](https://doi.org/10.1016/j.eswa.2024.123154). Expert Systems with Applications, 1-27.

    - ```bibtex
      @article{lofstrom2024calibrated,
        title = 	{Calibrated explanations: With uncertainty information and counterfactuals},
        journal = 	{Expert Systems with Applications},
        pages = 	{123154},
        year = 	{2024},
        issn = 	{0957-4174},
        doi = 	{https://doi.org/10.1016/j.eswa.2024.123154},
        url = 	{https://www.sciencedirect.com/science/article/pii/S0957417424000198},
        author = 	{Helena Löfström and Tuwe Löfström and Ulf Johansson and Cecilia Sönströd},
        keywords = 	{Explainable AI, Feature importance, Calibrated explanations, Venn-Abers, Uncertainty quantification, Counterfactual explanations},
        abstract = 	{While local explanations for AI models can offer insights into individual predictions, such as feature importance, they are plagued by issues like instability. The unreliability of feature weights, often skewed due to poorly calibrated ML models, deepens these challenges. Moreover, the critical aspect of feature importance uncertainty remains mostly unaddressed in Explainable AI (XAI). The novel feature importance explanation method presented in this paper, called Calibrated Explanations (CE), is designed to tackle these issues head-on. Built on the foundation of Venn-Abers, CE not only calibrates the underlying model but also delivers reliable feature importance explanations with an exact definition of the feature weights. CE goes beyond conventional solutions by addressing output uncertainty. It accomplishes this by providing uncertainty quantification for both feature weights and the model’s probability estimates. Additionally, CE is model-agnostic, featuring easily comprehensible conditional rules and the ability to generate counterfactual explanations with embedded uncertainty quantification. Results from an evaluation with 25 benchmark datasets underscore the efficacy of CE, making it stand as a fast, reliable, stable, and robust solution.}
      }
      ```
  - Added [Code and results](https://github.com/tuvelofstrom/calibrating-explanations) for the [Investigating the impact of calibration on the quality of explanations](https://link.springer.com/article/10.1007/s10472-023-09837-2) paper, inspiring the idea behind Calibrated Explanations.
  - Added a bibtex to the software repository:
    - ```bibtex
      @software{Lofstrom_Calibrated_Explanations_2024,
        author = 	{Löfström, Helena and Löfström, Tuwe and Johansson, Ulf and Sönströd, Cecilia and Matela, Rudy},
        license = 	{BSD-3-Clause},
        title = 	{Calibrated Explanations},
        url = 	{https://github.com/Moffran/calibrated_explanations},
        version = 	{v0.3.1},
        month = 	feb,
        year = 	{2024}
      }
      ```
  - Updated the [docs/citing.md](https://github.com/Moffran/calibrated_explanations/blob/main/docs/citing.md) with the above changes.
- Added a [CITATION.cff](https://github.com/Moffran/calibrated_explanations/blob/main/CITATION.cff) with citation data for the software repository.
### Fixes
- Extended `__repr__` to include additional fields when `verbose=True`.
- Fixed a minor bug in the example provided in the [README.md](https://github.com/Moffran/calibrated_explanations/blob/main/README.md#classification) and the [getting_started.md](https://github.com/Moffran/calibrated_explanations/blob/main/docs/getting_started.md#classification), as described in issue #26.
- Added `utils.transform_to_numeric` and a clarification about known limitations in [README.md](https://github.com/Moffran/calibrated_explanations/blob/main/README.md#classification) as a response to issue #28.
- Fixed a minor bug in `FactualExplanation.__plot_probabilistic` that was triggered when no features where to be shown.
- Fixed a bug with the discretizers in `core`.
- Fixed a bug with saving plots to file using the `filename` parameter.

## [v0.3.0](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.3.0) - 2024-01-02
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.2.3...v0.3.0)
### Features
- Updated to version 1.4.1 of venn_abers. Added `precision=4` to the fitting of the venn_abers model to increase speed.
- Preparation for weighted categorical rules implemented but not yet activated.
- Added a state-of-the-art comparison with scripts and notebooks for evaluating the performance of the method in comparison with `LIME` and `SHAP`: see [Classification_Experiment_sota.py](https://github.com/Moffran/calibrated_explanations/blob/main/evaluation/Classification_Experiment_sota.py) and [Classification_Analysis_sota.ipynb](https://github.com/Moffran/calibrated_explanations/blob/main/evaluation/Classification_Analysis_sota.ipynb) for running and evaluating the experiment. Unzip [results_sota.zip](https://github.com/Moffran/calibrated_explanations/blob/main/evaluation/results_sota.zip) and run [Classification_Analysis_sota.ipynb](https://github.com/Moffran/calibrated_explanations/blob/main/evaluation/Classification_Analysis_sota.ipynb) to get the results used in the paper [Calibrated Explanations: with Uncertainty Information and Counterfactuals](https://doi.org/10.1016/j.eswa.2024.123154).
- Updated the parameters used by `plot_all` and `plot_explanation`.
### Fixes
- Filtered out extreme target values in the quickstart notebook to make the regression examples more realistic.
- Fixed bugs related to how plots can be saved to file.
- Fixed an issue where add_conjunctions with `max_rule_size=3` did not work.

## [v0.2.3](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.2.3) - 2023-11-04
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.2.2...v0.2.3)
### Features
- Added an evaluation folder with scripts and notebooks for evaluating the performance of the method.
  - One evaluation focuses on stability and robustness of the method: see [Classification_Experiment_stab_rob.py](https://github.com/Moffran/calibrated_explanations/blob/main/evaluation/Classification_Experiment_stab_rob.py) and [Classification_Analysis_stab_rob.ipynb](https://github.com/Moffran/calibrated_explanations/blob/main/evaluation/Classification_Analysis_stab_rob.ipynb) for running and evaluating the experiment.
  - One evaluation focuses on how different parameters affect the method regarding time and robustness: see [Classification_Experiment_Ablation.py](https://github.com/Moffran/calibrated_explanations/blob/main/evaluation/Classification_Experiment_Ablation.py) and [Classification_Analysis_Ablation.ipynb](https://github.com/Moffran/calibrated_explanations/blob/main/evaluation/Classification_Analysis_Ablation.ipynb) for running and evaluating the experiment.

### Fixes
- Fix in `CalibratedExplainer` to ensure that greater-than works identical as less-than.
- Bugfix in `FactualExplanation._get_rules()` which caused an error when categorical labels where missing.

## [v0.2.2](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.2.2) - 2023-10-03
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.2.1...v0.2.2)
### Fixes
Smaller adjustments and fixes.

## [v0.2.1](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.2.1) - 2023-09-20
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.2.0...v0.2.1)
### Fixes
The wrapper file with helper classes `CalibratedAsShapExplainer` and `CalibratedAsLimeTabularExplanainer` has been removed. The `as_shap` and `as_lime` functions are still working.

## [v0.2.0](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.2.0) - 2023-09-19
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.1.1...v0.2.0)
### Features
- Added a `WrapCalibratedExplainer` class which can be used for both classificaiton and regression.
- Added [quickstart_wrap](https://github.com/Moffran/calibrated_explanations/blob/main/notebooks/quickstart_wrap.ipynb) to the notebooks folder.
- Added [LIME_comparison](https://github.com/Moffran/calibrated_explanations/notebooks/LIME_comparison.ipynb) to the notebooks folder.
### Fixes
- Removed the dependency on `shap` and `scikit-learn` and closed issue #8.
- Updated the weights to match LIME's weights (to ensure that a positive weight has the same meaning in both).
- Changed name of parameter `y` (representing the threshold in probabilistic regression) to `threshold`.

## [v0.1.1](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.1.1) - 2023-09-14
[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.1.0...v0.1.1)
### Features
- Exchanged the slow `VennABERS_by_def` function for the `VennAbers` class in the `venn-abers` package.
### Fixes
- Low and high weights are correctly assigned, so that low < high is always the case.
- Adjusted the number of decimals in counterfactual rules to 2.
## [v0.1.0](https://github.com/Moffran/calibrated_explanations/releases/tag/v0.1.0) - 2023-09-04

[Full changelog](https://github.com/Moffran/calibrated_explanations/compare/v0.0.2...v0.1.0)

### Features

- **Performance**: Fast, reliable, stable and robust feature importance explanations.
- **Calibrated Explanations**: Calibration of the underlying model to ensure that predictions reflect reality.
- **Uncertainty Quantification**: Uncertainty quantification of the prediction from the underlying model and the feature importance weights.
- **Interpretation**: Rules with straightforward interpretation in relation to the feature weights.
- **Factual and Counterfactual Explanations**: Possibility to generate counterfactual rules with uncertainty quantification of the expected predictions achieved.
- **Conjunctive Rules**: Conjunctive rules conveying joint contribution between features.
- **Multiclass Support**: Multiclass support has been added since the original version developed for the paper [Calibrated Explanations: with Uncertainty Information and Counterfactuals](https://arxiv.org/pdf/2305.02305.pdf).
- **Regression Support**: Support for explanations from standard regression was developed and is described in the paper [Calibrated Explanations for Regression](https://arxiv.org/pdf/2308.16245.pdf).
- **Probabilistic Regression Support**: Support for probabilistic explanations from standard regression was added together with regression and is described in the paper mentioned above.
- **Conjunctive Rules**: Since the original version, conjunctive rules has also been added.
- **Code Structure**: The code structure has been improved a lot. The `CalibratedExplainer`, when applied to a model and a collection of test instances, creates a collection class, `CalibratedExplanations`, holding `CalibratedExplanation` objects, which are either `FactualExplanation` or `CounterfactualExplanation` objects. Operations can be applied to all explanations in the collection directly through `CalibratedExplanations` or through each individual `CalibratedExplanation` (see the [documentation](https://calibrated-explanations.readthedocs.io)).

### Fixes
Numerous. The code has been refactored and improved a lot since the original version. The code is now also tested and documented.
