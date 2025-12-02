> **Status note (2025-10-24):** Last edited 2025-10-24 · Latest coverage run (`pytest --cov=src/calibrated_explanations --cov-report=term`) on 2025-10-24 reached **88.4%** line coverage with `fail_under=88` satisfied. Archive after: Re-evaluate post-v1.0.0 maintenance review · Implementation window: v0.9.0–v1.0.0.

# Coverage Uplift Plan (ADR-019)

This merged plan combines the coverage standardization roadmap with the module-level gap analysis so gating, baselines, and remediation tactics live together.

## Tooling and gating roadmap

### Phase 0 – Tooling foundation (Week 1)
1. Add `.coveragerc` with package/critical-path thresholds and explicitly documented excludes.
2. Update `pytest.ini` default `addopts` to include `--cov=src/calibrated_explanations --cov-report=term-missing --cov-fail-under=80` once debt burn-down reaches 90%.
3. Extend the `dev` optional dependency set so `pytest-cov` is installed for every contributor environment.【F:pyproject.toml†L38-L59】
4. Provide `make test-cov` (or `tox -e py-cov`) target mirroring CI invocation to ease local runs.

### Phase 1 – Debt burn-down (Weeks 2-4)
1. Author focused unit tests for the currently unreferenced runtime modules, prioritising:
   - `core.interval_regressor` happy-path predictions and error handling.【F:src/calibrated_explanations/core/interval_regressor.py†L1-L120】
   - Plugin CLI command smoke tests ensuring registry resolution works.
   - Discretizer utilities to cover edge cases around binning strategies.
   - External plugin extras installer (`external_plugins` folder, packaging extras) so aggregated installs remain tested.
2. Capture baseline coverage numbers and iterate until package-wide coverage stabilises above 90%.
3. Add lightweight fixtures (e.g., small sklearn datasets) so interval regression tests remain fast.

### Phase 2 – CI gating (Weeks 5-6)
1. Update `.github/workflows/test.yml` to run `pytest --cov=src/calibrated_explanations --cov-report=xml --cov-report=term --cov-fail-under=90` and upload the XML artifact for Codecov patch gating.【F:.github/workflows/test.yml†L33-L49】
2. Enable Codecov’s “patch coverage must be ≥88%” status check and make it required.
3. Document the waiver process in `CONTRIBUTING.md`, emphasising that waivers must link to follow-up issues.【F:CONTRIBUTING.md†L49-L58】

### Phase 3 – Continuous improvement (Ongoing)
1. Review `.coveragerc` exemptions quarterly, removing expired shims or adding TODO dates.
2. Track coverage deltas in release retrospectives and flag regressions >1% for root-cause analysis.
3. Explore mutation testing or fuzzing for calibration math once coverage stabilises above 95% on critical modules.

## Baseline snapshot
The baseline repository-wide coverage snapshot captured prior to the uplift (`pytest --cov=src/calibrated_explanations --cov-report=term`) reported **86.0%** line coverage (7,605 statements, 840 misses; branch coverage 78.1%), which kept the package short of the 90% policy target even though the interim `fail_under=85` gate passed. We retain the table below for historical context while the active gate now enforces `fail_under=88`.

| Module | Stmts | Miss | Branch miss | Coverage |
| --- | ---: | ---: | ---: | ---: |
| `plotting.py` | 666 | 227 | 262 | 63.1% |
| `core/calibrated_explainer.py` | 1,520 | 147 | 660 | 86.8% |
| `explanations/explanation.py` | 997 | 123 | 354 | 85.6% |
| `viz/builders.py` | 415 | 99 | 172 | 70.4% |
| `plugins/registry.py` | 756 | 77 | 250 | 86.3% |
| `plugins/builtins.py` | 387 | 55 | 118 | 80.4% |
| `core/wrap_explainer.py` | 355 | 28 | 150 | 90.7% |
| `plugins/cli.py` | 204 | 10 | 82 | 90.9% |
| `core/prediction_helpers.py` | 63 | 6 | 28 | 81.3% |
| `__init__.py` | 29 | 7 | 8 | 75.7% |

The same run highlighted thin coverage across gateway modules such as `core/__init__.py` (80.0%) and the root package initializer (75.7%), both of which guard public imports and configuration defaults.

## Gap analysis by subsystem

### 1. Plotting router and builder chain
* `_plot_probabilistic` and `_plot_global` orchestrate style negotiation, interval guards, triangular alternative rendering, and fallback to legacy drawing, but current tests barely exercise the override matrix (`style_override`, `use_legacy`) or the defensive logging paths, leaving large sections uncovered—including the triangular plot path.【F:src/calibrated_explanations/plotting.py†L601-L640】
* The modern rendering path through `viz/builders.py` constructs probability segments, pivot-aware colour roles, and ranking heuristics; coverage shows those branches are almost entirely cold, especially the uncertainty segment assembly near the bottom of the module.【F:src/calibrated_explanations/viz/builders.py†L431-L470】【F:src/calibrated_explanations/viz/builders.py†L838-L852】
* Save-extension handling still behaves differently on Windows vs. POSIX (the failing assertion that expects `tmp_path / "defaultsvg"`), signalling that IO-related branches lack coverage and break parity across platforms.【F:tests/unit/legacy/test_plotting_module.py†L205-L209】

**Remediation tactics**
1. Parameterise new plotting tests to drive combinations of `style_override`, `use_legacy`, interval flags, and `save_ext` inputs, asserting both figure assembly and normalised paths (use `pathlib.Path` to abstract separators).
2. Extract focused builder tests for `_build_probability_segments`, ranking heuristics, triangular plot assembly, and pivot-aware colouring so the cold loops in `viz/builders.py` receive deterministic coverage without heavy matplotlib integration.
3. Restore the Windows assertion by adjusting implementation (or test) path handling, then assert the branch that converts extension lists to filenames so the cross-platform guard remains covered.

### 2. Explanation assembly and validation
* `core.CalibratedExplainer.__init__` still lacks coverage over categorical target conversion, pyproject overrides, and fast-mode toggles; these guardrails drive label maps, difficulty estimators, and plugin registration state during runtime.【F:src/calibrated_explanations/core/calibrated_explainer.py†L321-L360】
* `explanations/explanation.py` continues to miss large swaths of logic when switching between probabilistic and regression outputs, ranking features, and hydrating cached metadata, particularly around the plotting bridge that now feeds the `plotting` router.【F:src/calibrated_explanations/explanations/explanation.py†L1409-L1445】【F:src/calibrated_explanations/explanations/explanation.py†L2291-L2313】
* `core/wrap_explainer.py` retains unexecuted flows for estimator wrappers (fast vs. lime vs. reject learners) and difficulty estimator assignment, so public APIs can regress without coverage alerts.【F:src/calibrated_explanations/core/wrap_explainer.py†L331-L376】【F:src/calibrated_explanations/core/wrap_explainer.py†L497-L506】

**Remediation tactics**
1. Build fixture-based tests that instantiate `CalibratedExplainer` with categorical labels, pyproject overrides, and custom plugin hints to cover initialization branches and label map handling.
2. Expand explanation tests to cover both thresholded and regression outputs, verifying caching invalidation, ranking heuristics, and shap/lime fallbacks with dummy explainers.
3. Exercise the wrapper APIs (`wrap_explainer.explain_fast`, `.explain_lime`, `.set_difficulty_estimator`) with mocked estimators to assert the gating predicates and integration points.

### 3. Plugin registry and builtins
* Registry trust-management helpers (`mark_plot_renderer_trusted/untrusted`) mutate shared registries and metadata, but there are no assertions that the trust sets or propagated metadata stay consistent.【F:src/calibrated_explanations/plugins/registry.py†L1211-L1234】
* Built-in plugins perform heavy payload normalisation—deriving feature indices, column names, and interval flags from heterogeneous payloads—yet coverage shows those branches are rarely executed.【F:src/calibrated_explanations/plugins/builtins.py†L661-L709】
* CLI emitters rely on registry lookups and metadata formatting; failure paths for missing identifiers and trust filtering remain untested, leaving CLI ergonomics brittle.【F:src/calibrated_explanations/plugins/cli.py†L51-L90】
* No coverage exists around the forthcoming `external_plugins` extras installer, leaving the aggregated installation path and folder bootstrap logic unverified.

**Remediation tactics**
1. Build registry tests that seed dummy descriptors, toggling trust flags to confirm `_PLOT_RENDERERS`, `_TRUSTED_PLOT_RENDERERS`, and metadata propagation stay in sync.
2. Add unit tests for the probabilistic built-ins that feed mixed payloads (mapping vs. array) to cover feature-index derivation, logging branches, and auto-selection of `features_to_plot`.
3. Exercise CLI commands via `CliRunner`, verifying that invalid identifiers raise the documented errors and that trusted/untrusted filters and JSON emitters behave as expected.
4. Add tests for the aggregated `external-plugins` extras installer that validate dependency resolution, folder discovery (`external_plugins/`), and opt-in semantics.

### 4. Prediction helpers and package gateways
* `core/prediction_helpers.initialize_explanation` hides several branch-heavy validation paths (Mondrian bins, regression thresholds, warning hooks) that remain untested, leaving subtle validation errors undetected.【F:src/calibrated_explanations/core/prediction_helpers.py†L82-L108】
* `core/__init__.__getattr__` emits deprecation warnings conditionally; without direct tests the branch that suppresses warnings during pytest runs can regress silently.【F:src/calibrated_explanations/core/__init__.py†L17-L30】
* The package-level `__getattr__` lazily imports interval and Venn-Abers helpers, but coverage misses the lazy import and caching behaviour that guard public API stability.【F:src/calibrated_explanations/__init__.py†L61-L75】

**Remediation tactics**
1. Add `prediction_helpers` unit tests covering Mondrian bin validation, regression threshold assertions, and warning emission so each branch remains exercised.
2. Monkeypatch environment flags in tests to assert the deprecation warning logic inside `core/__init__` behaves correctly for both pytest and runtime consumers.
3. Drive the package-level lazy imports under test, asserting that repeated attribute access returns cached objects and that unsupported names raise `AttributeError` as expected.

### 5. Legacy compatibility surfaces
* Deprecated re-export modules scheduled for removal have now been deleted (`legacy/_interval_regressor.py`, `legacy/_plots.py`, `_plots_legacy.py`), resolving the lingering coverage exceptions and leaving `legacy/plotting.py` as the sole supported legacy surface.【F:src/calibrated_explanations/legacy/__init__.py†L1-L6】【F:src/calibrated_explanations/legacy/plotting.py†L1-L120】
* The Windows-only plotting test failure shows legacy plotting utilities still influence default behaviours, so on-going maintenance needs to ratify which legacy paths stay supported.

**Remediation tactics**
1. Decide whether to surround the legacy modules with minimal smoke tests (ensuring imports succeed and wrappers delegate) or document permanent exclusions in `.coveragerc`.
2. Once plotting path handling is normalised, capture a regression test around the legacy save-extension ordering so the Windows behaviour remains stable.

## Proposed remediation roadmap

| Phase | Scope | Target modules | Success criteria |
| --- | --- | --- | --- |
| Sprint 1 | Plotting router + builder hardening | `plotting`, `viz/builders`, legacy plotting save paths | Windows save-extension assertions restored; plotting router coverage ≥80%; builder uncertainty segments executed in tests. |
| Sprint 2 | Explanation pipeline + wrappers | `core/calibrated_explainer`, `explanations/explanation`, `core/wrap_explainer` | Categorical init and caching branches covered; wrapper APIs exercised; module coverage ≥90%. |
| Sprint 3 | Plugin registry + built-ins | `plugins/registry`, `plugins/builtins`, `plugins/cli` | Trust toggles round-trip metadata; CLI smoke tests capture error paths; plugin modules reach ≥88% coverage. |
| Sprint 4 | Gateways + legacy surfaces | `core/prediction_helpers`, `core/__init__`, `calibrated_explanations/__init__`, `legacy/plotting.py` | Lazy import guards and Mondrian validation tested; decision logged for the remaining legacy plotting surface (tests vs. `.coveragerc`); residual modules ≥88% or explicitly excluded. |

## Supporting actions
1. Land shared fixtures in `tests/conftest.py` for synthetic calibration payloads, plugin descriptors, and plotting datasets so the new test suites stay concise.
2. Normalize filesystem assertions (paths, extension ordering) via helper utilities that wrap `pathlib.Path`, guaranteeing the Windows/Posix parity fix remains exercised.
3. Update `.coveragerc` after Sprint 4 to reflect the final decision on legacy re-exports, attaching expiry dates to any exclusions that remain.
4. Track per-module thresholds each sprint (e.g., `coverage report --fail-under=<module-threshold>`) and surface them in CI dashboards to prevent regressions while the plan is underway.
