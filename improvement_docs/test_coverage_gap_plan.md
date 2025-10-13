# Coverage Gap Remediation Plan (2025-02)

## Baseline snapshot
The latest repository-wide coverage run (`pytest --cov=src/calibrated_explanations`) reports 78.6% line coverage, below the 90% policy target. The table summarises the largest sources of uncovered code to prioritise remediation.

| Module | Stmts | Miss | Branch miss | Coverage |
| --- | ---: | ---: | ---: | ---: |
| `core/calibrated_explainer.py` | 1,442 | 277 | 727 | 77.4% |
| `explanations/explanation.py` | 818 | 106 | 318 | 85.1% |
| `viz/plots.py` | 366 | 114 | 177 | 64.0% |
| `legacy/_plots_legacy.py` | 428 | 121 | 169 | 68.9% |
| `viz/matplotlib_adapter.py` | 512 | 99 | 296 | 76.9% |
| `plugins/registry.py` | 672 | 145 | 288 | 72.2% |
| `_interval_regressor.py` | 142 | 49 | 35 | 59.1% |
| `core/validation.py` | 73 | 18 | 51 | 70.6% |
| `utils/helper.py` | 218 | 27 | 147 | 84.9% |
| `utils/perturbation.py` | 46 | 8 | 20 | 74.2% |

The same run highlighted thin coverage in `api/__init__.py` (48.1%) and `core/calibration_helpers.py` (70.8%), both of which expose public entry points.

## Gap analysis by subsystem

### 1. Interval calibration runtime
* `_interval_regressor.IntervalRegressor.predict_probability` contains an error path when test bins are supplied without calibration bins, plus iterative per-threshold execution that is never validated; no unit test ensures the branch importing `safe_first_element` behaves correctly when the relative import fails.【F:src/calibrated_explanations/core/interval_regressor.py†L104-L135】
* The singledispatch `compute_proba_cal` has an untested `TypeError` fallback and tuple threshold branch, and the `insert_calibration` routine mutates conformal predictor state and bins—scenarios absent from the suite.【F:src/calibrated_explanations/core/interval_regressor.py†L246-L378】
* `core/calibrated_explainer._resolve_interval_plugin` performs layered fallback resolution with metadata gating (fast mode, capability checks, bin requirements) yet lacks targeted mocks verifying override precedence and error aggregation.【F:src/calibrated_explanations/core/calibrated_explainer.py†L720-L789】

**Remediation tactics**
1. Create fast-running synthetic fixtures for interval regressors using small numpy arrays; exercise scalar, tuple, and vector thresholds, plus the bin-mismatch `ValueError`.
2. Add unit tests that patch the helper import to force the fallback import path and confirm `safe_first_element` usage.
3. Mock plugin descriptors in `core/calibrated_explainer` to validate preferred/fallback resolution, error accumulation, and fast-mode gating.

### 2. Explanation assembly and validation
* `core/calibrated_explainer.__init__` configures categorical handling, pyproject overrides, and plugin registries; misconfigurations such as non-numeric classification labels and absent difficulty estimators are not covered.【F:src/calibrated_explanations/core/calibrated_explainer.py†L320-L378】
* `core/validation.py` enforces schema checks (e.g., calibration split lengths, column name alignment), but branches raising `ConfigurationError` and `ValidationError` remain untested.【F:src/calibrated_explanations/core/validation.py†L20-L88】
* `explanations/explanation.py` includes thresholded vs. regression formatting and caching of feature contributions; numerous conditional branches handle missing metadata, cached predictions, and shap/lime fallbacks without coverage.【F:src/calibrated_explanations/explanations/explanation.py†L353-L607】

**Remediation tactics**
1. Build fixture-based tests that instantiate `CalibratedExplainer` with categorical labels, pyproject overrides, and custom plugin hints to cover initialization branches.
2. Expand validation tests using dummy pandas DataFrames to trigger shape mismatches, duplicate columns, and illegal percentile settings.
3. Test `Explanation` rendering helpers for thresholded/regression modes, ensuring caching branches (e.g., when prediction metadata is missing) behave as expected.

### 3. Plotting stack (modern + legacy)
* `_plots._plot_regression` performs numerous matplotlib operations, with early exits when `show=False` and no save path; none of the fill-between branches or axis labeling logic are validated through the new plot spec adapter.【F:src/calibrated_explanations/viz/plots.py†L700-L817】
* `_plots_legacy` still ships for backward compatibility yet lacks tests across its procedural drawing functions, including sampling-based label ordering and color interpolation.【F:src/calibrated_explanations/legacy/_plots_legacy.py†L217-L365】
* `viz/matplotlib_adapter.render` manages lazy imports, renderer fallbacks, DPI overrides, and multiple save-path workflows with little to no automated validation.【F:src/calibrated_explanations/viz/matplotlib_adapter.py†L54-L228】

**Remediation tactics**
1. Write headless matplotlib tests that use the Agg backend to assert figure creation, axis labeling, and colormap selections without requiring GUI display.
2. Cover the no-op branches (`show=False` with no `save_ext`) to keep fast paths stable in environments lacking matplotlib.
3. Backfill tests around the adapter to assert it honors DPI overrides, merges configuration dictionaries, and surfaces import errors consistently.
4. Treat `_plots_legacy` as deprecation candidate—either surround with targeted regression tests or mark sections for exclusion via `.coveragerc` once confirmed unused.

### 4. Plugin registry and CLI integration
* `plugins/registry.py` constructs composite plot plugins by pairing builders and renderers and exposes numerous descriptor listing utilities with trusted/untrusted filters, none of which are exercised.【F:src/calibrated_explanations/plugins/registry.py†L840-L910】
* CLI commands in `plugins/cli.py` depend on registry lookups and option parsing, but the branches handling invalid identifiers and default fallbacks remain untested.【F:src/calibrated_explanations/plugins/cli.py†L28-L138】
* `api/__init__.py` and `api/config.py` expose convenience imports and configuration hydration; failure modes when environment variables or pyproject sections are missing are uncovered.【F:src/calibrated_explanations/api/__init__.py†L53-L67】【F:src/calibrated_explanations/api/config.py†L94-L116】

**Remediation tactics**
1. Implement lightweight registry tests that register dummy descriptors and ensure trusted-only filters behave as documented.
2. Add CLI invocation tests via `CliRunner` (from `click.testing`) to exercise error messaging and JSON output modes.
3. Simulate missing config files and override environment variables to ensure the API layer surfaces descriptive exceptions.

### 5. Utilities and perturbation helpers
* `utils.helper.safe_first_element` and related utilities guard against ragged arrays, but the branches handling scalars vs. iterables are not asserted.【F:src/calibrated_explanations/utils/helper.py†L49-L88】
* Perturbation helpers expose multiple sampling strategies (Gaussian, uniform, salt-and-pepper) with dead branches for unsupported modes and deterministic seeding.【F:src/calibrated_explanations/utils/perturbation.py†L67-L205】
* Performance helpers in `perf/cache.py` and `perf/parallel.py` provide caching and parallelism toggles; their coverage deficits stem from missing unit tests for cache eviction and thread fallback logic.【F:src/calibrated_explanations/perf/cache.py†L1-L33】

**Remediation tactics**
1. Add focused unit tests for helper utilities covering scalar/list inputs, dtype preservation, and error paths.
2. Parameterize perturbation tests to validate each sampler and confirm deterministic behaviour under fixed seeds.
3. Mock timeouts and concurrency settings to assert caching/parallel wrappers respect configuration flags.

## Proposed remediation roadmap

| Phase | Scope | Target modules | Success criteria |
| --- | --- | --- | --- |
| Sprint 1 | Interval regression + validation foundations | `core.interval_regressor`, `core/calibrated_explainer` (init + plugin resolution), `core/validation` | Branch coverage for interval insertion and plugin resolution ≥80%; validation error branches exercised in new tests. |
| Sprint 2 | Registry + utility backfill | `plugins/registry`, `plugins/cli`, `api/config`, `utils/helper`, `utils/perturbation` | All public API/CLI entry points have smoke tests; helper/perturbation modules reach ≥90% line coverage. |
| Sprint 3 | Plotting stack | `_plots`, `_plots_legacy`, `viz/matplotlib_adapter`, `viz/builders` | Headless plotting tests cover both modern and legacy renderers; aggregate plotting coverage ≥85% with decisions on legacy exclusions documented. |
| Sprint 4 | Explanation rendering depth | `explanations/explanation`, `core/calibrated_explainer` (explanation caching paths) | Tests for regression vs. classification explanations, cache invalidation, and shap/lime fallbacks; explanation module coverage ≥90%. |

## Supporting actions
1. Introduce fixtures under `tests/conftest.py` for synthetic calibration datasets and dummy plugin descriptors to avoid repetition across new tests.
2. Capture matplotlib headless configuration in a shared fixture (Agg backend) to keep plotting tests fast and deterministic.
3. Extend `.coveragerc` to document intentional exclusions (e.g., optional third-party integrations) after auditing whether sections like `_plots_legacy` remain required in 2025.
4. Track coverage per phase by adding temporary thresholds for the touched modules (via `coverage report --fail-under=<module-threshold>` in CI) to prevent regressions while the plan is underway.
