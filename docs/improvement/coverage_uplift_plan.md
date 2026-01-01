> **Status note (2025-12-23):** Latest coverage run (`pytest --cov=src/calibrated_explanations --cov-report=term`) reached **89.5%** line coverage with `fail_under=88` satisfied.

# Coverage Uplift Plan (Standard-019)

This plan outlines the coverage standardization roadmap with module-level gap analysis, baselines, and remediation tactics.

## Tooling and gating roadmap

### Phase 0 – Tooling foundation
1. Add `.coveragerc` with package thresholds and documented excludes.
2. Update `pytest.ini` to include `--cov=src/calibrated_explanations --cov-report=term-missing --cov-fail-under=80` once coverage reaches 90%.
3. Ensure `pytest-cov` is in dev dependencies.
4. Provide `make test-cov` target for local runs.

### Phase 1 – Debt burn-down
1. Add unit tests for unreferenced modules, focusing on:
   - Core modules like interval regressor and prediction helpers.
   - Plugin CLI and registry resolution.
   - Discretizer edge cases.
   - External plugin installers.
2. Stabilize coverage above 90% with lightweight fixtures.

### Phase 2 – CI gating
1. Update CI to run coverage with XML upload and `fail_under=90`.
2. Enable Codecov patch gating at ≥88%.
3. Document waiver process in CONTRIBUTING.md.
4. Tie gating to releases.

### Phase 3 – Continuous improvement
1. Quarterly review of `.coveragerc` exemptions.
2. Track regressions in retrospectives.
3. Explore advanced testing like mutation testing at >95% coverage.

## Baseline snapshot
Current repository-wide coverage: **89.5%** line coverage (12,556 statements, 1,045 misses; branch coverage 89.5%).

| Module | Stmts | Miss | Coverage |
| --- | ---: | ---: | ---: |
| `plotting.py` | 788 | 169 | 73.8% |
| `viz/builders.py` | 423 | 42 | 86.1% |
| `explanations/explanation.py` | 1144 | 115 | 88.0% |
| `plugins/builtins.py` | 529 | 92 | 80.6% |
| `cache/cache.py` | 326 | 86 | 74.0% |
| `core/narrative_generator.py` | 247 | 50 | 76.1% |
| `parallel/parallel.py` | 363 | 30 | 88.6% |
| `core/wrap_explainer.py` | 392 | 41 | 89.2% |
| `plugins/registry.py` | 784 | 7 | 98.2% |
| `core/prediction_helpers.py` | 92 | 8 | 91.5% |

## Gap analysis by subsystem

### 1. Plotting and visualization
* Plotting router lacks coverage for style overrides, legacy fallbacks, and triangular plots.
* Viz builders miss uncertainty segments and ranking heuristics.
* Cross-platform save handling needs testing.

**Remediation:** Parameterize tests for overrides and builders; fix Windows assertions.

### 2. Explanations and core logic
* CalibratedExplainer init misses categorical handling and overrides.
* Explanation assembly lacks coverage for output switching and caching.
* Wrapper APIs have untested estimator paths.

**Remediation:** Add fixture-based tests for init variants and API exercises.

### 3. Plugins and registry
* Built-ins miss payload normalization branches.
* CLI lacks error path testing.
* External plugins untested.

**Remediation:** Test payload variations, CLI failures, and installer logic.

### 4. Helpers and gateways
* Prediction helpers miss validation branches.
* Lazy imports and deprecations untested.

**Remediation:** Cover validation paths and import behaviors.

### 5. Legacy surfaces
* Legacy plotting influences defaults but lacks tests.

**Remediation:** Decide on smoke tests or exclusions.

## Proposed remediation roadmap

| Iteration | Focus | Success criteria |
| --- | --- | --- |
| 1 | Plotting, viz builders, legacy paths | Router ≥80%, builders tested, cross-platform fixed |
| 2 | Core explainer, explanations, wrappers | Init and caching covered, APIs exercised |
| 3 | Plugins, builtins, CLI | Payloads and errors tested |
| 4 | Helpers, gateways, legacy | Validation and imports covered |

## Supporting actions
1. Add shared fixtures for payloads and datasets.
2. Normalize filesystem handling.
3. Update `.coveragerc` post-remediation.
4. Track thresholds to prevent regressions.
