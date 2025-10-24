> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Archived as of v0.8.x delivery · Implementation window: Historical (≤v0.8.x).

# Test Coverage Assessment

## Summary finding
- **Overall severity: High.** The repository aspires to ~90% coverage in contributor guidance, yet the toolchain does not enforce any numeric threshold or even load the coverage plugin locally. Several legacy modules remain completely unreferenced by the test suite, so regressions in those files could ship unnoticed.

## Evidence
1. `pytest.ini` configures quiet runs without coverage flags, signalling that local executions do not exercise `pytest-cov` by default.【F:pytest.ini†L1-L17】
2. The GitHub Actions `test` workflow invokes `pytest --cov=src/calibrated_explanations` but omits `--cov-report` or `--cov-fail-under`, so the job succeeds even if coverage collapses; it merely uploads a report to Codecov without gating merges.【F:.github/workflows/test.yml†L33-L49】
3. Contributor docs set an aspirational ~90% coverage target, but provide no enforcement mechanism beyond the non-blocking Codecov dashboard.【F:CONTRIBUTING.md†L49-L58】
4. Static inspection shows core runtime code such as `calibrated_explanations.core.interval_regressor` lacks any direct test imports, indicating entire execution paths are effectively untested.【F:src/calibrated_explanations/core/interval_regressor.py†L1-L120】

## Contributing factors
- Coverage tooling depends on optional extras (`pytest-cov`) that are not installed via the base dev dependencies, so `pytest` invocations (and pre-commit hooks) default to running without coverage data.【F:pyproject.toml†L32-L47】
- Several integration tests rely on expensive model fitting; without targeted smoke tests for interval regressors and CLI entry points, contributors avoid expanding coverage in those areas due to runtime costs.

## Risk assessment
- **Likelihood:** High – lack of gating allows merges with reduced coverage. Untested modules include calibration backbones whose regressions can silently affect predictions.
- **Impact:** High – production-facing components (interval regression, plugin CLI) shape user-facing explanations; defects here would degrade calibration guarantees without detection.

## Recommendation snapshot
1. Adopt a coverage ADR that codifies minimum thresholds per layer (core/api/plugins/viz) and enforces them in CI.
2. Seed smoke/unit tests for currently unreferenced modules, prioritizing `core.interval_regressor`, plugin CLI entry points, and discretizer utilities.
3. Wire a fast local `pytest --cov --cov-fail-under=<threshold>` target into developer tooling (makefile/pre-commit) to keep drift visible before CI.
