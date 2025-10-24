> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Re-evaluate post-v1.0.0 maintenance review · Implementation window: v0.9.0–v1.0.0.

# Test Coverage Standardization Plan

This plan complements ADR-019 and breaks the remediation into concrete workstreams.

See also the [Test Coverage Gap Plan](test_coverage_gap_plan.md) for the module-level
backlog associated with the current coverage report.

## Phase 0 – Tooling foundation (Week 1)
1. Add `.coveragerc` with package/critical-path thresholds and explicitly documented excludes.
2. Update `pytest.ini` default `addopts` to include `--cov=src/calibrated_explanations --cov-report=term-missing --cov-fail-under=80` once debt burn-down reaches 90%.
3. Extend the `dev` optional dependency set so `pytest-cov` is installed for every contributor environment.【F:pyproject.toml†L38-L59】
4. Provide `make test-cov` (or `tox -e py-cov`) target mirroring CI invocation to ease local runs.

## Phase 1 – Debt burn-down (Weeks 2-4)
1. Author focused unit tests for the currently unreferenced runtime modules, prioritising:
   - `core.interval_regressor` happy-path predictions and error handling.【F:src/calibrated_explanations/core/interval_regressor.py†L1-L120】
   - Plugin CLI command smoke tests ensuring registry resolution works.
   - Discretizer utilities to cover edge cases around binning strategies.
   - External plugin extras installer (`external_plugins` folder, packaging extras) so aggregated installs remain tested.
2. Capture baseline coverage numbers and iterate until package-wide coverage stabilises above 90%.
3. Add lightweight fixtures (e.g., small sklearn datasets) so interval regression tests remain fast.

## Phase 2 – CI gating (Weeks 5-6)
1. Update `.github/workflows/test.yml` to run `pytest --cov=src/calibrated_explanations --cov-report=xml --cov-report=term --cov-fail-under=90` and upload the XML artifact for Codecov patch gating.【F:.github/workflows/test.yml†L33-L49】
2. Enable Codecov’s “patch coverage must be ≥85%” status check and make it required.
3. Document the waiver process in `CONTRIBUTING.md`, emphasising that waivers must link to follow-up issues.【F:CONTRIBUTING.md†L49-L58】

## Phase 3 – Continuous improvement (Ongoing)
1. Review `.coveragerc` exemptions quarterly, removing expired shims or adding TODO dates.
2. Track coverage deltas in release retrospectives and flag regressions >1% for root-cause analysis.
3. Explore mutation testing or fuzzing for calibration math once coverage stabilises above 95% on critical modules.
