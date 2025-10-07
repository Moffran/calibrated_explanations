# ADR-019: Test Coverage Standardization

Status: Accepted
Date: 2025-10-06
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None

## Context

Pytest is the primary regression harness for the package, yet neither local defaults nor
continuous integration enforce minimum coverage. `pytest.ini` runs tests quietly without
loading `pytest-cov`, while the CI workflow executes `pytest --cov=src/calibrated_explanations`
without a `--cov-fail-under` threshold. Contributors are asked to target roughly 90% coverage,
with reports uploaded to Codecov, but there is no guardrail preventing significant regressions.
Legacy runtime modules (for example `_interval_regressor`) remain effectively untested, so
confidence in calibration guarantees erodes as the code evolves.【F:pytest.ini†L1-L17】【F:.github/workflows/test.yml†L33-L49】【F:CONTRIBUTING.md†L49-L58】【F:src/calibrated_explanations/_interval_regressor.py†L1-L120】

## Decision

Adopt a layered coverage policy that couples numeric thresholds with risk-based exceptions:

- **Package-wide floor:** Require `pytest --cov` runs to meet **90% statement coverage** across
  `src/calibrated_explanations`. CI will enforce this via `--cov-fail-under=90` when reaching the stable v1.0.0.
- **Critical paths:** Enforce **95% coverage** on calibrated prediction helpers, interval
  regression, serialization, and plugin registries by using `coverage report --fail-under` with
  per-path configuration.
- **Change-based gating:** Add a `coverage xml` step and integrate the Codecov “patch coverage”
  gate at **≥85%** for modified lines/files. Pull requests that lower patch coverage below the
  threshold must justify waivers in the review checklist.
- **Documented exemptions:** Generated code, visualization golden files, and deprecated
  shims can be excluded via `.coveragerc` with explicit comments that describe the rationale
  and expiry date.

## Alternatives Considered

1. **Status quo (Codecov dashboards only).** Rejected because it allows silent regressions and
   does not give reviewers an actionable pass/fail signal.
2. **Per-module 100% coverage.** Rejected as unrealistic for plotting backends and third-party
   wrappers, potentially discouraging contributions.
3. **Runtime smoke-only checks.** Rejected; these do not measure statement coverage and fail
   to capture unexecuted branches in calibration math.

## Consequences

Positive:
- Quantitative gate keeps critical calibration logic exercised by tests before release.
- Contributors receive immediate feedback locally and in CI when coverage slips.
- Patch coverage guard discourages untested features while permitting incremental debt paydown.

Negative/Risks:
- Initial CI failures until legacy debt is addressed; requires remediation sprints.
- Slightly longer test runtime from additional reporting/threshold checks.

## Adoption & Migration

1. Land this ADR and announce during contributor sync and release notes.
2. Introduce a shared `.coveragerc` that encodes thresholds and named exemptions.
3. Update CI (`test.yml`) to run `pytest --cov=src/calibrated_explanations --cov-report=xml \
   --cov-report=term --cov-fail-under=90` and pass the XML to Codecov with patch gating enabled.
4. Add a `make test-cov` (or invoke via `tox` target) so developers can trigger the same checks
   locally; ensure the dev extra installs `pytest-cov` by default.
5. Complete remediation tasks outlined in the coverage improvement plan so that historical debt
   does not block adoption.

## Open Questions

- What cadence should we use to review and prune `.coveragerc` exemptions?
- Do we require subpackage-specific thresholds (e.g., `viz` vs `core`), or is the
  critical-path list sufficient?
- Should we combine coverage gating with mutation testing for calibration modules in a
  future phase?

## Implementation Status

- 2025-10-06 – ADR accepted alongside the coverage remediation plan.
