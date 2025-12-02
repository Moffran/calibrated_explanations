> **Status note (2025-10-24):** Last edited 2025-10-24 · Archive after: Retain indefinitely as architectural record · Implementation window: Per ADR status (see Decision).

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
Legacy runtime modules (for example `core.interval_regressor`) remain effectively untested, so
confidence in calibration guarantees erodes as the code evolves.【F:pytest.ini†L1-L17】【F:.github/workflows/test.yml†L33-L49】【F:CONTRIBUTING.md†L49-L58】【F:src/calibrated_explanations/core/interval_regressor.py†L1-L120】

## Decision

Adopt a layered coverage policy that couples numeric thresholds with risk-based exceptions:

- **Package-wide floor:** Require `pytest --cov` runs to meet **90% statement coverage** across
  `src/calibrated_explanations`. CI will enforce this via `--cov-fail-under=90` when reaching the stable v1.0.0.
- **Critical paths:** Enforce **95% coverage** on calibrated prediction helpers, interval
  regression, serialization, and plugin registries by using `coverage report --fail-under` with
  per-path configuration.
- **Change-based gating:** Add a `coverage xml` step and integrate the Codecov “patch coverage”
  gate at **≥88%** for modified lines/files (raised from 85% as part of the v0.9.0 release).
  Pull requests that lower patch coverage below the
  threshold must justify waivers in the review checklist.
- **Documented exemptions:** Generated code, visualization golden files, and deprecated
  shims can be excluded via `.coveragerc` with explicit comments that describe the rationale
  and expiry date.
- **Public API guardrails:** Coverage thresholds MUST continue to exercise the
  WrapCalibratedExplainer contract (fit/calibrate/explain/predict flows,
  plotting helpers, uncertainty/threshold options). No part of the published
  API may be marked as deprecated or excluded from coverage unless a future ADR
  redefines the contract.

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
- Initial CI failures until legacy debt is addressed; requires remediation efforts.
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

- **Cadence:** Review and prune `.coveragerc` exemptions during the planning phase of each minor release (e.g., v0.10.0, v0.11.0).
- **Subpackage Thresholds:** The critical-path list defined in the Decision section is sufficient for v1.0.0. Subpackage-specific thresholds are deferred to avoid excessive configuration maintenance.
- **Mutation Testing:** Defer to v0.11+ or later. While valuable, it is not a blocking requirement for v1.0.0 stability.

## Implementation Status

- 2025-10-06 – ADR accepted alongside the coverage remediation plan and
  baseline assessment.
- v0.6.x – `.coveragerc` drafted with provisional exemptions and
  baseline metrics recorded to shape the remediation backlog while CI
  continues to run without fail-under gates.
- v0.7.0 – CI introduces `--cov-fail-under=80` with exit-zero preview
  reports, coverage dashboards are published, and contributor templates
  document the waiver workflow.
- v0.8.0 – Critical-path modules (`core`, calibration, serialization,
  registry) are raised to ≥95% coverage, Codecov patch gating at ≥85%
  becomes mandatory, and local tooling (`make test-cov`) mirrors the CI
  workflow.
- v0.9.0 – Package-wide floor raised to ≥88%, waiver inventory trimmed,
  Codecov patch gating tightened to ≥88%, and coverage enforcement is
  fully blocking on the release branch per the milestone gate.
- v1.0.0-rc – CI enforces the final ≥90% package floor, coverage
  dashboards become part of the release checklist, and branch protection
  rules require green coverage jobs before freeze.
- v1.0.0 – Stable release maintains ≥90% gating with scheduled audits of
  exemptions and telemetry-driven monitoring to detect regressions ahead
  of v1.0.x maintenance updates.
