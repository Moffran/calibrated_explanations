---
id: "STD-003"
title: "Test Coverage Standardization"
status: "active"
description: "Engineering standard for the 90% coverage gate, coverage configuration, and `# pragma: no cover` annotation policy. Remains active as long as the coverage standard governs CE test strategy; revised when the coverage policy is updated."
superseded_by: []
---
> **Status note (2025-12-22):** Last edited 2025-12-22 · Archive after: Retain indefinitely as an engineering standard · Implementation window: Per Standard status (see Decision).

# Standard-003: Test Coverage Standardization

Formerly ADR-019. Reclassified as an engineering standard to keep ADRs scoped to
architectural or contract decisions.

Date: 2025-10-06
Deciders: Core maintainers
Reviewers: TBD

## Context

Pytest remains the regression harness. Coverage configuration is owned by
`pyproject.toml`; the canonical command is `make test-cov`, while
`make local-checks-full` and the main-branch CI coverage job enforce the
repository threshold. The PR profile runs the required behavioral suite without
repeating the heavier main-only coverage run.

## Decision

Adopt a layered coverage policy that couples numeric targets with risk-based exceptions,
while right-sizing enforcement for OSS development:

- **Package-wide floor:** Enforce **90% statement coverage** across
  `src/calibrated_explanations` through `pyproject.toml`, `make test-cov`, the
  full local profile, and the main-branch CI coverage job.
- **Critical paths:** Target **95% coverage** on calibrated prediction helpers, interval
  regression, serialization, and plugin registries. The repository-owned
  `scripts/quality/check_coverage_gates.py` verifies scoped gates from
  `coverage.xml`.
- **Documented exemptions:** Coverage omissions live in `pyproject.toml` and
  require a documented rationale. ADR-023 owns the matplotlib-adapter
  instrumentation exemption; do not create a separate `.coveragerc`.
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
- OSS contributions are not blocked by legacy coverage gaps.

Negative/Risks:
- Initial CI failures until legacy debt is addressed; requires remediation efforts.
- Slightly longer test runtime from additional reporting/threshold checks.
- Advisory-only enforcement can slow convergence without clear ownership.

## Adoption & Migration

1. Keep coverage settings in `pyproject.toml`.
2. Keep the local entry points in `Makefile` and `scripts/local_checks.py`.
3. Keep the main-only CI coverage command aligned with those settings.
4. Review additions to `[tool.coverage.run].omit` as documented exceptions.
5. Run `make test-cov` before changing a coverage threshold or exemption.

## Open Questions

- **Cadence:** Review and prune `pyproject.toml` omissions during release planning.
- **Subpackage Thresholds:** The critical-path list defined in the Decision section is sufficient for v1.0.0. Subpackage-specific thresholds are deferred to avoid excessive configuration maintenance.
- **Mutation Testing:** Defer to v0.11+ or later. While valuable, it is not a blocking requirement for v1.0.0 stability.

## Implementation Status

- 2025-10-06 – Standard accepted with a staged coverage-remediation objective.
- v1.0.0 – `pyproject.toml` owns the 90% floor; local and CI entry points use
  the same configuration.
- Current – main-branch coverage and scoped module gates are blocking; PR
  behavior validation remains in the canonical `local-checks-pr` profile.
