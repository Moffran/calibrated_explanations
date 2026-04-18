> **Status note (2026-04-09):** Last edited 2026-04-09
> Archive after: Retain indefinitely as architectural record
> Implementation window: v0.11.1–v1.0.0

# ADR-035: CI Workflow Governance — CI Upgrade & Enforcement

Status: Accepted
Date: 2026-03-08
Deciders: Core maintainers
Reviewers: CI owners and governance maintainers
Supersedes: None
Superseded-by: None
Related: ADR-020, ADR-028, ADR-030

## Context

The repository has already migrated to modular GitHub Actions workflows with reusable primitives and a local reproducibility story (`make local-checks`). The implementation plan is documented in `docs/improvement/CI-upgrade.md`, including rollout, cleanup, least-privilege permissions, and path-gating expectations.

Without binding governance and automated enforcement, future PRs can reintroduce ad-hoc workflows that regress reproducibility, security, and CI feedback speed.

## Decision

### 1. Authoritative policy

This ADR is the authoritative policy for CI workflow governance for any change touching `.github/workflows/**`, `.github/actions/ci-policy/**`, or `scripts/local_checks.py`. `docs/improvement/CI-upgrade.md` remains the implementation appendix and migration playbook.

### 2. Merge blocking criteria

A PR that modifies CI-governed files MUST NOT merge unless all are satisfied:

1. `ci-policy/validate-workflows` succeeds,
2. CODEOWNERS approval for workflow/policy files is present,
3. PR includes CI checklist and short rationale.

### 3. CI policy rules

- **Reusable workflow first:** New entrypoints must call approved reusables (`reusable-python-test.yml`, `reusable-run-make.yml`, `reusable-build-docs.yml`) unless classified as experimental.
- **Least-privilege permissions:** default `contents: read`; write scopes only in approved maintenance workflows.
- **Pip constraints enforcement:** `pip install` in CI MUST include `-c constraints.txt` or a documented approved equivalent.
- **Heavy workload gating:** heavy jobs (`parity`, `perf`, `notebook-audit`, `docs`) MUST be path-gated and/or manual/scheduled (`workflow_dispatch` / `schedule`).
- **Local reproducibility parity:** CI changes that affect contributor-runnable checks MUST update `scripts/local_checks.py` and `Makefile` targets.
- **Cleanup process:** legacy workflow deletions should be grouped in a `ci:cleanup` PR and reference `docs/improvement/CI-upgrade.md`.

### 4. Exceptions and emergency path

- **Experimental workflows:** must be under `.github/workflows/experimental/` or include `experimental: true`, include expiry (<=30 days), and use label `ci-experimental`.
- **Urgent exception path:** `ci-exception` label + rollback/migration plan + two maintainer approvals. Exceptions are appended in ADR update notes.

### 5. Policy integrity

Changes to `.github/actions/ci-policy/**` are high-integrity and require two core maintainer approvals.

## Implementation

- Add local action `.github/actions/ci-policy/action.yml` and validator script `scripts/quality/validate_ci_policy.py`.
- Add workflow `.github/workflows/ci-policy.yml` to run on PRs touching CI-governed files.
- Add CODEOWNERS coverage for workflow and policy paths.
- Add CI PR template for mandatory checklist and audit metadata.

## Rollout

1. Advisory mode for approximately two weeks / two dev cycles.
2. Tune false positives and document accepted equivalent patterns.
3. Flip `ci-policy/validate-workflows` to required status check in branch protection.

## Consequences

**Positive**
- Codifies CI design constraints from CI-upgrade work.
- Prevents ad-hoc drift and insecure defaults.
- Improves local reproducibility and auditability.

**Negative / trade-offs**
- Increases PR overhead for CI-related changes.
- Requires active CI owner review capacity.
- Heuristic checks may need periodic maintenance.

## Implementation Appendix

Normative implementation details and migration sequence are documented in:

- `docs/improvement/CI-upgrade.md`
