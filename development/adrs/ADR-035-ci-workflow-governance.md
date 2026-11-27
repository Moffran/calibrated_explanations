---
id: "ADR-035"
title: "CI Workflow Governance \u2014 CI Upgrade & Enforcement"
status: "accepted"
description: "Governing architectural decision for CI workflow policy, job-level contracts, and the `validate_ci_policy` governance gate. The parity-reference scikit-learn pin (Task 7, v0.11.4) is an implementation milestone within this ADR's lifecycle."
superseded_by: []
---
> **Status note (2026-07-13):** Last edited 2026-07-13 (v0.11.6 Task 60 v1 consolidation)
> Archive after: Retain indefinitely as architectural record
> Implementation window: v0.11.1–v1.0.0

# ADR-035: CI Workflow Governance — CI Upgrade & Enforcement

Date: 2026-03-08
Deciders: Core maintainers
Reviewers: CI owners and governance maintainers
Related: ADR-020, ADR-028, ADR-030

## Context

The repository has already migrated to modular GitHub Actions workflows with reusable primitives and a local reproducibility story (`make local-checks`). The implementation plan was documented in `development/finished-work/CI-upgrade.md` (migration complete; retained as historical record), including rollout, cleanup, least-privilege permissions, and path-gating expectations.

Without binding governance and automated enforcement, future PRs can reintroduce ad-hoc workflows that regress reproducibility, security, and CI feedback speed.

## Decision

> **v1 consolidation note (2026-07-13, v0.11.6 Task 60):** sections 1–5 below
> were rewritten when the workflow inventory was consolidated from 14 files to
> 3 and the validator became blocking and full-inventory. The pre-consolidation
> rules (reusable-first, diff-based advisory validation, PR checklist
> metadata) are retained only in git history and in the migration record in
> `development/finished-work/CI-upgrade.md`.

### 1. Authoritative policy

This ADR is the authoritative policy for CI workflow governance for any change touching `.github/workflows/**`, `.github/actions/**`, `scripts/quality/validate_ci_policy.py`, or `scripts/local_checks.py`. `development/finished-work/CI-upgrade.md` is the implementation appendix and migration record.

### 2. Approved inventory (v1)

The complete top-level workflow inventory is exactly three files, each with one stated responsibility (`# Responsibility:` header):

| Workflow | Responsibility | Triggers | Write scopes |
|---|---|---|---|
| `ci.yml` | PR + push-to-main validation. Canonical PR gate = `make local-checks-pr` (min Python), full non-viz tests on newest Python, path-sensitive specialists (viz, docs, packaging, governance schema), always-run CI-policy job, aggregate `required` check. Main pushes add coverage + Codecov, package build + clean-wheel smoke, strict docs. | `pull_request`, `push` (main), `workflow_dispatch` | none |
| `scheduled.yml` | Weekly heavy assurance: full Python matrix, parity harnesses, core-vs-extras parity, perf regression, notebook execution, examples smoke, linkcheck, over-testing analysis, dependency audit. | `schedule` (weekly), `workflow_dispatch` | none |
| `maintenance.yml` | Manual performance-baseline refresh; opens a reviewable PR on a unique branch; requires a human reason. | `workflow_dispatch` only | `contents: write`, `pull-requests: write` (job-level) |

One composite action is approved: `.github/actions/setup-ce-python` (Python setup + constrained installs). Expanding either inventory is a deliberate governance decision: update `APPROVED_WORKFLOWS` / `APPROVED_ACTIONS` in `scripts/quality/validate_ci_policy.py` together with this ADR. A fourth normal workflow requires a documented responsibility that cannot fit the three above; convenience is not sufficient.

### 3. Blocking full-inventory validation (three identical layers)

`python scripts/quality/validate_ci_policy.py --full-inventory` validates the *complete* inventory (not a diff) and is blocking in repository code. The identical command runs in three layers so silent inventory expansion is caught everywhere:

1. the always-run `policy` job in `ci.yml`,
2. the `local-checks-pr` profile (`scripts/local_checks.py`, `make local-checks-pr` / `make check-ci-policy`),
3. the repo-local pre-commit hook scoped to `files: ^\.github/`.

The validator enforces at least: approved inventory only; no private-skill-repository or skill-sync references; no automated writes to agent skill directories; top-level `permissions: contents: read`; write scopes only in `maintenance.yml`; full-40-char SHA pins for external actions; constrained pip installs; a `# Responsibility:` header per workflow; heavy assurance scheduled/manual only; maintenance dispatch-only; no PR-opening write mechanism outside maintenance; no publishing (packages, tags, releases, docs deploys); no `continue-on-error`; job timeouts; PR-run concurrency cancellation (never for maintenance); unique stable workflow names; the `CI / required` aggregate check; and valid local-profile↔CI command mappings.

### 4. Merge blocking criteria

A PR that modifies CI-governed files MUST NOT merge unless:

1. the `CI / required` aggregate check succeeds (which includes the blocking full-inventory policy job),
2. CODEOWNERS approval for workflow/policy files is present.

### 5. Policy integrity

Changes to `scripts/quality/validate_ci_policy.py`, `.github/actions/**`, or this ADR's inventory table are high-integrity and require core maintainer approval (CODEOWNERS).

## Implementation

- `scripts/quality/validate_ci_policy.py` — stdlib-only blocking full-inventory validator (`--full-inventory`).
- `.github/workflows/ci.yml` `policy` job, `local-checks-pr` profile step, and `.pre-commit-config.yaml` `ci-policy-full-inventory` hook — the three guard layers.
- CODEOWNERS coverage for workflow and policy paths.
- Focused tests in `tests/scripts/test_validate_ci_policy.py` (accepted inventory + representative forbidden cases).

## Required check and remaining platform actions

- The single stable required branch-protection check is **`CI / required`** (workflow `CI`, job `required`). It fails when any required child job fails or is cancelled and treats legitimate path-skips as success; the workflow triggers on every PR so the check can never hang pending.
- **Platform action (repository administrators):** update branch protection to require exactly `CI / required` and remove the pre-consolidation required checks (the former `CI — Pull Request checks` jobs and `ci-policy/validate-workflows`). This is platform-governed and cannot be enforced from repository code; it remains the only outstanding admin step (carried over from the v0.11.3 re-evaluation record below).


## Governed claims

- `CE-CAP-CI-001` — CI workflow changes comply with reusable workflow, least-privilege, pinning, constraints, and local reproducibility policy.

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

- `development/finished-work/CI-upgrade.md` (migration complete; retained as historical record)

## v0.11.3 Re-evaluation Record (2026-06-02)

**Gap 1 — Advisory-to-required branch-protection flip:** Re-evaluated as required by `v0.11.3_plan.md` Task 9 Workstream D. Outcome: the advisory-to-required promotion of `ci-policy/validate-workflows` to a required branch-protection status check is a **platform-governed** setting that cannot be enforced from repository code alone. It requires repository administrator access and GitHub branch-protection rule changes that are outside the scope of PR-level governance. This is recorded as an **accepted operational constraint** for v0.11.3:

- The `ci-policy/validate-workflows` check runs on PRs touching CI-governed files and reports violations (advisory mode, Rollout step 1).
- The validator logic and CODEOWNERS coverage for workflow/policy paths are complete.
- Promotion to required status is recorded as a pending platform action for repository administrators; it is not a code or ADR gap.
- No milestone-blocking work remains in-repo; further promotion follows the Rollout plan in ADR-035 §Rollout when administrators apply the change.

This re-evaluation closes the v0.11.3 appendix gap with an accepted-constraint record.
