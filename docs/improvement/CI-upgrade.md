# CI Upgrade Operations Guide

## Scope and authority

- `kristinebergs/calibrated_explanations` CI is a **validation + artifact build verification** surface.
- `Moffran/calibrated_explanations` remains authoritative for versions, tags, GitHub releases, PyPI publication, changelog, security advisories, and documentation.
- Nothing in this document changes release authority allocation.

## Current architecture (state as of 2026-04-06)

Modular CI is the current architecture, not a future target.

- Fast PR lane: `.github/workflows/ci-pr.yml`
- Full PR viz/parity lane (path/manual): `.github/workflows/ci-full.yml`
- Main-branch safety and drift detection: `.github/workflows/ci-main.yml`
- Nightly/scheduled heavy jobs: `.github/workflows/ci-nightly.yml`
- Legacy duplicate wrappers have been decommissioned; the mapping table below is retained as migration evidence.

## Required PR path (critical lane)

The required PR lane is intentionally narrow and fast. Required checks are:

1. Lint
2. MyPy
3. Core tests
4. Private-member scan
5. Anti-pattern audit
6. Governance-event schema checks **only when governance paths are touched**

Heavy/manual/scheduled checks stay off the critical PR path unless explicitly promoted in planning docs:

- perf guard
- notebook execution
- full viz-focused checks
- over-testing density
- other heavy/manual/scheduled jobs

## Notebook execution policy

- Blocking on release branches only.
- Advisory/manual/non-blocking outside release-boundary contexts unless explicitly promoted by a milestone plan.
- `notebook-audit` strict mode (`--check`) is release-branch enforcement; non-release contexts remain advisory.

## Packaging verification policy (validation only)

For the development-mirror CI role, packaging verification means:

1. Build wheel and sdist.
2. Install from built artifacts in a clean environment.
3. Inspect built artifact contents.

This is a verification gate only and does not authorize publication.

## Local reproduction policy

Two-tier local reproduction is mandatory:

- Routine work: `make local-checks-pr`
- Milestone closure or branch-gate changes: `make local-checks`

CI/planning changes must keep local checks aligned (`scripts/local_checks.py` + Make targets).

## Branch protection policy

Branch protection should require:

- fast PR lane checks
- selected safety checks

Do **not** require nearly every non-nightly job. Keep required checks practical and stable.

## Workflow/check-name freeze policy during migration

- Workflow names and required check names are frozen during migration.
- Any rename must include, in the same change:
  1. branch-protection update,
  2. mapping-table update in this document,
  3. validation evidence that required checks still map 1:1.

## Legacy migration and decommission policy

1. Legacy workflows fall into two classes:
   - **Removal-eligible duplicates**: wrappers with complete replacement and no active parity purpose.
   - **Parity-retained legacy workflows**: intentionally retained for comparison/evidence.
2. Parity-retained legacy workflows **MUST NOT** be removed until their parity purpose is formally retired in planning docs.
3. CI migration readiness requires:
   - quantitative evidence, and
   - maintainer judgment.

Quantitative evidence is necessary but not sufficient on its own.

## Legacy → replacement mapping (mandatory control table)

| Legacy workflow file | Legacy required check name(s) | Replacement workflow file | Replacement required check name(s) | Retained for parity? | Artifact/check parity expectation | Validation status | Removal eligibility |
|---|---|---|---|---|---|---|---|
| `.github/workflows/test.yml` | `test (compat wrapper)`; `Anti-pattern Audit` | `.github/workflows/ci-pr.yml`, `.github/workflows/ci-main.yml`, `.github/workflows/ci-full.yml` | `CI — Pull Request checks / core-tests`; `CI — Pull Request checks / Anti-pattern Audit (ADR-030)`; `CI — Main branch gates / perf-guard`; `CI — Full PR checks (includes viz)` | No | Required PR + selected safety checks fully covered by modular workflows. | Completed | Removed |
| `.github/workflows/coverage.yml` | `Coverage (wrapper) / coverage` | `.github/workflows/ci-main.yml` | `CI — Main branch gates / core-with-coverage` and per-module coverage gate jobs | No | Coverage thresholds and reports matched branch-gate expectations. | Completed | Removed |
| `.github/workflows/examples.yml` | `Examples QA / examples` | `.github/workflows/ci-nightly.yml` | `CI — Nightly heavy jobs / examples-smoke` | No | Example smoke validity retained on nightly/manual schedule. | Completed | Removed |
| `.github/workflows/scan-private-members.yml` | `Scan Private Members in Tests / scan-private-members` | `.github/workflows/ci-pr.yml` | `CI — Pull Request checks / private-member-scan` | No | PR-path private-member enforcement parity confirmed. | Completed | Removed |
| `.github/workflows/notebook-audit.yml` | `notebook-audit / audit` | `.github/workflows/ci-nightly.yml` + release-branch notebook blocking policy | `CI — Nightly heavy jobs / notebook-audit` (advisory outside release boundary) | No | Release-branch blocking + non-release advisory split preserved. | Completed | Removed |
| `.github/workflows/docs.yml` | `docs / build` | `.github/workflows/ci-main.yml` and local milestone docs gates | `CI — Main branch gates / core-with-coverage` plus milestone docs gating policy | No | Docs gate retained without standalone duplicate wrapper. | Completed | Removed |
| `.github/workflows/lint.yml` | `Lint / lint` | `.github/workflows/ci-pr.yml` | `CI — Pull Request checks / lint` | No | Same lint stack and ADR checks covered in required PR lane. | Completed | Removed |
| `.github/workflows/mypy.yml` | `mypy / typecheck` | `.github/workflows/ci-pr.yml` | `CI — Pull Request checks / mypy` | No | Same typed-scope check retained in required PR lane. | Completed | Removed |
| `.github/workflows/dependency-audit.yml` | `dependency-audit / pip-audit` | `.github/workflows/ci-main.yml` local/milestone safety checks | Dependency audit retained as non-critical safety check outside required PR lane | No | Safety check retained without duplicate standalone wrapper. | Completed | Removed |

## Migration exit criteria (evidence-based)

A removal-eligible legacy wrapper may be removed only when all criteria below are met:

1. **Consecutive green runs:** at least 10 consecutive green replacement runs.
2. **Representative PR coverage:** evidence across at least 5 PRs spanning docs, runtime, tests, and workflow-touching changes.
3. **Check-name parity:** required check names are present and match branch protection expectations.
4. **Artifact/check parity:** expected artifacts and gate outcomes match legacy behavior for the same change set.
5. **No open parity defects:** any mismatch tickets are resolved or explicitly waived with dated rationale.
6. **Maintainer judgment recorded:** explicit go/no-go call recorded in the active milestone plan.

## Operational process

1. Keep mapping table current with every workflow/check change.
2. Do not reintroduce standalone duplicate wrappers once modular replacements exist.
3. Update branch-protection requirements only after replacement checks are proven and names are frozen.
4. Any new parity-retention exception must be explicitly approved in planning docs before adding a duplicate workflow.

## Local/CI parity maintenance

When workflow entrypoints change under `.github/workflows/`:

- update `scripts/local_checks.py` for equivalent local reproduction,
- keep `make local-checks-pr` and `make local-checks` behavior aligned,
- document changed commands in the milestone plan/checklist.
