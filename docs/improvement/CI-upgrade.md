# CI Upgrade Plan

This document describes the planned reorganization of GitHub Actions workflows for the project, the steps to decommission older workflows, and the branch-protection flip procedure to be executed before the v1.0.0-rc release.

Summary
- Move to a modular CI layout using reusable workflows under `.github/workflows/reusable/` and thin top-level entrypoints under `.github/workflows/`.
- Preserve current capabilities: PR fast checks, full PR checks (viz), parity reference, nightly/perf runs, audits, and maintenance tasks.
- Run old CI in parallel with the new layout during validation; remove old workflows before the `v1.0.0-rc` freeze.

Design and rationale
- Reusable workflows centralize Python setup, cache keys, and installation rules (always using `-c constraints.txt` where pip installs occur).
- Top-level entrypoints are small orchestration nodes that call the reusables; this reduces repeated install and matrix logic.
- Path filters and `concurrency` groups reduce unnecessary runs and cancel in-progress superseded runs.
- Jobs run with least-privilege `permissions` (default `contents: read`), escalating to `write` only in maintenance/update workflows.

Implementation notes (local trial)
- Pip caching is standardized via `actions/setup-python` with `cache: pip` and explicit `cache-dependency-path` entries.
- `CI — Full PR checks (includes viz)` is path-gated for viz/plot-related changes and can always be run manually via `workflow_dispatch`.
- The manual maintenance baseline task generates micro benchmark baselines via `scripts/micro_bench_perf.py` and can optionally open a PR.

Workflows introduced
- `.github/workflows/reusable/python-test.yml` — canonical python setup + pytest runner.
- `.github/workflows/reusable/run-make.yml` — standardized `make` target runner.
- `.github/workflows/reusable/build-docs.yml` — Sphinx/linkcheck builder.
- `.github/workflows/ci-pr.yml` — fast PR checks (lint, mypy, core tests matrix).
- `.github/workflows/ci-full.yml` — viz-focused PR checks (path-gated) + manual parity run.
- `.github/workflows/ci-nightly.yml` — scheduled heavy jobs (parity, notebook audits, example smoke runs).
- `.github/workflows/ci-main.yml` — main-branch gates (coverage upload placeholder, perf-guard, anti-pattern audits).
- `.github/workflows/maintenance.yml` — manual maintenance actions (update-baseline via micro benchmarks, regen-docs).

Workflows to be decommissioned before v1.0.0-rc
(note: names refer to files in `.github/workflows/`)
- `test.yml` — replaced by `ci-pr.yml` + `ci-full.yml`; kept as a compat wrapper during rollout but scheduled for removal.
- `coverage.yml` — replaced by `ci-main.yml`/coverage wrapper; remove when coverage upload + gates confirmed working in new pipeline.
- `examples.yml` — functionality moved into `ci-nightly.yml` / top-level example checks; remove old file after validation.
- `docs.yml` — replaced by new `docs` entrypoint (already cleaned); old variants removed.
- `dependency-audit.yml`, `notebook-audit.yml`, `scan-private-members.yml`, `update_baseline.yml` — keep as scheduled/manual wrappers but unify to use reusables; remove legacy duplicates.

Rollback & validation window
1. Deploy new workflows in a feature branch and/or to `main` with no branch-protection changes.
2. Let both old and new pipelines run in parallel for at least 3 full dev cycles (recommended: 2 weeks) to surface differences and flaky tests.
3. Fix any issues; iterate on reusables where per-job needs diverge.
4. When new pipelines are stable and produce the same artifacts and checks, schedule branch-protection flip (see below).

Branch-protection flip checklist (flip to new CI)
- Identify required checks to replace: typically `test (compat wrapper)`, `lint`, `mypy`, `coverage`, `examples`, `docs`.
- Create an explicit mapping from old check names to new check names and verify exact job names appearing in GitHub Actions UI.
- In a maintenance window:
  1. Add new required checks to branch protection (do not remove old yet).
  2. Wait for two green runs of the new checks on `main`.
  3. Remove old checks from branch protection.
  4. Monitor for 48 hours; if any regressions occur, revert branch-protection changes and investigate.
- Communicate the change via README/CHANGELOG and create a short PR template note describing the new checks.

Removal schedule
- Immediately after successful validation and after branch-protection flip: delete the old workflow files listed above in a single PR that documents the removal and references this `CI-upgrade.md` and the release plan.
- Tag the removal PR with `ci:cleanup` and require one approving review from core maintainers.

Operational notes
- Always use `-c constraints.txt` for pip installs in CI to avoid dependency drift in the runner environment.
- Keep heavy workloads (parity, perf) scheduled or manual where sensible; avoid running them on every PR unless path filters / labels request it.
- Keep `workflow_dispatch` available on heavy jobs to assist debugging.
- Ensure all reusables emit informative logs and warnings for fallback paths (per the repo's fallback visibility policy).

Contact
- For questions or to request the validation window be extended, contact the release manager and CI owner listed in `GOVERNANCE.md`.
