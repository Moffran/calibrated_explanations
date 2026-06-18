> **Finished:** Standalone release checklist superseded by the `## Release preparation` section that is now part of each `vX.Y.Z_plan.md` milestone file. Retained as a historical record of the gate format used through v0.11.3. Do not update; consult the active milestone plan instead.

# Release Checklist (historical — retired after v0.11.3)

This checklist tracked compliance with release gates defined in `development/current-work/RELEASE_PLAN_v1.md`.
It was reviewed and completed for every pull request that touched the user-facing API or documentation.

Replaced by the unnumbered `## Release preparation` section at the end of each milestone plan (see `development/finished-work/v0.11.3_plan.md` Task 18 as the template reference).

---

## CI role and local reproduction

- **Mirror CI scope confirmed?**
  - [ ] Yes — validation + artifact build verification only
  - [ ] Yes — authoritative release/publication remains `Moffran/calibrated_explanations`
- **Local checks run with correct tier?**
  - [ ] `make local-checks-pr` (routine PR validation)
  - [ ] `make local-checks` (milestone closure or branch-gate changes)

## Packaging verification (validation only)

- **Build artifacts produced?**
  - [ ] Yes (wheel + sdist)
- **Install from built artifacts in clean environment?**
  - [ ] Yes
- **Artifact contents inspected?**
  - [ ] Yes

## Legacy Stability (ADR-020)

- **Legacy contract touched?**
  - [ ] No
  - [ ] Yes (requires verification against `development/finished-work/legacy_user_api_contract.md`)
- **Notebooks audited?**
  - [ ] Yes (run `python scripts/quality/audit_notebook_api.py notebooks --check`)
  - [ ] No (only allowed if notebooks are untouched)
- **API snapshot updated?**
  - [ ] N/A (no public API changes)
  - [ ] Yes (run `python scripts/quality/snapshot_public_api.py`)

## Documentation & Gallery (ADR-012)

- **Documentation builds?**
  - [ ] Yes - without warnings
- **Gallery examples execute? (within time ceilings)**
  - [ ] Yes
- **Notebook execution policy applied correctly?**
  - [ ] Yes — release branches blocking
  - [ ] Yes — non-release contexts advisory/non-blocking unless explicitly promoted

## Test Quality (ADR-030)

- **Anti-patterns check** (`python scripts/anti-pattern-analysis/detect_test_anti_patterns.py --check --baseline .github/test-quality-baseline.json`)
  - [ ] Passed

## Formatting & Standards

- **Linting** (`pre-commit run --all-files`)
  - [ ] Ruff passes
- **Typing**
  - [ ] MyPy passes
