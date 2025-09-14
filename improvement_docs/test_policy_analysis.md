# Test Suite Analysis & Cleanup Plan (2025-09-14)

## Summary of Current State
- Tests are all under top-level `tests/`, not `tests/unit/<package>` as per new policy mapping. This is consistent repo-wide and can be accepted as canonical for now.
- Shared fixtures are in `tests/conftest.py`, but many test modules define their own dataset fixtures, causing duplication.
- Several test files exceed the policy thresholds (>400 lines, >50 cases), notably `test_wrap_regression.py` and `test_wrap_classification.py`.
- Test style is good: pytest functions, parametrize, deterministic, mocking for plotting and expensive ops.
- Helpers are sometimes defined in test modules and imported elsewhere, which is brittle.

## Issues vs. Policy
- **A. Layout:** Not under `tests/unit/<package>`, but repo convention is top-level `tests/`.
- **B. Duplicate fixtures:** Dataset fixtures are duplicated across modules.
- **C. Large files:** Some files exceed size/case thresholds.
- **D. Helpers:** Shared helpers are defined in test modules, not a central helpers file.
- **E. Enforcement:** No CI guard for new test files or PR justification.

## Cleanup & Alignment Plan
**Phase 0 — Quick Wins**
- Add CI guard job to require PR justification for new test files and verify placement.
- Add `tests/_helpers.py` for shared helpers.
- Add CONTRIBUTING checklist or README snippet referencing `.github/copilot-instructions.md`.

**Phase 1 — Fixtures Consolidation**
- Migrate dataset fixtures to `conftest.py` or `tests/_fixtures.py`.
- Replace duplicated fixtures in test modules with imports.

**Phase 2 — Split Large Files**
- Split large files into smaller modules grouped by feature area.
- Aim for <400 lines and <50 cases per file.

**Phase 3 — QA & Performance Tuning**
- Mark heavy tests as `@pytest.mark.integration` or `@pytest.mark.slow`.
- Add/Refine test markers in `pytest.ini`.

**Phase 4 — Ongoing Guardrails & Developer UX**
- Add `CODEOWNERS` for `tests/**`.
- Add PR template checklist for new test files.
- (Optional) Add pre-commit hook for test file naming/placement.

## Future Test Development Actions
- Adopt per-scope directories for new tests (`tests/unit/`, `tests/integration/`, etc.).
- Add test templates for AAA pattern and deterministic tests.
- Add performance markers and nightly jobs for benchmarks.
- Add property-based tests for calibration invariants (optional).
- Add coverage metrics to CI.

## Risks & Mitigations
- Splitting files and moving fixtures may cause transient failures; mitigate with incremental PRs and local test runs.
- Moving fixtures may cause name collisions; use explicit names and conservative scopes.
- Enforcing strict mapping would require large reorg; accept current layout for now, require new tests to follow mapping.

## Immediate Deliverables
- Add CI guard job.
- Create `tests/_helpers.py` and migrate helpers.
- Create `tests/_fixtures.py` and migrate one dataset fixture.
- Add `CODEOWNERS` and `tests/README.md`.

---
Analysis performed on 2025-09-14. See `.github/copilot-instructions.md` for policy details.
