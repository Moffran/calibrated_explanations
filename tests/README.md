# Test Suite Policy & Checklist

This repository enforces strict guardrails for test creation and editing. All contributors must follow the rules in `.github/copilot-instructions.md` and related instruction files.

## Test File Creation & Grouping Policy
- **Golden rule:** Extend existing test files whenever possible. Creating new test files is a last resort.
- **Directory mapping:**
  - Unit: `tests/unit/<package>/test_<module>.py` (preferred for new tests)
  - Integration: `tests/integration/<feature>/test_<feature>.py`
  - E2E: `tests/e2e/<flow>/test_<flow>.py`
  - Existing tests are under top-level `tests/` — this is accepted as canonical for now.
- **File creation gate:** Only create a new test file if:
  - No suitable file exists for the SUT
  - Adding to an existing file would exceed ~400 lines or ~50 test cases
  - Scope differs (unit vs integration)
  - New file follows the directory/naming mapping
- **Reuse fixtures and helpers:** Import from `conftest.py` or shared helpers. Do not duplicate fixtures.
- **Test style:**
  - Use `pytest` function style
  - Name tests: `test_<func>__should_<behavior>_when_<condition>`
  - Prefer `@pytest.mark.parametrize` for input matrices
  - Keep tests deterministic (no real network, clock, or randomness)
  - Use AAA (Arrange–Act–Assert) structure

## Checklist for Adding/Editing Tests
- [ ] Extend an existing test file unless all new-file criteria are met
- [ ] Use the correct directory and naming mapping for the language
- [ ] Reuse fixtures and helpers; do not duplicate them
- [ ] Keep diffs minimal and focused on the SUT under edit
- [ ] If creating a new test file, add a "Why a new test file?" section to your PR description (see `.github/copilot-instructions.md`)
- [ ] Use deterministic test patterns (seed RNG, mock time/network)
- [ ] Use AAA structure and clear test names

## Reference
- See `.github/copilot-instructions.md` for full policy
- See `.github/instructions/tests.instructions.md` and `.github/instructions/python-tests.instructions.md` for details
- See `.github/prompts/generate-tests-strict.prompt.md` for the strict test generation prompt

## Migration Notes
- Existing tests under `tests/` are accepted as canonical for now
- New tests should be placed under `tests/unit/`, `tests/integration/`, or `tests/e2e/` as appropriate
- Large files (>400 lines or >50 cases) should be split by feature area
- Shared fixtures should be moved to `conftest.py` or `tests/_fixtures.py`

---
For questions, see the improvement docs or ask a test steward.

## Faster Local Runs

- The test fixtures include a small CSV read cache to reduce repeated disk IO when many tests load the same datasets.
- To run a faster test session that uses smaller sample sizes, use the helper script:

```powershell
# From the repository root (PowerShell)
.
\scripts\run_fast_tests.ps1
```

This sets `FAST_TESTS=1` and `SAMPLE_LIMIT=200` which makes fixtures return smaller datasets and skips slow tests marked `@pytest.mark.slow`.
