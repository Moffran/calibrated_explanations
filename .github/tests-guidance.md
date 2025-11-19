# Centralized Test Guidance

This repository uses a single source of truth for all test-related instructions. Whether you are writing tests manually, using Copilot, or invoking automation prompts, follow the guidance below without deviation unless a maintainer signs off in the PR description.

## Scope, Frameworks, and Style
- Python tests use `pytest` with existing fixtures and `pytest-mock` when mocking is required. Do not add alternative frameworks.
- Favor behaviorally focused assertions. A pure refactor that preserves behavior should not break your tests; if it would, rewrite the test.
- Avoid targeting `_private` helpers directly. Either elevate them to a public contract or cover them indirectly through public callers.

## Test File Creation & Grouping Policy
1. Start by identifying the source under test (SUT) and the nearest existing test file that matches it. Extend that file whenever possible.
2. Only create a new test file when **all** of these are true:
   - No appropriate file exists for the SUT and appending elsewhere would mix unrelated concerns.
   - The candidate file would exceed ~400 lines or ~50 test cases after your changes, or it is already flaky/slow.
   - The new tests have a different scope (unit vs integration vs e2e) than the existing file.
   - You can place the new file in the canonical path: `tests/unit/<package>/test_<module>.py`, `tests/integration/<feature>/test_<feature>.py`, or `tests/e2e/<flow>/test_<flow>.py`.
3. Group tests by scope first, then by SUT. Keep fixtures/helpers co-located and reuse shared fixtures rather than duplicating them.
4. Creating a new file requires a **“Why a new test file?”** section in the PR describing scope, justification, and exact path.

## Test Content Rubric
- **Naming:** prefer `should_<behavior>_when_<condition>` style.
- **Structure:** follow Arrange–Act–Assert with one logical assertion block per behavior.
- **Determinism:** no real network, clock, or randomness. Use mocks/fakes, freeze time, or seed RNG, and pair heavily mocked unit tests with at least one integration test.
- **Snapshots/Round-trips:** acceptable only for stable structures; always combine with semantic assertions (e.g., serialize → deserialize → verify invariants).
- **Coverage focus:** emphasize branches and edge cases near recent changes. Link each test to its SUT via a short comment if not obvious.
- **Performance:** keep unit tests <100 ms, integration tests <2 s when feasible. Mark slow tests using existing repo conventions.

## Coverage & Tooling Expectations
- Local gate: `pytest --cov=src/calibrated_explanations --cov-config=.coveragerc --cov-fail-under=88`.
- Lint/mypy requirements still apply to any touched modules; update docs when behavior changes.
- For deep domain context, see `improvement_docs/test_analysis/TEST_GUIDELINES_ENHANCED.md`.

Adhering to this document keeps Copilot policies, automation prompts, and human contributors aligned. If a scenario demands deviating, document the reasoning explicitly in the PR.
