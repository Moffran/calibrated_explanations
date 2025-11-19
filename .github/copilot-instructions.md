> **Scope:** Repository-wide defaults for Copilot when it touches any tests. **Always** pair these notes with `.github/tests-guidance.md`, which contains the authoritative rules.

## How to use the centralized guidance
- Before generating or editing tests, open `.github/tests-guidance.md` and follow every rule in the “Test File Creation & Grouping Policy”, “Test Content Rubric”, and “Coverage & Tooling Expectations” sections.
- If a change would deviate from that document (e.g., adding a new framework or file), stop and document the justification in the PR template section “Why a new test file?”.

## Quick reminders for Copilot
1. **Modify existing files first.** Locate the nearest test file for the SUT and extend it unless every creation criterion in `tests-guidance.md` is met.
2. **Respect scope + naming.** Use `tests/unit|integration|e2e/...` paths with `test_<module>.py` naming. Never fork new structures.
3. **Content rubric.** Output deterministic, AAA-structured pytest tests named `should_<behavior>_when_<condition>`, mirroring fixtures and style already in the file.
4. **Mocking & snapshots.** Mock only to avoid slow I/O and always assert behaviors, not mock-call details. Pair snapshots with semantic checks.
5. **Coverage context.** Prioritize edge cases tied to the change and keep an eye on the `pytest --cov=... --cov-fail-under=88` expectation from the PR checklist.

These reminders intentionally mirror the centralized document so Copilot stays in sync with humans and automation.
