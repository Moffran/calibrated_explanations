# /generate-tests-strict

All generation must comply with `.github/tests-guidance.md`. Summarize at the end which existing file you extended (or document the required “Why a new test file?” justification when you truly create one).

Key reminders pulled from the central doc:
- Identify the SUT and reuse the nearest existing pytest file unless every new-file criterion is satisfied.
- Keep naming and directories canonical (`tests/unit|integration|e2e/.../test_<module>.py`) and reuse fixtures/helpers instead of duplicating them.
- Tests must be deterministic, use Arrange–Act–Assert, and focus on behavior over implementation.

Inputs (optional):
- `target=<path/to/source_file>`
- `scope=unit|integration|e2e`
- `framework=pytest` (the only supported test framework here)
