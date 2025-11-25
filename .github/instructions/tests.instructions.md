---
applyTo:
  - "tests/**"
  - "**/*test*.{py,ts,tsx,js,java,cs}"
  - "**/*.spec.{ts,tsx,js}"
priority: 100
---

All actionable guidance for tests now lives in `.github/tests-guidance.md`. Follow that document for:

- Framework + style expectations (pytest + pytest-mock, behavior-first assertions)
- File creation and grouping policy (when you can/can't create new files, required directory mapping, PR justification)
- Content rubric (AAA structure, naming, determinism, mocking, snapshot usage)
- Coverage and tooling requirements, including the `pytest --cov=... --cov-fail-under=88` gate

If something is unclear in that document, call it out in your PR rather than inventing new local conventions. Additional historical context still exists in `improvement_docs/test_analysis/TEST_GUIDELINES_ENHANCED.md`.
