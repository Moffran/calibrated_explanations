---
applyTo:
  - "tests/**"
  - "**/*test*.{py,ts,tsx,js,java,cs}"
  - "**/*.spec.{ts,tsx,js}"
priority: 100
---


**Framework:** pytest (+pytest-mock if already present). Do not introduce new libraries.

**Where to put tests**
- Unit → `tests/unit/<package>/test_<module>.py` (append to existing file if present)
- Integration → `tests/integration/<feature>/test_<feature>.py`

**File creation gate**
- Create new file only if: no suitable file exists **and** post-change size would exceed ~400 lines/50 cases **or** scope differs (unit vs integration).

**Style**
- Use `pytest` style functions; avoid `unittest.TestCase` unless already used in the target file.
- Name tests: `test_<func>__should_<behavior>_when_<condition>`.
- Prefer `@pytest.mark.parametrize` over loops.
- Use `freezegun`/time-freeze patterns if present; otherwise stub the clock.
- No I/O, env, or network in unit tests—use monkeypatch/mocks.
- Prefer testing behaviour over implementation details.

**Fixtures**
- Import shared fixtures from `conftest.py` or existing fixture modules; only create a new fixture file when SUT-specific and not reusable.

**Examples**
- If editing `pkg/module.py`, target `tests/unit/pkg/test_module.py`.
- If adding a new integration around an HTTP client, target `tests/integration/http/test_client.py`.

**You are generating or editing tests. Follow this policy strictly:**

1. Prefer amending existing files over creating new ones.
2. Only create a new file if **all** creation criteria in the repo policy are satisfied.
3. Use the correct directory and naming mapping for the detected language.
4. Reuse fixtures/helpers; do not duplicate them.
5. Keep diffs minimal: focus the change on the SUT under edit.
6. If a rule would be violated, output a short justification block for the PR and then proceed by **modifying an existing file instead**.


**Snapshots**
- Keep snapshots minimal and stable; prefer explicit assertions for logic.
