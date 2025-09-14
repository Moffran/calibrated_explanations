> **Scope:** Repository-wide rules Copilot must follow for any test generation or editing.

### 📐 Test File Creation & Grouping Policy

**Golden rule:** *Modify existing test files whenever possible. Creating new test files is a last resort.*

**Copilot must adhere to all of the following:**

1. **Default to appending** tests to the nearest existing file that matches the source under test (SUT) using the repo’s naming scheme (see mapping below). Do **not** open a new file if a suitable one exists.

2. **Only create a new test file when *all* of these are true:**

   * There is **no existing file** for the SUT’s module/component **and** placing tests into a neighboring file would mix unrelated concerns.
   * The existing file would exceed **~400 lines** or **~50 test cases** after changes, *or* it is already **slow/flake-prone** and should not grow.
   * The new tests are a **different scope** (e.g., integration vs unit) than the existing file’s scope.
   * The new file will **follow the directory structure and naming conventions** in the mapping below.

3. **Never fork file patterns.** Respect the canonical structure and naming rules—if unsure, ask in the PR description (see template) and propose 1 option only.

4. **Group by scope first, then by SUT.**

   * *Unit* tests live beside or under `tests/unit/` with one file per module/component.
   * *Integration* tests live under `tests/integration/` with one file per feature or external boundary.
   * *E2E* tests live under `tests/e2e/` grouped by user flow.

5. **Co-locate fixtures and helpers.** Reuse shared fixtures; do **not** duplicate them. If a fixture exists that fits, import it. Only create a new fixture file when it’s SUT-specific and not reusable elsewhere.

6. **Avoid introducing frameworks or mixing styles.** Use the repo’s existing test framework and style (assertion libs, runners, mock libraries). If any change is needed, propose it in the PR description; do not implement it ad‑hoc.

7. **Atomic diffs.** Keep changes small, focused, and limited to ⩽ 3 files unless adding a new, warranted test file per (2).

### 🗂 Directory & naming mapping (adjust to your stack)

> Copilot: infer the language from the edited file and apply the correct mapping.

**Python**

* Unit: `tests/unit/<package>/test_<module>.py`
* Integration: `tests/integration/<feature>/test_<feature>.py`
* E2E: `tests/e2e/<flow>/test_<flow>.py`

**TypeScript/JavaScript**

* Unit: `tests/unit/<feature>.spec.ts(x)|.test.ts(x)`
* Integration: `tests/integration/<feature>.spec.ts`
* E2E (Playwright/Cypress): `tests/e2e/<flow>.spec.ts`

**Java**

* Unit: `src/test/java/<pkg>/<ClassName>Test.java`
* Integration: `src/it/java/<pkg>/<ClassName>IT.java`

*(Add other stacks as needed with the same pattern.)*

### ✅ Test content rubric

* **Naming:** `should_<behavior>_when_<condition>`.
* **Arrange–Act–Assert:** enforce AAA structure; one logical assertion block per behavior.
* **Determinism:** no real network, clock, or randomness. Use fakes/mocks; freeze time; seed RNG.
* **Coverage:** prioritize branches & edge cases near recent changes; link each test to SUT function/class in a comment.
* **Performance:** unit tests < 100ms typical; integration < 2s typical. Skip or mark slow tests with existing project conventions.
* **Snapshots:** only for stable UI/serialization; keep snapshots small and focused.

### 🔁 When modifying existing tests

* Extend the **nearest** matching file and section for the SUT (e.g., class block, describe/Context).
* Keep **imports, fixtures, and style** consistent with the file.
* If the file has clear internal sections, add to the correct one; else create a minimal new section with a header comment.

### 🧾 PR description requirements (for any new test file)

Create a section titled **“Why a new test file?”** containing:

* Scope justification (unit/integration/e2e) and why existing file can’t be extended.
* Evidence (line count, number of cases, flakiness, or scope mismatch).
* Directory & name used from mapping above.

If any of the above are missing, **do not** create the new file; append to an existing one instead.

---

Optional: tweak thresholds (400 lines/50 cases) to your repo norms in a PR.
