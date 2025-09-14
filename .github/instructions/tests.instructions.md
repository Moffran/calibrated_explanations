---
applyTo:
  - "tests/**"
  - "**/*test*.{py,ts,tsx,js,java,cs}"
  - "**/*.spec.{ts,tsx,js}"
priority: 100
---

**You are generating or editing tests. Follow this policy strictly:**

1. Prefer amending existing files over creating new ones.
2. Only create a new file if **all** creation criteria in the repo policy are satisfied.
3. Use the correct directory and naming mapping for the detected language.
4. Reuse fixtures/helpers; do not duplicate them.
5. Keep diffs minimal: focus the change on the SUT under edit.
6. If a rule would be violated, output a short justification block for the PR and then proceed by **modifying an existing file instead**.
