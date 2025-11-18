---
applyTo:
  - "tests/**"
  - "**/*test*.{py,ts,tsx,js,java,cs}"
  - "**/*.spec.{ts,tsx,js}"
priority: 100
---

**Framework:** pytest (+pytest-mock if already present). Do not add new testing frameworks.

Behavior vs implementation — rules of thumb
- Test observable outcomes (user-facing behavior), not implementation details. If a test would break under a pure refactor that preserves behavior, rewrite it.
- Quick decision: is it a user contract/invariant? → test it. Is it a private dict key or a specific helper call? → don't test directly.

Private helpers
- Avoid testing `_private` helpers directly. If a helper is a stable, reusable domain concept, make it public and test the public API.
- If only one module uses the helper, verify it through the public caller.

Snapshots / roundtrip + mocking
- Snapshots are fine for structural stability but always pair them with semantic assertions (serialize → deserialize → assert invariants).
- Use mocks to isolate slow I/O/network/RNG; assert on outcomes (not mock call details), and pair mock-heavy unit tests with at least one real integration test.

For deeper guidance see: `improvement_docs/test_analysis/TEST_GUIDELINES_ENHANCED.md`

