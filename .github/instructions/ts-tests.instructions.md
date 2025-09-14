---
applyTo:
  - "**/*.{ts,tsx,js}"
  - "tests/**"
priority: 120
---

**Framework:** Jest or Vitest (match existing). Do not mix both.

**Where to put tests**
- Unit → `tests/unit/<feature>.spec.ts(x)` (or `.test.ts(x)` if that is the repo norm)
- Integration → `tests/integration/<feature>.spec.ts`
- E2E → `tests/e2e/<flow>.spec.ts`

**File creation gate**
- Create new file only if: no suitable file exists **and** adding to it would exceed ~400 lines/50 cases **or** scope differs.

**Style**
- `describe` blocks per module/feature; `it("should ...")` names.
- Use fake timers for time; mock network; no real FS/network in unit tests.
- Keep imports/framework match with existing file.

**Snapshots**
- Keep snapshots minimal and stable; prefer explicit assertions for logic.
