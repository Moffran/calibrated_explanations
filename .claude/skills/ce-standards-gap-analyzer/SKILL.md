---
name: ce-standards-gap-analyzer
description: >
  Audit standards compliance across implementation and docs, reporting only
  unresolved gaps.
---

# CE Standards Gap Analyzer

You are verifying that the **intent and decisions** of a given STD are fully
realized in code and documentation. Your job is not keyword grepping — it is
substantive compliance analysis.

---

## Inputs

- STD identifier (e.g. `STD-004`), or `all` to sweep every STD in the appendix.
- Repository files (source code, tests, docs) accessible in workspace.

---

## Phase 1 — Understand the STD intent

1. **Read the full STD text** (`development/standards/STD-XXX-*.md`). Extract:
   - **Decisions** — what the STD mandates, forbids, or constrains.
   - **Requirements** — concrete deliverables (APIs, contracts, invariants,
     enforcement mechanisms, documentation, tests).
   - **Rationale** — why these decisions were made. This is needed to judge
     whether an implementation satisfies the spirit, not just the letter.
   - **Scope** — which modules, subsystems, or surfaces the STD governs.
2. Build a **checklist of required outcomes** from the decisions. Each outcome
   is a testable statement, e.g.:
   - "Plugin trust flags must be immutable after registration."
   - "RTD must document the fallback chain."
   - "CI must enforce the two-release deprecation window."

## Phase 2 — Verify implementation against intent

3. For each required outcome, **search and read the codebase** (`src/`,
   `tests/`, configuration files):
   - Read the relevant source files. Understand what the code actually does
     and whether it satisfies the STD requirement.
   - Check for invariant enforcement (assertions, validators, guards) where
     the STD requires them.
   - Check for correct API surfaces, signatures, and contracts where the STD
     specifies them.
   - Check for tests that exercise the required behavior.
4. For each required outcome, **check RTD and documentation** (`docs/`) when
   the STD has documentation requirements:
   - Verify that docs describe the behavior accurately and consistently with
     the implementation.
   - Flag docs that contradict the code or the STD text.
   - Flag missing documentation for STD-mandated surfaces.

## Phase 3 — Classify and score gaps

5. For each required outcome that is **not fully satisfied**, record a gap:
   - Violation impact (1–5): how seriously the gap violates the STD intent.
   - Code scope (1–5): breadth of code affected.
   - Unified severity = impact × scope.
   - One-line recommendation with file/line pointers where evidence was found
     (or expected but missing).
6. Mark outcomes that **are fully satisfied** as completed — these will be
   purged from the output (see Phase 4).

## Phase 4 — Record the gap report in the STD itself

There is no shared, continuously-updated Standards status appendix in
`development/current-work/` (the pre-v1.0.0 build-out appendix is archived at
`development/finished-work/RELEASE_PLAN_status_appendix.md` as a historical
record and is not updated further). Record gap results as a dated report
directly in the target Standard's own status note, following the pattern
already used by individual Standards (see the `> **Status note (YYYY-MM-DD):**`
line near the top of most STD files, e.g. STD-004).

Rules:

7. **Purge completed gaps.** When updating a Standard's own gap material:
   - **Remove every row whose gap is resolved.** Do not keep completed rows
     with zeroed-out scores — delete them entirely.
   - If a table mixes completed and open rows, rewrite it with **only the
     remaining open rows**, ranks renumbered from 1.
   - If **all** rows are completed, **replace the entire table** (header,
     separator, data rows) with a single compliance verification line.
8. **Only unresolved gaps appear.** Never list completed, resolved, or
   zeroed-out rows in the table.
9. **Date stamp — mandatory on every update.** Every time gap status is
   updated (single STD or full sweep), stamp today's date:
   - Compliance lines use format:
     `**Compliance verification (YYYY-MM-DD):** Reviewed code and RTD — no STD-XXX gaps found; STD-XXX is fully compliant. No further action required.`
   - Gap tables: add `_Last gap analysis: YYYY-MM-DD_` immediately above
     the table.
10. **If NO gaps remain** the compliance verification line must be
    unambiguous — it replaces the entire table and makes clear that no
    further action is required.
11. If the gap changes what is committed for the release currently being
    coordinated, propose a row update in the active
    `development/current-work/vX.Y.Z_plan.md`, or file/link a GitHub issue
    when no release is currently scoping this work.

---

## Key principles

- **Intent over keywords.** Do not rely on grepping for `TODO` or `COMPLETED`.
  Read the STD decisions, understand what they require, and verify that code
  and docs deliver. A missing `TODO` does not mean compliance; present code
  does not mean the STD intent is satisfied.
- **Substance over ceremony.** A gap exists when implementation diverges from
  what the STD decided — not when a keyword is absent. A gap is closed when
  code and docs genuinely satisfy the requirement — not when someone wrote
  `COMPLETED` next to it.
- **Conservative severity.** Prefer conservative estimates when evidence is
  ambiguous.
- **Evidence-based.** Every gap claim must cite specific files and lines (or
  their absence). Every compliance claim must cite the evidence that satisfies
  the STD requirement.
- **Keep the Standard's own gap material tidy.** No duplicate gap entries within a Standard.

---

## Files to read

```
development/standards/STD-XXX-*.md          ← the STD itself (primary source of intent and gap-report target)
development/current-work/vX.Y.Z_plan.md     ← update a row if the gap changes the active release's scope
src/                                         ← implementation evidence
tests/                                       ← test coverage of STD requirements
docs/                                        ← RTD evidence (when STD has doc requirements)
```

---

## Notes and constraints

- This skill is an evidence-gathering assistant, not an authoritative arbiter.
  Impact/scope ratings are suggestions for STD owner review.
- The date is always stamped — there is no opt-out.
- When sweeping all STDs, process each section independently and apply the
  same four phases to every one.
