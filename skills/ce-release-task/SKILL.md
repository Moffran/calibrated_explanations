---
name: ce-release-task
description: >
  Implement and verify one active-plan deliverable with its ADRs, tests, gates,
  and closure evidence.
---

# CE Release Task

You are identifying, implementing, and verifying a single deliverable from
the active version plan's `## Included work` table.

## Required references

- `development/current-work/vX.Y.Z_plan.md` (the sole active version plan)
- The deliverable's linked GitHub issue, if any
- Governing ADRs and standards for the selected deliverable
- `CONTRIBUTOR_INSTRUCTIONS.md` (coding and testing rules)

## Use this skill when

- Picking the next actionable row from the current version plan.
- Implementing a specific deliverable end-to-end.
- Verifying that a deliverable is firmly closed.

## Workflow

### Phase 1: Selection

1. Read the active `vX.Y.Z_plan.md`'s `## Included work` table.
2. Identify rows by Status:
   - **Done**: has verification evidence (tests green, code merged).
   - **In progress**: partially implemented.
   - **Not started**: no implementation yet.
3. Select the highest-priority `Not started` or `In progress` row, considering:
   - `## Dependencies` ordering in the plan
   - the linked issue's severity/priority
   - user preference (if specified)

### Phase 2: Planning

4. Read the linked GitHub issue (if any) for the deliverable's acceptance criteria.
5. Read all governing ADRs, standards, and source files referenced.
6. Identify the specific code changes needed:
   - which files to modify or create
   - which tests to add
   - which documentation to update
7. Consult `ce-adr-consult` if the deliverable touches ADR-governed behavior.
8. Present the implementation plan to the user for approval.

### Phase 3: Implementation

9. Implement the code changes following CE coding standards.
10. Write tests per the `ce-test-author` rubric.
11. Run `make local-checks-task TASK=<n>` to validate task closure when the
    plan maps one, or `make local-checks-pr` otherwise.
12. If tests fail, diagnose and fix before proceeding.

### Phase 4: Verification

13. Confirm the verification commands from the deliverable's linked GitHub
    issue (or its governing ADR/Standard) pass.
14. Confirm:
    - all new tests pass
    - no coverage regression
    - no existing test breakage
15. Update the row's Status to `Done` in `vX.Y.Z_plan.md`, and close or
    comment on the linked issue.

## Output contract

For each completed deliverable, provide:
- summary of changes made
- files modified
- tests added/modified
- verification evidence (test output, coverage)
- updated Status in `vX.Y.Z_plan.md`

## Constraints

- One deliverable at a time. Do not batch unrelated rows.
- Always run verification before declaring a row `Done`.
- Do not add rows to the plan that are not backed by the approved milestone
  or an explicit maintainer instruction.
- Follow CE-first coding rules: use `WrapCalibratedExplainer` and the public
  CE API directly, not ad-hoc wrappers.
