---
name: ce-release-check
description: >
  Read the active version plan and select the next ADR-compliant actionable
  development step.
model: haiku
---

# CE Release Check

You are determining the next development step from the active version plan.
This skill implements the "proceed according to plan" workflow defined in
`execution-plan.instructions.md`.

**Mandatory sequence:**
1. Read the sole active `development/current-work/vX.Y.Z_plan.md`.
2. Identify the release/development version and the plan's `## Included work`
   table.
3. List rows whose Status is not `Done`.
4. Verify that the proposed next step is allowed by all relevant ADRs.
5. If an ADR constraint and a plan row conflict, the ADR wins.

---

## Files to read

```
development/current-work/vX.Y.Z_plan.md   ← primary source: included work, exclusions, gates, decision
development/adrs/                          ← governance constraints (ADR takes precedence)
CHANGELOG.md                               ← completed items; do not duplicate
```

---

## Step 1 — Identify current state

```markdown
Active version plan:  development/current-work/vX.Y.Z_plan.md
Release version:      X.Y.Z   (from the plan's front matter)
```

---

## Step 2 — Scan the Included work table

```
| ID | Deliverable | Issue | ADR/Standard | Status |
```

Rows with Status `Not started` or `In progress` are outstanding. List them
with their linked issue and governing ADR/Standard.

---

## Step 3 — Cross-reference ADRs

For each outstanding row, identify the governing ADR(s) using `ce-adr-consult`.
If the proposed work conflicts with an ADR decision, **stop** and flag the
conflict rather than proceeding.

---

## Step 4 — Select the next actionable item

Priority order:
1. **Blocking gates** — rows explicitly required by `## Release-specific gates`.
2. **Linked-issue work** — rows with an open governing issue.
3. **Remaining rows** — anything else still `Not started`.

---

## Step 5 — Verify against CHANGELOG

```bash
# Check what was delivered recently
head -100 CHANGELOG.md
```

Do not propose work that is already present in `CHANGELOG.md`.

When recommending closure evidence for a planned item, prefer:
- `make local-checks-task TASK=<n>` for task completion (when the plan maps one)
- `make local-checks-pr` for PR preflight
- `make local-checks-release` for release-boundary validation

---

## Output format

```
Release Check: <date>
======================
Active version plan:  vX.Y.Z_plan.md (release X.Y.Z)

Outstanding rows:
  1. [T2] <deliverable> — issue #<n> — ADR-NNN
  2. [T3] <deliverable> — issue #<n> — ADR-NNN
  ...

Next actionable step:
  Work item: <title>
  ADR(s):    ADR-NNN (Decision section: <binding rule>)
  Plan ref:  vX.Y.Z_plan.md, row <ID>
  Rationale: <one sentence>

ADR conflicts detected: NONE | <list if any>
```

---

## Completing a work item (CHANGELOG update)

When an item is completed satisfactorily, add it to `CHANGELOG.md` under
the appropriate section header:

```markdown
## [Unreleased]

### Added
- Implemented `to_primitive` / `from_primitive` calibrator serialization (ADR-031).

### Fixed
- ...
```

Then update the row's Status in `vX.Y.Z_plan.md` to `Done`.

---

## Evaluation Checklist

- [ ] The active `vX.Y.Z_plan.md` read before proposing any step.
- [ ] Release version and outstanding rows clearly identified.
- [ ] Outstanding rows listed with linked issue and ADR references.
- [ ] No ADR constraint violated by the proposed next step.
- [ ] `CHANGELOG.md` checked to avoid duplicate work.
- [ ] Completed items added to `CHANGELOG.md` under `[Unreleased]` and marked `Done` in the plan.
