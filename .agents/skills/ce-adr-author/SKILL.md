---
name: ce-adr-author
description: >
  Author or revise ADR files and release-plan ADR entries; use for new ADR drafts,
  status transitions, and ADR governance updates.
model: haiku
---
<!-- Generated from skills/ce-adr-author/SKILL.md by scripts/quality/sync_skill_packages.py; do not edit. -->

# CE ADR Author

Use this skill for any architectural record work in
`development/adrs/`.

## Core assets

- `assets/adr_template.md` - canonical ADR template (copy this first).
- `development/adrs/` - authoritative ADR directory.
- The active `development/current-work/vX.Y.Z_plan.md`, if the decision
  changes what is in scope for the release currently being coordinated.

## Workflow

1. Determine whether this is a new ADR or an update to an existing ADR.
2. For new ADRs, calculate the next ADR number:

```bash
Get-ChildItem development/adrs/ADR-*.md |
  ForEach-Object { [int]($_.Name -replace 'ADR-(\d+).*','$1') } |
  Measure-Object -Maximum | Select-Object -ExpandProperty Maximum
```

3. Copy the template asset to the target file:

```bash
Copy-Item skills/ce-adr-author/assets/adr_template.md `
  development/adrs/ADR-<NNN>-<kebab-slug>.md
```

4. Fill every required field and section in the template. Keep language
   normative when needed (`MUST`, `MUST NOT`, `SHOULD`).
5. Add/validate `Related:` ADR links and update references to superseded ADRs.
6. If the decision changes what is committed for the release currently being
   coordinated, propose adding/updating a row in the active
   `development/current-work/vX.Y.Z_plan.md`'s `## Included work` table (only
   after maintainer confirmation) — or file/link a GitHub issue for work not
   yet scheduled into a milestone. Do not create or reference a master
   roadmap document.
7. If status changes to `Superseded`, rename the replaced ADR with
   `superseded ` prefix and set `Superseded-by`.

## Status lifecycle

| Status | Meaning |
|---|---|
| `Draft` | Under review; not yet binding |
| `Accepted` | Binding and enforceable |
| `Accepted (scoped)` | Binding only for explicit scope |
| `Deprecated` | Visible but replaced guidance exists |
| `Superseded` | Replaced by newer ADR |

## Required quality checks

- Filename uses `ADR-<NNN>-<kebab-slug>.md`.
- Template sections are complete (Context, Decision, Alternatives, Consequences,
  Adoption, Open Questions).
- At least two alternatives are documented with rejection rationale.
- Decision text states enforceable constraints clearly.
- `Related:` entries point to real ADR files.
- Active-plan or issue linkage is updated when applicable.

## Output contract

Return:

1. ADR file path and status.
2. Summary of decision and alternatives.
3. Any companion updates made to the active version plan or linked issues.
