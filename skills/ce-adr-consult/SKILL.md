---
name: ce-adr-consult
description: >
  Map a proposed change to governing ADRs and constraints before implementation;
  use for ADR applicability and conflict checks.
model: haiku
---

# CE ADR Consult

Identify the accepted ADRs and engineering standards that govern a proposed
change. An ADR takes precedence over the active plan, a skill, and contributor
workflow guidance.

## Consultation workflow

1. Read `development/adrs/README.md`, which is generated from canonical metadata.
2. Search its titles and applicability descriptions for the proposed behavior,
   boundary, or lifecycle change.
3. Treat `accepted` records in the active section as governing. Treat `proposed`
   records as non-final; `superseded` and `withdrawn` records are historical.
4. Read the selected ADR source, especially its `Decision`, `Normative`, or
   equivalently binding section and its consequences.
5. Repeat with `development/standards/README.md` for engineering and quality
   requirements.
6. Check the active version plan only after identifying the governing records.
   A plan row cannot override a binding decision.
7. Validate supersession links and status in the metadata before relying on a
   historical record.

Do not maintain a topic-to-ADR table in this skill. The generated index is the
single discovery inventory, and the ADR source remains normative.

## Required output

Return:

1. the task boundary being evaluated;
2. applicable accepted ADRs and standards with source paths;
3. binding constraints extracted from each source;
4. conflicts, exemptions, or missing decisions;
5. the implementation consequence and verification command.

## Conflict resolution

When a plan and an ADR conflict, the ADR wins. Record the conflict in the
implementation summary. If a binding decision must change, obtain maintainer
approval and revise or supersede the ADR before implementing the behavior.
