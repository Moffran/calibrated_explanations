---
name: ce-adr-author
description: >
  Write a new Architectural Decision Record (ADR) for calibrated_explanations.
  Use when asked to 'write a new ADR', 'document this architectural decision',
  'create an ADR', 'ADR template', 'new architectural record', 'start an ADR',
  'draft an ADR'. Covers the canonical ADR format, status lifecycle, and the
  process for numbering, linking, and filing the document.
---

# CE ADR Author

You are writing a new Architectural Decision Record. ADRs are the authoritative
governance documents for calibrated_explanations. They take precedence over
release-plan steps when they conflict.

ADRs live in `docs/improvement/adrs/` as Markdown files named
`ADR-<NNN>-<kebab-slug>.md`.

---

## File naming

```
ADR-<NNN>-<brief-kebab-title>.md
```

- `<NNN>` is the next sequential number. Check the existing files to find it.
- `<brief-kebab-title>`: lowercase words connected by hyphens; ≤ 6 words.

```bash
# Find next ADR number
Get-ChildItem docs/improvement/adrs/ADR-*.md |
  ForEach-Object { [int]($_.Name -replace 'ADR-(\d+).*','$1') } |
  Measure-Object -Maximum | Select-Object -ExpandProperty Maximum
# Add 1 for the new ADR number
```

---

## Canonical template

```markdown
> **Status note (<YYYY-MM-DD>):** Last edited <YYYY-MM-DD> · Archive after: Retain indefinitely as architectural record · Implementation window: <version or TBD>.

# ADR-<NNN>: <Title>

Status: Draft
Date: <YYYY-MM-DD>
Deciders: Core maintainers
Reviewers: TBD
Supersedes: None
Superseded-by: None
Related: <comma-separated list of ADR-NNN slugs, if any>

## Context

<1-3 paragraphs describing the problem, current state, and why a decision is needed.
Reference specific files or code paths where relevant. Be specific about the
constraint or risk that motivated this ADR.>

## Decision

<The decision itself. Use numbered or bullet points for multi-part decisions.
Be unambiguous. Use "MUST", "MUST NOT", "SHOULD" for binding vs advisory rules.
Include code samples if they clarify the contract.>

## Alternatives Considered

<At least 2 alternatives with a brief rationale for why each was rejected.>

1. **Alternative A** — <describe>. Rejected because <reason>.
2. **Alternative B** — <describe>. Rejected because <reason>.

## Consequences

### Positive
- <benefit 1>
- <benefit 2>

### Negative / Risks
- <risk 1>
- <risk 2>

## Adoption & Migration

<What changes are needed to adopt this decision? Include release milestones
and any backward/forward compatibility notes.>

## Open Questions

- <question 1>
- <question 2>
```

---

## Status lifecycle

| Status | Meaning |
|---|---|
| `Draft` | Under review; not yet binding |
| `Accepted` | Binding; must be followed |
| `Accepted (scoped)` | Binding for a defined scope |
| `Deprecated` | Still visible; guidance replaced elsewhere |
| `Superseded` | Replaced by another ADR; prefix filename with `superseded ` |

---

## Linking to the release plan

After accepting an ADR, add a one-line entry to `docs/improvement/RELEASE_PLAN_v1.md`
under the "ADR roadmap summary" section:

```markdown
**ADR-<NNN> – <Title>:** <One-line status>. Remaining work targeted for v<X.Y.Z>.
```

---

## Cross-referencing other ADRs

In the `Related:` frontmatter field, list related ADRs by slug:
```
Related: ADR-006-plugin-registry-trust-model, ADR-013-interval-calibrator-plugin-strategy
```

In body text, reference as `(ADR-006)` or link as
`[ADR-006](./ADR-006-plugin-registry-trust-model.md)`.

---

## Committing an ADR

1. Open a PR with just the ADR; implementation follows in a separate PR.
2. Set status to `Draft` in the PR; reviewers change it to `Accepted` on merge.
3. The implementation PR must reference the ADR in the PR description.
4. Update `docs/improvement/RELEASE_PLAN_v1.md` entry in the same implementation PR.

---

## Evaluation Checklist

- [ ] Filename follows `ADR-<NNN>-<kebab-slug>.md` naming.
- [ ] Status note header present with date.
- [ ] All required sections present: Context, Decision, Alternatives, Consequences, Adoption.
- [ ] Decision uses MUST / MUST NOT / SHOULD clearly for binding vs advisory.
- [ ] At least 2 alternatives considered with rejection rationale.
- [ ] `Related:` field populated with existing ADR slugs.
- [ ] `RELEASE_PLAN_v1.md` updated with one-line entry.
- [ ] Implementation PR references the ADR number.
