# CE Version Plan Reference (`vX.Y.Z_plan.md`)

Use this as the canonical scaffold for the sole active release-coordination
file under `development/current-work/`.

This file coordinates the active release. It does not hold proposed work
(GitHub issues), approved scope decisions (GitHub milestones), architecture
decisions (ADRs), or the publication procedure (`release.md`). Link to those
instead of duplicating them.

## Required front matter

```markdown
# vX.Y.Z Plan

> **Release version:** `X.Y.Z`
> **Development version:** `X.Y.Z-dev`
> **Status:** Active
> **Theme:** <one short phrase>
```

For prereleases, the filename label and package version may differ. For
example, `v1.0.0-rc_plan.md` may declare release version `1.0.0rc1` and
development version `1.0.0-rc-dev`. The explicit fields are authoritative for
automation, not the filename.

## Mandatory sections

1. `## Outcome` — a short paragraph on what users and maintainers gain.
2. `## Included work` — a table, one row per deliverable.
3. `## Excluded` — a short list preventing scope creep.
4. `## Dependencies` — only dependencies that affect task order or release
   readiness. Omit or state "None" when there are none.
5. `## Release-specific gates` — only gates specific to this release. Do not
   repeat standard checks already owned by `release.md` / `make
   release-preflight`.
6. `## Release decision` (optional) — non-authoritative summary prose. Release
   automation does not parse this section; it derives readiness solely from
   the `## Included work` table (see below).

Do not add per-task subsections (goal, status assessment, references, current
anchors, gaps, implementation steps, verification checklist). Detailed
acceptance criteria, commands, expected observations, and closure evidence
belong on the linked GitHub issue, the governing ADR/Standard, the source
diff, and the PR — not duplicated here.

## Included work table

```markdown
## Included work

| ID | Deliverable | Issue | ADR/Standard | Status |
|---|---|---|---|---|
| T1 | <short deliverable statement> | #<issue> or `—` | ADR-NNN / STD-NNN or `—` | Not started |
```

- `ID` is a short stable token (`T1`, `T2`, ...) referenced from commits/PRs.
- `Status` is one of `Not started`, `In progress`, `Done`. Release automation
  (`make release-preflight`) requires every row to read `Done` before it will
  proceed; keep the column literal so it stays machine-readable. This is the
  sole readiness signal — there is no separate manually-synchronized switch.
- Mark a row `Done` only with verifiable evidence (tests green, code merged) —
  not prior status prose alone.
- Every deliverable traces to a GitHub issue when one exists. Work with no
  issue and no clear ADR/Standard anchor should not be in this table — open an
  issue first, or leave it out.

## Release decision (optional, non-authoritative)

If you want a human-readable summary, add:

```markdown
## Release decision

`Not ready` — <short reason, e.g. which row(s) are still open>.
```

This section is prose for readers, not a machine-parsed gate: `make
release-preflight` reads only the `## Included work` statuses and the
executable release gates. Update or drop this section freely; it never needs
to be kept manually in sync with a separate "Ready" switch.

## Standard release procedure

Do not copy the `release.md` step sequence into this file. Reference it:

- `make release-preflight` — release.md steps 1-10 (readiness, release-file
  and changelog updates, tests, notebooks, alignment, clean build,
  Twine/artifact checks, clean-wheel smoke).
- `make release-finalize` — verifies the preflight snapshot is still current.
- Maintainer-only steps 11-13 — commit/tag/push, publish/verify on Read the
  Docs, upload to PyPI.
- `make release-postcommit` — release.md steps 14-17 (PyPI page/metadata
  verification, published-install smoke, archive this plan to
  `development/finished-work/`, bump the development version). It does not
  scaffold a next version plan; open one only once maintainers have selected
  the next GitHub milestone.

## Example

```markdown
# v1.0.1 Plan

> **Release version:** `1.0.1`
> **Development version:** `1.0.1-dev`
> **Status:** Active
> **Theme:** Post-v1.0 stabilisation

## Outcome

One short paragraph.

## Included work

| ID | Deliverable | Issue | ADR/Standard | Status |
|---|---|---|---|---|
| T1 | ... | #201 | ADR-003 | Not started |

## Excluded

- New public APIs.
- Feature work intended for a later release.

## Dependencies

None.

## Release-specific gates

Only gates specific to this release.

## Release decision

`Not ready` — T1 has not started.
```
