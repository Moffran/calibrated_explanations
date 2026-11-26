---
name: ce-release-planner
description: >
  Plan a versioned CE release from an approved milestone or an explicit
  maintainer-selected issue set.
---
<!-- Generated from skills/ce-release-planner/SKILL.md by scripts/quality/sync_skill_packages.py; do not edit. -->

# CE Release Planner

You are creating the sole active version plan for an upcoming CE release.
The plan coordinates the release; it does not hold proposed work or approved
scope — those live in GitHub issues and a GitHub milestone.

## Required references

- The GitHub milestone the maintainer has selected for this release (its
  issue list is the scope), or an explicit maintainer-supplied issue list
  when no milestone exists yet.
- `release.md` (maintainer release sequence; steps 1-10 automated by preflight,
  11-13 manual, and 14-17 automated by postcommit)
- `scripts/local_checks.py` and `Makefile` (executable release workflow)
- `references/version_plan_reference.md` (canonical structure/template for
  `development/current-work/vX.Y.Z_plan.md` files)
- Governing ADRs and Standards for each selected issue
- The most recently archived `development/finished-work/vX.Y.Z_plan.md` for
  formatting reference only (not for scope — do not mine it for carry-over work)

## Use this skill when

- Planning the next release version.
- Creating a new `vX.Y.Z_plan.md` after maintainers have chosen a GitHub
  milestone (or an explicit issue list) for the release.
- Confirming there is exactly one active version plan before release work starts.

## Workflow

1. **Confirm exactly one active plan.**
   - Check `development/current-work/` for existing `vX.Y.Z_plan.md` files.
     There must be exactly one. If one already exists and is still open, work
     within it instead of creating a second one.

2. **Identify target version and scope.**
   - Ask the maintainer which GitHub milestone (or explicit issue list) this
     plan implements if it is not already clear from the conversation.
   - Do not promote unapproved work to a committed version without an
     explicit maintainer decision — this skill does not decide release scope
     by itself, and it does not search archived plans for "carry-over" work.
   - Record both the exact PEP 440 release version and its development
     placeholder. Do not infer `X.Y.(Z+1)` when the maintainer names a minor,
     major, or prerelease milestone.

3. **Read governing ADRs and Standards** for each selected issue. Note
   current implementation status against the relevant source and tests.

4. **Check current codebase state** for each deliverable — relevant modules,
   existing tests, anything already partially or fully done.

5. **Draft the plan.**
   - Create `development/current-work/vX.Y.Z_plan.md` following
     `references/version_plan_reference.md`.
   - Populate the `## Included work` table with one row per deliverable,
     each linked to its GitHub issue where one exists.
   - Keep `## Excluded`, `## Dependencies`, and `## Release-specific gates`
     short and specific to this release.
   - `## Release decision` is optional, non-authoritative prose (not parsed
     by automation) — add it only if a short human-readable summary helps
     readers; readiness derives solely from `## Included work` statuses.
   - Do not duplicate the `release.md` step sequence in the plan; the
     template already links to it.

6. **Cross-check completeness.**
   - Verify every issue selected for this milestone has a row.
   - Verify no command, path, expected version, or next-version rule is
     hard-coded to the release that happened to precede this one.

## Output contract

Produce `development/current-work/vX.Y.Z_plan.md` matching
`references/version_plan_reference.md`: front matter, `Outcome`,
`Included work`, `Excluded`, `Dependencies`, `Release-specific gates`, and
optionally `Release decision`.

## Constraints

- Do not invent deliverables that are not backed by an approved GitHub
  milestone or an explicit maintainer-supplied issue list.
- Do not create a second active version plan; exactly one must exist under
  `development/current-work/`.
- Do not create another master roadmap document (by any name) — proposed
  work belongs in GitHub issues, approved scope in a GitHub milestone.
- Mark rows `Done` only when verification evidence exists in the codebase.
- Do not copy the `release.md` step sequence into the plan; link to it.
