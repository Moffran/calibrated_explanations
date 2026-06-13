# CE Version Plan Reference (`vX.Y.Z_plan.md`)

Use this file as the canonical scaffold for release implementation plans under:

- `docs/improvement/vX.Y.Z_plan.md`

This reference exists so old detailed plan files can be removed while keeping
the planning structure stable and repeatable.

---

## Required front matter

1. Title:
   - `# vX.Y.Z Release Task Implementation Plan`
2. Scope paragraph:
   - state that this plan expands milestone tasks from
     `docs/improvement/RELEASE_PLAN_v1.md`.
3. Milestone framing:
   - state milestone type (for example ADR gap closure, hardening, RC, etc.).
4. Authoritative task source:
   - explicitly cite the target milestone section in `RELEASE_PLAN_v1.md`.

---

## Mandatory sections

1. `## Source references reviewed`
2. `## Release tasks covered (from RELEASE_PLAN_v1.md)`
3. `## Global rules` (only if applicable for the milestone)
4. Numbered task sections matching milestone tasks:
   - `## 1) ...`
   - `## 2) ...`
   - ...
5. `## N) Release preparation` — always the final numbered task; see template below.
6. `## Release gate summary`
7. `## Minimal new tests required`

---

## Task section contract (for every numbered task)

Each task section must include:

1. Goal:
   - one concise paragraph.
2. Status assessment:
   - `Not started`, `Partial`, or `Implemented with evidence`.
3. Relevant references:
   - ADRs, standards, and key source files.
4. Current anchors in code/docs:
   - concrete modules/files currently implementing related behavior.
5. Gaps:
   - what is still missing vs task intent.
6. Implementation steps:
   - concrete, ordered, actionable steps.
7. Verification checklist:
   - tests, scripts, and expected pass criteria.

---

## Evidence rules

1. Mark a task as completed only with verifiable code/doc/test evidence.
2. Do not rely on prior plan status text alone.
3. When uncertain, classify as `Partial` and list blocking evidence gaps.
4. Keep assumptions explicit.

---

## Release gate summary requirements

The summary must:

1. Map each release-gate criterion to specific evidence.
2. Identify unresolved blockers explicitly.
3. State final recommendation:
   - `Ready to close` or `Not ready`, with blockers.

---

## Minimal tests section requirements

List only new or updated tests/scripts that are strictly required to close
remaining gaps, grouped by task number.

---

## Suggested heading skeleton

```md
# vX.Y.Z Release Task Implementation Plan

## Source references reviewed

## Release tasks covered (from RELEASE_PLAN_v1.md)

## 1) <Task title>
### 1.0 Goal
### 1.1 Status assessment
### 1.2 Relevant references
### 1.3 Current anchors in code/docs
### 1.4 Gaps
### 1.5 Implementation steps
### 1.6 Verification checklist

## 2) <Task title>
...

## N) Release preparation
### N.0 Goal
### N.1 Status assessment
### N.2 Relevant references
### N.3 Current anchors in code/docs
### N.4 Gaps
### N.5 Implementation steps
### N.6 Verification checklist

## Release gate summary

## Minimal new tests required
```

---

## Release preparation task template

Copy this verbatim as the final numbered task section and substitute `N` with the
actual task number. Adjust the milestone version tag (`vX.Y.Z`) wherever it appears.

```md
## N) Release preparation

### N.0 Goal

Complete all release-preparation checks before tagging vX.Y.Z. This task runs the
automated gates that `make local-checks` does not cover, audits structural/doc hygiene,
and ensures the milestone is recorded as closed in all living documents. It is always
the final task in a release plan and is always executed last.

### N.1 Status assessment

Not started.

### N.2 Relevant references

- `docs/improvement/RELEASE_PLAN_v1.md` — milestone task list and packaging decision record
- `docs/improvement/RELEASE_PLAN_status_appendix.md` — per-ADR gap rows
- `docs/migration/deprecations.md` — active/history deprecation ledger
- `scripts/local_checks.py` — umbrella check runner
- `scripts/quality/snapshot_public_api.py` — public API surface snapshot

### N.3 Current anchors in code/docs

- `make local-checks` already covers: coverage, private members, anti-patterns, docstring
  coverage, parameter naming, logging domains, import graph, mypy, marker hygiene,
  report local-path guard, dependency audit (advisory), notebook audit (advisory), docs build (advisory).
- `make deprecation-closure` covers: active deprecation table empty check + focused deprecation tests.
- The following are NOT wired into either target and require explicit runs or manual review.

### N.4 Gaps

All items below are outside the `make local-checks` / `make local-checks-pr` umbrella
and must be checked manually or via explicit invocations at milestone closure.

### N.5 Implementation steps

**Automated gates (run and confirm exit 0):**

1. `make local-checks` — confirm passes clean with no advisory failures of concern.
2. `make deprecation-closure` — active deprecation table empty; focused deprecation tests pass.
3. `make uv-install-smoke` — wheel builds and installs in a fresh venv; `import calibrated_explanations` succeeds.
4. `make warning-policy` — no unclassified warning emission sites remain.
5. Docs strict build: `python -m sphinx -W --keep-going docs docs/_build/html` — zero warnings
   (the plain `docs build` step in `make local-checks` is advisory; this is the strict gate).
6. API snapshot diff: `python scripts/quality/snapshot_public_api.py` — review diff for
   unintended public additions or removals.

**Manual / structural hygiene:**

7. Version bump: confirm `__version__` in `src/calibrated_explanations/__init__.py` and
   `[tool.poetry] version` in `pyproject.toml` match the release tag.
8. CHANGELOG: entry present, complete, correctly dated, correct category
   (Breaking / Added / Fixed / Removed).
9. Migration guide (`docs/migration/deprecations.md`): symbols removed in this version moved
   from active to history table; no phantom active entries from prior milestones remain.
10. ADR gap tables (`docs/improvement/RELEASE_PLAN_status_appendix.md`): all gaps closed in
    this milestone reflected as closed; no open row still targets `vX.Y.Z`.
11. `RELEASE_PLAN_v1.md` milestone closure: all vX.Y.Z tasks marked `[x]`; milestone section
    records closure date.
12. `docs/improvement/` archive: review all files in `docs/improvement/` and its sub-folders;
    move obsolete plan files, completed-version artefacts, and superseded standalone design docs
    to `docs/improvement/archive/` or delete. Only files that are active references for future
    milestones should remain at the top level. (Already-superseded ADRs carry a `superseded`
    filename prefix — no action needed for those.)
13. GitHub release draft: tag prepared; release notes drafted from CHANGELOG; PyPI release
    decision recorded in `RELEASE_PLAN_v1.md` under the packaging workflow section.

### N.6 Verification checklist

- [ ] `make local-checks` exits 0 with no advisory failures of concern.
- [ ] `make deprecation-closure` exits 0; active deprecation table is empty.
- [ ] `make uv-install-smoke` exits 0; version string printed matches release tag.
- [ ] `make warning-policy` exits 0.
- [ ] Sphinx strict build (`-W --keep-going`) exits 0; no broken cross-references.
- [ ] `snapshot_public_api.py` diff reviewed; no unintended additions or removals.
- [ ] `__version__` and `pyproject.toml` version match the release tag.
- [ ] CHANGELOG entry present, dated, complete.
- [ ] `docs/migration/deprecations.md` active table empty (or contains only post-vX.Y.Z
    deprecations); removed symbols in history table.
- [ ] `RELEASE_PLAN_status_appendix.md` has no open gap row targeting `vX.Y.Z`.
- [ ] `RELEASE_PLAN_v1.md` vX.Y.Z milestone section marked closed with date.
- [ ] `docs/improvement/` top level and sub-folders reviewed; obsolete artefacts archived or deleted.
- [ ] GitHub release draft exists with tag and CHANGELOG-sourced release notes.
```
