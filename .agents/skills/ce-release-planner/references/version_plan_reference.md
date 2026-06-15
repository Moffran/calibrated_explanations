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
5. `## Release gate summary`
6. `## Minimal new tests required`

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

## Release gate summary

## Minimal new tests required
```
