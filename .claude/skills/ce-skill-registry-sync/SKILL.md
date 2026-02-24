---
name: ce-skill-registry-sync
description: >
  Keep CONTRIBUTOR_INSTRUCTIONS.md §6A (Shared Skill Registry) and
  .claude/skills/ce-onboard/SKILL.md §4 (Skill catalogue) and
  docs/contributor/agent_skills.md (RTD skill catalogue) synchronized with the
  actual .claude/skills catalog. Use when any skill is added, removed, renamed,
  or moved, or when the user asks for a skill listing update. Enforces updating
  all registries in the same patch as the skill change.
---

# CE Skill Registry Sync

Use this skill whenever `.claude/skills/` changes or whenever a user requests
any skill listing update.

## Mandatory policy

If any skill directory or `SKILL.md` file is added, removed, renamed, or moved:

1. Update `CONTRIBUTOR_INSTRUCTIONS.md` Table 6A in the same patch.
2. Update `.claude/skills/ce-onboard/SKILL.md` section 4 skill catalogue in the same patch.
3. Update `docs/contributor/agent_skills.md` in the same patch.
4. Ensure the skill name, path, and primary-use text are accurate in Table 6A.
5. Ensure intent-to-skill mapping is accurate in ce-onboard section 4.
6. Ensure each RTD skill row contains a valid skill link.
7. Keep registry lists sorted alphabetically by skill name where applicable.
8. Do not leave stale rows for removed or renamed skills.
9. Do not complete the task unless all registries and filesystem state match.

Always perform **full-set reconciliation**:
- Enumerate all directories under `.claude/skills/`.
- Ensure each directory has exactly one corresponding entry in:
  - `CONTRIBUTOR_INSTRUCTIONS.md` section 6A table, and
  - `.claude/skills/ce-onboard/SKILL.md` section 4 catalogue, and
  - `docs/contributor/agent_skills.md` table.
- Ensure neither registry contains entries for non-existent skills.

## Verification steps (required)

Run all checks:

```bash
rg --files .claude/skills
rg -n "## 6A|ce-skill-registry-sync|Maintenance rule" CONTRIBUTOR_INSTRUCTIONS.md
rg -n "## 4\\. Skill catalogue|ce-skill-registry-sync" .claude/skills/ce-onboard/SKILL.md
rg -n "Agent skill catalogue|ce-skill-registry-sync|\\.claude/skills/.*/SKILL.md" docs/contributor/agent_skills.md
```

Then compare the discovered skill folders to:
- `CONTRIBUTOR_INSTRUCTIONS.md` Table 6A entries
- `.claude/skills/ce-onboard/SKILL.md` section 4 catalogue entries
- `docs/contributor/agent_skills.md` table entries
- Confirm set equality (filesystem set == registry set for all three files).

## Failure behavior

If any registry is not updated correctly, fail the task and report:
- which skill entries are missing
- which entries are stale
- the exact row(s) that must be changed in each file
