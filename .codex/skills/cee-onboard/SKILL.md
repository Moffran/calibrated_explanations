---
name: cee-onboard
description: >
  Read-only session primer for CEE-first invariants, key files, and skill routing at session start.
---

## Inputs

- **`content`** (text, required): The input relevant to this skill. See instructions for details.

## Output Format

Format: `markdown`

Required sections:
- output

# Cee Onboard — Core Instructions

# CEE Session Onboard

## Use this skill when
- Starting any new Claude Code session on the calibrated-explanations-enterprise repo
- Resuming work after a break and needing to re-establish context
- Before making any changes to ensure baseline is understood
- When asked "what should I know before working on this repo?"

## Inputs
- `AGENTS.md` — canonical agent instructions (mandatory read)
- `development/current-work/agent_feedback_log.md` — lessons from previous sessions
- `development/current-work/IMPLEMENTATION_GUIDE.md` — active stage and implementation status
- `development/current-work/agent_acceptance_checklists.md` — definition of done

## Workflow

1. **Read AGENTS.md** — internalize the full canonical instructions including:
   - OSS CE alignment rules and CE-First workflow
   - Repository structure and package roles
   - Package dependency policy (common/adaptive/governance isolation)
   - Critical rules (10+ items) and coding conventions
   - OSS terminology table (allowed/restricted/forbidden terms)

2. **Read agent_feedback_log.md** — note any recent recurring mistakes to avoid

3. **Establish test baseline** — run `pytest -q` to see current pass/fail state before making any changes

4. **Read IMPLEMENTATION_GUIDE.md** — identify current stage and active milestones

5. **Route to correct skill** — based on task type, select the appropriate enterprise or OSS CE skill:
   - New feature touching adaptive/governance: start with `cee-layer-placement`
   - Package import question: start with `cee-package-isolation`
   - OSS CE bug found: start with `cee-upstream-log`
   - Code review needed: start with `cee-code-review`
   - Parity regression: start with `cee-parity-test`
   - V2 protocol work: start with `cee-v2-protocol`
   - Queue-aware defer/review policy design: start with `cee-capacity-aware-deferral-designer`
   - Validity contract design: start with `cee-calibration-validity-contract-designer`
   - Drift detection work: start with `cee-drift-detection`
   - Checkpoint/persistence work: start with `cee-checkpoint`
   - Minimal decision-ledger design: start with `cee-decision-ledger-minimality-designer`
   - Telemetry/audit work: start with `cee-governance-telemetry`

## Verification
```bash
pytest -q   # establish baseline — note the count of passed/failed/errors
ruff check . 2>&1 | head -20   # note any existing lint issues
```

## Output contract
Return a session brief with:
1. Current stage and milestone status from IMPLEMENTATION_GUIDE.md
2. Top 3 items from agent_feedback_log.md to watch out for
3. Baseline test count (passed / failed / errors)
4. Recommended skill for the user's stated task

## Constraints
- This skill is READ-ONLY: do not make any changes during onboarding
- Always run the test baseline BEFORE any code changes, never after
- If `calibrated-explanations` is not importable, report it immediately — this is a critical environment issue
- Do not proceed with implementation until the baseline is established

## CEE-Specific Invariants to Verify

| Rule | Check |
|---|---|
| CE-First | `from calibrated_explanations import WrapCalibratedExplainer` succeeds |
| Package isolation | No cross-package imports (see AGENTS.md §Package Dependency Policy) |
| No runtime artifacts staged | `git status` shows no `.parquet`, `.pkl`, `.db` files |
| Terminology | No "counterfactual" in new code (should be "alternatives") |
| No "Orchestrator" | Search for forbidden term if working on managers |