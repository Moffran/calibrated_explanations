---
name: cee-layer-placement
description: >
  Decide whether a proposed change belongs in CEE or upstream OSS CE, preventing misplaced technical debt before implementation begins.
---

## Inputs

- **`content`** (text, required): The input relevant to this skill. See instructions for details.

## Output Format

Format: `markdown`

Required sections:
- output

# Cee Layer Placement — Core Instructions

# CEE vs OSS CE Layer Placement

## Use this skill when
- About to implement a new feature and unsure which repo it belongs in
- A bug is found and it's unclear whether to fix it in CEE or report it upstream
- Reviewing a PR and a change looks like it might be patching OSS behaviour locally
- Asked "does this belong in CEE or in calibrated-explanations?"

## Inputs
- Description of the proposed change (feature, bug fix, performance improvement, etc.)
- `AGENTS.md` §"CEE vs OSS CE Scope" decision table
- `development/oss_ce_upstream_log.md` — existing upstream items for deduplication

## Workflow

1. **Classify the change** using the decision table from AGENTS.md:

   | Belongs in CEE | Belongs in OSS CE |
   |---|---|
   | Enterprise wrapping, governance, telemetry, audit | Bug in `calibrate()`, `explain_factual()`, or any core OSS method |
   | Adaptive / online calibration, drift detection | Missing public method CEE needs to call |
   | KServe V2 protocol, deployment, Kubernetes | New explanation type or calibration algorithm |
   | Checkpointing, MLflow persistence | Performance regression in OSS library |
   | Security controls | Documentation gap in calibrated-explanations |
   | | Numerical correctness / parity issue in OSS predictions |

2. **Apply the rule of thumb**: If the change would make CEE's parity tests pass by altering OSS internals → OSS CE. If it adds enterprise capability *around* OSS CE without changing OSS maths → CEE.

3. **If CEE**: Proceed with implementation in the correct package (see `cee-package-isolation` for which package to use).

4. **If OSS CE**:
   - Do NOT implement it in CEE
   - Do NOT monkey-patch or copy-paste OSS code
   - Invoke `cee-upstream-log` to record the item
   - Add a `# TODO upstream: <title>` comment if a CEE workaround is temporarily unavoidable
   - Notify a human maintainer

5. **If ambiguous**: Default to OSS CE for anything touching mathematical correctness or OSS public API surface.

## Verification
```bash
# After deciding CEE: confirm correct package
grep -r "from calibrated_explanations_enterprise.adaptive" packages/common/
grep -r "from calibrated_explanations_enterprise.governance" packages/common/
# Both must return no results
```

## Output contract
Return a placement decision with:
1. **Decision**: CEE or OSS CE (not both, not "maybe")
2. **Rationale**: One sentence citing the specific decision table row
3. **If CEE**: Which package (`common`, `adaptive`, or `governance`) and why
4. **If OSS CE**: Next action (invoke `cee-upstream-log`)

## Constraints
- Never place a change in both repos simultaneously
- Never monkey-patch OSS methods in CEE even as a temporary workaround
- Never copy OSS source code into CEE packages
- If the answer is genuinely ambiguous, consult a human maintainer before proceeding
