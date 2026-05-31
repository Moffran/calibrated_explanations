---
name: cee-upstream-log
description: >
  Log OSS CE bugs and feature requests to the upstream log with correct classification, preventing silent CEE workarounds that diverge from upstream.
---

## Inputs

- **`content`** (text, required): The input relevant to this skill. See instructions for details.

## Output Format

Format: `markdown`

Required sections:
- output

# Cee Upstream Log — Core Instructions

# CEE Upstream Log

## Use this skill when
- A bug is found in the OSS `calibrated-explanations` library
- A missing OSS public method is blocking CEE implementation
- A numerical correctness or parity issue is traced to OSS code
- A performance regression in OSS affects CEE
- `cee-layer-placement` has determined the change belongs in OSS CE

## Inputs
- `development/oss_ce_upstream_log.md` — existing log to append to
- Description of the OSS issue found
- The CEE code location where the issue was discovered (if any)

## Workflow

1. **Read** `development/oss_ce_upstream_log.md` to understand the log format and check for duplicates

2. **Classify** the item:
   - `BUG` — existing OSS behaviour is incorrect or diverges from docs
   - `FEATURE REQUEST` — OSS is missing a method/capability CEE needs

3. **Draft the log entry** using the template in the log file:
   ```
   ## [OPEN] <Short title> — <BUG|FEATURE REQUEST>
   **Discovered**: YYYY-MM-DD
   **CEE context**: <file:line where discovered>
   **Description**: <what is wrong or missing>
   **Impact on CEE**: <what CEE cannot do until this is fixed>
   **Workaround**: <temporary CEE workaround if any, or "None">
   **OSS Issue**: <link once created, or "Pending">
   ```

4. **Add `# TODO upstream:` comment** in CEE code if a workaround was implemented:
   ```python
   # TODO upstream: <exact title from log entry>
   ```

5. **Notify maintainer** — remind the user to create the GitHub issue in the OSS repo

6. **Update the log** — append the new entry to `development/oss_ce_upstream_log.md`

## Verification
```bash
# Confirm the entry was added and is correctly formatted
grep -A 8 "OPEN.*$(date +%Y)" development/oss_ce_upstream_log.md
```

## Output contract
Return:
1. The drafted log entry (ready to append)
2. The `# TODO upstream:` comment to add to CEE code (if applicable)
3. Reminder to human: create GitHub issue at https://github.com/kristinebergs/calibrated_explanations

## Constraints
- Never implement the OSS fix in CEE code — only log it
- Classification must be exactly `BUG` or `FEATURE REQUEST` (not both)
- If a workaround is added to CEE, it MUST have the `# TODO upstream:` comment
- The title in the log and the comment must match exactly
