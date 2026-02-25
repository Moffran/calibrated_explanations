# RTD Audit Checklist (CE)

Source basis: Claude skill authoring guidance (`docs.claude.com`, "Create a
skill") + CE standards (`STD-004`, CE-first policy, ADR governance).

## Critical checks

1. Navigation
- Page appears in correct toctree and audience hub.
- New page has inbound and outbound links.

2. Accuracy
- API names and signatures align with current public surface.
- CE-first lifecycle is preserved: fit -> calibrate -> explain/predict.
- No private-member access instructions.

3. Audience fit (STD-004)
- Audience is explicit (practitioner/researcher/contributor/etc.).
- Depth and jargon level match the target audience.

4. Governance
- ADR/standard links exist and are not stale.
- Claims do not contradict accepted ADR decisions.

5. Operational quality
- Commands are copy/paste safe.
- Paths in backticks exist.
- Link targets resolve.

## Reporting format

- Findings first, by severity.
- Include file path + line, impact, and concrete fix.
- Note residual risks/testing gaps after findings.
