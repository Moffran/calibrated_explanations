---
name: cee-decision-ledger-minimality-designer
description: >
  Design or critique a minimal CEE governance decision ledger that records human decision, review, and escalation events with enough structure to reconstruct what happened without collapsing into narrative sludge. Use when choosing ledger fields, rationale codebooks, linkage between review and escalation events, or audit-sampling rules. Triggers on: "design the decision ledger", "what fields do we actually need", "is this audit schema too bloated", "reduce this rationale form", "make this ledger minimal but reconstructable".
---

## Inputs

- **`workflow_context`** (text, required): Current human decision, review, or escalation flow and what the organization needs to reconstruct later.
  - Example: `Reviewers can approve, defer, or escalate model-assisted decisions and we need sampled-case reconstructability.`
- **`current_schema`** (text, optional): Existing ledger, form, or audit schema to shrink or critique.
  - Example: `Current form has 18 fields, 3 long free-text boxes, and copies the raw payload.`
- **`audit_constraints`** (text, optional): Sampling, retention, privacy, or accountability constraints that bound the ledger.
  - Example: `Keep only 90 days for routine cases, sample 5 percent for review, and avoid storing raw explanations.`

## Output Format

Format: `markdown`

Required sections:
- minimal_schema
- codebook
- recommended_field_deletions
- audit_sampling_guidance
- burden_risk_assessment
- baseline_comparison

# cee-decision-ledger-minimality-designer — Core Instructions

Design or critique a minimal governance decision ledger for human review and escalation.

## Use this skill when
- Asked to design the decision ledger for review, approval, deferral, or escalation events.
- Asked which fields are actually needed to reconstruct sampled cases later.
- Asked to reduce a bloated rationale form or audit schema without losing accountability.
- Asked how to link decision, review, and escalation events while keeping the artifact minimal.

## Do not use this skill when
- The task is generic logging or observability design. Use `ce-logging-observability` or `cee-governance-telemetry`.
- The task is broad regulatory mapping without a concrete ledger artifact. Use `ce-regulatory-compliance`.
- The task is UX form design or narrative documentation.

## Boundary rule
This skill owns the minimal human decision record in the governance layer. It does not own machine telemetry, evidence-pack generation, or long-form narrative workflows.

## Ledger design procedure
1. Start with the reconstructability questions the ledger must answer:
   - who made the decision
   - what action they took
   - when they took it
   - what case or event it linked to
   - what rationale category applied
2. Write down the simpler baseline first: structured logs with no separate human decision artifact.
3. Define the smallest schema that adds value beyond logs-only operation.
4. Link decision, review, and escalation events explicitly. Prefer IDs and references over copied payloads.
5. Build a rationale codebook. Use free text only where a code cannot preserve a legally or operationally important distinction.
6. Delete fields that duplicate telemetry, raw explanations, or user-input narrative that nobody can review consistently.
7. Recommend audit sampling, retention, and privacy limits in proportion to burden and risk.
8. Compare the ledger against the logs-only baseline and say when the ledger should not exist.

## Minimality rules
- Prefer coded fields over prose.
- Prefer references to source systems over duplication.
- Keep the schema answerable by humans under real workload, not idealized compliance theater.
- If the same question can be answered from telemetry plus one ledger pointer, do not duplicate it in the ledger.

## Output structure

### MINIMAL SCHEMA
Return a concrete schema with field name, type, required or optional status, and why each field survives.

### CODEBOOK
Provide the rationale codebook and any small controlled vocabularies needed for reconstructability.

### RECOMMENDED FIELD DELETIONS
Name fields to remove, merge, or demote to references, and explain why they do not earn their burden.

### AUDIT SAMPLING GUIDANCE
State what should be logged for every case, what should be sampled, and what should be retained only on escalation.

### BURDEN RISK ASSESSMENT
Assess reviewer burden, privacy or retention risk, and reconstructability tradeoffs directly.

### BASELINE COMPARISON
Compare the proposed ledger against structured logs with no separate human decision artifact and say when logs-only is the better design.


## Constraints

- Do not turn the ledger into generic logging, regulatory mapping in the abstract, or long-form journaling.
- Prefer references or IDs to raw payload duplication.
- Minimize free text and justify every retained free-text field.
- Keep the artifact in the governance layer, separate from adaptive routing logic and UI flow design.
- If a minimal ledger does not beat the structured-logs baseline, say so plainly.

## Self-Check Before Responding

- [ ] Can a reviewer reconstruct who decided what, when, and why from the proposed schema?
- [ ] Are decision, review, and escalation events linked cleanly?
- [ ] Are unnecessary fields explicitly deleted or moved out of the ledger?
- [ ] Are sampling, retention, and privacy limits proportionate?
- [ ] Is burden-versus-reconstructability assessed honestly?
- [ ] Is the structured-logs-only baseline compared explicitly?