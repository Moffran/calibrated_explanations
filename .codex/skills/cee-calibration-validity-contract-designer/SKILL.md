---
name: cee-calibration-validity-contract-designer
description: >
  Design or critique the CEE adaptive validity contract that downstream UX, governance, and policy code consume. Use when defining validity states, reason codes, score semantics, or detector-to-state mappings and when deciding what belongs in a shared `ValidityResult`-style type without giving false reassurance. Triggers on: "design the validity result", "what should count as likely invalid", "which signals belong in ValidityResult", "is this a reason code or UI detail", "map detector outputs to validity states".
---

## Inputs

- **`contract_context`** (text, required): Current detector outputs, routing decisions, and downstream consumers that need the validity contract.
  - Example: `Adaptive routing, governance audit records, and UX banners all consume a shared validity result today.`
- **`current_signals`** (text, optional): Existing states, booleans, scores, or reason strings to preserve, rename, or remove.
  - Example: `We currently return is_valid, confidence, and a free-form reason string from three detectors.`
- **`downstream_consumers`** (text, optional): Packages or surfaces that will consume the contract and any ownership constraints.
  - Example: `Adaptive uses it for routing, governance stores it, and UX renders a simplified banner.`

## Output Format

Format: `markdown`

Required sections:
- minimal_contract_proposal
- state_definitions
- reason_code_taxonomy
- ownership_split_by_package
- excluded_fields
- baseline_comparison

# cee-calibration-validity-contract-designer — Core Instructions

Design or critique the runtime validity contract that downstream CEE consumers rely on.

## Use this skill when
- Asked to design a shared `ValidityResult` or equivalent runtime type.
- Asked what should count as valid, degraded, or likely invalid in routing logic.
- Asked which reason codes belong in the contract versus in UX or detector internals.
- Asked how detector outputs should map to stable validity states or bands.

## Do not use this skill when
- The task is implementing or tuning a detector. Use `cee-drift-detection`.
- The task is writing UX banners, status copy, or remediation text.
- The task is generic package-isolation enforcement outside the contract question. Use `cee-package-isolation`.

## Boundary rule
This skill owns the shared contract between detector or policy logic and downstream consumers. It does not own detector algorithms, product copy, or telemetry schemas.

## Contract design procedure
1. List the downstream consumers and the decisions they need to make.
2. Start from the simpler baseline: ad hoc booleans and undocumented reason strings.
3. Propose the smallest shared shape that still supports routing and governance use cases.
4. Define each validity state or band with plain semantics:
   - what evidence tends to produce it
   - what it does and does not imply
5. If a numeric score exists, define its direction, scale, and limits. If you cannot explain the score semantics cleanly, exclude it.
6. Keep reason codes stable and low-cardinality. They should survive detector swaps and UI redesigns.
7. Split ownership explicitly:
   - keep adaptive or common types neutral
   - keep UX wording, color, and remediation hints downstream
   - keep detector diagnostics outside the shared contract unless they are stable and broadly useful
8. Compare the design against the boolean-plus-string baseline and say when the baseline is enough.

## Ownership rule
- Keep the contract in `adaptive` if only adaptive policy code consumes it.
- Promote neutral enums or dataclasses to `common` only when adaptive, governance, and UX all need the same semantics.
- Keep presentation models, view helpers, and package-specific metadata out of the shared type.

## Output structure

### MINIMAL CONTRACT PROPOSAL
Return the smallest field set that downstream consumers genuinely need.

### STATE DEFINITIONS
Define each state or band, its intended meaning, and its non-guarantees.

### REASON CODE TAXONOMY
Provide stable reason-code groups and explain what collapses into the same code versus what stays detector-local.

### OWNERSHIP SPLIT BY PACKAGE
State what belongs in `common`, what stays in `adaptive`, and what must be mapped only in UX or governance.

### EXCLUDED FIELDS
Name fields that do not belong in the shared contract and explain why.

### BASELINE COMPARISON
Compare the proposed contract against booleans plus undocumented string reasons and say when the simpler baseline should win.


## Constraints

- Never describe validity as a guarantee, proof, or certification.
- Do not implement detectors, write UX copy, or offer broad package-placement guidance beyond the contract boundary.
- Keep shared types minimal and stable across detectors and downstream consumers.
- Keep reason codes low-cardinality, stable, and independent of UI wording.
- Promote fields to common only when multiple packages genuinely need the same semantics.

## Self-Check Before Responding

- [ ] Are state definitions explicit about what they do and do not imply?
- [ ] Are reason codes stable and non-UI?
- [ ] Is score semantics defined if a numeric score is included?
- [ ] Is the ownership split across adaptive, common, UX, and governance explicit?
- [ ] Are excluded fields justified with concrete reasons?
- [ ] Is the ad hoc boolean-plus-string baseline compared explicitly?
