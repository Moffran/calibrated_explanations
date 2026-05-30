---
name: cee-capacity-aware-deferral-designer
description: >
  Design or critique CEE adaptive defer, review, and escalate policies that combine calibrated uncertainty signals, validity posture, and explicit review-capacity limits. Use when setting review thresholds, deferral budgets, or queue-aware abstention and when translating intervals into proceed, review, or escalate routing under finite manual-review capacity. Triggers on: "design a deferral budget", "set review thresholds", "capacity-aware abstention policy", "queue-aware deferral policy", "will this defer too many cases", "translate intervals into proceed review escalate".
---

## Inputs

- **`task_context`** (text, required): Current defer, review, or escalate problem, available runtime signals, and desired action routes.
  - Example: `Interval width, validity band, and freshness posture should determine whether we proceed, send to review, or escalate.`
- **`capacity_constraints`** (text, optional): Reviewer throughput, queue limits, SLA, daily review budget, or escalation reserve.
  - Example: `300 reviews per day, backlog cap 600, 24-hour SLA, hard escalation budget of 40 cases per day.`
- **`current_policy`** (text, optional): Existing thresholds, SOP baseline, or pain points to critique.
  - Example: `We currently review any case with interval width above 0.25 and do not cap queue growth.`

## Output Format

Format: `markdown`

Required sections:
- policy_table
- threshold_rationale
- capacity_assumptions
- overload_failure_modes
- baseline_comparison
- simpler_baseline_preferred_when

# cee-capacity-aware-deferral-designer — Core Instructions

Design or critique capacity-aware defer, review, and escalate routing for CEE adaptive systems.

## Use this skill when
- Asked to set review thresholds or deferral budgets from calibrated intervals, uncertainty bands, or validity posture.
- Asked whether a queue-aware abstention policy will overload manual reviewers.
- Asked to turn interval or validity outputs into a proceed, review, or escalate policy table.
- Asked to compare a nuanced routing policy against a single-threshold plus SOP manual-review baseline.

## Do not use this skill when
- The task is only configuring `RejectPolicy` or interpreting `RejectResult` mechanics. Use `ce-reject-policy`.
- The task is detector implementation, drift-trigger tuning, or monitoring strategy selection. Use `cee-drift-detection`.
- The task is UI design, staffing planning without runtime signals, or governance ledger design.

## Boundary rule
This skill owns adaptive routing logic with explicit queue and budget assumptions. It does not own detector implementation, UI surfaces, or governance artifacts.

## Design procedure
1. Name the operational actions first. Use the smallest action set that fits the case, usually `proceed`, `review`, and `escalate`.
2. Write down the simpler baseline before proposing anything more complex: one threshold plus SOP manual review.
3. List the runtime gating signals that really exist:
   - interval width or score band
   - validity posture or freshness state
   - drift or staleness flags if already available
4. State explicit capacity assumptions:
   - case arrival rate
   - reviewer throughput
   - queue or backlog cap
   - review SLA
   - daily deferral or escalation budget
5. Build a monotonic policy table. Better validity and tighter intervals should not route to stricter actions than worse validity and wider intervals without a stated reason.
6. Define overload behavior explicitly. Do not leave queue growth as an implied operations problem.
7. Compare the design against the simpler baseline and say when the baseline wins.

## Output structure

### POLICY TABLE
Return a concrete table with signal bands, validity posture, action, and overload fallback.

### THRESHOLD RATIONALE
Explain why each threshold or band exists and why fewer thresholds would or would not be enough.

### CAPACITY ASSUMPTIONS
State every operational assumption directly. If a number is missing, make a conservative assumption and label it as such.

### OVERLOAD FAILURE MODES
Name the failure modes that matter operationally: queue collapse, hidden labor, KPI gaming, silent over-deferral, or escalation saturation.

### BASELINE COMPARISON
Compare the proposed policy against the threshold-plus-SOP manual review baseline on simplicity, review burden, and failure containment.

### SIMPLER BASELINE PREFERRED WHEN
State the exact conditions where the simpler baseline is better and the nuanced policy should be rejected.


## Constraints

- Do not recommend deferral or escalation without explicit review-capacity assumptions.
- Do not drift into UI design, governance schema design, or generic staffing advice detached from CE or CEE uncertainty signals.
- Treat validity posture as a routing input, not as a guarantee of correctness.
- Prefer the fewest thresholds or bands that materially improve on the simpler baseline.
- Always specify overload behavior when review demand exceeds the stated budget.

## Self-Check Before Responding

- [ ] Are capacity assumptions explicit rather than implied?
- [ ] Is the policy table operationally executable and monotonic?
- [ ] Are overload and queue-collapse failure modes named directly?
- [ ] Is the simpler threshold-plus-SOP manual review baseline compared explicitly?
- [ ] Does the answer say when the simpler baseline is preferable?