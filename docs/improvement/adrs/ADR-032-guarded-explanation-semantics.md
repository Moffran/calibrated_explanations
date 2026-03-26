> **Status note (2026-03-24):** Last edited 2026-03-24 · Archive after: Retain indefinitely as architectural record · Implementation window: v0.11.x.

# ADR-032: Guarded Explanation Semantics and Schema-Compatible Dense-Bin Two-Probe Guarding

## Status
Status: Accepted (scoped)
Date: 2026-03-20
Deciders: Core maintainers
Reviewers: Core maintainers
Supersedes: None
Superseded-by: None
Related: ADR-020, ADR-021, ADR-026, ADR-029

## Context
Guarded explanations were introduced to reduce implausible or out-of-distribution rule candidates in factual and alternative CE workflows.

Guarded mode must satisfy two competing goals:

1. preserve CE-compatible containers, helper surfaces, and prediction payloads so existing downstream workflows keep working;
2. apply a guard rule that is conservative enough to avoid over-trusting interval candidates, especially when numerical bins are wide or merges create broader emitted conditions.

A single median representative is adequate for sparse bins and categorical candidates, where the candidate itself is either explicit or tightly localized. For dense numerical bins, however, a single median probe can understate risk near the edges of the emitted interval. A conservative two-probe check at `q10` and `q90` gives a better interval-level screening rule while still allowing the median representative to remain the payload anchor for calibrated prediction values and CE helper compatibility.

This ADR therefore defines guarded mode as a CE-compatible extension with a mixed guard rule chosen to balance conservative interval screening, stable payload semantics, and backward-compatible helper interoperability.

## Decision
1. **Schema compatibility and helper interoperability are the guarded contract.**
   - `explain_guarded_factual(...)` and `explore_guarded_alternatives(...)` keep their public signatures.
   - Guarded entrypoints return standard CE collection classes with guarded subclasses:
     - `GuardedFactualExplanation` subclasses `FactualExplanation`.
     - `GuardedAlternativeExplanation` subclasses `AlternativeExplanation`.
   - The following compatibility surfaces are guaranteed to keep working on guarded outputs:
     - collection/explanation containers,
     - `get_guarded_audit()`,
     - conjunction helpers,
     - plotting,
     - narratives,
     - reject-policy wrapping.

2. **Guarded mode is not semantically identical to standard CE internals or metrics.**
   - Guarded execution is a sanctioned core-side guarded path, not an explanation-plugin mode.
   - Guarded outputs are not contracted to preserve metric identity, perturbation identity, or internal-data-structure identity with standard CE.
   - Compatibility shims may populate CE helper caches and payload fields so downstream helper surfaces continue to operate.

3. **Guardedness is defined by a mixed guard rule chosen to match candidate structure.**
   - Categorical candidates are evaluated directly at the candidate value.
   - Sparse numerical candidates are evaluated at a single representative perturbed point, typically the median calibration value in that candidate.
   - Dense numerical candidates are evaluated conservatively using two guard probes, `q10` and `q90`, with the decision p-value defined as `min(p(q10), p(q90))` and the acceptance threshold adjusted accordingly.
   - The stored `representative` remains the median value used for the candidate's calibrated prediction payloads and helper compatibility surfaces.
   - Merged dense intervals are re-checked using the same dense-interval two-probe rule.
   - An emitted guarded interval therefore means this guard rule accepted that interval candidate. It does **not** certify that every point inside the interval would pass the guard.

4. **`merge_adjacent` is a heuristic compaction step, not a semantic-preservation guarantee.**
   - In factual mode, adjacent conforming bins may be merged freely, including across the factual region.
   - Such merged rules can produce interval conditions that standard factual CE would never emit.
   - Emitted interval conditions may also be narrower than raw bin bounds via `emitted_lower` / `emitted_upper` when dense-bin guarding truncates the displayed condition.
   - This behavior is allowed and should be documented as a guarded-specific rule-compaction heuristic.

5. **Structural interval validity follows ADR-021.**
   - Guarded outputs must continue to satisfy ADR-021’s structural invariant `low <= predict <= high`.
   - This invariant is a structural validity rule for emitted predictions and intervals.
   - It must not be described as a coverage or correctness guarantee for whole guarded intervals.

6. **Exchangeability alignment is enforced as a hard precondition.**
   - The guard and the active interval learner must use the same calibration feature matrix values and shape.
   - Guarded entrypoints must fail with `ValidationError` when backend calibration features are unavailable or differ from `explainer.x_cal`.
   - Equality is defined by array shape plus value equality, not Python object identity.

7. **Guarded explanations are not supported for fast explainers.**
   - Fast interval calibrators are trained on per-feature blends of `scaled_x_cal` / `fast_x_cal`, not on `explainer.x_cal` directly.
   - The `InDistributionGuard` always uses `explainer.x_cal` as its reference distribution.
   - These two distributions cannot be aligned, so the ADR-032 precondition (decision 6) cannot be reliably enforced for fast explainers.
   - Calling any guarded entrypoint (`explain_guarded_factual`, `explore_guarded_alternatives`) on a fast explainer must hard-fail with `ConfigurationError` before any calibration-alignment check proceeds.
   - This prohibition is enforced in `_require_guarded_calibration_alignment` and is not subject to configuration or opt-out.

## Consequences

Positive:

- The repository now makes one defensible guarded claim instead of implying “same math.”
- Public CE helper surfaces remain usable on guarded outputs.
- Audit payloads can be interpreted without overstating what guarded conformity means.

Negative / Risks:

- Guarded mode can no longer be described as mathematically equivalent to standard CE.
- Users must understand that emitted guarded intervals reflect this conservative candidate-level guard rule, not whole-interval certification.
- For dense numerical intervals, users must not read `representative` and `p_value` as if they always came from the same single perturbation.
- Calibration-feature divergence now fails fast instead of degrading with a warning.
- Fast explainers cannot use guarded entrypoints at all; users who need guarded filtering must use a standard (non-fast) explainer.

## Addendum: Guarded Auditability
To support transparent guarded diagnostics without breaking CE payload contracts:

1. Guarded explanations provide a dedicated audit API (`get_guarded_audit()`), separate from `get_rules()`.
2. The audit payload includes:
   - interval-level records (bounds, representative, p-value, conforming, emitted, emission reason),
   - optional emitted-condition bounds (`emitted_lower`, `emitted_upper`) when the displayed condition is narrower than the raw bin bounds,
   - summary counts including:
     - `intervals_conforming`: candidate intervals accepted by the guarded decision rule,
     - `intervals_removed_guard`: candidate intervals rejected by the guarded decision rule.
3. `CalibratedExplanations.get_guarded_audit()` aggregates per-instance guarded audits and raises an actionable error on non-guarded collections.
4. In dense numerical candidates, the audit `p_value` is the decision p-value used by the guarded decision rule and is not necessarily the p-value of the median `representative` alone.
5. Existing rule payload schemas remain unchanged to preserve backward compatibility and CE helper interoperability. This addendum does **not** imply semantic identity with standard CE internals.
