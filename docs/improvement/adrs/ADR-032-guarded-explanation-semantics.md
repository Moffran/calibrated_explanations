> **Status note (2026-03-24):** Last edited 2026-03-24 · Archive after: Retain indefinitely as architectural record · Implementation window: v0.11.x.

# ADR-032: Guarded Explanation Semantics and Single-Median-Probe Guarding

## Status
Status: Accepted (scoped)
Date: 2026-03-20
Updated: 2026-03-24
Deciders: Core maintainers
Reviewers: Core maintainers
Supersedes: None
Superseded-by: None
Related: ADR-020, ADR-021, ADR-026, ADR-029

## Context
Guarded explanations were introduced to reduce implausible or out-of-distribution rule candidates in factual and alternative CE workflows.

Guarded mode must satisfy two competing goals:

1. preserve CE-compatible containers, helper surfaces, and prediction payloads so existing downstream workflows keep working;
2. apply a guard rule that is conservative enough to avoid over-trusting interval candidates.

All bins (numerical and categorical) are evaluated via a single guard probe at the median calibration value within the bin. This provides a uniform, simple, and interpretable guard rule. The median representative also serves as the payload anchor for calibrated prediction values and CE helper compatibility.

This ADR therefore defines guarded mode as a CE-compatible extension with a single-probe guard rule chosen to balance conservative interval screening, stable payload semantics, and backward-compatible helper interoperability.

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

3. **Guardedness is defined by a single median-probe guard rule.**
   - All candidates (categorical and numerical) are evaluated at a single representative perturbed point: the median calibration value within the bin.
   - Empty bins (no calibration samples) receive `p_value=0.0` and are marked non-conforming.
   - The stored `representative` is the median value used for both the guard probe and the candidate's calibrated prediction payloads.
   - Merged intervals are re-checked using the same single-median-probe rule on the merged range.
   - An emitted guarded interval means the guard rule accepted that interval candidate. It does **not** certify that every point inside the interval would pass the guard.

4. **`merge_adjacent` is a heuristic compaction step, not a semantic-preservation guarantee.**
   - In factual mode, adjacent conforming bins may be merged freely, including across the factual region.
   - Such merged rules can produce interval conditions that standard factual CE would never emit.
   - Emitted interval conditions use the raw bin bounds (`lower` / `upper`).
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
- Single-probe logic is simpler, more predictable, and easier to audit.

Negative / Risks:

- Guarded mode can no longer be described as mathematically equivalent to standard CE.
- Users must understand that emitted guarded intervals reflect this candidate-level guard rule, not whole-interval certification.
- Calibration-feature divergence now fails fast instead of degrading with a warning.
- Fast explainers cannot use guarded entrypoints at all; users who need guarded filtering must use a standard (non-fast) explainer.

## Addendum: Guarded Auditability
To support transparent guarded diagnostics without breaking CE payload contracts:

1. Guarded explanations provide a dedicated audit API (`get_guarded_audit()`), separate from `get_rules()`.
2. The audit payload includes:
   - interval-level records (bounds, representative, p-value, conforming, emitted, emission reason),
   - summary counts including:
     - `intervals_conforming`: candidate intervals accepted by the guarded decision rule,
     - `intervals_removed_guard`: candidate intervals rejected by the guarded decision rule.
3. `CalibratedExplanations.get_guarded_audit()` aggregates per-instance guarded audits and raises an actionable error on non-guarded collections.
4. The audit `p_value` is always the p-value from the single median-probe guard check.
5. Existing rule payload schemas remain unchanged to preserve backward compatibility and CE helper interoperability. This addendum does **not** imply semantic identity with standard CE internals.
