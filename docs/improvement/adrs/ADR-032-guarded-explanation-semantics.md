# ADR-032: Guarded Explanation Semantics and Semantic Identity

## Status
Proposed

## Context
Guarded (in-distribution) explanations were introduced to prevent misleading feature attribution or alternatives for data points that fall outside the high-density regions of the training data.

The goal is to provide a "guarded" version of both factual and alternative explanations without diverging from the core mathematical and API contracts established by the standard Calibrated Explanations (CE) library.

## Decision
1. **Semantic Identity**: Guarded explanations MUST be semantically identical to standard CE explanations in all internal data structures, metrics, and API surfaces.
   - The only permitted differences are:
     - The **bin conditions** (must use interval rules `a < x <= b` from multi-bin discretisation).
     - The **perturbed instances** (only those classified as in-distribution are selected).
2. **Feature Parity**: All existing features (parallelism, caching, narratives, conjunctions, reject policies, etc.) must function identically for guarded explanations.
3. **Internal Data Structures**:
   - `GuardedFactualExplanation` must subclass `FactualExplanation`.
   - `GuardedAlternativeExplanation` must subclass `AlternativeExplanation`.
   - Resulting `CalibratedExplanations` containers must adhere to the standard schema.
4. **Invariant Enforcement**:
   - The "Interval Invariant" (`low <= predict <= high`) must be strictly enforced for all selected perturbed bins. Any violation must either trigger an immediate validation error or be handled according to established core invariants (ADR-021).
5. **Exchangeability**:
   - The `InDistributionGuard` must operate on the same `x_cal` set used for conformal prediction intervals in the `CalibratedExplainer`.

## Consequences
- Users gain trust that "Guarded" means "fewer, better rules" rather than "different math".
- Plotting plugins and narrative generators do not need modification to support guarded explanations.
- Any improvement to core CE (like new plot styles or caching) automatically benefits guarded explanations.

## Addendum: Guarded Auditability (2026-02-21)
To support transparent guarded diagnostics without breaking CE payload contracts:

1. Guarded explanations provide a dedicated audit API (`get_guarded_audit()`), separate from `get_rules()`.
2. The audit payload includes:
   - interval-level records (bounds, representative, p-value, conforming, emitted, emission reason),
   - summary counts including `intervals_removed_guard`, defined strictly as `conforming == False`.
3. `CalibratedExplanations.get_guarded_audit()` aggregates per-instance guarded audits and raises an actionable error on non-guarded collections.
4. Existing rule payload schemas remain unchanged to preserve backward compatibility and CE semantic identity.
