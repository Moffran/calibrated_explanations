> **Status note (2026-06-09):** Last edited 2026-06-09 · Archive after: Retain indefinitely as architectural record · Implementation window: v0.11.x.

# ADR-032: Guarded Explanation Semantics and Single-Median-Probe Guarding

## Status
Status: Accepted (scoped)
Date: 2026-03-20
Updated: 2026-06-09
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
   - The canonical guarded API is parameterized:
     - `explain_factual(..., guarded=True)` for guarded factual explanations.
     - `explore_alternatives(..., guarded=True)` for guarded alternative explanations.
   - `explain_guarded_factual(...)` and `explore_guarded_alternatives(...)` are deprecated
     compatibility wrappers that delegate to the canonical parameterized API.  They will
     be removed in v1.0.0.  Callers must migrate to the parameterized form.
   - Guarded is not an explanation mode.  The explanation mode taxonomy remains
     factual / alternative / fast.  `guarded` is a boolean policy flag on the request.
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

7. **Conjunctions are guarded via a joint representative perturbation.**
   - When `add_conjunctions()` is called on a guarded explanation, each generated conjunction
     is tested by building a **joint perturbation**: all constituent features are simultaneously
     set to the median of their respective bin's sampled values (the same median-probe rule as
     single-feature bins).
   - The joint point is evaluated by `InDistributionGuard.p_values()` at `significance`
     (unadjusted — a conjunction is one holistic test, not a multi-bin Bonferroni candidate).
   - Conjunctions whose joint perturbation fails (`p_value < significance`) are removed from
     the rule set before `add_conjunctions` returns.
   - Single-feature bins that already passed the original guard are not re-tested; only the
     joint representative point is evaluated for the conjunction.
   - Conjunction audit records are appended to `get_guarded_audit()["conjunctions"]` and include
     `features`, `p_value`, `conforming`, and `emission_reason` (`"emitted"` or
     `"removed_guard"`).
   - Summary keys `conjunctions_tested`, `conjunctions_guard_removed`, and
     `conjunctions_emitted` are added to `get_guarded_audit()["summary"]` when conjunctions
     were tested.

8. **The plugin contract exposes exactly one guarded-support metadata field.**
   - Plugin metadata may include `"supports_guarded": True` to declare that the plugin
     is compatible with guarded execution.
   - The default for missing metadata is `"supports_guarded": False`.
   - The runtime request carries `guarded: bool` on `ExplanationRequest`.
   - No additional guarded-specific capability tags (e.g. `"explanation:guarded"`) are
     permitted.  The single boolean `supports_guarded` is the complete contract.
   - The plugin resolver must not silently select an unguarded-only plugin for guarded
     execution.  If no `supports_guarded=True` plugin is available for a given
     modality/mode/task combination, `ValidationError` is raised.

9. **Guarded explanations are not supported for fast explainers.**
   - Fast interval calibrators are trained on per-feature blends of `scaled_x_cal` / `fast_x_cal`, not on `explainer.x_cal` directly.
   - The `InDistributionGuard` always uses `explainer.x_cal` as its reference distribution.
   - These two distributions cannot be aligned, so the ADR-032 precondition (decision 6) cannot be reliably enforced for fast explainers.
   - Calling any guarded entrypoint (`explain_factual(..., guarded=True)`, `explore_alternatives(..., guarded=True)`) on a fast explainer must hard-fail with `ConfigurationError` before any calibration-alignment check proceeds.
   - This prohibition is enforced in `_require_guarded_calibration_alignment` and is not subject to configuration or opt-out.

## Consequences

Positive:

- The repository now makes one defensible guarded claim instead of implying “same math.”
- Public CE helper surfaces remain usable on guarded outputs.
- Audit payloads can be interpreted without overstating what guarded conformity means.
- Single-probe logic is simpler, more predictable, and easier to audit.
- A single public API entry point (`explain_factual` / `explore_alternatives`) handles both
  guarded and unguarded execution, eliminating the parallel method surface.
- The plugin contract has exactly one guarded-support field (`supports_guarded`), preventing
  redundant or contradictory guarded declarations.

Negative / Risks:

- Guarded mode can no longer be described as mathematically equivalent to standard CE.
- Users must understand that emitted guarded intervals reflect this candidate-level guard rule, not whole-interval certification.
- Calibration-feature divergence now fails fast instead of degrading with a warning.
- Fast explainers cannot use guarded entrypoints at all; users who need guarded filtering must use a standard (non-fast) explainer.
- The deprecated `explain_factual(..., guarded=True)` / `explore_alternatives(..., guarded=True)` wrappers must not
  be used in new code.  Remove usage before v1.0.0.

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

## Addendum: Target-Confidence Filter for Alternative Explanations (v0.11.3)

**Decision date:** 2026-06-10
**Option chosen:** post-processing filter method on the returned collection.

### Rationale

The existing `reject_policy` mechanism operates at the source-instance level (which inputs to
explain); the `guarded` filter operates at the interval level (which perturbation intervals are
in-distribution).  Neither answers: *"Would acting on this suggested change actually land in a
confidently accepted outcome?"*  A target-confidence filter fills that gap as a complementary,
opt-in post-processing step using conformal decision-making.

Option C was selected over Option A (new keyword parameter) and Option B (new `RejectPolicySpec`
factory) because it keeps the core generation API unchanged.  The downside (generating intervals
that may later be discarded) is acceptable at v0.11.3 scope.

### Conformal classification semantics

The filter applies conformal classification using the hinge non-conformity function (NCF)
and the stored calibration set.  The Venn-Abers calibrated probability `predict` for the
positive class is used to derive per-class NCF scores; proper conformal p-values are then
obtained by comparing these scores against the calibration NCF distribution.

**Calibration NCF scores** (computed once per `filter_by_target_confidence` call):

```
alpha_cal[i] = 1 − P(true_class_i | x_cal_i)
             = 1 − proba_cal[i]   if y_cal[i] == 1   (NCF for class 1)
             = proba_cal[i]        if y_cal[i] == 0   (NCF for class 0)
```

**Per-interval NCF scores** for a perturbation representative with `predict = p`:

```
alpha_1 = 1 − p   (NCF for class 1: 1 − P(class 1 | x))
alpha_0 = p        (NCF for class 0: 1 − P(class 0 | x) = p)
```

**Conformal p-values** (fraction of calibration NCF scores at least as non-conforming):

```
p_val_k = (|{i : alpha_cal[i] >= alpha_k}| + 1) / (n_cal + 1)
```

**Singleton test** (epsilon = 1 − confidence):

| Condition | in_set_1 | in_set_0 | Decision |
|---|---|---|---|
| Accepted class 1 | `p_val_1 >= epsilon` | `p_val_0 < epsilon` | retain |
| Accepted class 0 | `p_val_1 < epsilon` | `p_val_0 >= epsilon` | retain |
| Ambiguity-rejected | `p_val_1 >= epsilon` | `p_val_0 >= epsilon` | discard |
| Novelty-rejected | `p_val_1 < epsilon` | `p_val_0 < epsilon` | discard |

At `confidence=1.0` (epsilon=0.0), both p-values always exceed the threshold, so all
intervals are ambiguity-rejected.  At `confidence=0.0` (epsilon=1.0), only intervals
with the most calibration-extreme predictions survive.

### API contract

```python
AlternativeExplanations.filter_by_target_confidence(
    self,
    confidence: float = 0.8,
) -> "AlternativeExplanations"
```

**Parameters**

- `confidence` (`float`, default `0.8`, range `[0.0, 1.0]`): Conformal confidence level.
  Maps to significance `epsilon = 1 - confidence`.  An interval is retained only when its
  conformal prediction set is a singleton at `epsilon` — i.e. exactly one class has a
  conformal p-value `>= epsilon`.  Higher values apply a stricter filter.

**Returns**

- A new `AlternativeExplanations` instance (same concrete type as `self`) containing only
  the intervals that pass the singleton-prediction-set criterion at `confidence`.
  The original collection is **not mutated**.

**Raises**

- `ValidationError` if `confidence` is outside `[0.0, 1.0]`.
- `ValidationError` if the underlying model does not produce probability outputs (i.e.
  `is_probabilistic()` is `False` for any explanation in the collection).

**Stability obligation (ADR-011)**

This method name and signature are stable as of v0.11.3 and must not be changed or removed
before v1.0.0.  Any semantic change requires a deprecation notice per ADR-011.
