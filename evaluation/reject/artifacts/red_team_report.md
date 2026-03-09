# Red Team Report — Reject Framework
<!-- Generated 2026-03-08. Read-only findings. No fixes proposed. -->

**Scope:** Implementation · Evaluation · Documentation · Requirements
**Files attacked:** `orchestrator.py` · `explanations/reject.py` · `core/reject/policy.py` · ADR-029 · evaluation scenarios A–D · test suite (13 files, ~936 lines)

---

## Executive Summary

The reject framework has a well-structured skeleton: a clean enum API, a strategy registry, a metadata-rich envelope, and four evaluation scenarios covering binary classification, multiclass, and regression. The core integration is usable. However, **three P0 issues warrant blocking the v0.11.x release**: the blended-NCF design breaks conformal nesting (evidenced by the already-reported monotonicity violations), the multiclass NCF computation is architecturally inconsistent with its advertised semantics, and the error-rate formula can return negative values with no guard. Beyond these, twelve additional implementation, evaluation, documentation, and requirements gaps require resolution before the framework can be claimed correct.

---

## P0 — Release Blockers

### P0-1 · Blended NCF Breaks Conformal Nesting (Implementation / Scientific)

**Location:** `orchestrator.py:78–89` (`_ncf_scores_test`), `orchestrator.py:58–75` (`_ncf_scores_cal`)

When `w < 1.0` and `ncf != "hinge"`, calibration scores use `(1-w)*base_ncf(proba) + w*hinge(proba, classes, labels)` (1-D) while test scores use `(1-w)*base_ncf(proba) + w*hinge(proba)` (2-D, class-wise). The two blends are **structurally different**: calibration scores are scalar per instance; test scores are one column per class. Standard conformal prediction requires that calibration and test non-conformity scores are exchangeable under the null. Mixing a 1-D calibration blend with a 2-D class-specific test blend violates this exchangeability. The consequence is that prediction sets are **not nested** as the confidence level varies — which is exactly what Scenario C reports (1 coverage violation, 1 accepted-confidence violation). This is not a data artifact: it is a structural guarantee violation. The violations are reported in `artifacts/summary.md` without investigation or explanation, signalling the problem was observed but not understood.

**Why it matters:** A conformal rejection framework that does not produce nested prediction sets cannot guarantee monotonicity of coverage w.r.t. confidence. Any downstream governance or reliability claim depends on this property.

---

### P0-2 · Multiclass NCF Claims vs. Actual Computation (Implementation / Scientific)

**Location:** `orchestrator.py:229–236` (`initialize_reject_learner`), `orchestrator.py:263–267` (`_compute_prediction_set`)

For multiclass inputs, both calibration and test probability matrices are binarized:
```python
proba = np.array([[1 - proba[i, c], proba[i, c]] for i, c in enumerate(predicted_labels)])
```
This creates a (n, 2) matrix where only the predicted class probability is retained. All NCFs — including `entropy` and `margin` — then operate on this 2-column binarized representation, not the full K-class distribution.

**Consequence for `entropy`:** With a 2-column matrix, entropy reduces to `-p*log2(p) - (1-p)*log2(1-p)`, which is binary entropy of the top-class confidence. This is not Shannon entropy over K classes as documented.

**Consequence for `margin`:** Operates on `[1-p_argmax, p_argmax]`, computing `1 - (p_argmax - (1-p_argmax)) = 2*(1-p_argmax)`. This is a confidence score, not the "top-two probability gap across K classes" as the docstring states.

**Consequence for Scenario B:** The "best NCF = ensured for multiclass" conclusion is an artifact of this binarized architecture. The relative NCF ranking is dataset-specific and architecture-specific, not a general finding.

**Why it matters:** The NCF descriptions mislead users. Documentation and marketing claims about entropy/margin for multiclass are inaccurate. Scenario B's conclusions cannot be generalised.

---

### P0-3 · Error Rate Formula Returns Negative Values (Implementation)

**Location:** `orchestrator.py:500–503`

```python
if num_instances == 0 or singleton == 0:
    error_rate = 0.0
else:
    error_rate = (num_instances * epsilon - empty) / singleton
```

When `empty > num_instances * epsilon` (many novelty/OOD rejections, small epsilon), the formula returns a **negative error rate**. Example: n=10, confidence=0.99 (epsilon=0.01), empty=3, singleton=7 → `error_rate = (0.1 - 3)/7 = -0.414`. There is no clamp, no guard, and no flag in metadata indicating the formula produced a nonsensical value. The sentinel for the undefined case (`singleton==0`) is `0.0`, indistinguishable from "perfect error rate". This means callers see negative numbers in `metadata["error_rate"]` with no indication of invalidity.

**Why it matters:** Any downstream consumer checking `error_rate < threshold` will silently pass highly-novel datasets through gating checks. Governance audit trails containing negative error rates are meaningless.

---

## P1 — Serious Issues (Should Fix Before v0.11.x)

### P1-1 · Stale Aggregate Rates After Slicing (Implementation)

**Location:** `explanations/reject.py:236–239` (`_slice_reject_fields`)

```python
# "For simple parity, we copy the original metadata dict reference or values."
self._metadata = copy(source._metadata)
```

The comment acknowledges the problem. After slicing a `RejectCalibratedExplanations` to a subset, per-instance arrays (`rejected`, `ambiguity_mask`, `novelty_mask`, `prediction_set_size`) are correctly sliced, but aggregate rates (`reject_rate`, `error_rate`, `ambiguity_rate`, `novelty_rate`) are copied from the original and no longer match the slice. A caller accessing `result.metadata["reject_rate"]` after slicing to only accepted instances will see the rate for the full original batch. The property at `explanations/reject.py:189–201` merges `_metadata` with sliced arrays but does not recompute rates.

**Why it matters:** Any downstream monitoring or reporting that slices an explanation collection and reads aggregate metadata will silently get wrong numbers.

---

### P1-2 · Plain RejectPolicy Bypasses NCF Reinitialization (Implementation / Requirements)

**Location:** `orchestrator.py:117–118` (`resolve_policy_spec`)

```python
if not isinstance(reject_policy_kw, RejectPolicySpec):
    return reject_policy_kw   # returned unchanged; no NCF check
```

Only `RejectPolicySpec` triggers NCF comparison and reinitialization. A plain `RejectPolicy` enum value (e.g., `RejectPolicy.FLAG`) passes through without any check of whether the current reject learner's NCF matches any intended configuration. This creates a silent configuration mismatch: a user who calls `explain(..., reject_policy=RejectPolicy.FLAG)` after previously using `RejectPolicySpec.flag(ncf="entropy")` will silently use the entropy-trained reject learner with no warning.

**ADR-029 conflict:** The "Operational rule" (ADR-029, §Consequences) states "Selecting any non-`NONE` `RejectPolicy` MUST implicitly enable the reject orchestration". Implicitly enabling it with a stale NCF configuration is not equivalent to "enabling" it correctly.

---

### P1-3 · Auto-Selected NCF Not Recorded in Metadata (Implementation / Reproducibility)

**Location:** `orchestrator.py:197–203` (`initialize_reject_learner`)

```python
ncf_explicit = ncf is not None
if ncf is None:
    ncf = "margin" if self.explainer.is_multiclass() else "hinge"
_ = ncf_explicit  # suppress warning
```

The `ncf_explicit` flag is computed but then discarded (`_ = ncf_explicit`). No metadata field records which NCF was auto-selected vs. explicitly chosen. A user who calls `initialize_reject_learner()` with no arguments and later inspects `explainer.reject_ncf` gets the resolved value, but has no record that it was auto-selected. Evaluation artifacts and logs do not capture this selection.

**Why it matters:** Two users running the same pipeline on the same data but with different `is_multiclass()` return values get different NCF behaviours silently. Not reproducible from metadata alone.

---

### P1-4 · Empty Subset Returns None Without Distinguishable Signal (Implementation)

**Location:** `orchestrator.py:666–675`, `orchestrator.py:676–685`

```python
if idx:
    explanation = explain_fn(subset, **kwargs)
else:
    explanation = None
```

When `ONLY_REJECTED` finds no rejected instances, or `ONLY_ACCEPTED` finds no accepted instances, `RejectResult.explanation` is `None`. This is the same value as "explain_fn was not provided" (also `None`). The metadata does not include a count of matched instances or a flag indicating "empty subset". Callers must independently check `rejected.sum() == 0` or `(~rejected).sum() == 0` to distinguish these cases, and this logic is undocumented.

---

### P1-5 · Class-Swap in from_collection() Is Fragile (Implementation)

**Location:** `explanations/reject.py:254–265`

```python
obj = copy(base)
obj.__class__ = cls
```

`copy(base)` creates a shallow copy of a `CalibratedExplanations` instance, then reassigns its class to `RejectCalibratedExplanations`. This does not call `RejectCalibratedExplanations.__init__` or any MRO initialization. If `CalibratedExplanations` has slots, `__init_subclass__` hooks, or `__post_init__` logic in any mixin, the class-swapped object may be in an inconsistent state. The pattern is also refactoring-hostile: adding any field initialization to `RejectCalibratedExplanations.__init__` in the future would silently be skipped for all objects created via `from_collection`.

---

### P1-6 · w=0.0 Is Accepted Without Hard Guard (Implementation)

**Location:** `orchestrator.py:209–216`

Warning fires at `w < 0.1` for non-hinge NCFs, but `w=0.0` is a valid configuration. With `w=0.0`, test scores become `(1.0)*base_ncf(proba)` — class-independent scalars broadcast across all classes. The resulting prediction sets are either all classes or no classes for every instance, producing 100% rejection or 100% ambiguity. The warning text says "consider w >= 0.1" but does not block execution. There is no documented lower bound in the API.

---

### P1-7 · Scenario C Monotonicity Violations Unreported and Unexplained (Evaluation / Scientific)

**Location:** `evaluation/reject/artifacts/summary.md:7–12`

The summary reports `coverage_monotonicity_violations: 1` and `accepted_confidence_monotonicity_violations: 1` but provides no analysis. These violations are not flagged as failures, warnings, or open issues. They are presented as descriptive statistics alongside the positive results. Given that P0-1 identifies a structural reason why these violations can occur (non-nested prediction sets from blended NCF), an evaluation that reports violations without investigation provides false confidence.

---

### P1-8 · ADR-029 Envelope Prediction Contract Untested for Regression (Requirements / Evaluation)

**Location:** ADR-029, §Consequences: "envelope's `prediction` field MUST mirror the invoked method's legacy payload (including tuple shapes used by regression UQ, e.g., `(proba, (low, high))`)"

Scenario D exercises regression rejection but validates threshold-based binary accuracy and coverage metrics. The `RejectResult.prediction` field shape for regression (which may be a tuple `(proba, (low, high))`) is not checked in any evaluation scenario or referenced test. The ADR's binding requirement for this shape is therefore unverified.

---

## P2 — Minor Issues (Should Fix Before v1.0.0)

### P2-1 · RejectContext Has No Template System (Implementation / Documentation)

**Location:** `explanations/reject.py:172–175`

```python
# Rendered strings (optional) - templates preferred instead of hardcoding
beginner_text: str | None = None
```

The comment promises "templates preferred instead of hardcoding" but no template system exists. Callers must manually generate three expertise-level strings per instance, creating duplication risk across any downstream consumers.

---

### P2-2 · Error Rate Semantics When Singleton==0 Are Undocumented (Documentation)

**Location:** `orchestrator.py:499–501`; ADR-029 §Consequences (no mention of undefined case)

The `error_rate = 0.0` sentinel for undefined state is not specified in ADR-029 or any user-facing documentation. The docstring for `predict_reject_breakdown` does not describe this edge case.

---

### P2-3 · w Parameter Guidance Absent from User Docs (Documentation)

**Location:** `docs/improvement/reject_policy_usage.md`, `docs/practitioner/advanced/reject-policy.md`

No practical guidance on `w` values for each NCF type exists. The only signal is the `w < 0.1` warning. Users have no basis for selecting `w` without empirical trial and error.

---

### P2-4 · Auto-NCF Selection Undocumented in User-Facing Docs (Documentation)

**Location:** `docs/practitioner/advanced/reject-policy.md`

The auto-selection rule (`margin` for multiclass, `hinge` for binary/regression) is not described in user-facing documentation. A user calling `initialize_reject_learner()` with no arguments cannot predict the selected NCF without reading source code.

---

### P2-5 · WrapCalibratedExplainer Constructor Constraint Has No Enforced Test (Requirements)

**Location:** ADR-029 §Consequences: "`WrapCalibratedExplainer.__init__` MUST NOT accept `default_reject_policy`"

This constraint has no corresponding `pytest.raises` test that validates constructor rejection of `default_reject_policy`. The constraint can be silently violated by future refactoring.

---

### P2-6 · Thread Safety During Unpickling Has a Race Window (Implementation)

**Location:** `orchestrator.py:154–158` (`__setstate__`)

```python
def __setstate__(self, state: dict) -> None:
    self.__dict__.update(state)          # strategies dict loaded here
    self._strategies_lock = threading.RLock()   # lock created after dict exposed
```

Between `__dict__.update` and `_strategies_lock = threading.RLock()` creation, the `_strategies` dict is accessible without a lock. Concurrent access during multi-threaded unpickling (e.g., joblib deserialization in parallel explain calls) can race.

---

## Missing Tests

| ID | Gap | Risk |
|----|-----|------|
| MT-1 | Conformal nesting property: monotonicity of prediction sets across confidence levels for all NCFs | High — P0-1 is undetected without this |
| MT-2 | `error_rate` is clipped to `[0, 1]` or flagged as undefined when formula goes negative | High — P0-3 undetected |
| MT-3 | `RejectResult.metadata["error_rate"]` is non-negative for all evaluation scenarios | High |
| MT-4 | After slicing `RejectCalibratedExplanations`, aggregate rates match recomputed values from sliced masks | Medium — P1-1 |
| MT-5 | `explain(..., reject_policy=RejectPolicy.FLAG)` after `RejectPolicySpec.flag(ncf="entropy")` uses entropy learner | Medium — P1-2 |
| MT-6 | `RejectResult.explanation is None` when `ONLY_REJECTED` finds 0 rejected instances; caller can distinguish from "no explain_fn" | Medium — P1-4 |
| MT-7 | `WrapCalibratedExplainer.__init__(default_reject_policy=...)` raises `TypeError` | Medium — P2-5 |
| MT-8 | Regression `RejectResult.prediction` shape matches legacy tuple `(proba, (low, high))` | High — P1-8 |
| MT-9 | Multiclass NCF scores with K>2 classes differ from binarized scores (document or fix the binarization) | High — P0-2 |
| MT-10 | `w=0.0` produces 100% rejection or 100% ambiguity (document hard limit) | Medium — P1-6 |

---

## Evaluation Blind Spots

| ID | Scenario | Issue |
|----|----------|-------|
| EB-1 | Scenario C (4 rows) | 4 confidence levels is insufficient to claim monotonicity. Standard conformal monotonicity tests use ≥10 levels; 4 levels with 1 violation at `n=38` (coverage ~0.67) has no statistical power. |
| EB-2 | Scenario B (8 rows, 2 seeds) | Iris (150 samples, 3 classes) with 2 seeds. NCF ranking is not stable enough to claim "best NCF = ensured". A 2-seed bootstrap at n=54 per fold is not adequate for a comparative claim. |
| EB-3 | Scenario B | NCF comparison conclusion is invalid given P0-2: all NCFs operate on a binarized (n,2) matrix in multiclass mode. The comparison does not distinguish NCF behaviour from binarization artefacts. |
| EB-4 | Scenario D | The regression threshold is derived from the training/calibration data quantile, not a holdout set. The "best threshold = 0.25" may overfit the calibration distribution. |
| EB-5 | All | No evaluation of explainer-level `default_reject_policy` vs. per-call `reject_policy` override precedence. |
| EB-6 | All | No evaluation of the strategy registry: only `builtin.default` is tested. Extensibility claims are assertions, not measured. |
| EB-7 | Scenario C | The two monotonicity violations are reported as descriptive statistics, not as test failures. An evaluation that tolerates conformal guarantee violations as "findings" conflates empirical exploration with verification. |

---

## Documentation Fixes Required

| ID | File | Fix Required |
|----|------|-------------|
| DF-1 | `docs/practitioner/advanced/reject-policy.md` | Document NCF auto-selection rule and resulting NCF stored in `explainer.reject_ncf` |
| DF-2 | `docs/improvement/reject_policy_usage.md` | Add `w` guidance per NCF with empirical reference ranges |
| DF-3 | ADR-029 §Consequences | Specify `error_rate` semantics when undefined (negative or `singleton==0`); either prohibit or define a sentinel contract |
| DF-4 | ADR-029 §Open Questions | Close the open question on `RejectContext` template system or mark as deferred with an explicit ADR |
| DF-5 | ADR-029 §Consequences | Specify `RejectResult.explanation = None` semantics: distinguish "no explain_fn provided" from "0 instances matched filter" |
| DF-6 | `docs/practitioner/advanced/reject-policy.md` | Warn that blended NCF (`w < 1.0`) breaks conformal nesting; monotonicity is not guaranteed |

---

## Requirements Conflicts

| ID | Requirement (ADR-029) | Implementation | Conflict |
|----|----------------------|----------------|---------|
| RC-1 | "envelope `prediction` MUST mirror legacy payload including tuple shapes for regression UQ" | Prediction is obtained via `explainer.prediction_orchestrator.predict(x)` (line 647); shape is not validated or enforced | Shape contract is unenforced and untested |
| RC-2 | "Selecting any non-NONE RejectPolicy MUST implicitly enable reject orchestration" | Plain `RejectPolicy` enum bypasses NCF reinitialization in `resolve_policy_spec` | Enabling rejection with stale NCF ≠ correctly enabling rejection |
| RC-3 | "`WrapCalibratedExplainer.__init__` MUST NOT accept `default_reject_policy`" | No enforced test exists | Constraint can be silently violated |
| RC-4 | "Structured wrapper/envelope C3 — governance-friendly, versionable, allows strict type-checking" | `RejectResult` fields are all `Optional[Any]` with no schema versioning field | No version field; schema evolution is opaque; type-checking is impossible |

---

## Verdict

> **The reject framework is not ready for a v0.11.x feature-complete release.**

The framework is useful for exploration but carries three release-blocking scientific and numerical correctness issues (P0-1 through P0-3). The blended NCF design has a fundamental exchangeability problem that invalidates the conformal coverage guarantee; the multiclass NCF descriptions do not match their computation; and the error-rate formula can return negative values silently. The evaluation suite reports evidence of these problems (the monotonicity violations in Scenario C) without recognising them as failures.

**The strongest single objection:** The evaluation already observes that conformal monotonicity is violated (Scenario C, `coverage_monotonicity_violations: 1`) but treats this as a descriptive metric rather than a test failure. A conformal rejection framework that does not guarantee monotonic coverage as confidence varies does not provide the safety property it advertises, and the current evaluation methodology is incapable of detecting this because it treats violations as acceptable.
