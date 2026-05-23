# ADR-032: Guarded Explanation Semantics — Reference Extract

> **Source:** `docs/improvement/adrs/ADR-032-guarded-explanation-semantics.md`
> Loaded by `ce-factual-explain` and `ce-alternatives-explore` skills.

---

## Core Principle: Semantic Identity

Guarded explanations are **semantically identical** to standard CE explanations
in all internal data structures, metrics, and API surfaces.

Permitted differences (and ONLY these):
1. **Bin conditions** — must use interval rules `a < x <= b` from multi-bin discretisation.
2. **Perturbed instances** — only instances classified as in-distribution are selected.

Everything else (parallelism, caching, narratives, conjunctions, reject policies,
plotting) works identically.

## Class Hierarchy

```
FactualExplanation
  └── GuardedFactualExplanation     (subclass; same schema)

AlternativeExplanation
  └── GuardedAlternativeExplanation (subclass; same schema)
```

## Interval Invariant (ADR-032 §4 + ADR-021 §4)

The invariant `low <= predict <= high` is **strictly enforced** for all
selected perturbed bins in guarded explanations. Violations trigger an
immediate validation error (not silent clamping).

## When to Use Guarded vs Standard

| Scenario | Use |
|---|---|
| Training/development/research | `explain_factual` / `explore_alternatives` |
| Production endpoint (arbitrary user input) | `explain_guarded_factual` / `explore_guarded_alternatives` |
| Unknown input distribution | Guarded variants |
| Limited calibration set coverage | Guarded variants |

## Audit API (added 2026-02-21)

```python
audit = explanations.get_guarded_audit()
# Returns per-instance audit records:
# {
#   "intervals_removed_guard": int,   # count where conforming == False
#   "interval_records": [...],        # per-bin: bounds, p-value, emitted, reason
# }
```

Note: `get_guarded_audit()` raises an actionable error if called on
non-guarded `CalibratedExplanations` collections.

## InDistributionGuard Contract

The `InDistributionGuard` operates on the same `x_cal` set used for conformal
prediction intervals in `CalibratedExplainer`. It must NOT use a separate
calibration set.
