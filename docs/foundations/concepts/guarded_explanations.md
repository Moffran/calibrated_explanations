# Guarded explanations

Guarded explanations are a CE-compatible extension built around the shipped
guard rule for interval candidates.

Semantics and non-guarantees are defined in
{doc}`calibrated_interval_semantics` and
{doc}`../../improvement/adrs/ADR-032-guarded-explanation-semantics`.

## What guarded mode changes

- Uses a mixed guard rule on interval candidates:
  categorical values are checked directly,
  sparse numerical bins are checked at the median representative,
  and dense numerical bins are checked conservatively with `q10` and `q90`.
- Returns standard CE containers and helper-compatible subclasses.
- Removes candidate intervals rejected by the shipped guard rule.
- May merge adjacent conforming bins into wider interval rules when `merge_adjacent=True`.

## Core calls

```python
guarded_factual = explainer.explain_guarded_factual(X, significance=0.1)
guarded_alts = explainer.explore_guarded_alternatives(X, significance=0.1)
```

## What guarded mode does not guarantee

- It is not semantically identical to standard CE internals, metrics, or perturbation math.
- It does not certify that every point in an emitted interval would pass the guard.
- In factual mode, `merge_adjacent=True` can emit wider interval rules than standard CE would produce.

## Guarded audit

```python
audit = guarded_factual.get_guarded_audit()
```

The audit payload reports tested, conforming, removed, and emitted
guarded interval candidates.

- `intervals_conforming`: candidates accepted by the shipped guard rule.
- `intervals_removed_guard`: candidates rejected by the shipped guard rule.
- `emitted_lower` / `emitted_upper`: the displayed interval condition when dense-bin guarding narrows the raw bin bounds.

## Parameters

| Parameter | Meaning |
| --- | --- |
| `significance` | Larger values apply stricter filtering. |
| `merge_adjacent` | Merge adjacent conforming bins when enabled. |
| `n_neighbors` | KNN neighbors for conformity scoring. |
| `normalize_guard` | Apply per-feature normalization before distance calculations. |

## Related pages

- {doc}`../../get-started/quickstart_guarded`
- {doc}`alternatives`
- {doc}`../how-to/interpret_explanations`

Entry-point tier: Tier 3.
