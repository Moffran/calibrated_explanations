# Guarded explanations

Guarded explanations keep only in-distribution perturbations.

Semantics and non-guarantees are defined in
{doc}`calibrated_interval_semantics`.

## What guarded mode changes

- Uses conformal filtering on perturbations.
- Keeps calibrated prediction semantics.
- Removes non-conforming rule candidates.

## Core calls

```python
guarded_factual = explainer.explain_guarded_factual(X, significance=0.1)
guarded_alts = explainer.explore_guarded_alternatives(X, significance=0.1)
```

## Guarded audit

```python
audit = guarded_factual.get_guarded_audit()
```

The audit payload reports tested, conforming, removed, and emitted interval
candidates.

## Parameters

| Parameter | Meaning |
| --- | --- |
| `significance` | Lower values apply stricter filtering. |
| `merge_adjacent` | Merge adjacent conforming bins when enabled. |
| `n_neighbors` | KNN neighbors for conformity scoring. |
| `normalize_guard` | Apply per-feature normalization before distance calculations. |

## Related pages

- {doc}`../../get-started/quickstart_guarded`
- {doc}`alternatives`
- {doc}`../how-to/interpret_explanations`

Entry-point tier: Tier 3.
