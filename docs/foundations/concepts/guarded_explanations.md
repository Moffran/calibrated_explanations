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

## Limitations

**Interval-level false positive rate (FPR).**
The conformal guarantee operates at the representative-point level, not over
the full interval.  Empirically, the interval-level FPR can exceed the nominal
significance level, especially for wide bins whose representative point is
near the distribution boundary.  Do not treat `significance` as a certified
per-interval false positive bound.

**Guard tests interval plausibility, not instance OOD-ness.**
The guard evaluates perturbed instances — copies of the test instance with
one feature replaced by the interval's representative value (a calibration-derived
median or percentile).  It does not directly test whether the test instance
itself is out-of-distribution.  When the feature being replaced is the primary
source of distributional shift, the resulting probe is in-distribution by
construction and the guard cannot detect the OOD-ness of the original instance.
In low-dimensional settings this is especially pronounced: for a 1-feature
dataset, replacing the only feature with a calibration representative makes
every probe in-distribution regardless of how far the test instance lies from
the calibration support.  The guard therefore measures *interval plausibility*
(is this hypothetical perturbation a plausible input?) rather than *instance
OOD detection* (is this test instance OOD?).  To test whether a test instance
is itself OOD, use ``InDistributionGuard.is_conforming(x_instance)`` directly.

## Guarded audit

```python
audit = guarded_factual.get_guarded_audit()
```

The audit payload reports tested, conforming, removed, and emitted
guarded interval candidates.

- `intervals_conforming`: candidates accepted by the shipped guard rule.
- `intervals_removed_guard`: candidates rejected by the shipped guard rule.

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
