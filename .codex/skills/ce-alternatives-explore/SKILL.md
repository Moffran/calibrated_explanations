---
name: ce-alternatives-explore
description: >
  Generate and interpret CE alternative and counterfactual explanations, including ensured filters and alternatives-specific plot workflows.
---

## Inputs

- **`content`** (text, required): The input relevant to this skill. See instructions for details.

## Output Format

Format: `markdown`

Required sections:
- output

# Ce Alternatives Explore — Core Instructions

# CE Alternatives Explorer

You are producing alternative calibrated explanations — counterfactual rules
that show what feature changes would produce a different prediction.

## Preconditions — Fail Fast Here

Before calling any explore entry point, verify all three conditions.
**If any check fails, stop and resolve it — do not proceed to explore calls.**

```python
assert isinstance(explainer, WrapCalibratedExplainer), (
    "CE-First violation: use WrapCalibratedExplainer, not a subclass or raw CalibratedExplainer"
)
assert explainer.fitted is True, (
    "Explainer not fitted — call explainer.fit(x_proper, y_proper) first"
)
assert explainer.calibrated is True, (
    "Explainer not calibrated — call explainer.calibrate(x_cal, y_cal) first"
)
```

If the fitted or calibrated state is missing, invoke `ce-pipeline-builder` and
complete the `fit → calibrate` lifecycle before returning here. Do not soften or
skip these assertions.

For all post-generation interaction (plot, narrative, `add_conjunctions`,
`filter_rule_sizes`, `filter_features`) see `ce-explain-interact`.

Load `references/alternatives_api_reference.md` for the ensured framework,
plot styles, ranking controls, and conjunction API.

---

## Two Entry Points

### Standard path
```python
alternatives = explainer.explore_alternatives(X_query)
```

### Explaining all classes (multiclass)
```python
multi_alts = explainer.explore_alternatives(X_query, multi_labels_enabled=True)
```

### Guarded path (interval plausibility filter — ADR-032)
```python
alternatives = explainer.explore_guarded_alternatives(X_query)
```

Guarded variant: candidate alternative rules whose probe points are
non-conforming to calibration data are filtered out of emitted explanations.
Use when you need plausibility filtering for hypothetical perturbations.
This is not an instance-level OOD detection API.

---

## Threshold Semantics

For **classification** — change the decision boundary:
```python
alternatives = explainer.explore_alternatives(X_query, threshold=0.7)
```

For **regression** — see `ce-regression-intervals` for full semantics:
```python
alternatives = explainer.explore_alternatives(X_query, threshold=50.0)
alternatives = explainer.explore_alternatives(X_query, threshold=(40.0, 60.0))
```

---

## Output Types

```
AlternativeExplanations      (collection — returned by explore_alternatives)
   [i] -> AlternativeExplanation  (per-instance)
```

The prediction dict structure and interval invariant are identical to factual
explanations (`pred['low'] <= pred['predict'] <= pred['high']`), but the
*direction* of the rules is counterfactual (opposing, not supporting).

---

## Out of Scope

- Factual / supporting rule generation (see `ce-factual-explain`).
- Regression interval configuration (see `ce-regression-intervals`).
- Generic plot / narrative / filter API (see `ce-explain-interact`).
- Building the pipeline (see `ce-pipeline-builder`).

## Evaluation Checklist

- [ ] `WrapCalibratedExplainer` instance confirmed (not raw `CalibratedExplainer` or subclass).
- [ ] `explainer.fitted is True` asserted before explore call — fail fast if not.
- [ ] `explainer.calibrated is True` asserted before explore call — fail fast if not.
- [ ] Correct variant (`explore_alternatives` vs `explore_alternatives(..., guarded=True)`).
- [ ] Threshold provided if the user wants boundary-crossing alternatives for regression.
- [ ] Ensured-framework filter selected appropriately for the use case.
- [ ] `only_ensured=True` used when narrower-uncertainty alternatives are required.
- [ ] `include_potential` set to `True`/`False` per user intent.
- [ ] Plot style is `"triangular"` or `"ensured"` (not `"regular"`) when showing alternatives in a confidence-uncertainty view.
- [ ] `rnk_metric="ensured"` with appropriate `rnk_weight` when ranking by output/uncertainty trade-off.
- [ ] Interval invariant `low <= predict <= high` verified on at least one output.
- [ ] No uncalibrated output returned.


## Self-Check Before Responding

- [ ] `WrapCalibratedExplainer` instance confirmed (not raw `CalibratedExplainer` or subclass).
- [ ] `explainer.fitted is True` asserted before explore call — fail fast if not.
- [ ] `explainer.calibrated is True` asserted before explore call — fail fast if not.
- [ ] Correct variant (`explore_alternatives` vs `explore_alternatives(..., guarded=True)`).
- [ ] Threshold provided if the user wants boundary-crossing alternatives for regression.
- [ ] Ensured-framework filter selected appropriately for the use case.
- [ ] `only_ensured=True` used when narrower-uncertainty alternatives are required.
- [ ] `include_potential` set to `True`/`False` per user intent.
- [ ] Plot style is `"triangular"` or `"ensured"` (not `"regular"`) when showing alternatives in a confidence-uncertainty view.
- [ ] `rnk_metric="ensured"` with appropriate `rnk_weight` when ranking by output/uncertainty trade-off.
- [ ] Interval invariant `low <= predict <= high` verified on at least one output.
- [ ] No uncalibrated output returned.
