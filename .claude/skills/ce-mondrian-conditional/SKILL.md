---
name: ce-mondrian-conditional
description: >
  Configure Mondrian (conditional) calibration for group-specific uncertainty
  estimates and fairness-aware predictions and explanations. Use when asked
  about 'conditional calibration', 'Mondrian', 'MondrianCategorizer',
  'fairness', 'bias', 'group-specific calibration', 'protected attribute',
  'subgroup calibration', 'bins parameter', 'demographic parity',
  'conditional conformal', 'per-group intervals', 'fairness-aware XAI',
  'conditional explanations'. Covers the full calibrate → predict → explain
  workflow for Mondrian calibration using both the bins parameter and
  MondrianCategorizer.
---

# CE Mondrian Conditional

You are setting up Mondrian (conditional) calibration, which partitions
calibration data into subgroups so that each group receives its own
uncertainty estimate. This reveals group-specific prediction quality and is
the foundational technique for fairness-aware deployments in CE.

**Research**: [Conditional Calibrated Explanations (xAI 2024)](https://link.springer.com/chapter/10.1007/978-3-031-63787-2_17)

---

## Why Mondrian matters for fairness

Without conditional calibration, the CPS/Venn-Abers calibrator averages over
all calibration instances. A minority group with harder prediction patterns
may receive the same interval width as an easy majority group, hiding bias.

Mondrian splits calibration by a grouping key and fits a separate calibrator
per bin. Resulting intervals are:
- **Narrower** for groups the model predicts reliably.
- **Wider** for groups the model predicts poorly.

This makes uncertainty visible at the group level, which is a prerequisite
for auditing fairness.

---

## Option A — Inline `bins` array (quickest)

Pass an array of group labels directly at calibrate time. This is the
simplest approach when bins are known before calibration.

```python
import numpy as np
from calibrated_explanations import WrapCalibratedExplainer

explainer = WrapCalibratedExplainer(model)
explainer.fit(x_proper, y_proper)

# Create group labels for calibration set (e.g., from a protected attribute)
# Must be integers or categories
group_cal = x_cal[:, gender_feature_idx].astype(int)   # e.g. 0 / 1

explainer.calibrate(
    x_cal,
    y_cal,
    feature_names=feature_names,
    bins=group_cal,          # Mondrian bins for calibration set
)

# Explain with group-specific uncertainty
group_test = x_test[:, gender_feature_idx].astype(int)
explanations = explainer.explain_factual(x_test, bins=group_test)
```

**Rule**: the `bins` array at explain/predict time must use the **same
label space** as at calibrate time.

---

## Option B — `MondrianCategorizer` (recommended for continuous features)

`MondrianCategorizer` from `crepes.extras` auto-bins a continuous feature
(or a callable) into discrete groups. Use this when grouping is based on a
continuous score like model confidence or age.

```python
from crepes.extras import MondrianCategorizer

explainer.fit(x_proper, y_proper)

# Auto-bin by the model's predicted confidence on the calibration set
mc = MondrianCategorizer()
mc.fit(x_cal, f=explainer.learner.predict_proba, no_bins=5)

explainer.calibrate(x_cal, y_cal, mc=mc, feature_names=feature_names)

# Explain — mc knows how to bin test instances automatically
explanations = explainer.explain_factual(x_test)
predictions  = explainer.predict(x_test)
```

### `MondrianCategorizer` constructor options

```python
# Auto-bin a callable on x_cal
mc = MondrianCategorizer()
mc.fit(x_cal, f=callable_returning_scores, no_bins=5)

# Auto-bin a single feature column
mc = MondrianCategorizer()
mc.fit(x_cal[:, feature_idx])

# Explicit bin boundaries (e.g., percentiles of age)
mc = MondrianCategorizer(bins=[0, 25, 50, 75, 100])
mc.fit(x_cal[:, age_idx])
```

---

## Option C — Lambda as `mc` (ad-hoc, no object)

For one-off scripts you may pass a callable directly as `mc`:

```python
explainer.calibrate(
    x_cal, y_cal,
    mc=lambda x: (explainer.learner.predict(x) > 0.5).astype(int),
)

# At explain/predict time, the lambda is called on the input to determine bins automatically
explanations = explainer.explain_factual(x_test)
```

---

## Calibration → predict → explain consistency rules

| Step | Bins argument |
|---|---|
| `calibrate(...)` | `mc=` (MondrianCategorizer or callable) OR `bins=` (integer array) |
| `predict(x, ...)` | nothing if `mc` was used at calibrate time OR `bins=group_labels_test` |
| `predict_proba(x, ...)` | same as above |
| `explain_factual(x, ...)` | same as above |
| `explore_alternatives(x, ...)` | same as above |

**CRITICAL**: always pass `bins=` at inference time whenever the explainer
was calibrated with Mondrian bins. Omitting it silently falls back to global
calibration, which defeats fairness analysis.

---

## Minimum bin size warning

Mondrian calibration splits the calibration set by group. Too few samples
per bin leads to unreliable or very wide intervals.

> **Rule of thumb**: aim for ≥ 30–50 calibration samples per bin.

Check bin sizes before calibrating:
```python
import numpy as np

group_cal = x_cal[:, protected_idx].astype(int)
unique, counts = np.unique(group_cal, return_counts=True)
for g, n in zip(unique, counts):
    if n < 30:
        print(f"WARN: group {g} has only {n} calibration samples — intervals may be unstable")
```

---

## Fairness analysis workflow

After calibrating with Mondrian bins, compare interval widths and coverage
across protected groups:

```python
import numpy as np

group_test = x_test[:, protected_idx].astype(int)
explanations = explainer.explain_factual(x_test, bins=group_test)

# Per-group average interval width
for group_id in np.unique(group_test):
    mask = group_test == group_id
    widths = [
        exp.prediction["high"] - exp.prediction["low"]
        for exp, in_group in zip(explanations, mask)
        if in_group
    ]
    print(f"Group {group_id}: mean width = {np.mean(widths):.3f}")
```

Systematic differences in interval width indicate that the model is less
reliable for some groups — a signal for further fairness audit.

---

## Comparing global vs conditional calibration

```python
# Global calibration (no Mondrian)
explainer_global = WrapCalibratedExplainer(model)
explainer_global.fit(x_proper, y_proper)
explainer_global.calibrate(x_cal, y_cal)

# Mondrian calibration (per group)
explainer_mondrian = WrapCalibratedExplainer(model)
explainer_mondrian.fit(x_proper, y_proper)
mc = MondrianCategorizer(); mc.fit(x_cal, f=explainer_mondrian.learner.predict_proba, no_bins=5)
explainer_mondrian.calibrate(x_cal, y_cal, mc=mc)

# Compare interval widths per group
for gid in np.unique(group_test):
    mask = group_test == gid
    global_w = float(np.mean([
        e.prediction["high"] - e.prediction["low"]
        for e, m in zip(explainer_global.explain_factual(x_test[mask]), [True]*mask.sum())
        if m
    ]))
    mondrian_w = float(np.mean([
        e.prediction["high"] - e.prediction["low"]
        for e, m in zip(explainer_mondrian.explain_factual(x_test[mask], bins=group_test[mask]), [True]*mask.sum())
        if m
    ]))
    print(f"Group {gid}: global={global_w:.3f}  mondrian={mondrian_w:.3f}")
```

---

## Out of Scope

- DifficultyEstimator (per-instance sigma scaling for regression; see `ce-regression-intervals`).
- Reject policies (deciding whether to defer uncertain predictions; see `ce-reject-policy`).
- Fairness constraint enforcement (CE reveals uncertainty; it does not enforce fairness automatically).

---

## Evaluation Checklist

- [ ] Group labels at explain/predict time match the label space used at calibrate time.
- [ ] Minimum bin size verified (≥ 30 samples per bin recommended).
- [ ] Both `calibrate(mc=...)` and `explain_factual(bins=...)` consistently use the same categorizer.
- [ ] If comparing global vs Mondrian: separate `WrapCalibratedExplainer` instances used.
- [ ] Per-group interval widths inspected to surface differential uncertainty.
- [ ] Coverage rate verified per group (not just pooled).
