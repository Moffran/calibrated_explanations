# Mondrian Calibration Code Examples

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

## Minimum bin size check

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
