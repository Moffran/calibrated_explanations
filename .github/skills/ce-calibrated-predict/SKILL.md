---
name: ce-calibrated-predict
description: >
  Produce calibrated predict and predict_proba outputs, with optional uncertainty intervals, without generating explanations.
---

## Inputs

- **`content`** (text, required): The input relevant to this skill. See instructions for details.

## Output Format

Format: `markdown`

Required sections:
- output

# Ce Calibrated Predict — Core Instructions

# CE Calibrated Predict

You are obtaining calibrated predictions **without explanations** from a CE
explainer. This is the lightest entry point — no explanation rules are
generated, just the calibrated prediction and (optionally) its uncertainty
interval.

## Preconditions — Fail Fast Here

Before calling `predict` or `predict_proba`, verify all three conditions.
**If any check fails, stop — do not proceed.**

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

If the lifecycle is incomplete, invoke `ce-pipeline-builder` first.

---

## Quick reference

| Method | Returns | Use for |
|---|---|---|
| `predict(x)` | Point prediction (class or y-value) | Classification and regression |
| `predict(x, uq_interval=True)` | `(prediction, (low, high))` | Point pred + uncertainty bounds |
| `predict_proba(x)` | Class probability array | Classification |
| `predict_proba(x, uq_interval=True)` | `(proba, (low, high))` | Probability + bounds |
| `predict_proba(x, threshold=t)` | `P(y ≤ t)` | Probabilistic regression |
| `predict_proba(x, threshold=t, uq_interval=True)` | `(P(y ≤ t), (low, high))` | Regression with probability bounds |

---

## Classification — `predict`

```python
# Point prediction (returns class labels for classification)
y_hat = explainer.predict(x_test)

# With uncertainty interval (returns tuple)
y_hat, (low, high) = explainer.predict(x_test, uq_interval=True)
# Interpretation: low ≤ p(class) ≤ high (Venn-Abers probability bounds)
#   Binary: p(class=1) bounds (always positive class)
#   Multiclass: p(class=y_hat) bounds (predicted class)

# Force uncalibrated predictions (bypasses calibration; warns if calibrated)
y_uncal = explainer.predict(x_test, calibrated=False)
```

---

## Classification — `predict_proba`

```python
# Calibrated probability array (shape: [n, n_classes])
proba = explainer.predict_proba(x_test)

# With uncertainty interval (tuple of arrays)
proba, (low, high) = explainer.predict_proba(x_test, uq_interval=True)
#   Binary: 1D probability arrays (low_1, high_1) for positive class 1
#   Multiclass: 2D arrays (n_samples, n_classes) for all classes

# Uncalibrated (uses underlying learner's predict_proba directly)
proba_uncal = explainer.predict_proba(x_test, calibrated=False)
```

---

## Regression — `predict`

```python
# Calibrated point prediction (CPS median by default)
y_hat = explainer.predict(x_test)

# With 90% conformal interval (target variable space)
# Interpretation: low ≤ y_hat ≤ high (e.g., in dollars or years)
y_hat, (low, high) = explainer.predict(x_test, uq_interval=True)

# Custom percentile interval
y_hat, (low, high) = explainer.predict(
    x_test,
    uq_interval=True,
    low_high_percentiles=(10, 90),
)
```

---

## Regression — `predict_proba` (thresholded)

```python
# P(y ≤ 50.0) for each instance — scalar threshold
proba = explainer.predict_proba(x_test, threshold=50.0)

# With uncertainty interval
proba, (low, high) = explainer.predict_proba(
    x_test, threshold=50.0, uq_interval=True
)
# Interpretation: low ≤ proba ≤ high are probability bounds (not y-space)

# Two-sided window: P(40.0 < y ≤ 60.0)
proba = explainer.predict_proba(x_test, threshold=(40.0, 60.0))
```

---

## Conditional predictions (Mondrian bins)

If the explainer was calibrated with `mc=` or `bins=`, pass the same group
assignments at predict time:

```python
# Predict with group-specific calibration
y_hat = explainer.predict(x_test, bins=group_labels_test)
proba = explainer.predict_proba(x_test, bins=group_labels_test)
```

See `ce-mondrian-conditional` for step-by-step group calibration setup.

---

## Reject-aware predictions

```python
from calibrated_explanations.core.reject.policy import RejectPolicy

# Returns RejectResult envelope instead of plain array
result = explainer.predict(x_test, reject_policy=RejectPolicy.FLAG)
y_hat = result.prediction
rejected = result.rejected  # bool mask
```

See `ce-reject-policy` for full policy documentation.

---

## Uncalibrated escape hatch — non-canonical output

`calibrated=False` is an **explicit opt-out**, not a canonical output mode.
Never present it as a default, a shortcut, or a fallback without a visible warning.

```python
# Explicit opt-out — only use when the caller understands they are bypassing calibration:
y_uncal = explainer.predict(x_test, calibrated=False)
proba_uncal = explainer.predict_proba(x_test, calibrated=False)
```

Rules for uncalibrated usage:
- Never return `calibrated=False` output in a CE-First pipeline as a normal result.
- Never document it as the first example or the recommended path.
- Always label uncalibrated results clearly so downstream consumers know the semantics differ.

Calling `predict` / `predict_proba` with the default `calibrated=True` on an
**uncalibrated** explainer emits a `UserWarning` and falls back to the underlying
learner. This warning is mandatory (fallback visibility policy — `ce-fallback-impl`).

```python
# Before calibrate() — will warn and return uncalibrated output:
with pytest.warns(UserWarning):
    y_hat = explainer.predict(x_test)
```

Do not suppress this warning. Do not treat the result as a calibrated output.

---

## Choosing between predict and explain_factual

| Need | Use |
|---|---|
| Calibrated prediction only (fastest) | `predict()` or `predict_proba()` |
| Prediction + factual rules | `explain_factual()` (see `ce-factual-explain`) |
| Prediction + alternatives | `explore_alternatives()` (see `ce-alternatives-explore`) |
| Prediction set for reject decision | `predict_reject()` (see `ce-reject-policy`) |

`predict` and `predict_proba` do **not** generate explanation rules; they are
significantly cheaper than the explain* entry points.

---

## Evaluation Checklist

- [ ] `WrapCalibratedExplainer` instance confirmed (not raw `CalibratedExplainer` or subclass).
- [ ] `explainer.fitted is True` asserted — fail fast if not.
- [ ] `explainer.calibrated is True` asserted — fail fast if not.
- [ ] Return type identified: array (no `uq_interval`) vs tuple (with `uq_interval`).
- [ ] For regression with `uq_interval=True`: invariant `low ≤ predict ≤ high` verified.
- [ ] For regression `predict_proba(threshold=t)`: result is a probability (0–1), not y-space.
- [ ] Mondrian `bins=` passed at predict time if calibrated with `mc=`.
- [ ] `calibrated=False` not used as a default, shortcut, or undocumented result.
- [ ] Uncalibrated fallback warning handled (not suppressed) in tests.


## Self-Check Before Responding

- [ ] `WrapCalibratedExplainer` instance confirmed (not raw `CalibratedExplainer` or subclass).
- [ ] `explainer.fitted is True` asserted — fail fast if not.
- [ ] `explainer.calibrated is True` asserted — fail fast if not.
- [ ] Return type identified: array (no `uq_interval`) vs tuple (with `uq_interval`).
- [ ] For regression with `uq_interval=True`: invariant `low ≤ predict ≤ high` verified.
- [ ] For regression `predict_proba(threshold=t)`: result is a probability (0–1), not y-space.
- [ ] Mondrian `bins=` passed at predict time if calibrated with `mc=`.
- [ ] `calibrated=False` not used as a default, shortcut, or undocumented result.
- [ ] Uncalibrated fallback warning handled (not suppressed) in tests.